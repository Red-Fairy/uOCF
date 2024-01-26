from itertools import chain

import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection
from .model import Decoder, raw2outputs
from .model_general import dualRouteEncoderSeparate, singleRouteEncoder
from util.util import AverageMeter
from sklearn.metrics import adjusted_rand_score
import lpips
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr


class uorfEvalDinoModel(BaseModel):

	@staticmethod
	def modify_commandline_options(parser, is_train=True):
		"""Add new model-specific options and rewrite default values for existing options.

		Parameters:
			parser -- the option parser
			is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

		Returns:
			the modified parser.
		"""
		parser.add_argument('--num_slots', metavar='K', type=int, default=8, help='Number of supported slots')
		parser.add_argument('--z_dim', type=int, default=64, help='Dimension of individual z latent per slot')
		parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
		parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
		parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
		parser.add_argument('--obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects')
		parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
		parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
		parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
		parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
		parser.add_argument('--input_size', type=int, default=64)
		parser.add_argument('--frustum_size', type=int, default=128, help='Size of rendered images')
		parser.add_argument('--near_plane', type=float, default=6)
		parser.add_argument('--far_plane', type=float, default=20)
		parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
		parser.add_argument('--dual_route_encoder', action='store_true', help='use dual route encoders')
		parser.add_argument('--shape_dim', type=int, default=48, help='shape dimension')
		parser.add_argument('--color_dim', type=int, default=16, help='color dimension')

		parser.set_defaults(batch_size=1, lr=3e-4, niter_decay=0,
							dataset_mode='multiscenes', niter=1200, custom_lr=True, lr_policy='warmup')

		parser.set_defaults(exp_id='run-{}'.format(time.strftime('%Y-%m-%d-%H-%M-%S')))

		return parser

	def __init__(self, opt):
		"""Initialize this model class.

		Parameters:
			opt -- training/test options

		A few things can be done here.
		- (required) call the initialization function of BaseModel
		- define loss function, visualization images, model names, and optimizers
		"""
		BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
		self.loss_names = ['psnr', 'ssim', 'lpips'] if not self.opt.no_loss else []
		if not opt.recon_only and not opt.video:
			self.loss_names += ['ari', 'fgari', 'nvari']
		if opt.show_recon_stats and not opt.no_loss:
			self.loss_names += ['recon_psnr', 'recon_ssim', 'recon_lpips']
		n = opt.n_img_each_scene
		self.set_visual_names()
		self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
		render_size = (opt.render_size, opt.render_size)
		frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
		self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
									 frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
									 
		z_dim = opt.z_dim if not opt.dual_route_encoder else opt.color_dim + opt.shape_dim
		self.num_slots = opt.num_slots

		self.pretrained_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device).eval()
		dino_dim = 768
		if opt.dual_route_encoder:
			from .model_general import SlotAttentionAblation as SlotAttention
			self.netEncoder = networks.init_net(dualRouteEncoderSeparate(input_nc=3, pos_emb=opt.pos_emb, bottom=opt.bottom, shape_dim=opt.shape_dim, color_dim=opt.color_dim, input_dim=dino_dim),
					   								gpu_ids=self.gpu_ids, init_type='normal')
			if opt.color_in_attn:
				self.netSlotAttention = networks.init_net(
					SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, color_dim=0, iters=opt.attn_iter), gpu_ids=self.gpu_ids, init_type='normal')
			else:
				self.netSlotAttention = networks.init_net(
					SlotAttention(num_slots=opt.num_slots, in_dim=opt.shape_dim, slot_dim=opt.shape_dim, color_dim=opt.color_dim, iters=opt.attn_iter), gpu_ids=self.gpu_ids, init_type='normal')
		else:
			from .model import SlotAttention
			self.netEncoder = networks.init_net(singleRouteEncoder(pos_emb=opt.pos_emb, bottom=opt.bottom, out_dim=opt.z_dim, input_dim=dino_dim),
												gpu_ids=self.gpu_ids, init_type='normal')
			self.netSlotAttention = networks.init_net(
				SlotAttention(num_slots=opt.num_slots, in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter), gpu_ids=self.gpu_ids, init_type='normal')

		self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=z_dim, n_layers=opt.n_layer, locality=False,
													locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality), gpu_ids=self.gpu_ids, init_type='xavier')
		self.L2_loss = torch.nn.MSELoss()
		self.LPIPS_loss = lpips.LPIPS().to(self.device)

	def setup(self, opt):
		"""Load and print networks; create schedulers
		Parameters:
			opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
		"""
		if self.isTrain:
			self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
		if not self.isTrain or opt.continue_train:
			load_suffix = 'iter_{}'.format(opt.load_iter) if opt.load_iter > 0 else opt.epoch
			self.load_networks(load_suffix)
		self.print_networks(opt.verbose)

	def set_visual_names(self):
		n = self.opt.n_img_each_scene
		n_slot = self.opt.num_slots
		self.visual_names =	['gt_novel_view{}'.format(i+1) for i in range(n-1)] + \
							['x_rec{}'.format(i) for i in range(n)] + \
							['input_image'] + \
							['slot{}_view{}_unmasked'.format(k, i) for k in range(n_slot) for i in range(n)] + \
							['slot{}_view{}'.format(k, i) for k in range(n_slot) for i in range(n)]

		if self.opt.vis_mask:
			self.visual_names += ['mask_gt{}'.format(i) for i in range(n)]
		
		if self.opt.vis_render_mask or self.opt.vis_mask:
			self.visual_names += ['mask_rec{}'.format(i) for i in range(n)]

		if self.opt.vis_attn:
			self.visual_names += ['slot{}_attn'.format(k) for k in range(n_slot)]

		if self.opt.vis_disparity:
			self.visual_names += ['disparity_{}'.format(i) for i in range(n)]

		if self.opt.vis_disparity or self.opt.vis_render_disparity:
			self.visual_names += ['disparity_rec{}'.format(i) for i in range(n)]


	def set_input(self, input):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input: a dictionary that contains the data itself and its metadata information.
		"""
		self.x = input['img_data'].to(self.device)
		self.cam2world = input['cam2world'].to(self.device)
		if not self.opt.fixed_locality:
			self.cam2world_azi = input['azi_rot'].to(self.device)
		self.image_paths = input['paths']
		self.x_large = input['img_data_large'].to(self.device)
		if 'masks' in input:
			self.gt_masks = input['masks']
			self.mask_idx = input['mask_idx']
			self.fg_idx = input['fg_idx']
			self.obj_idxs = input['obj_idxs']  # NxKxHxW

	def forward(self, epoch=0):
		"""Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
		dev = self.x[0:1].device
		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

		# Encoding images
		with torch.no_grad():
			feat_size = 64
			feature_map = self.pretrained_encoder.forward_features(self.x_large[0:1].to(dev))['x_norm_patchtokens'].reshape(1, feat_size, feat_size, -1).permute([0, 3, 1, 2]).contiguous() # 1xCxHxW

		if not self.opt.dual_route_encoder:
			feature_map = self.netEncoder(feature_map)
			feat = feature_map.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC
			# Slot Attention
			z_slots, attn = self.netSlotAttention(feat)  # 1xKxC, 1xKxN
			z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN
		else:
			feature_map_shape, feature_map_color = self.netEncoder(feature_map,
					F.interpolate(self.x[0:1], size=self.opt.input_size, mode='bilinear', align_corners=False))  # Bxshape_dimxHxW, Bxcolor_dimxHxW
			feat_shape = feature_map_shape.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC
			feat_color = feature_map_color.flatten(start_dim=2).permute([0, 2, 1])  # BxNxC
			# Slot Attention
			if not self.opt.color_in_attn:
				z_slots, attn = self.netSlotAttention(feat_shape, feat_color=feat_color)  # 1xKxC, 1xKxN
			else:
				z_slots, attn = self.netSlotAttention(torch.cat([feat_shape, feat_color], dim=-1))  # 1xKxC, 1xKxN
			z_slots, attn = z_slots.squeeze(0), attn.squeeze(0)  # KxC, KxN
		K = attn.shape[0]

		cam2world = self.cam2world
		N = cam2world.shape[0]

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size
		frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world, partitioned=True)
		# 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
		x = self.x
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev)
		for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale
			sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
			sampling_coor_bg_ = frus_nss_coor_  # Px3

			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
			raws_ = raws_.view([N, D, H_, W_, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, D, H_, W_, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, D, H_, W_, 4])
			masked_raws[..., h::scale, w::scale, :] = masked_raws_
			unmasked_raws[..., h::scale, w::scale, :] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_

		if not self.opt.no_loss and not self.opt.video:
			x_recon_novel, x_novel = x_recon[1:], x[1:]
			self.loss_recon = self.L2_loss(x_recon_novel, x_novel)
			self.loss_lpips = self.LPIPS_loss(x_recon_novel, x_novel).mean()
			self.loss_psnr = compute_psnr(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)
			self.loss_ssim = compute_ssim(x_recon_novel/2+0.5, x_novel/2+0.5, data_range=1.)

		if not self.opt.no_loss and self.opt.show_recon_stats:
			x_recon_ori, x_ori = x_recon[0:1], x[0:1]
			self.loss_recon_psnr = compute_psnr(x_recon_ori/2+0.5, x_ori/2+0.5, data_range=1.)
			self.loss_recon_ssim = compute_ssim(x_recon_ori/2+0.5, x_ori/2+0.5, data_range=1.)
			self.loss_recon_lpips = self.LPIPS_loss(x_recon_ori, x_ori).mean()

		with torch.no_grad():
			attn = attn.detach().cpu()  # KxN
			H_, W_ = feature_map.shape[2], feature_map.shape[3]
			attn = attn.view(self.opt.num_slots, 1, H_, W_)
			for i in range(self.opt.n_img_each_scene):
				setattr(self, 'x_rec{}'.format(i), x_recon[i])
				if i == 0:
					setattr(self, 'input_image', x[i])
				else:
					setattr(self, 'gt_novel_view{}'.format(i), x[i])
			setattr(self, 'masked_raws', masked_raws.detach())
			setattr(self, 'unmasked_raws', unmasked_raws.detach())
			setattr(self, 'attn', attn)
			setattr(self, 'z_slots', z_slots.detach())

	def visual_cam2world(self, cam2world):
		'''
		render the scene given the cam2world matrix (1x4x4)
		must be called after forward()
		'''
		dev = self.x[0:1].device
		cam2world = cam2world.to(dev)
		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
		K = self.attn.shape[0]
		N = cam2world.shape[0]

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size
		frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world, partitioned=True)
		# 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev), torch.zeros([K, N, D, H, W, 4], device=dev)
		for (j, (frus_nss_coor_, z_vals_, ray_dir_)) in enumerate(zip(frus_nss_coor, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale
			sampling_coor_fg_ = frus_nss_coor_[None, ...].expand(K - 1, -1, -1)  # (K-1)xPx3
			sampling_coor_bg_ = frus_nss_coor_  # Px3

			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(sampling_coor_bg_, sampling_coor_fg_, self.z_slots, nss2cam0)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
			raws_ = raws_.view([N, D, H_, W_, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, D, H_, W_, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, D, H_, W_, 4])
			masked_raws[..., h::scale, w::scale, :] = masked_raws_
			unmasked_raws[..., h::scale, w::scale, :] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_

		with torch.no_grad():
			for i in range(self.opt.n_img_each_scene):
				setattr(self, 'x_rec{}'.format(i), x_recon[i])
			setattr(self, 'masked_raws', masked_raws.detach())
			setattr(self, 'unmasked_raws', unmasked_raws.detach())

	def compute_visuals(self):
		with torch.no_grad():
			cam2world = self.cam2world[:self.opt.n_img_each_scene]
			_, N, D, H, W, _ = self.masked_raws.shape
			masked_raws = self.masked_raws  # KxNxDxHxWx4
			unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
			mask_maps = []
			for k in range(self.num_slots):
				raws = masked_raws[k]  # NxDxHxWx4
				_, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
				raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
				rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True)
				mask_maps.append(mask_map.view(N, H, W))
				rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
				x_recon = rendered * 2 - 1
				for i in range(self.opt.n_img_each_scene):
					setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])
				
				setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

			for k in range(self.num_slots):
				raws = unmasked_raws[k]  # NxDxHxWx4
				_, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
				raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
				rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals, ray_dir, render_mask=True)
				mask_maps.append(mask_map.view(N, H, W))
				rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
				x_recon = rendered * 2 - 1
				for i in range(self.opt.n_img_each_scene):
					setattr(self, 'slot{}_view{}_unmasked'.format(k, i), x_recon[i])

			if not self.opt.recon_only:
				mask_maps = torch.stack(mask_maps)  # KxNxHxW
				mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW
				predefined_colors = []
				obj_idxs = self.obj_idxs  # Kx1xHxW
				gt_mask0 = self.gt_masks[0]  # 3xHxW
				for k in range(self.num_slots):
					mask_idx_this_slot = mask_idx[0:1] == k  # 1xHxW
					iou_this_slot = []
					for kk in range(self.num_slots):
						try:
							obj_idx = obj_idxs[kk, ...]  # 1xHxW
						except IndexError:
							break
						iou = (obj_idx & mask_idx_this_slot).type(torch.float).sum() / (obj_idx | mask_idx_this_slot).type(torch.float).sum()
						iou_this_slot.append(iou)
					target_obj_number = torch.tensor(iou_this_slot).argmax()
					target_obj_idx = obj_idxs[target_obj_number, ...].squeeze()  # HxW
					obj_first_pixel_pos = target_obj_idx.nonzero()[0]  # 2
					obj_color = gt_mask0[:, obj_first_pixel_pos[0], obj_first_pixel_pos[1]]
					predefined_colors.append(obj_color)
				predefined_colors = torch.stack(predefined_colors).permute([1,0])
				mask_visuals = predefined_colors[:, mask_idx]  # 3xNxHxW

				nvari_meter = AverageMeter()
				for i in range(N):
					setattr(self, 'render_mask{}'.format(i), mask_visuals[:, i, ...])
					setattr(self, 'gt_mask{}'.format(i), self.gt_masks[i])
					this_mask_idx = mask_idx[i].flatten(start_dim=0)
					gt_mask_idx = self.mask_idx[i]  # HW
					fg_idx = self.fg_idx[i]
					fg_idx_map = fg_idx.view([self.opt.frustum_size, self.opt.frustum_size])[None, ...]
					fg_map = mask_visuals[0:1, i, ...].clone()
					fg_map[fg_idx_map] = -1.
					fg_map[~fg_idx_map] = 1.
					setattr(self, 'bg_map{}'.format(i), fg_map)
					if i == 0:
						ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
						fg_ari = adjusted_rand_score(gt_mask_idx[fg_idx], this_mask_idx[fg_idx])
						self.loss_ari = ari_score
						self.loss_fgari = fg_ari
					else:
						ari_score = adjusted_rand_score(gt_mask_idx, this_mask_idx)
						nvari_meter.update(ari_score)
					self.loss_nvari = nvari_meter.val

	def backward(self):
		pass

	def optimize_parameters(self, ret_grad=False, epoch=0):
		"""Update network weights; it will be called in every training iteration."""
		self.forward(epoch)
		for opm in self.optimizers:
			opm.zero_grad()
		self.backward()
		avg_grads = []
		layers = []
		if ret_grad:
			for n, p in chain(self.netEncoder.named_parameters(), self.netSlotAttention.named_parameters(), self.netDecoder.named_parameters()):
				if p.grad is not None and "bias" not in n:
					with torch.no_grad():
						layers.append(n)
						avg_grads.append(p.grad.abs().mean().cpu().item())
		for opm in self.optimizers:
			opm.step()
		return layers, avg_grads

	def save_networks(self, surfix):
		"""Save all the networks to the disk.

		Parameters:
			surfix (int or str) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		super().save_networks(surfix)
		for i, opm in enumerate(self.optimizers):
			save_filename = '{}_optimizer_{}.pth'.format(surfix, i)
			save_path = os.path.join(self.save_dir, save_filename)
			torch.save(opm.state_dict(), save_path)

		for i, sch in enumerate(self.schedulers):
			save_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
			save_path = os.path.join(self.save_dir, save_filename)
			torch.save(sch.state_dict(), save_path)

	def load_networks(self, surfix):
		"""Load all the networks from the disk.

		Parameters:
			surfix (int or str) -- current epoch; used in he file name '%s_net_%s.pth' % (epoch, name)
		"""
		super().load_networks(surfix)

		if self.isTrain:
			for i, opm in enumerate(self.optimizers):
				load_filename = '{}_optimizer_{}.pth'.format(surfix, i)
				load_path = os.path.join(self.save_dir, load_filename)
				print('loading the optimizer from %s' % load_path)
				state_dict = torch.load(load_path, map_location=str(self.device))
				opm.load_state_dict(state_dict)

			for i, sch in enumerate(self.schedulers):
				load_filename = '{}_lr_scheduler_{}.pth'.format(surfix, i)
				load_path = os.path.join(self.save_dir, load_filename)
				print('loading the lr scheduler from %s' % load_path)
				state_dict = torch.load(load_path, map_location=str(self.device))
				sch.load_state_dict(state_dict)


if __name__ == '__main__':
	pass
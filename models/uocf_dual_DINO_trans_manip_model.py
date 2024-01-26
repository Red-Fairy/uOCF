from itertools import chain
from pickle import FALSE

import torch
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection, pixel2world
from .model_general import MultiDINOStackEncoder
from .transformer_attn import SlotAttentionTransformer, DecoderIPE
from .utils import *
from util.util import AverageMeter
from sklearn.metrics import adjusted_rand_score
import lpips
from piq import ssim as compute_ssim
from piq import psnr as compute_psnr
import numpy as np


class uocfDualDINOTransManipModel(BaseModel):

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
		parser.add_argument('--shape_dim', type=int, default=48, help='Dimension of individual z latent per slot')
		parser.add_argument('--color_dim', type=int, default=16, help='Dimension of individual z latent per slot texture')
		parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
		parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
		parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
		parser.add_argument('--obj_scale', type=float, default=4.5, help='slot-centric locality constraint')
		parser.add_argument('--world_obj_scale', type=float, default=4.5, help='locality constraint in world space')
		parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
		parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
		parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
		parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
		parser.add_argument('--input_size', type=int, default=64)
		parser.add_argument('--frustum_size', type=int, default=128, help='Size of rendered images')
		parser.add_argument('--near_plane', type=float, default=6)
		parser.add_argument('--far_plane', type=float, default=20)
		parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
		parser.add_argument('--fg_in_world', action='store_true', help='foreground objects are in world space')
		parser.add_argument('--manipulate_mode', default='translation', choices=['translation', 'removal'], help='manipulation mode')

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
		self.loss_names = ['psnr_moving', 'ssim_moving', 'lpips_moving'] if not opt.no_loss else []
		n = opt.n_img_each_scene
		self.set_visual_names()
		self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
		render_size = (opt.render_size, opt.render_size)
		frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
		self.intrinsics = np.loadtxt(os.path.join(opt.dataroot, 'camera_intrinsics_ratio.txt')) if opt.load_intrinsics else None
		self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
									 frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size, intrinsics=self.intrinsics)

		z_dim = opt.color_dim + opt.shape_dim
		self.num_slots = opt.num_slots

		self.pretrained_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(self.device).eval()
		dino_dim = 768
		self.netEncoder = MultiDINOStackEncoder(shape_dim=opt.shape_dim, color_dim=opt.color_dim, input_dim=dino_dim, 
						n_feat_layer=opt.n_feat_layers, global_bg_feature=opt.global_bg_feature, 
						kernel_size=opt.enc_kernel_size, mode=opt.enc_mode)

		self.netSlotAttention = SlotAttentionTransformer(num_slots=opt.num_slots, in_dim=opt.shape_dim+opt.color_dim if opt.color_in_attn else opt.shape_dim, 
					slot_dim=opt.shape_dim+opt.color_dim if opt.color_in_attn else opt.shape_dim, 
					color_dim=0 if opt.color_in_attn else opt.color_dim, momentum=opt.attn_momentum, pos_init=opt.pos_init,
					learnable_pos=not opt.no_learnable_pos, iters=opt.attn_iter, depth_scale_pred=self.opt.depth_scale_pred,
					camera_modulation=opt.camera_modulation, camera_dim=16)
							  
		self.netDecoder = DecoderIPE(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=z_dim, n_layers=opt.n_layer, locality=False,
													locality_ratio=opt.obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality,
													mlp_act=opt.dec_mlp_act, density_act=opt.dec_density_act,)

		self.netEncoder = self.netEncoder.to(self.device)
		self.netSlotAttention = self.netSlotAttention.to(self.device)
		self.netDecoder = self.netDecoder.to(self.device)
		
		self.L2_loss = torch.nn.MSELoss()
		self.LPIPS_loss = lpips.LPIPS().to(self.device)

	def set_visual_names(self):
		n = self.opt.n_img_each_scene
		n_slot = self.opt.num_slots
		self.visual_names = ['input_view', 'our_recon'] + \
							['ours_moved{}'.format(i) for i in range(n)]

		if not self.opt.no_loss:
			self.visual_names += ['gt_moved{}'.format(i) for i in range(n)] + ['fg_idx'] + \
								['individual_mask{}'.format(i) for i in range(n_slot-1)] 
			
		if self.opt.vis_mask:
			self.visual_names += ['gt_mask{}'.format(i) for i in range(n)] + \
								 ['render_mask{}'.format(i) for i in range(n)]
		if self.opt.vis_attn:
			self.visual_names += ['slot{}_attn'.format(k) for k in range(n_slot)]

		if self.opt.vis_disparity:
			self.visual_names += ['disparity_{}'.format(i) for i in range(n)] + \
								 ['disparity_rec{}'.format(i) for i in range(n)]

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

	def set_input(self, input):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input: a dictionary that contains the data itself and its metadata information.
		"""
		self.x = input['img_data'].to(self.device)
		self.x_large = input['img_data_large'].to(self.device)
		self.cam2world = input['cam2world'].to(self.device)
		if not self.opt.fixed_locality:
			self.cam2world_azi = input['azi_rot'].to(self.device)
		self.image_paths = input['paths']
		if 'intrinsics' in input:
			self.intrinsics = input['intrinsics'][0].to(self.device).squeeze(0) # overwrite the default intrinsics

		if 'masks' in input:
			self.gt_masks = input['masks']
			self.mask_idx = input['mask_idx']
			self.fg_idx = input['fg_idx']
			self.gt_moved = input['img_data_moved'].to(self.device)  # Nx3xHxW

		if self.opt.vis_disparity:
			self.disparity = input['depth'].to(self.device)

		if self.opt.manipulate_mode == 'translation' and 'movement' in input:
			self.movement = input['movement'].to(self.device)  # Nx3

	def encode(self, idx=0, return_global_feature=False):
		"""Encode the input image into a feature map.
		Parameters:
			idx: idx of the image to be encoded (typically 0, if position loss is used, we may use 1)
		Returns:
			feat_shape, feat_color (BxHxWxC), (BxHxWxC)
			class_token (BxC)
		"""
		feature_maps, class_tokens = [], []
		with torch.no_grad(): # B*C*H*W
			outputs = self.pretrained_encoder.get_intermediate_layers(self.x_large[idx:idx+1], n=self.opt.n_feat_layers, reshape=True, return_class_token=True)
		
		for feature_map, class_token in outputs:
			feature_maps.append(feature_map)
			if return_global_feature:
				class_tokens.append(class_token)
			else:
				class_tokens.append(None)

		feature_map_shape, feature_map_color, feature_global = self.netEncoder(feature_maps, feature_maps[-1], class_tokens[-1])  # Bxshape_dimxHxW, Bxcolor_dimxHxW

		feat_shape = feature_map_shape.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC
		feat_color = feature_map_color.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC

		return feat_shape, feat_color, feature_global
		

	def forward(self, epoch=0):
		"""Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
		dev = self.x[0:1].device
		cam2world_viewer = self.cam2world[0]
		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
		if self.opt.fixed_locality: # divide the translation part by self.opt.nss_scale
			nss2cam0 = torch.cat([torch.cat([nss2cam0[:, :3, :3], nss2cam0[:, :3, 3:4]/self.opt.nss_scale], dim=2), 
									nss2cam0[:, 3:4, :]], dim=1) # 1*4*4

		# Encoding images
		feat_shape, feat_color, feat_global = self.encode(0, return_global_feature=self.opt.global_bg_feature)

		# calculate camera cond (R, T, fx, fy, cx, cy) , 1*camera_dim, camera_dim=5, assert camera_normalized
		camExt = torch.cat([cam2world_viewer[:3, :3].flatten(), cam2world_viewer[:3, 3:4].flatten()/self.opt.nss_scale], dim=0)
		camInt = torch.Tensor([self.intrinsics[0, 0], self.intrinsics[1, 1], self.intrinsics[0, 2], self.intrinsics[1, 2]]).to(dev) \
			if self.intrinsics is not None else torch.Tensor([350./320., 350./240., 0., 0.]).to(dev)
		camera_modulation = torch.cat([camExt, camInt], dim=0).unsqueeze(0)

		# transformer attention
		z_slots, attn, fg_slot_position, fg_depth_scale = self.netSlotAttention(feat_shape, feat_color=feat_color, camera_modulation=camera_modulation, 
														  remove_duplicate=self.opt.remove_duplicate)  # 1xKxC, 1xKxN, 1xKx2, 1xKx1
		z_slots, attn, fg_slot_position, fg_depth_scale = z_slots.squeeze(0), attn.squeeze(0), fg_slot_position.squeeze(0), fg_depth_scale.squeeze(0)  # KxC, KxN, Kx2, Kx1

		K = z_slots.shape[0] # num_slots - n_remove
		self.num_slots = K

		cam2world = self.cam2world
		N = cam2world.shape[0]

		W, H, D = self.projection.frustum_size

		if not self.opt.scaled_depth:
			fg_slot_nss_position = pixel2world(fg_slot_position, cam2world_viewer, intrinsics=self.intrinsics, nss_scale=self.opt.nss_scale)  # Kx3
		else: 
			depth_scale = self.opt.depth_scale if self.opt.depth_scale is not None else torch.norm(cam2world_viewer[:3, 3:4])
			slot_depth = torch.ones_like(fg_slot_position[:, 0:1]).to(self.x.device) * depth_scale * fg_depth_scale
			fg_slot_nss_position = pixel2world(fg_slot_position, cam2world_viewer, intrinsics=self.intrinsics, 
													nss_scale=self.opt.nss_scale, depth=slot_depth) # Kx3

		scale = H // self.opt.render_size
		(mean, var), z_vals, ray_dir = self.projection.sample_along_rays(cam2world, partitioned=True, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
		# scale**2x(NxHxW)xDx3, scale**2x(NxHxW)xDx3x3, scale**2x(NxHxW)xD, scale**2x(NxHxW)x3
		x = self.x
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		if self.opt.vis_disparity:
			disparity_rec = torch.zeros([N, 1, H, W], device=dev)
		# print(z_vals.shape)
		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			# print(z_slots.shape, sampling_coor_bg_.shape, sampling_coor_fg_.shape, nss2cam0.shape, fg_slot_nss_position.shape)
			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(mean_, var_, z_slots, 
								 nss2cam0, fg_slot_nss_position, fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			masked_raws[:, :, h::scale, w::scale, ...] = masked_raws_
			unmasked_raws[:, :, h::scale, w::scale, ...] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_
			if self.opt.vis_disparity:
				disparity_rec_ = 1 / depth_map_.view(N, 1, H_, W_)
				disparity_rec[..., h::scale, w::scale] = disparity_rec_

		setattr(self, 'our_recon', x_recon[0])
		setattr(self, 'input_view', self.x[0])
		setattr(self, 'fg_slot_image_position', fg_slot_position.detach())
		setattr(self, 'fg_slot_nss_position', fg_slot_nss_position.detach())

		""" moving object """
		if not self.opt.no_loss:
			mask_maps = []
			for k in range(self.num_slots):
				raws = masked_raws[k]  # NxHxWxDx4
				_, z_vals_, ray_dir_ = self.projection.sample_along_rays(cam2world, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
				raws = raws.flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
				rgb_map, depth_map, _, mask_map = raw2outputs(raws, z_vals_, ray_dir_, render_mask=True, mip=True)
				mask_maps.append(mask_map.view(N, H, W))

			mask_maps = torch.stack(mask_maps)  # KxNxHxW
			mask_idx = mask_maps.cpu().argmax(dim=0)  # NxHxW, i.e., 1xHxW
			predefined_colors = torch.tensor([[0.2510, 1.0000, 0.0000, 0.0000, 1.0000, 1.0000, 0.0000, 0.7373],
											[0.2510, 0.0000, 1.0000, 0.0000, 1.0000, 0.0000, 1.0000, 0.7373],
											[0.2510, 0.0000, 0.0000, 1.0000, 0.0000, 1.0000, 1.0000, 0.7373]])
			mask_visuals = predefined_colors[:, mask_idx]  # 3xNxHxW
			mask_visuals = mask_visuals * 2 - 1

			ious = []
			fg_idx = self.fg_idx[0]
			for k in range(1, self.num_slots):
				mask_this_slot = mask_idx[0:1] == k
				mask_this_slot_visual = mask_this_slot.type(torch.float32)  # 1xHxW, {0,1}
				mask_this_slot_visaul = mask_this_slot_visual * 2 - 1
				setattr(self, 'individual_mask{}'.format(k - 1), mask_this_slot_visaul)
				setattr(self, 'fg_idx', fg_idx.type(torch.float32) * 2 - 1)
				iou = (fg_idx & mask_this_slot).type(torch.float).sum() / (fg_idx | mask_this_slot).type(torch.float).sum()
				print('{}-th slot IoU: {}'.format(k, iou))
				ious.append(iou)
			move_slot_idx = torch.tensor(ious).argmax()
			print('to move: {}'.format(move_slot_idx))

		else:
			move_slot_idx = 2
			self.movement = torch.tensor([1.5, 1.5, 0.], device=self.device)

		# insert objects
		# n_add = 6
		# fg_slot_nss_position_ = torch.zeros([K-1+n_add, 3], device=dev)
		# fg_slot_nss_position_[:-n_add] = fg_slot_nss_position
		# z_slots_ = torch.zeros([K+n_add, self.opt.shape_dim+self.opt.color_dim], device=dev)
		# z_slots_[:-n_add] = z_slots
		# K += n_add
		
		# fg_slot_nss_position_[-1] = fg_slot_nss_position[0] + torch.tensor([0.020, 0.020, 0.], device=self.device)
		# fg_slot_nss_position_[-2] = fg_slot_nss_position[3] + torch.tensor([0.020, 0.020, 0.], device=self.device)
		# fg_slot_nss_position_[-3] = fg_slot_nss_position[3] + torch.tensor([-0.015, -0.01, 0.], device=self.device)
		# fg_slot_nss_position_[-4] = fg_slot_nss_position[0] + torch.tensor([0.0, 0.015, 0.], device=self.device)
		# fg_slot_nss_position_[-5] = fg_slot_nss_position[3] + torch.tensor([0.0, 0.015, 0.], device=self.device)
		# fg_slot_nss_position_[-6] = fg_slot_nss_position[1] + torch.tensor([-0.01, 0, 0.], device=self.device)

		# fg_slot_nss_position_[-1][2] = 0.08
		# fg_slot_nss_position_[-2][2] = 0.16
		# fg_slot_nss_position_[-3][2] = 0.08
		# fg_slot_nss_position_[-4][2] = 0.15
		# fg_slot_nss_position_[-5][2] = 0.22
		# fg_slot_nss_position_[-6][2] = 0.06

		# z_slots_[-1] = z_slots[4]
		# z_slots_[-2] = z_slots[4]
		# z_slots_[-3] = z_slots[1]
		# z_slots_[-4] = z_slots[2]
		# z_slots_[-5] = z_slots[3]
		# z_slots_[-6] = z_slots[3]

		# stack objects
		# fg_slot_nss_position_ = torch.zeros_like(fg_slot_nss_position)
		# fg_slot_nss_position_[3][2] = 0.08
		# fg_slot_nss_position_[1][2] = 0.21
		# fg_slot_nss_position_[2][2] = 0.14
		# fg_slot_nss_position_[1:2] += torch.tensor([0.0, 0., 0.], device=self.device)
		# fg_slot_nss_position_[2:3] += torch.tensor([-0.005, 0., 0.], device=self.device)
		# fg_slot_nss_position_[3:4] += torch.tensor([0.02, 0., 0.], device=self.device)
		# z_slots_ = z_slots.clone()	

		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		
		fg_slot_nss_position[move_slot_idx] += self.movement / self.opt.nss_scale if self.opt.manipulate_mode == 'translation' else 100 # move it away
		z_slots_ = z_slots.clone()	
		fg_slot_nss_position_ = fg_slot_nss_position.clone()

		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(mean_, var_, z_slots_, 
								 nss2cam0, fg_slot_nss_position_, fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			
			masked_raws[:, :, h::scale, w::scale, ...] = masked_raws_
			unmasked_raws[:, :, h::scale, w::scale, ...] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_

		if not self.opt.no_loss:
			x_recon_novel, x_novel = x_recon, self.gt_moved
			self.loss_recon_moving = self.L2_loss(x_recon_novel, x_novel)
			self.loss_lpips_moving = self.LPIPS_loss(x_recon_novel, x_novel).mean()
			self.loss_psnr_moving = compute_psnr(x_recon_novel / 2 + 0.5, x_novel / 2 + 0.5, data_range=1.)
			self.loss_ssim_moving = compute_ssim(x_recon_novel / 2 + 0.5, x_novel / 2 + 0.5, data_range=1.)

		for i in range(self.opt.n_img_each_scene):
			if not self.opt.no_loss:
				setattr(self, 'gt_moved{}'.format(i), self.gt_moved[i])
			setattr(self, 'ours_moved{}'.format(i), x_recon[i])


	def visual_cam2world(self, cam2world):
		'''
		render the scene given the cam2world matrix (1x4x4)
		must be called after forward()
		'''
		dev = self.x[0:1].device
		cam2world = cam2world.to(dev)

		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
		if self.opt.fixed_locality: # divide the translation part by self.opt.nss_scale
			nss2cam0 = torch.cat([torch.cat([nss2cam0[:, :3, :3], nss2cam0[:, :3, 3:4]/self.opt.nss_scale], dim=2), 
									nss2cam0[:, 3:4, :]], dim=1) # 1*4*4
			
		K = self.attn.shape[0]
		N = cam2world.shape[0]

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size
		(mean, var), z_vals, ray_dir = self.projection.sample_along_rays(cam2world, partitioned=True, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
		#  4x(Nx(H/2)x(W/2))xDx3, 4x(Nx(H/2)x(W/2))xDx3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			# print(z_slots.shape, sampling_coor_bg_.shape, sampling_coor_fg_.shape, nss2cam0.shape, fg_slot_nss_position.shape)
			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(mean_, var_, self.z_slots, 
								 nss2cam0, self.fg_slot_nss_position, fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			masked_raws[..., h::scale, w::scale, :, :] = masked_raws_
			unmasked_raws[..., h::scale, w::scale, :, :] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True)
			# (NxHxW)x3, (NxHxW)
			rendered_ = rgb_map_.view(N, H_, W_, 3).permute([0, 3, 1, 2])  # Nx3xHxW
			rendered[..., h::scale, w::scale] = rendered_
			x_recon_ = rendered_ * 2 - 1
			x_recon[..., h::scale, w::scale] = x_recon_

		with torch.no_grad():
			# for i in range(self.opt.n_img_each_scene):
			for i in range(1):
				setattr(self, 'x_rec{}'.format(i), x_recon[i])
			setattr(self, 'masked_raws', masked_raws.detach())
			setattr(self, 'unmasked_raws', unmasked_raws.detach())

   
	def forward_position(self, fg_slot_image_position=None, fg_slot_nss_position=None, z_slots=None):
		assert fg_slot_image_position is None or fg_slot_nss_position is None
		x = self.x
		dev = x.device
		cam2world = self.cam2world
		N = cam2world.shape[0]

		nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()
		if self.opt.fixed_locality: # divide the translation part by self.opt.nss_scale
			nss2cam0 = torch.cat([torch.cat([nss2cam0[:, :3, :3], nss2cam0[:, :3, 3:4]/self.opt.nss_scale], dim=2), 
									nss2cam0[:, 3:4, :]], dim=1) # 1*4*4
		
		if fg_slot_image_position is not None:
			fg_slot_nss_position = pixel2world(fg_slot_image_position.to(dev), self.cam2world[0], intrinsics=self.intrinsics, nss_scale=self.opt.nss_scale)
		elif fg_slot_nss_position is not None:
			if fg_slot_nss_position.shape[1] == 2:
				fg_slot_nss_position = torch.cat([fg_slot_nss_position, torch.zeros_like(fg_slot_nss_position[:, :1])], dim=1)
			fg_slot_nss_position = fg_slot_nss_position.to(dev)
		else:
			fg_slot_nss_position = self.fg_slot_nss_position.to(dev)

		if z_slots is not None:
			z_slots = z_slots.to(dev)
		else:
			z_slots = self.z_slots.to(dev)

		K = z_slots.shape[0]

		W, H, D = self.projection.frustum_size
		scale = H // self.opt.render_size

		(mean, var), z_vals, ray_dir = self.projection.sample_along_rays(cam2world, partitioned=True, intrinsics=self.intrinsics if (self.intrinsics is not None and not self.opt.load_intrinsics) else None)
		# 4x(NxDx(H/2)x(W/2))x3, 4x(Nx(H/2)x(W/2))xD, 4x(Nx(H/2)x(W/2))x3
		x_recon, rendered, masked_raws, unmasked_raws = \
			torch.zeros([N, 3, H, W], device=dev), torch.zeros([N, 3, H, W], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev), torch.zeros([K, N, H, W, D, 4], device=dev)
		
		for (j, (mean_, var_, z_vals_, ray_dir_)) in enumerate(zip(mean, var, z_vals, ray_dir)):
			h, w = divmod(j, scale)
			H_, W_ = H // scale, W // scale

			# print(z_slots.shape, sampling_coor_bg_.shape, sampling_coor_fg_.shape, nss2cam0.shape, fg_slot_nss_position.shape)
			raws_, masked_raws_, unmasked_raws_, masks_ = self.netDecoder(mean_, var_, z_slots, 
								 nss2cam0, fg_slot_nss_position, fg_object_size=self.opt.fg_object_size/self.opt.nss_scale)  # (NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x4, Kx(NxHxWxD)x1
			raws_ = raws_.view([N, H_, W_, D, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
			masked_raws_ = masked_raws_.view([K, N, H_, W_, D, 4])
			unmasked_raws_ = unmasked_raws_.view([K, N, H_, W_, D, 4])
			masked_raws[..., h::scale, w::scale, :, :] = masked_raws_
			unmasked_raws[..., h::scale, w::scale, :, :] = unmasked_raws_
			rgb_map_, depth_map_, _ = raw2outputs(raws_, z_vals_, ray_dir_, mip=True)
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
			if fg_slot_image_position is not None:
				setattr(self, 'fg_slot_image_position', fg_slot_image_position.detach())
			setattr(self, 'fg_slot_nss_position', fg_slot_nss_position.detach())

	def compute_visuals(self, cam2world=None):
		pass

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
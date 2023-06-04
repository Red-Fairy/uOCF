from itertools import chain
from math import e
from sympy import N

import torch
from torch import nn, optim
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks
import os
import time
from .projection import Projection, pixel2world
from torchvision.transforms import Normalize
from .model_T_sam_fgmask import SlotAttention
from .model_general import DinoEncoder
from .model_general import DecoderFG as Decoder
from .utils import *

import torchvision

class uorfNoGanTDINOFGMaskModel(BaseModel):

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
        parser.add_argument('--z_dim', type=int, default=64, help='Dimension of latent vector')
        parser.add_argument('--attn_iter', type=int, default=3, help='Number of refine iteration in slot attention')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
        parser.add_argument('--nss_scale', type=float, default=7, help='Scale of the scene, related to camera matrix')
        parser.add_argument('--render_size', type=int, default=64, help='Shape of patch to render each forward process. Must be Frustum_size/(2^N) where N=0,1,..., Smaller values cost longer time but require less GPU memory.')
        parser.add_argument('--supervision_size', type=int, default=64)
        parser.add_argument('--world_obj_scale', type=float, default=4.5, help='Scale for locality on foreground objects in world coordinates')
        parser.add_argument('--obj_scale', type=float, default=3.5, help='Scale for locality on foreground objects in object-centric coordinates')
        parser.add_argument('--n_freq', type=int, default=5, help='how many increased freq?')
        parser.add_argument('--n_samp', type=int, default=64, help='num of samp per ray')
        parser.add_argument('--n_layer', type=int, default=3, help='num of layers bef/aft skip link in decoder')
        parser.add_argument('--weight_percept', type=float, default=0.006)
        parser.add_argument('--percept_in', type=int, default=100)
        parser.add_argument('--mask_in', type=int, default=0)
        parser.add_argument('--no_locality_epoch', type=int, default=1000)
        parser.add_argument('--locality_in', type=int, default=1000)
        parser.add_argument('--locality_full', type=int, default=0)
        parser.add_argument('--bottom', action='store_true', help='one more encoder layer on bottom')
        parser.add_argument('--input_size', type=int, default=64)
        parser.add_argument('--frustum_size', type=int, default=64)
        parser.add_argument('--frustum_size_fine', type=int, default=128)
        parser.add_argument('--attn_decay_steps', type=int, default=2e5)
        parser.add_argument('--coarse_epoch', type=int, default=600)
        parser.add_argument('--near_plane', type=float, default=6)
        parser.add_argument('--far_plane', type=float, default=20)
        parser.add_argument('--fixed_locality', action='store_true', help='enforce locality in world space instead of transformed view space')
        parser.add_argument('--fg_in_world', action='store_true', help='foreground objects are in world space')
        parser.add_argument('--dens_noise', type=float, default=1., help='Noise added to density may help in mitigating rank collapse')
        parser.add_argument('--invariant_in', type=int, default=0, help='when to start translation invariant decoding')
        parser.add_argument('--lr_encoder', type=float, default=6e-5, help='learning rate for encoder')
        parser.add_argument('--feature_aggregate', action='store_true', help='aggregate features from encoder')
        parser.add_argument('--surface_loss', action='store_true', help='surface loss')
        parser.add_argument('--weight_surface', type=float, default=0.1)
        parser.add_argument('--surface_in', type=int, default=0)
        parser.add_argument('--dino_type', type=str, default='dinov2_vitb14') # vitb14 or vitl14
        parser.add_argument('--centered', action='store_true', help='centered')

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
        self.loss_names = ['recon', 'perc']
        if opt.surface_loss:
            self.loss_names.append('surface')
            self.surfaceLoss = surfaceLoss()
        self.set_visual_names()
        self.model_names = ['Encoder', 'SlotAttention', 'Decoder']
        self.perceptual_net = get_perceptual_net().to(self.device)
        self.vgg_norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        render_size = (opt.render_size, opt.render_size)
        frustum_size = [self.opt.frustum_size, self.opt.frustum_size, self.opt.n_samp]
        self.projection = Projection(device=self.device, nss_scale=opt.nss_scale,
                                     frustum_size=frustum_size, near=opt.near_plane, far=opt.far_plane, render_size=render_size)
        frustum_size_fine = [self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp]
        self.projection_fine = Projection(device=self.device, nss_scale=opt.nss_scale,
                                          frustum_size=frustum_size_fine, near=opt.near_plane, far=opt.far_plane, render_size=render_size)

        z_dim = opt.z_dim

        if not opt.preextract:
            self.DinoViT = torch.hub.load('facebookresearch/dinov2', opt.dino_type).cuda()
            self.dino_dim = 768 if opt.dino_type == 'dinov2_vitb14' else 1024
            print("Pretrained DINO model loaded")
        self.netEncoder = networks.init_net(DinoEncoder(dino_dim=self.dino_dim, z_dim=z_dim), gpu_ids=self.gpu_ids, init_type='normal')
        if not opt.feature_aggregate:
            self.netSlotAttention = networks.init_net(
                SlotAttention(in_dim=z_dim, slot_dim=z_dim, iters=opt.attn_iter, centered=opt.centered), gpu_ids=self.gpu_ids, init_type='normal')
        else:
            assert False
            # self.netSlotAttention = networks.init_net(
            #     FeatureAggregate(in_dim=z_dim, out_dim=z_dim), gpu_ids=self.gpu_ids, init_type='normal')
        self.netDecoder = networks.init_net(Decoder(n_freq=opt.n_freq, input_dim=6*opt.n_freq+3+z_dim, z_dim=z_dim, n_layers=opt.n_layer,
                                                    locality_ratio=opt.world_obj_scale/opt.nss_scale, fixed_locality=opt.fixed_locality, # set locality_ratio to 1
                                                    project=opt.project, rel_pos=opt.relative_position, fg_in_world=opt.fg_in_world
                                                    ), gpu_ids=self.gpu_ids, init_type='xavier')

        if self.isTrain:  # only defined during training time
            requires_grad = lambda x: x.requires_grad
            params = chain(self.netEncoder.parameters(),self.netSlotAttention.parameters(), self.netDecoder.parameters())
            self.optimizer = optim.Adam(filter(requires_grad, params), lr=opt.lr)
            self.optimizers = [self.optimizer]

        self.L2_loss = nn.MSELoss()

    def set_visual_names(self):
        n = self.opt.n_img_each_scene
        n_slot = self.opt.num_slots
        self.visual_names = ['x{}'.format(i) for i in range(n)] + \
                            ['x_rec{}'.format(i) for i in range(n)] + \
                            ['slot{}_view{}'.format(k, i) for k in range(n_slot) for i in range(n)] + \
                            ['unmasked_slot{}_view{}'.format(k, i) for k in range(n_slot) for i in range(n)]
        if not self.opt.feature_aggregate:
            self.visual_names += ['slot{}_attn'.format(k) for k in range(n_slot)]

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
        self.x = input['img_data'].to(self.device) # N*3*H*W
        if self.opt.preextract:
            self.x_feats = input['img_feats'].to(self.device) # 1*H'*W'*C (H'=W'=64, C=1024)
        else:
            self.x_large = input['img_data_large'].to(self.device) # 1*3*H*W (H=W=1024)
        self.cam2world = input['cam2world'].to(self.device)
        self.masks = input['obj_idxs'].float().to(self.device) # K*1*H*W (no background mask)
        self.num_slots = self.masks.shape[0]
        self.bg_mask = input['bg_mask'].float().to(self.device) # N*1*H*W
        if not self.opt.fixed_locality:
            self.cam2world_azi = input['azi_rot'].to(self.device)

    def forward(self, epoch=0):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        self.weight_percept = self.opt.weight_percept if epoch >= self.opt.percept_in else 0
        dens_noise = self.opt.dens_noise if (epoch <= self.opt.percept_in and self.opt.fixed_locality) else 0
        self.loss_recon = 0
        self.loss_perc = 0
        if self.opt.surface_loss:
            self.loss_surface = 0
        dev = self.x[0:1].device
        cam2world_viewer = self.cam2world[0]
        nss2cam0 = self.cam2world[0:1].inverse() if self.opt.fixed_locality else self.cam2world_azi[0:1].inverse()

        # Encoding images
        if not self.opt.preextract:
            with torch.no_grad():
                feat_size = 64
                feature_map = self.DinoViT.forward_features(self.x_large[0:1].to(dev))['x_norm_patchtokens'].reshape(-1, feat_size, feat_size, self.dino_dim)  # 1xHxWxC
        else:
            feature_map = self.x_feats[0:1].to(dev)  # 1xHxWxC, C: 1024 for DinoViT_L
        feature_map = self.netEncoder(feature_map.permute([0, 3, 1, 2]).contiguous())  # BxCxHxW

        feat = feature_map.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC
        self.masks = F.interpolate(self.masks, size=feat.shape[1:3], mode='nearest')  # Kx1xHxW
    
        # Slot Attention
        use_mask = epoch < self.opt.mask_in
        if not self.opt.feature_aggregate:
            z_slots, fg_slot_position, attn = self.netSlotAttention(feat, self.masks, use_mask=use_mask)  # 1xKxC, 1xKx2, 1xKxN
            z_slots, fg_slot_position, attn = z_slots.squeeze(0), fg_slot_position.squeeze(0), attn.squeeze(0)  # KxC, Kx2, KxN
        else:
            z_slots, fg_slot_position = self.netSlotAttention(feat, self.masks, use_mask=use_mask)  # KxC, Kx2
            # z_slots, fg_slot_position = z_slots.squeeze(0), fg_slot_position.squeeze(0)  # KxC, Kx2
        fg_slot_nss_position = pixel2world(fg_slot_position, cam2world_viewer)  # Kx3
        
        K = z_slots.shape[0]
            
        cam2world = self.cam2world
        N = cam2world.shape[0]
        if self.opt.stage == 'coarse':
            frus_nss_coor, z_vals, ray_dir = self.projection.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            x = F.interpolate(self.x, size=self.opt.supervision_size, mode='bilinear', align_corners=False)
            self.z_vals, self.ray_dir = z_vals, ray_dir
        else:
            W, H, D = self.opt.frustum_size_fine, self.opt.frustum_size_fine, self.opt.n_samp
            start_range = self.opt.frustum_size_fine - self.opt.render_size
            rs = self.opt.render_size
            frus_nss_coor, z_vals, ray_dir = self.projection_fine.construct_sampling_coor(cam2world)
            # (NxDxHxW)x3, (NxHxW)xD, (NxHxW)x3
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor.view([N, D, H, W, 3]), z_vals.view([N, H, W, D]), ray_dir.view([N, H, W, 3])
            H_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            W_idx = torch.randint(low=0, high=start_range, size=(1,), device=dev)
            frus_nss_coor_, z_vals_, ray_dir_ = frus_nss_coor[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], z_vals[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :], ray_dir[..., H_idx:H_idx + rs, W_idx:W_idx + rs, :]
            frus_nss_coor, z_vals, ray_dir = frus_nss_coor_.flatten(0, 3), z_vals_.flatten(0, 2), ray_dir_.flatten(0, 2)
            x = self.x[:, :, H_idx:H_idx + rs, W_idx:W_idx + rs]
            self.z_vals, self.ray_dir = z_vals, ray_dir

        sampling_coor_fg = frus_nss_coor[None, ...].expand(K, -1, -1)  # KxPx3

        local_locality_ratio = 1 - min((epoch-self.opt.locality_in) / self.opt.locality_full, 1) * (1 - self.opt.obj_scale) if epoch >= self.opt.locality_in else None
        W, H, D = self.opt.supervision_size, self.opt.supervision_size, self.opt.n_samp
        invariant = epoch >= self.opt.invariant_in
        raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_fg, z_slots, nss2cam0, fg_slot_nss_position, dens_noise=dens_noise, invariant=invariant, local_locality_ratio=local_locality_ratio)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
        # raws, masked_raws, unmasked_raws, masks = self.netDecoder(sampling_coor_fg, z_slots, nss2cam0, fg_slot_nss_position, dens_noise=dens_noise, invariant=invariant)  # (NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x4, Kx(NxDxHxW)x1
        raws = raws.view([N, D, H, W, 4]).permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
        masked_raws = masked_raws.view([K, N, D, H, W, 4])
        unmasked_raws = unmasked_raws.view([K, N, D, H, W, 4])
        rgb_map, _, weights = raw2outputs(raws, z_vals, ray_dir)
        # (NxHxW)x3, (NxHxW)
        rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
        x_recon = rendered * 2 - 1

        self.loss_recon = self.L2_loss(x_recon, x)
        x_norm, rendered_norm = self.vgg_norm((x + 1) / 2), self.vgg_norm(rendered)
        rendered_feat, x_feat = self.perceptual_net(rendered_norm), self.perceptual_net(x_norm)
        self.loss_perc = self.weight_percept * self.L2_loss(rendered_feat, x_feat)
        if self.opt.surface_loss and epoch >= self.opt.surface_in:
            self.loss_surface = self.opt.weight_surface * self.surfaceLoss(weights)

        with torch.no_grad():
            if not self.opt.feature_aggregate:
                attn = attn.detach().cpu()  # KxN
                H_, W_ = feature_map.shape[2], feature_map.shape[3]
                attn = attn.view(self.opt.num_slots, 1, H_, W_)
                if H_ != H:
                    attn = F.interpolate(attn, size=[H, W], mode='bilinear')
                setattr(self, 'attn', attn)
            for i in range(self.opt.n_img_each_scene):
                setattr(self, 'x_rec{}'.format(i), x_recon[i])
                setattr(self, 'x{}'.format(i), x[i])
            setattr(self, 'masked_raws', masked_raws.detach())
            setattr(self, 'unmasked_raws', unmasked_raws.detach())
            

    def compute_visuals(self):
        with torch.no_grad():
            _, N, D, H, W, _ = self.masked_raws.shape
            masked_raws = self.masked_raws  # KxNxDxHxWx4
            unmasked_raws = self.unmasked_raws  # KxNxDxHxWx4
            for k in range(self.num_slots):
                raws = masked_raws[k]  # NxDxHxWx4
                z_vals, ray_dir = self.z_vals, self.ray_dir
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), x_recon[i])
                raws = unmasked_raws[k]  # (NxDxHxW)x4
                raws = raws.permute([0, 2, 3, 1, 4]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xDx4
                rgb_map, depth_map, _ = raw2outputs(raws, z_vals, ray_dir)
                rendered = rgb_map.view(N, H, W, 3).permute([0, 3, 1, 2])  # Nx3xHxW
                x_recon = rendered * 2 - 1
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), x_recon[i])
                if not self.opt.feature_aggregate:
                    setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

            for k in range(self.num_slots, self.opt.num_slots):
                # add dummy images
                for i in range(self.opt.n_img_each_scene):
                    setattr(self, 'slot{}_view{}'.format(k, i), torch.zeros_like(x_recon[i]))
                    setattr(self, 'unmasked_slot{}_view{}'.format(k, i), torch.zeros_like(x_recon[i]))
                setattr(self, 'slot{}_attn'.format(k), self.attn[k] * 2 - 1)

                

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        loss = self.loss_recon + self.loss_perc
        if self.opt.surface_loss:
            loss += self.loss_surface
        loss.backward()
        self.loss_perc = self.loss_perc / self.weight_percept if self.weight_percept > 0 else self.loss_perc

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
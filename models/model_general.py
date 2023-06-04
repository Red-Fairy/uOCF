import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd

from models.resnet import resnet34, resnet18
from .utils import PositionalEncoding, sin_emb, build_grid

class Encoder(nn.Module):
	def __init__(self, input_nc=3, z_dim=64, bottom=False, pos_emb=False):

		super().__init__()

		self.bottom = bottom
		print('Bottom for Encoder: ', self.bottom)

		input_nc = input_nc + 4 if pos_emb else input_nc
		self.pos_emb = pos_emb

		if self.bottom:
			self.enc_down_0 = nn.Sequential(nn.Conv2d(input_nc, z_dim, 3, stride=1, padding=1),
											nn.ReLU(True))
		self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc, z_dim, 3, stride=2 if bottom else 1, padding=1),
										nn.ReLU(True))
		self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
										nn.ReLU(True))
		self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
										nn.ReLU(True))
		self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
									  nn.ReLU(True),
									  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
									  nn.ReLU(True),
									  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
									#   nn.ReLU(True)
									  )
		
		

	def forward(self, x):
		"""
		input:
			x: input image, Bx3xHxW
		output:
			feature_map: BxCxHxW
		"""

		if self.pos_emb:
			W, H = x.shape[3], x.shape[2]
			X = torch.linspace(-1, 1, W)
			Y = torch.linspace(-1, 1, H)
			y1_m, x1_m = torch.meshgrid([Y, X])
			x2_m, y2_m = -x1_m, -y1_m  # Normalized distance in the four direction
			pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).unsqueeze(0)  # 1x4xHxW
			x_ = torch.cat([x, pixel_emb], dim=1)
		else:
			x_ = x

		if self.bottom:
			x_down_0 = self.enc_down_0(x_)
			x_down_1 = self.enc_down_1(x_down_0)
		else:
			x_down_1 = self.enc_down_1(x_)
		x_down_2 = self.enc_down_2(x_down_1)
		x_down_3 = self.enc_down_3(x_down_2)
		x_up_3 = self.enc_up_3(x_down_3)
		x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
		feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
		
		feature_map = feature_map
		return feature_map

class SAMViT(nn.Module):
	def __init__(self, sam_model):
		super(SAMViT, self).__init__()

		self.sam = sam_model.image_encoder
		self.sam.requires_grad_(False)
		self.sam.eval()

	def forward(self, x_sam):
		
		return self.sam(x_sam) # (B, 256, 64, 64)

class dualRouteEncoder(nn.Module):
	def __init__(self, bottom=False, pos_emb=False, input_nc=3, shape_dim=48, color_dim=16):
		super().__init__()

		self.Encoder = Encoder(bottom=bottom, z_dim=color_dim, pos_emb=pos_emb, input_nc=input_nc)

		vit_dim = 256
		self.shallow_encoder = nn.Sequential(nn.Conv2d(vit_dim, vit_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(vit_dim, shape_dim, 3, stride=1, padding=1))

	def forward(self, sam_feature, x):
		'''
		input:
			sam_feature: (B, 256, 64, 64)
			x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True
		output:
			spatial feature (B, shape_dim+color_dim, 64, 64)
		'''
		feat_color = self.Encoder(x)
		feat_shape = self.shallow_encoder(sam_feature)

		return torch.cat([feat_shape, feat_color], dim=1)

class dualRouteEncoderSeparate(nn.Module):
	def __init__(self, bottom=False, pos_emb=False, input_nc=3, shape_dim=48, color_dim=16, input_dim=256, hidden_dim=256):
		super().__init__()

		self.Encoder = Encoder(bottom=bottom, z_dim=color_dim, pos_emb=pos_emb, input_nc=input_nc)

		self.shallow_encoder = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(hidden_dim, shape_dim, 3, stride=1, padding=1))

	def forward(self, input_feat, x):
		'''
		input:
			input_feat: (B, input_dim, 64, 64)
			x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True
		output:
			spatial feature (B, shape_dim+color_dim, 64, 64)
		'''
		feat_color = self.Encoder(x)
		feat_shape = self.shallow_encoder(input_feat)

		return feat_shape, feat_color

class dualRouteEncoderSDSeparate(nn.Module):
	def __init__(self, bottom=False, pos_emb=False, input_nc=3, shape_dim=48, color_dim=16, input_dim=256, hidden_dim=256):
		super().__init__()

		self.Encoder = Encoder(bottom=bottom, z_dim=color_dim, pos_emb=pos_emb, input_nc=input_nc)

		self.conv1 = nn.Sequential(nn.Conv2d(512, 128, 1, stride=1, padding=0),
								   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.conv2 = nn.Sequential(nn.Conv2d(640, 128, 1, stride=1, padding=0),
									  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.conv3 = nn.Conv2d(512, 128, 1, stride=1, padding=0)

		self.out = nn.Sequential(nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
									nn.ReLU(True),
								nn.Conv2d(128, shape_dim, 3, stride=1, padding=1))

	def forward(self, input_feat, x):
		'''
		input:
			input_feat: list of Tensors: B*512*32*32, B*640*32*32, B*512*64*64
			output: B*z_dim*64*64
			x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True
		output:
			spatial feature (B, shape_dim+color_dim, 64, 64)
		'''
		feat_color = self.Encoder(x)

		x1 = self.conv1(input_feat[0])
		x2 = self.conv2(input_feat[1])
		x3 = self.conv3(input_feat[2])
		feat_shape = torch.cat([x1, x2, x3], dim=1)
		feat_shape = self.out(feat_shape)

		return feat_shape, feat_color

class SAMEncoder(nn.Module):
	def __init__(self, z_dim=64, vit_dim=256):
		super().__init__()

		self.shallow_encoder = nn.Sequential(nn.Conv2d(vit_dim, vit_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(vit_dim, z_dim, 3, stride=1, padding=1))

	def forward(self, sam_feature):
		'''
		input:
			sam_feature: (B, 256, 64, 64)
		output:
			spatial feature (B, z_dim, 64, 64)
		'''
		feat = self.shallow_encoder(sam_feature)

		return feat

class DinoEncoder(nn.Module):
	def __init__(self, dino_dim=768, z_dim=64, hidden_dim=128):
		super().__init__()

		self.shallow_encoder = nn.Sequential(nn.Conv2d(dino_dim, hidden_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(hidden_dim, z_dim, 3, stride=1, padding=1))

	def forward(self, dino_feats):
		'''
		input:
			dino_feature: (B, dino_dim, 64, 64)
		output:
			spatial feature (B, z_dim, 64, 64)
		'''
		return self.shallow_encoder(dino_feats)

class SDEncoder(nn.Module):
	def __init__(self, z_dim=64):
		super().__init__()
		'''
		input: list of Tensors: B*512*32*32, B*640*32*32, B*512*64*64
		output: B*z_dim*64*64
		'''

		self.conv1 = nn.Sequential(nn.Conv2d(512, 128, 1, stride=1, padding=0),
								   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.conv2 = nn.Sequential(nn.Conv2d(640, 128, 1, stride=1, padding=0),
									  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
		self.conv3 = nn.Conv2d(512, 128, 1, stride=1, padding=0)

		self.out = nn.Sequential(nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
									nn.ReLU(True),
								nn.Conv2d(128, z_dim, 3, stride=1, padding=1))

	def forward(self, x):
		x1 = self.conv1(x[0])
		x2 = self.conv2(x[1])
		x3 = self.conv3(x[2])
		x = torch.cat([x1, x2, x3], dim=1)
		return self.out(x)

class EncoderPosEmbedding(nn.Module):
	def __init__(self, dim, slot_dim, hidden_dim=128):
		super().__init__()
		self.grid_embed = nn.Linear(4, dim, bias=True)
		self.input_to_k_fg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_fg = nn.Linear(dim, dim, bias=False)

		self.input_to_k_bg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_bg = nn.Linear(dim, dim, bias=False)

		self.MLP_fg = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, slot_dim),
		)

		self.MLP_bg = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, slot_dim),
		)
		
	def apply_rel_position_scale(self, grid, position):
		"""
		grid: (1, h, w, 2)
		position (batch, number_slots, 2)
		"""
		b, n, _ = position.shape
		h, w = grid.shape[1:3]
		grid = grid.view(1, 1, h, w, 2)
		grid = grid.repeat(b, n, 1, 1, 1)
		position = position.view(b, n, 1, 1, 2)
		
		return grid - position # (b, n, h, w, 2)

	def forward(self, x, h, w, position_latent=None):

		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		if position_latent is not None:
			rel_grid = self.apply_rel_position_scale(grid, position_latent)
		else:
			rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)

		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, n_slot-1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, n_slot-1, h*w, d)

		k, v = self.input_to_k_fg(x).unsqueeze(1), self.input_to_v_fg(x).unsqueeze(1) # (b, 1, h*w, d)

		k, v = k + grid_embed, v + grid_embed
		k, v = self.MLP_fg(k), self.MLP_fg(v)

		return k, v # (b, n, h*w, d)

	def forward_bg(self, x, h, w):
		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)
		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, 1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, 1, h*w, d)
		
		k_bg, v_bg = self.input_to_k_bg(x).unsqueeze(1), self.input_to_v_bg(x).unsqueeze(1) # (b, 1, h*w, d)
		k_bg, v_bg = self.MLP_bg(k_bg + grid_embed), self.MLP_bg(v_bg + grid_embed)

		return k_bg, v_bg # (b, 1, h*w, d)

class SlotAttention(nn.Module):
	def __init__(self, num_slots, in_dim=64, slot_dim=64, color_dim=8, iters=4, eps=1e-8, hidden_dim=128, learnable_pos=True, n_feats=64*64):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5

		self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
		self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
		init.xavier_uniform_(self.slots_logsigma)
		self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
		self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
		init.xavier_uniform_(self.slots_logsigma_bg)

		self.learnable_pos = learnable_pos
		self.fg_position = nn.Parameter(torch.rand(1, num_slots-1, 2) * 1.5 - 0.75)
		if self.learnable_pos:
			self.attn_to_pos_bias = nn.Linear(n_feats, 2, bias=False)
			self.attn_to_pos_bias.weight.data.zero_()

		self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)
		self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
		self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

		self.gru_fg = nn.GRUCell(slot_dim, slot_dim)
		self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

		self.to_res_fg = nn.Sequential(nn.LayerNorm(slot_dim),
										nn.Linear(slot_dim, slot_dim))
		self.to_res_bg = nn.Sequential(nn.LayerNorm(slot_dim),
										nn.Linear(slot_dim, slot_dim))

		self.norm_feat = nn.LayerNorm(in_dim)
		self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

	def forward(self, feat, feat_color, num_slots=None):
		"""
		input:
			feat: visual feature with position information, BxHxWxC
			feat_color: texture feature with position information, BxHxWxC'
			output: slots: BxKxC, attn: BxKxN
		"""
		B, H, W, _ = feat.shape
		N = H * W
		feat = feat.flatten(1, 2) # (B, N, C)
		K = num_slots if num_slots is not None else self.num_slots

		mu = self.slots_mu.expand(B, K-1, -1)
		sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
		slot_fg = mu + sigma * torch.randn_like(mu)
		
		fg_position = self.fg_position if self.fg_position is not None else torch.rand(1, K-1, 2) * 2 - 1
		fg_position = fg_position.expand(B, -1, -1).to(feat.device) # Bx(K-1)x2
		
		mu_bg = self.slots_mu_bg.expand(B, 1, -1)
		sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
		slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

		feat = self.norm_feat(feat)
		feat_color = self.norm_feat_color(feat_color)

		k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2) # (1,N,2)

		# attn = None
		for it in range(self.iters):
			slot_prev_bg = slot_bg
			slot_prev_fg = slot_fg
			q_fg = self.to_q(slot_fg)
			q_bg = self.to_q_bg(slot_bg) # (B,1,C)
			
			attn = torch.empty(B, K, N, device=feat.device)
			
			k, v = self.to_kv(feat, H, W, fg_position) # (B,K-1,N,C)
			
			for i in range(K):
				if i != 0:
					k_i = k[:, i-1] # (B,N,C)
					slot_qi = q_fg[:, i-1] # (B,C)
					attn[:, i] = torch.einsum('bd,bnd->bn', slot_qi, k_i) * self.scale
				else:
					attn[:, i] = torch.einsum('bd,bnd->bn', q_bg.squeeze(1), k_bg.squeeze(1)) * self.scale
			
			attn = attn.softmax(dim=1) + self.eps  # BxKxN
			attn_fg, attn_bg = attn[:, 1:, :], attn[:, 0:1, :]  # Bx(K-1)xN, Bx1xN
			attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN
			attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
			
			# update slot position
			# print(attn_weights_fg.shape, grid.shape, fg_position.shape)
			fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
			if self.learnable_pos: # add a bias term
				fg_position = fg_position + self.attn_to_pos_bias(attn_weights_fg) / 5 # (B,K-1,2)
			
			if it != self.iters - 1:
			
				updates_fg = torch.empty(B, K-1, self.slot_dim, device=k.device) # (B,K-1,C)
				for i in range(K-1):
					v_i = v[:, i] # (B,N,C)
					attn_i = attn_weights_fg[:, i] # (B,N)
					updates_fg[:, i] = torch.einsum('bn,bnd->bd', attn_i, v_i)

				updates_bg = torch.einsum('bn,bnd->bd',attn_weights_bg.squeeze(1), v_bg.squeeze(1)) # (B,N,C) * (B,N) -> (B,C)
				updates_bg = updates_bg.unsqueeze(1) # (B,1,C)

				slot_bg = self.gru_bg(
					updates_bg.reshape(-1, self.slot_dim),
					slot_prev_bg.reshape(-1, self.slot_dim)
				)
				slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
				slot_bg = slot_bg + self.to_res_bg(slot_bg)

				slot_fg = self.gru_fg(
					updates_fg.reshape(-1, self.slot_dim),
					slot_prev_fg.reshape(-1, self.slot_dim)
				)
				slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
				slot_fg = slot_fg + self.to_res_fg(slot_fg)

			else:
				# calculate slot texture feature
				feat_color = feat_color.flatten(1, 2) # (B,N,C')
				slot_fg_color = torch.einsum('bkn,bnd->bkd', attn_weights_fg, feat_color) # (B,K-1,N) * (B,N,C') -> (B,K-1,C')
				slot_bg_color = torch.einsum('bn,bnd->bd', attn_weights_bg.squeeze(1), feat_color).unsqueeze(1) # (B,N) * (B,N,C') -> (B,C'), (B,1,C')

		slots_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # (B,K-1,C+C')
		slots_bg = torch.cat([slot_bg, slot_bg_color], dim=-1) # (B,1,C+C')
		slots = torch.cat([slots_bg, slots_fg], dim=1) # (B,K,C+C')
				
		return slots, attn, fg_position

class Decoder(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, 
					project=False, rel_pos=True, fg_in_world=False):
		"""
		freq: raised frequency
		input_dim: pos emb dim + slot dim
		z_dim: network latent dim
		n_layers: #layers before/after skip connection.
		locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
		locality_ratio: if locality, what value is the boundary to clamp?
		fixed_locality: if True, compute locality in world space instead of in transformed view space
		"""
		super().__init__()
		super().__init__()
		self.n_freq = n_freq
		self.locality = locality
		self.locality_ratio = locality_ratio
		self.fixed_locality = fixed_locality
		self.out_ch = 4
		self.z_dim = z_dim
		before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
		after_skip = [nn.Linear(z_dim+input_dim, z_dim), nn.ReLU(True)]
		for i in range(n_layers-1):
			before_skip.append(nn.Linear(z_dim, z_dim))
			before_skip.append(nn.ReLU(True))
			after_skip.append(nn.Linear(z_dim, z_dim))
			after_skip.append(nn.ReLU(True))
		self.f_before = nn.Sequential(*before_skip)
		self.f_after = nn.Sequential(*after_skip)
		self.f_after_latent = nn.Linear(z_dim, z_dim)
		self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
		self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim//4),
									 nn.ReLU(True),
									 nn.Linear(z_dim//4, 3))
		before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
		after_skip = [nn.Linear(z_dim + input_dim, z_dim), nn.ReLU(True)]
		for i in range(n_layers - 1):
			before_skip.append(nn.Linear(z_dim, z_dim))
			before_skip.append(nn.ReLU(True))
			after_skip.append(nn.Linear(z_dim, z_dim))
			after_skip.append(nn.ReLU(True))
		after_skip.append(nn.Linear(z_dim, self.out_ch))
		self.b_before = nn.Sequential(*before_skip)
		self.b_after = nn.Sequential(*after_skip)

		if project:
			self.position_project = nn.Linear(2, self.z_dim)
			self.position_project.weight.data.zero_()
			self.position_project.bias.data.zero_()
		else:
			self.position_project = None
		self.rel_pos = rel_pos
		self.fg_in_world = fg_in_world

	def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, fg_slot_position, dens_noise=0., invariant=True, local_locality_ratio=None):
		"""
		1. pos emb by Fourier
		2. for each slot, decode all points from coord and slot feature
		input:
			sampling_coor_bg: Px3, P = #points, typically P = NxDxHxW
			sampling_coor_fg: (K-1)xPx3
			z_slots: KxC, K: #slots, C: #feat_dim
			z_slots_texture: KxC', K: #slots, C: #texture_dim
			fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
			fg_slot_position: (K-1)x3 in nss space
			dens_noise: Noise added to density
		"""
		K, C = z_slots.shape
		P = sampling_coor_bg.shape[0]

		if self.fixed_locality:
			assert False
			# first compute the originallocality constraint
			outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			if self.rel_pos and invariant:
				# then compute the transformed locality constraint
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
			sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # KxPx4
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # KxPx4x1
			sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # KxPx3
			if local_locality_ratio is not None:
				outsider_idx = outsider_idx | torch.any(sampling_coor_fg.abs() > local_locality_ratio, dim=-1)  # KxP
		else:
			# currently do not support fg_in_world
			# first compute the original locality constraint
			if self.locality:
				sampling_coor_fg_temp = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # (K-1)xPx3
				outsider_idx = torch.any(sampling_coor_fg_temp.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
			# relative position with fg slot position
			if self.rel_pos and invariant:
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :] # KxPx3
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # KxPx3
			if local_locality_ratio is not None:
				outsider_idx = outsider_idx | torch.any(sampling_coor_fg.abs() > local_locality_ratio, dim=-1)  # KxP

		z_bg = z_slots[0:1, :]  # 1xC
		z_fg = z_slots[1:, :]  # (K-1)xC

		if self.position_project is not None and invariant:
			z_fg = z_fg + self.position_project(fg_slot_position[:, :2]) # (K-1)xC

		query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
		input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)

		sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
		query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
		z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
		input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # ((K-1)xP)x(60+C)

		tmp = self.b_before(input_bg)
		bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.out_ch])  # Px5 -> 1xPx5
		tmp = self.f_before(input_fg)
		tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # ((K-1)xP)x64
		latent_fg = self.f_after_latent(tmp)  # ((K-1)xP)x64
		fg_raw_rgb = self.f_color(latent_fg).view([K-1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
		fg_raw_shape = self.f_after_shape(tmp).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
		if self.locality:
			fg_raw_shape[outsider_idx] *= 0
		fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)xPx4

		all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # KxPx4
		raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
		masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
		raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
		raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

		unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
		masked_raws = unmasked_raws * masks
		raws = masked_raws.sum(dim=0)

		return raws, masked_raws, unmasked_raws, masks

class EncoderPosEmbeddingFG(nn.Module):
	def __init__(self, dim, slot_dim, hidden_dim=128):
		super().__init__()
		self.grid_embed = nn.Linear(4, dim, bias=True)
		self.input_to_k_fg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_fg = nn.Linear(dim, dim, bias=False)

		self.MLP_fg = nn.Sequential(
			nn.LayerNorm(dim),
			nn.Linear(dim, slot_dim),
			# nn.Linear(dim, hidden_dim),
			# nn.ReLU(),
			# nn.Linear(hidden_dim, slot_dim)
		)
		
	def apply_rel_position_scale(self, grid, position):
		"""
		grid: (1, h, w, 2)
		position (batch, number_slots, 2)
		"""
		b, n, _ = position.shape
		h, w = grid.shape[1:3]
		grid = grid.view(1, 1, h, w, 2)
		grid = grid.repeat(b, n, 1, 1, 1)
		position = position.view(b, n, 1, 1, 2)
		
		return grid - position # (b, n, h, w, 2)

	def forward(self, x, h, w, position_latent=None):

		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		if position_latent is not None:
			rel_grid = self.apply_rel_position_scale(grid, position_latent)
		else:
			rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)

		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, n_slot-1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, n_slot-1, h*w, d)

		k, v = self.input_to_k_fg(x).unsqueeze(1), self.input_to_v_fg(x).unsqueeze(1) # (b, 1, h*w, d)

		k, v = k + grid_embed, v + grid_embed
		k, v = self.MLP_fg(k), self.MLP_fg(v)

		return k, v # (b, n, h*w, d)

class SlotAttentionFG(nn.Module):
	def __init__(self, in_dim=64, slot_dim=64, color_dim=16, iters=4, eps=1e-8, hidden_dim=64, centered=False):
		'''
		in_dim: dimension for input image feature (shape feature dim)
		color_dim: dimension for color feature (color feature dim)
		slot_dim: dimension for slot feature (output slot dim), final output dim is slot_dim + color_dim. Currently slot_dim == in_dim
		'''
		super().__init__()
		# self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5

		self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
		self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
		init.xavier_uniform_(self.slots_logsigma)
		
		self.to_kv = EncoderPosEmbeddingFG(in_dim, slot_dim)

		self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

		self.gru_fg = nn.GRUCell(slot_dim, slot_dim)

		self.norm_feat = nn.LayerNorm(in_dim)
		if color_dim != 0:
			self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

		self.to_res_fg = nn.Sequential(nn.LayerNorm(slot_dim),
										nn.Linear(slot_dim, slot_dim))
		
		self.centered = centered
		

	def get_fg_position(self, mask):
		'''
		Compute the weighted mean of the grid points as the position of foreground objects.
		input:
			mask: mask for foreground objects. shape: K*1*H*W, K: number of slots
		output:
			fg_position: position of foreground objects. shape: K*2
		'''
		K, _, H, W = mask.shape
		grid = build_grid(H, W, device=mask.device) # 1*H*W*2
		grid = grid.expand(K, -1, -1, -1).permute(0, 3, 1, 2) # K*2*H*W
		grid = grid * mask # K*2*H*W

		fg_position = grid.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5) # K*2
		return fg_position

	def forward(self, feat, feat_color=None, mask=None, use_mask=False):
		"""
		input:
			feat: visual feature with position information, BxHxWxC
			mask: mask for foreground objects, KxHxW, K: number of foreground objects (exclude background)
			feat_color: color feature, BxHxWxC'
		output:
			slot_feat: slot feature, BxKx(C+C') if feat_color is not None else BxKxC
		"""
		B, H, W, _ = feat.shape
		N = H * W
		feat = feat.flatten(1, 2) # (B, N, C)
		K = mask.shape[0] if mask is not None else 1

		mu = self.slots_mu.expand(B, K, -1)
		sigma = self.slots_logsigma.exp().expand(B, K, -1)
		slot_fg = mu + sigma * torch.randn_like(mu)
		
		if self.centered:
			fg_position = torch.zeros(B, K, 2, device=feat.device)
		else:
			fg_position = self.get_fg_position(mask) # Kx2
			fg_position = fg_position.expand(B, -1, -1).to(feat.device) # BxKx2
		
		feat = self.norm_feat(feat)
		if feat_color is not None:
			feat_color = self.norm_feat_color(feat_color)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2).squeeze(0) # Nx2

		# attn = None
		for it in range(self.iters):
			slot_prev_fg = slot_fg
			q_fg = self.to_q(slot_fg)
			
			attn = torch.empty(B, K, N, device=feat.device)
			
			k, v = self.to_kv(feat, H, W, fg_position) # (B,K,N,C)

			# compute foreground attention and updates, each slot only compute attention on corresponding mask
			updates_fg = torch.empty(B, K, self.slot_dim, device=feat.device)
			
			for i in range(K):
				attn_this_slot = torch.einsum('bd,bnd->bn', q_fg[:, i, :], k[:, i, :, :]) * self.scale # BxN
				# we will use softmax after masking, so we need to set the masked values to a very small value
				if use_mask:
					mask_this_slot = mask[i].flatten() # N
					attn_this_slot = attn_this_slot.masked_fill(mask_this_slot.unsqueeze(0) == 0, -1e9)
				attn_this_slot = attn_this_slot.softmax(dim=1) # BxN
				updates_fg[:, i, :] = torch.einsum('bn,bnd->bd', attn_this_slot, v[:, i, :, :]) # BxC
				# update the position of this slot (weighted mean of the grid points, with attention as weights)
				# fg_position[:, i, :] = torch.einsum('bn,nd->bd', attn_this_slot, grid) # Bx2
				attn[:, i, :] = attn_this_slot
				
			if it != self.iters - 1: # do not update slot for the last iteration
				slot_fg = self.gru_fg(updates_fg.reshape(-1, self.slot_dim), slot_fg.reshape(-1, self.slot_dim)).reshape(B, K, self.slot_dim) # BxKxC
				slot_fg = self.to_res_fg(slot_fg) + slot_prev_fg # BxKx2
			else: # last iteration, compute the slots' color feature if feat_color is not None
				if feat_color is not None:
					# weighted mean of the color feature, with attention as weights
					slot_fg_color = torch.einsum('bkn,bnc->bkc', attn, feat_color.flatten(1, 2)) # BxKxC'
					slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # BxKx(C+C')

		# normalize attn for visualization (min-max normalization along the N dimension)
		attn = (attn - attn.min(dim=2, keepdim=True)[0]) / (attn.max(dim=2, keepdim=True)[0] - attn.min(dim=2, keepdim=True)[0] + 1e-5)
				
		return slot_fg, fg_position, attn


class SlotAttentionFGKobj(nn.Module):
	def __init__(self, in_dim=64, slot_dim=64, color_dim=16, iters=4, eps=1e-8, hidden_dim=64, num_slots=3, n_feats=64*64):
		'''
		in_dim: dimension for input image feature (shape feature dim)
		color_dim: dimension for color feature (color feature dim)
		slot_dim: dimension for slot feature (output slot dim), final output dim is slot_dim + color_dim. Currently slot_dim == in_dim
		'''
		super().__init__()
		# self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5

		self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
		self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
		init.xavier_uniform_(self.slots_logsigma)
		
		self.to_kv = EncoderPosEmbeddingFG(in_dim, slot_dim)

		self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

		self.gru_fg = nn.GRUCell(slot_dim, slot_dim)

		self.norm_feat = nn.LayerNorm(in_dim)
		if color_dim != 0:
			self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

		self.to_res_fg = nn.Sequential(nn.LayerNorm(slot_dim),
										nn.Linear(slot_dim, slot_dim))

		self.num_slots = num_slots # foreground slots
		self.fg_position = nn.Parameter(torch.rand(1, num_slots, 2) * 1.5 - 0.75) # initialize the position of foreground slots to [-0.75, 0.75]
		# map the attention map of a slot to its position from H*W to 2
		self.attn_to_pos_bias = nn.Linear(n_feats, 2, bias=False)
		self.attn_to_pos_bias.weight.data.zero_()

	def forward(self, feat, feat_color=None):
		"""
		input:
			feat: visual feature with position information, BxHxWxC
			mask: mask for foreground objects, KxHxW, K: number of foreground objects (exclude background)
			feat_color: color feature, BxHxWxC'
		output:
			slot_feat: slot feature, BxKx(C+C') if feat_color is not None else BxKxC
		"""
		B, H, W, _ = feat.shape
		N = H * W
		feat = feat.flatten(1, 2) # (B, N, C)
		K = self.num_slots

		mu = self.slots_mu.expand(B, K, -1)
		sigma = self.slots_logsigma.exp().expand(B, K, -1)
		slot_fg = mu + sigma * torch.randn_like(mu)
		
		fg_position = self.fg_position.expand(B, -1, -1).to(feat.device) # BxKx2
		
		feat = self.norm_feat(feat)
		if feat_color is not None:
			feat_color = self.norm_feat_color(feat_color)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2) # (1,N,2)

		# attn = None
		for it in range(self.iters):
			slot_prev_fg = slot_fg
			q_fg = self.to_q(slot_fg)
			
			attn = torch.empty(B, K, N, device=feat.device)
			
			k, v = self.to_kv(feat, H, W, fg_position) # (B,K,N,C)

			updates_fg = torch.empty(B, K, self.slot_dim, device=feat.device)
			
			for i in range(K):
				attn[:, i] = torch.einsum('bd,bnd->bn', q_fg[:, i, :], k[:, i, :, :]) * self.scale # BxN
			
			if K != 1:
				attn = attn.softmax(dim=1) # BxKxN, first normalize along the slot dimension
				attn_weights = attn / attn.sum(dim=2, keepdim=True) # BxKxN, then normalize along the N dimension
			else:
				attn = attn.softmax(dim=-1) # BxKxN, only one slot, directly normalize along the N dimension
				attn_weights = attn

			for i in range(K):
				updates_fg[:, i, :] = torch.einsum('bn,bnd->bd', attn_weights[:, i, :], v[:, i, :, :]) # BxC
			
			# update position, fg_position: BxKx2
			# compute the weighted mean of the attention map of a slot, with attention as weights
			fg_position = torch.einsum('bkn,bnd->bkd', attn_weights, grid) # BxKx2
			fg_position = self.attn_to_pos_bias(attn_weights) / 5 + fg_position # BxKx2

			if it != self.iters - 1: # do not update slot for the last iteration
				slot_fg = self.gru_fg(updates_fg.reshape(-1, self.slot_dim), slot_fg.reshape(-1, self.slot_dim)).reshape(B, K, self.slot_dim) # BxKxC
				slot_fg = self.to_res_fg(slot_fg) + slot_prev_fg # BxKx2
			else: # last iteration, compute the slots' color feature if feat_color is not None
				if feat_color is not None:
					# weighted mean of the color feature, with attention as weights
					slot_fg_color = torch.einsum('bkn,bnc->bkc', attn_weights, feat_color.flatten(1, 2)) # BxKxC'
					slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # BxKx(C+C')
		
		if K == 1: # normalize attn for visualization (min-max normalization along the N dimension), when K==1
			attn = (attn - attn.min(dim=2, keepdim=True)[0]) / (attn.max(dim=2, keepdim=True)[0] - attn.min(dim=2, keepdim=True)[0] + 1e-5)
				
		return slot_fg, fg_position, attn

class DecoderFG(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, 
					project=False, rel_pos=True, fg_in_world=False):
		"""
		freq: raised frequency
		input_dim: pos emb dim + slot dim
		z_dim: network latent dim
		n_layers: #layers before/after skip connection.
		locality: if True, for each obj slot, clamp sigma values to 0 outside obj_scale.
		locality_ratio: if locality, what value is the boundary to clamp?
		fixed_locality: if True, compute locality in world space instead of in transformed view space
		"""
		super().__init__()
		super().__init__()
		self.n_freq = n_freq
		self.locality = locality
		self.locality_ratio = locality_ratio
		self.fixed_locality = fixed_locality
		self.out_ch = 4
		self.z_dim = z_dim
		before_skip = [nn.Linear(input_dim, z_dim), nn.ReLU(True)]
		after_skip = [nn.Linear(z_dim+input_dim, z_dim), nn.ReLU(True)]
		for i in range(n_layers-1):
			before_skip.append(nn.Linear(z_dim, z_dim))
			before_skip.append(nn.ReLU(True))
			after_skip.append(nn.Linear(z_dim, z_dim))
			after_skip.append(nn.ReLU(True))
		self.f_before = nn.Sequential(*before_skip)
		self.f_after = nn.Sequential(*after_skip)
		self.f_after_latent = nn.Linear(z_dim, z_dim)
		self.f_after_shape = nn.Linear(z_dim, self.out_ch - 3)
		self.f_color = nn.Sequential(nn.Linear(z_dim, z_dim//4),
									 nn.ReLU(True),
									 nn.Linear(z_dim//4, 3))

		if project:
			self.position_project = nn.Linear(2, self.z_dim)
			self.position_project.weight.data.zero_()
			self.position_project.bias.data.zero_()
		else:
			self.position_project = None
		self.rel_pos = rel_pos
		self.fg_in_world = fg_in_world

	def forward(self, sampling_coor_fg, z_slots, fg_transform, fg_slot_position, dens_noise=0., invariant=True, local_locality_ratio=None):
		"""
		1. pos emb by Fourier
		2. for each slot, decode all points from coord and slot feature
		input:
			sampling_coor_fg: KxPx3, P = #points, typically P = NxDxHxW
			z_slots: KxC, K: #slots, C: #feat_dim
			z_slots_texture: KxC', K: #slots, C: #texture_dim
			fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
			fg_slot_position: Kx3 in nss space
			dens_noise: Noise added to density
		"""
		K, C = z_slots.shape
		P = sampling_coor_fg.shape[1]

		if self.fixed_locality:
			assert False
			# first compute the originallocality constraint
			outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			if self.rel_pos and invariant:
				# then compute the transformed locality constraint
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
			sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # KxPx4
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # KxPx4x1
			sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # KxPx3
			if local_locality_ratio is not None:
				outsider_idx = outsider_idx | torch.any(sampling_coor_fg.abs() > local_locality_ratio, dim=-1)  # KxP
		else:
			# currently do not support fg_in_world
			# first compute the original locality constraint
			if self.locality:
				sampling_coor_fg_temp = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # (K-1)xPx3
				outsider_idx = torch.any(sampling_coor_fg_temp.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
			# relative position with fg slot position
			if self.rel_pos and invariant:
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :] # KxPx3
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # KxPx3
			if local_locality_ratio is not None:
				outsider_idx = outsider_idx | torch.any(sampling_coor_fg.abs() > local_locality_ratio, dim=-1)  # KxP
			# if not self.fg_in_world:
			# 	sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # KxPx3x1
			# outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP

		z_fg = z_slots

		if self.position_project is not None and invariant:
			z_fg = z_fg + self.position_project(fg_slot_position[:, :2]) # KxC

		sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # (KxP)x3
		query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # (KxP)x60
		z_fg_ex = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # (KxP)xC
		input_fg = torch.cat([query_fg_ex, z_fg_ex], dim=1)  # (KxP)x(60+C)

		tmp = self.f_before(input_fg)
		tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # (KxP)x64
		latent_fg = self.f_after_latent(tmp)  # (KxP)x64
		fg_raw_rgb = self.f_color(latent_fg).view([K, P, 3])  # (KxP)x3 -> KxPx3
		fg_raw_shape = self.f_after_shape(tmp).view([K, P])  # (KxP)x1 -> KxP, density
		if self.locality:
			fg_raw_shape[outsider_idx] *= 0
		fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # KxPx4

		all_raws = fg_raws
		raw_masks = F.relu(all_raws[:, :, -1:], True)  # KxPx1
		masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # KxPx1
		raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
		raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

		unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # KxPx4
		masked_raws = unmasked_raws * masks
		raws = masked_raws.sum(dim=0)

		return raws, masked_raws, unmasked_raws, masks

class FeatureAggregate(nn.Module):
	def __init__(self, in_dim=64, out_dim=64):
		super().__init__()
		self.convs = nn.Sequential(nn.Conv2d(in_dim, in_dim, 3, 1, 1), 
								nn.ReLU(inplace=True),
								nn.Conv2d(in_dim, out_dim, 3, 1, 1))
		self.pool = nn.AdaptiveAvgPool2d(1)

	def get_fg_position(self, mask):
		'''
		Compute the weighted mean of the grid points as the position of foreground objects.
		input:
			mask: mask for foreground objects. shape: K*1*H*W, K: number of slots
		output:
			fg_position: position of foreground objects. shape: K*2
		'''
		K, _, H, W = mask.shape
		grid = build_grid(H, W, device=mask.device) # 1*H*W*2
		grid = grid.expand(K, -1, -1, -1).permute(0, 3, 1, 2) # K*2*H*W
		grid = grid * mask # K*2*H*W

		fg_position = grid.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5) # K*2
		return fg_position

	def forward(self, x, mask, use_mask=True, x_color=None):
		if x_color is not None:
			x = torch.cat([x, x_color], dim=1)
		x = self.convs(x)
		if use_mask: # only aggregate foreground features
			x = x * mask
			x = x.sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-5) # BxC
		else: 
			x = self.pool(x).squeeze(-1).squeeze(-1) # BxC
		fg_position = self.get_fg_position(mask) # K*2
		return x, fg_position
	
						
	
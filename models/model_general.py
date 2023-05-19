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
	def __init__(self, in_dim=64, slot_dim=64, iters=4, eps=1e-8, hidden_dim=128):
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

		hidden_dim = max(slot_dim, hidden_dim)

		self.to_res_fg = nn.Sequential(
			nn.LayerNorm(slot_dim),
			nn.Linear(slot_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, slot_dim)
		)

		self.norm_feat = nn.LayerNorm(in_dim)
		self.slot_dim = slot_dim

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

	def forward(self, feat, mask, use_mask=True, feat_color=None):
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
		K = mask.shape[0]

		mu = self.slots_mu.expand(B, K, -1)
		sigma = self.slots_logsigma.exp().expand(B, K, -1)
		slot_fg = mu + sigma * torch.randn_like(mu)
		
		fg_position = self.get_fg_position(mask) # Kx2
		# fg_position = torch.rand(1, K, 2) * 2 - 1
		fg_position = fg_position.expand(B, -1, -1).to(feat.device) # BxKx2
		
		feat = self.norm_feat(feat)

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
				fg_position[:, i, :] = torch.einsum('bn,nd->bd', attn_this_slot, grid) # Bx2
				attn[:, i, :] = attn_this_slot
				
			if it != self.iters - 1: # do not update slot for the last iteration
				slot_fg = self.gru_fg(updates_fg.reshape(-1, self.slot_dim), slot_fg.reshape(-1, self.slot_dim)).reshape(B, K, self.slot_dim) # BxKxC
				slot_fg = self.to_res_fg(slot_fg) + slot_prev_fg # BxKx2
				# normalize attn for visualization (min-max normalization along the N dimension)
			else: # last iteration, compute the slots' color feature if feat_color is not None
				if feat_color is not None:
					# weighted mean of the color feature, with attention as weights
					slot_fg_color = torch.einsum('bkn,bnc->bkc', attn, feat_color.flatten(1, 2)) # BxKxC'
					slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # BxKx(C+C')

			# calculate attn for visualization
			attn = (attn - attn.min(dim=2, keepdim=True)[0]) / (attn.max(dim=2, keepdim=True)[0] - attn.min(dim=2, keepdim=True)[0] + 1e-5)
				
		return slot_fg, fg_position, attn

						
	
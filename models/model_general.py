import math
from os import X_OK

from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd

from .utils import PositionalEncoding, sin_emb, build_grid, debug, integrated_pos_enc, integrated_pos_enc_360

class Encoder(nn.Module):
	def __init__(self, input_nc=3, z_dim=64, bottom=False, double_bottom=False, pos_emb=False):

		super().__init__()

		self.bottom = bottom
		self.double_bottom = double_bottom
		assert double_bottom == False or bottom == True
		print('Bottom for Encoder: ', self.bottom)
		print('Double Bottom for Encoder: ', self.double_bottom)

		input_nc = input_nc + 4 if pos_emb else input_nc
		self.pos_emb = pos_emb

		if self.bottom and self.double_bottom:
			self.enc_down_00 = nn.Sequential(nn.Conv2d(input_nc, z_dim // 2, 3, stride=1, padding=1),
																nn.ReLU(True))
			self.enc_down_01 = nn.Sequential(nn.Conv2d(z_dim // 2, z_dim, 3, stride=2, padding=1),
																nn.ReLU(True))
	
		elif self.bottom:
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

		if self.bottom and self.double_bottom:
			x_down_00 = self.enc_down_00(x_)
			x_down_01 = self.enc_down_01(x_down_00)
			x_down_1 = self.enc_down_1(x_down_01)
		elif self.bottom:
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
	def __init__(self, bottom=False, pos_emb=False, input_nc=3, shape_dim=48, color_dim=16, input_dim=256):
		super().__init__()

		self.Encoder = Encoder(bottom=bottom, z_dim=color_dim, pos_emb=pos_emb, input_nc=input_nc)

		self.shallow_encoder = nn.Sequential(nn.Conv2d(input_dim, input_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(input_dim, shape_dim, 3, stride=1, padding=1))

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

class singleRouteEncoder(nn.Module):
	def __init__(self, bottom=False, pos_emb=False, out_dim=64, input_dim=256, hidden_dim=256):
		super().__init__()

		self.shallow_encoder = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(hidden_dim, out_dim, 3, stride=1, padding=1))

	def forward(self, input_feat):
		'''
		input:
			input_feat: (B, input_dim, 64, 64)
			x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True
		output:
			spatial feature (B, shape_dim, 64, 64)
		'''
		feat = self.shallow_encoder(input_feat)

		return feat

class MultiDINOEncoder(nn.Module):
	def __init__(self, n_feat_layer=1, shape_dim=64, input_dim=256, hidden_dim=256):
		super().__init__()

		self.shallow_encoders = nn.ModuleList([nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1),
											nn.ReLU(True),) for _ in range(n_feat_layer)]
											)
		
		self.combine = nn.Conv2d(hidden_dim, shape_dim, 3, stride=1, padding=1)

	def forward(self, input_feats):
		'''
		input:
			input_feat: (B, input_dim, 64, 64)
		output:
			spatial feature (B, shape_dim, 64, 64)
		'''
		feats_shape = [shallow_encoder(input_feat) for shallow_encoder, input_feat in zip(self.shallow_encoders, input_feats)]
		feat_shape = torch.sum(torch.stack(feats_shape), dim=0) / len(feats_shape)
		feat_shape = self.combine(feat_shape)

		return feat_shape

class MultiRouteEncoderSeparate(nn.Module):
	def __init__(self, bottom=False, pos_emb=False, n_feat_layer=1, input_nc=3, shape_dim=48, color_dim=16, input_dim=256, hidden_dim=128):
		super().__init__()

		self.Encoder = Encoder(bottom=bottom, z_dim=color_dim, 
			 					pos_emb=pos_emb, input_nc=input_nc)

		self.shallow_encoders = nn.ModuleList([nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1),
											nn.ReLU(True),) for _ in range(n_feat_layer)]
											)
		
		self.combine = nn.Conv2d(hidden_dim, shape_dim, 3, stride=1, padding=1)

	def forward(self, input_feats, x):
		'''
		input:
			input_feats: list of Tensors: B*input_dim*64*64
			x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True
		output:
			shape feature (B, shape_dim, 64, 64), color feature (B, color_dim, 64, 64)
		'''
		feat_color = self.Encoder(x)
		feats_shape = [shallow_encoder(input_feat) for shallow_encoder, input_feat in zip(self.shallow_encoders, input_feats)]
		feat_shape = torch.sum(torch.stack(feats_shape), dim=0) / len(feats_shape)
		feat_shape = self.combine(feat_shape)

		return feat_shape, feat_color

class dualRouteEncoderSeparate(nn.Module):
	def __init__(self, bottom=False, double_bottom=False, pos_emb=False, input_nc=3, shape_dim=48, color_dim=16, input_dim=256, hidden_dim=256):
		super().__init__()

		self.Encoder = Encoder(bottom=bottom, double_bottom=double_bottom, z_dim=color_dim, 
			 					pos_emb=pos_emb, input_nc=input_nc)

		self.shallow_encoder = nn.Sequential(nn.Conv2d(input_dim, hidden_dim, 3, stride=1, padding=1),
											nn.ReLU(True),
											nn.Conv2d(hidden_dim, shape_dim, 3, stride=1, padding=1))

	def forward(self, input_feat, x):
		'''
		input:
			input_feat: (B, input_dim, 64, 64)
			x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True 
				or (B, 3, 256, 256) if bottom and double_bottom are True
		output:
			shape feature (B, shape_dim, 64, 64), color feature (B, color_dim, 64, 64)
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

class InputPosEmbedding(nn.Module):
	def __init__(self, in_dim):
		super().__init__()
		self.point_conv = nn.Conv2d(in_dim+4, in_dim, 1, bias=False)
		# init as eye matrix
		self.point_conv.weight.data.zero_()
		for i in range(in_dim):
			self.point_conv.weight.data[i, i, 0, 0] = 1

	def forward(self, x): # x: B*H*W*C
		x = x.permute(0, 3, 1, 2) # B*C*H*W
		W, H = x.shape[3], x.shape[2]
		X = torch.linspace(-1, 1, W)
		Y = torch.linspace(-1, 1, H)
		y1_m, x1_m = torch.meshgrid([Y, X])
		x2_m, y2_m = -x1_m, -y1_m  # Normalized distance in the four direction
		pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).unsqueeze(0)  # 1x4xHxW
		x_ = torch.cat([x, pixel_emb], dim=1)
		# print(x_.shape)
		return self.point_conv(x_).permute(0, 2, 3, 1) # B*H*W*C

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

	def forward(self, x, h, w, position_latent=None, dropout_shape_dim=48, dropout_shape_rate=None, dropout_all_rate=None):

		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		if position_latent is not None:
			rel_grid = self.apply_rel_position_scale(grid, position_latent)
		else:
			rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)

		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, n_slot-1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, n_slot-1, h*w, d)

		if dropout_shape_rate is not None or dropout_all_rate is not None:
			x_ = x.clone()
			if dropout_shape_rate is not None:
				# randomly dropout the first few dimensions of the feature, i.e., keep only the information from the shallow encoder
				drop = (torch.rand(1, 1, 1, device=x.device) > dropout_shape_rate).expand(1, 1, dropout_shape_dim)
				drop = torch.cat([drop, torch.ones(1, 1, x.shape[-1] - dropout_shape_dim, device=x.device)], dim=-1).expand(x.shape[0], x.shape[1], -1)
				x_ = x_ * drop # zeroing out some dimensions of the feature
			if dropout_all_rate is not None:
				# randomly dropout all dimensions of the feature, i.e., keep only the position information
				drop = torch.rand(1, 1, 1, device=x.device) > dropout_all_rate
				x_ = x_ * drop # zeroing out all dimensions of the feature
			k, v = self.input_to_k_fg(x_).unsqueeze(1), self.input_to_v_fg(x).unsqueeze(1) # (b, 1, h*w, d)
		else:
			k, v = self.input_to_k_fg(x).unsqueeze(1), self.input_to_v_fg(x).unsqueeze(1)

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
	def __init__(self, num_slots, in_dim=64, slot_dim=64, color_dim=8, iters=4, eps=1e-8, hidden_dim=128,
		  learnable_pos=True, n_feats=64*64, global_feat=False, pos_emb=False, feat_dropout_dim=None,
		  random_init_pos=False, pos_no_grad=False, diff_fg_init=False, learnable_init=False):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5

		self.learnable_init = learnable_init

		if not self.learnable_init:
			self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
			self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
			init.xavier_uniform_(self.slots_logsigma)
			self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
			self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
			init.xavier_uniform_(self.slots_logsigma_bg)
		else:
			self.slots_init_fg = nn.Parameter((torch.randn(1, num_slots-1, slot_dim)))
			self.slots_init_bg = nn.Parameter((torch.randn(1, 1, slot_dim)))
			init.xavier_uniform_(self.slots_init_fg)
			init.xavier_uniform_(self.slots_init_bg)

		# self.diff_fg_init = diff_fg_init

		# if not self.diff_fg_init:
		# 	self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
		# 	self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
		# 	init.xavier_uniform_(self.slots_logsigma)
		# else:
		# 	self.slots_mu = nn.Parameter(torch.randn(1, num_slots-1, slot_dim))
		# 	self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots-1, slot_dim))
		# 	init.xavier_uniform_(self.slots_logsigma)

		self.learnable_pos = learnable_pos
		self.fg_position = nn.Parameter(torch.rand(1, num_slots-1, 2) * 1.5 - 0.75)
		if self.learnable_pos:
			self.attn_to_pos_bias = nn.Sequential(nn.Linear(n_feats, 2), nn.Tanh()) # range (-1, 1)
			# nn.Linear(n_feats, 2, bias=False)
			self.attn_to_pos_bias[0].weight.data.zero_()
			self.attn_to_pos_bias[0].bias.data.zero_()

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
		if color_dim != 0:
			self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

		self.global_feat = global_feat
		self.random_init_pos = random_init_pos
		self.pos_no_grad = pos_no_grad
		self.pos_emb = pos_emb
		if self.pos_emb:
			self.input_pos_emb = InputPosEmbedding(in_dim)
		self.dropout_shape_dim = feat_dropout_dim

	def forward(self, feat, feat_color=None, num_slots=None, dropout_shape_rate=None, 
			dropout_all_rate=None, init_mask=None):
		"""
		input:
			feat: visual feature with position information, BxHxWxC
			feat_color: texture feature with position information, BxHxWxC'
			output: slots: BxKxC, attn: BxKxN
		"""
		B, H, W, _ = feat.shape
		N = H * W
		if self.pos_emb:
			feat = self.input_pos_emb(feat)
		feat = feat.flatten(1, 2) # (B, N, C)

		if init_mask is not None:
			init_mask = F.interpolate(init_mask, size=(H, W), mode='bilinear', align_corners=False) # (K-1, 1, H, W)
			if init_mask.shape[0] != self.num_slots - 1:
				init_mask = torch.cat([init_mask, torch.ones(self.num_slots - 1 - init_mask.shape[0], 1, H, W, device=init_mask.device)], dim=0)
			init_mask = init_mask.flatten(1,3).unsqueeze(0).expand(B, -1, -1) # (B, K-1, N)

		# if self.feat_dropout_dim is not None:
		# 	# randomly dropout the first few dimensions of the feature
		# 	drop = torch.rand(B, N, self.feat_dropout_dim, device=feat.device) > feat_dropout_rate
		# 	drop = torch.cat([drop, torch.ones(B, N, feat.shape[-1] - self.feat_dropout_dim, device=feat.device)], dim=-1)
		# 	feat = feat * drop # zeroing out some dimensions of the feature

		K = num_slots if num_slots is not None else self.num_slots

		if not self.learnable_init:
			mu = self.slots_mu.expand(B, K-1, -1)
			sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
			slot_fg = mu + sigma * torch.randn_like(mu)

			mu_bg = self.slots_mu_bg.expand(B, 1, -1)
			sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
			slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)
		else:
			slot_fg = self.slots_init_fg.expand(B, K-1, -1)
			slot_bg = self.slots_init_bg.expand(B, 1, -1)

		# if not self.diff_fg_init:
		# 	mu = self.slots_mu.expand(B, K-1, -1)
		# 	sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
		# 	slot_fg = mu + sigma * torch.randn_like(mu)
		# else:
		# 	mu = self.slots_mu.expand(B, -1, -1)[:, 0:K-1, :]
		# 	sigma = self.slots_logsigma.exp().expand(B, -1, -1)[:, 0:K-1, :]
		# 	slot_fg = mu + sigma * torch.randn_like(mu)
		
		feat = self.norm_feat(feat)
		if feat_color is not None:
			feat_color = self.norm_feat_color(feat_color)

		k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2) # (1,N,2)
		
		fg_position = self.fg_position if (self.fg_position is not None and not self.random_init_pos) else torch.zeros(1, K-1, 2).to(feat.device)
		fg_position = fg_position.expand(B, -1, -1)[:, :K-1, :].to(feat.device) # Bx(K-1)x2

		# attn = None
		for it in range(self.iters):
			slot_prev_bg = slot_bg
			slot_prev_fg = slot_fg
			q_fg = self.to_q(slot_fg)
			q_bg = self.to_q_bg(slot_bg) # (B,1,C)
			
			attn = torch.empty(B, K, N, device=feat.device)
			
			k, v = self.to_kv(feat, H, W, fg_position, 
			 			dropout_shape_dim=self.dropout_shape_dim, 
			 			dropout_shape_rate=dropout_shape_rate,
						dropout_all_rate=dropout_all_rate) # (B,K-1,N,C), (B,K-1,N,C)
			
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
			if init_mask is not None: # (K, 1, H, W)
				fg_position = torch.einsum('bkn,bnd->bkd', init_mask, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
			else:
				if self.pos_no_grad:
					with torch.no_grad():
						fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
				else:
					fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)

			if self.learnable_pos: # add a bias term
				fg_position = fg_position + self.attn_to_pos_bias(attn_weights_fg) / 5 # (B,K-1,2)
				fg_position = fg_position.clamp(-1, 1) # (B,K-1,2)

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
				if feat_color is not None:
					# calculate slot color feature
					feat_color = feat_color.flatten(1, 2) # (B,N,C')
					if not self.global_feat:
						slot_fg_color = torch.einsum('bkn,bnd->bkd', attn_weights_fg, feat_color) # (B,K-1,N) * (B,N,C') -> (B,K-1,C')
						slot_bg_color = torch.einsum('bn,bnd->bd', attn_weights_bg.squeeze(1), feat_color).unsqueeze(1) # (B,N) * (B,N,C') -> (B,C'), (B,1,C')
					else:
						slot_fg_color = feat_color.repeat(1, K-1, 1) # (B,K-1,C')
						slot_bg_color = feat_color

		if feat_color is not None:
			slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # (B,K-1,C+C')
			slot_bg = torch.cat([slot_bg, slot_bg_color], dim=-1) # (B,1,C+C')
			
		slots = torch.cat([slot_bg, slot_fg], dim=1) # (B,K,C+C')
				
		return slots, attn, fg_position

class Decoder(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, 
					project=False, rel_pos=True, fg_in_world=False, no_transform=False):
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
		self.no_transform = no_transform
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

		if self.no_transform:
			if self.locality:
				outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
		elif self.fixed_locality:
			# first compute the originallocality constraint
			if self.locality:
				outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # KxPx4
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # KxPx4x1
			sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # KxPx3
			if self.rel_pos and invariant:
				# transform fg_slot_position to the camera coordinate
				fg_slot_position = torch.cat([fg_slot_position, torch.ones_like(fg_slot_position[:, 0:1])], dim=-1)  # (K-1)x4
				fg_slot_position = torch.matmul(fg_transform.squeeze(0), fg_slot_position.t()).t() # (K-1)x4
				# print('Debug: ', fg_slot_position.shape, sampling_coor_fg.shape)
				fg_slot_position = fg_slot_position[:, :3]  # (K-1)x3
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
			# if self.locality:
			# 	outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			# if self.rel_pos and invariant:
			# 	# then compute the transformed locality constraint
			# 	sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
			# sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # KxPx4
			# sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # KxPx4x1
			# sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # KxPx3
			if self.locality and local_locality_ratio is not None:
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
			if self.locality and local_locality_ratio is not None:
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


class DecoderBox(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, 
					project=False, rel_pos=True, fg_in_world=False, no_transform=False):
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
		self.no_transform = no_transform
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

	def processQueries(self, sampling_coor_fg, sampling_coor_bg, z_fg, z_bg, 
		    keep_ratio=0.0, mask_ratio=0., fg_object_size=None):
		'''
		Process the query points and the slot features
		1. If self.fg_object_size is not None, do:
			Remove the query point that is too far away from the slot center, 
			the bouding box is defined as a cube with side length 2 * self.fg_object_size
			for the points outside the bounding box, keep only keep_ratio of them
			store the new sampling_coor_fg and the indices of the remaining points
		2. Do the pos emb by Fourier
		3. Concatenate the pos emb and the slot features
		4. If self.fg_object_size is not None, return the new sampling_coor_fg and their indices

		input: sampling_coor_fg: (K-1)xPx3 (already transformed into the camera coordinate, and deducted the slot center)
				sampling_coor_bg: Px3 (in world coordinate)
				z_fg: (K-1)xC
				z_bg: 1xC
				ssize: supervision size (64)
		return: input_fg: M * (60 + C) (M is the number of query points inside bbox), C is the slot feature dim, and 60 means increased-freq feat dim
				input_bg: Px(60+C)
				idx: M (indices of the query points inside bbox)
		'''

		P = sampling_coor_bg.shape[0]
		K = z_fg.shape[0] + 1
		sampling_coor_fg = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3

		# 1. Remove the query points too far away from the slot center
		if fg_object_size is not None:
			# remove abs(x) > fg_object_size | abs(y) > fg_object_size | z > fg_object_size
			mask = torch.all(torch.abs(sampling_coor_fg) < fg_object_size, dim=-1)  # ((K-1)xP)
			if mask.sum() == 0:
				mask[0] = True
			sampling_coor_fg = sampling_coor_fg[mask]  # Update the coordinates using the mask
			idx = mask.nonzero().squeeze()  # Indices of valid points
			# print("Number of points inside bbox: ", idx.size(0), " out of ", (K-1)*P, 
			# 		'\n, ratio: ', idx.size(0) / ((K-1)*P))
		else:
			idx = torch.arange(sampling_coor_fg.size(0))

		# 2. Compute Fourier position embeddings
		pos_emb_fg = sin_emb(sampling_coor_fg, n_freq=self.n_freq)  # Mx(6*n_freq+3)
		pos_emb_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px(6*n_freq+3)

		if mask_ratio > 0.0:
			# mask the last ratio of the pos emb
			n_mask = int(pos_emb_fg.shape[-1] * (1 - mask_ratio)) // 3 * 3 + 3
			# pos_emb_fg[..., n_mask:] *= 0
			pos_emb_bg[..., n_mask:] *= 0

		# 3. Concatenate the embeddings with z_fg and z_bg features
		# Assuming z_fg and z_bg are repeated for each query point
		# Also assuming K is the first dimension of z_fg and we need to repeat it for each query point
		
		z_fg = z_fg[:, None, :].expand(-1, P, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xP)xC
		z_fg = z_fg[idx]  # MxC

		input_fg = torch.cat([pos_emb_fg, z_fg], dim=-1)
		input_bg = torch.cat([pos_emb_bg, z_bg.repeat(sampling_coor_bg.size(0), 1)], dim=-1)

		# 4. Return required tensors
		return input_fg, input_bg, idx

	def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, fg_slot_position, dens_noise=0., invariant=True, 
		 		local_locality_ratio=None, fg_object_size=None, keep_ratio=0.0, mask_ratio=0.):
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

		# debug(sampling_coor_fg, fg_slot_position, save_name='before_transform')

		# keep_idx = sampling_coor_fg.flatten(start_dim=0, end_dim=1)[:, 2] > -0.05 \
		# 			if fg_object_size is not None else None

		if self.no_transform:
			if self.locality:
				outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
		elif self.fixed_locality:
			# first compute the original locality constraint
			if self.locality:
				outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # KxP
			sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # KxPx4
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # KxPx4x1
			sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # KxPx3
			if self.rel_pos and invariant:
				# transform fg_slot_position to the camera coordinate
				fg_slot_position = torch.cat([fg_slot_position, torch.ones_like(fg_slot_position[:, 0:1])], dim=-1)  # (K-1)x4
				fg_slot_position = torch.matmul(fg_transform.squeeze(0), fg_slot_position.t()).t() # (K-1)x4
				# print('Debug: ', fg_slot_position.shape, sampling_coor_fg.shape)
				fg_slot_position = fg_slot_position[:, :3]  # (K-1)x3
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :]  # KxPx3
		else:
			if self.locality:
				sampling_coor_fg_temp = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # (K-1)xPx3
				outsider_idx = torch.any(sampling_coor_fg_temp.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
			# relative position with fg slot position
			if self.rel_pos and invariant:
				sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :] # KxPx3
			sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # KxPx3

		z_bg = z_slots[0:1, :]  # 1xC
		z_fg = z_slots[1:, :]  # (K-1)xC

		# debug(sampling_coor_fg, save_name='after_transform')

		input_fg, input_bg, idx = self.processQueries(sampling_coor_fg, sampling_coor_bg, z_fg, z_bg, 
						keep_ratio=keep_ratio, mask_ratio=mask_ratio, fg_object_size=fg_object_size)

		tmp = self.b_before(input_bg)
		bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P, self.out_ch])  # Px4 -> 1xPx4

		tmp = self.f_before(input_fg)
		tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # Mx64
		latent_fg = self.f_after_latent(tmp)  # Mx64
		fg_raw_rgb = self.f_color(latent_fg) # Mx3
		# put back the removed query points, for indices between idx[i] and idx[i+1], put fg_raw_rgb[i] at idx[i]
		fg_raw_rgb_full = torch.zeros((K-1)*P, 3, device=fg_raw_rgb.device) # ((K-1)xP)x3
		fg_raw_rgb_full[idx] = fg_raw_rgb
		fg_raw_rgb = fg_raw_rgb_full.view([K-1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3
		''' Compute the lengths of intervals to be filled with each row of fg_raw_rgb, Use repeat_interleave to fill fg_raw_rgb_full'''
		# lengths = torch.cat([idx[1:] - idx[:-1], torch.tensor([(K-1)*P - idx[-1]], device=fg_raw_rgb.device)])
		# fg_raw_rgb = fg_raw_rgb.repeat_interleave(lengths, dim=0).view([K-1, P, 3])  # ((K-1)xP)x3 -> (K-1)xPx3

		fg_raw_shape = self.f_after_shape(tmp) # Mx1
		fg_raw_shape_full = torch.zeros((K-1)*P, 1, device=fg_raw_shape.device) # ((K-1)xP)x1
		fg_raw_shape_full[idx] = fg_raw_shape
		fg_raw_shape = fg_raw_shape_full.view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density
		# fg_raw_shape = fg_raw_shape.repeat_interleave(lengths, dim=0).view([K - 1, P])  # ((K-1)xP)x1 -> (K-1)xP, density

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
	
class DecoderIPE(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, 
		  			locality_ratio=4/7, fixed_locality=False, use_viewdirs=False):
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
		assert self.fixed_locality == True
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

		self.pos_enc = PositionalEncoding(max_deg=n_freq)

	def processQueries(self, mean, var, fg_transform, fg_slot_position, z_fg, z_bg, 
					keep_ratio=0.0, mask_ratio=0.0, fg_object_size=None):
		'''
		Process the query points and the slot features
		1. If self.fg_object_size is not None, do:
			Remove the query point that is too far away from the slot center, 
			the bouding box is defined as a cube with side length 2 * self.fg_object_size
			for the points outside the bounding box, keep only keep_ratio of them
			store the new sampling_coor_fg and the indices of the remaining points
		2. Do the pos emb by Fourier
		3. Concatenate the pos emb and the slot features
		4. If self.fg_object_size is not None, return the new sampling_coor_fg and their indices

		input: 	mean: PxDx3
				var: PxDx3
				fg_transform: 1x4x4
				fg_slot_position: (K-1)x3
				z_fg: (K-1)xC
				z_bg: 1xC
				ssize: supervision size (64)
				mask_ratio: frequency mask ratio to the pos emb
		return: input_fg: M * (60 + C) (M is the number of query points inside bbox), C is the slot feature dim, and 60 means increased-freq feat dim
				input_bg: Px(60+C)
				idx: M (indices of the query points inside bbox)
		'''
		P, D = mean.shape[0], mean.shape[1]
		K = z_fg.shape[0] + 1
		
		sampling_mean_fg = mean[None, ...].expand(K-1, -1, -1, -1).flatten(1, 2) # (K-1)*(P*D)*3
		sampling_mean_fg = torch.cat([sampling_mean_fg, torch.ones_like(sampling_mean_fg[:, :, 0:1])], dim=-1)  # (K-1)*(P*D)*4
		sampling_mean_fg = torch.matmul(fg_transform[None, ...], sampling_mean_fg[..., None]).squeeze(-1)  # (K-1)*(P*D)*4x1
		sampling_mean_fg = sampling_mean_fg[:, :, :3]  # (K-1)*(P*D)*3

		# transform fg_slot_position to the camera coordinate
		fg_slot_position = torch.cat([fg_slot_position, torch.ones_like(fg_slot_position[:, 0:1])], dim=-1)  # (K-1)x4
		fg_slot_position = torch.matmul(fg_transform.squeeze(0), fg_slot_position.t()).t() # (K-1)x4
		# print('Debug: ', fg_slot_position.shape, sampling_coor_fg.shape)
		fg_slot_position = fg_slot_position[:, :3]  # (K-1)x3
		sampling_mean_fg = sampling_mean_fg - fg_slot_position[:, None, :]  # (K-1)x(P*D)x3
		sampling_mean_fg = sampling_mean_fg.view([K-1, P, D, 3]).flatten(0, 1)  # ((K-1)xP)xDx3

		sampling_var_fg = var[None, ...].expand(K-1, -1, -1, -1).flatten(0, 1)  # ((K-1)xP)xDx3

		sampling_mean_bg, sampling_var_bg = mean, var

		sampling_mean_fg_ = sampling_mean_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xPxD)x3

		# 1. Remove the query points too far away from the slot center
		if fg_object_size is not None:
			# remove abs(x) > fg_object_size | abs(y) > fg_object_size | z > fg_object_size
			mask = torch.all(torch.abs(sampling_mean_fg_) < fg_object_size, dim=-1)  # ((K-1)xP) --> M
			# M == 0 / 1, keep at least two points to avoid error
			if mask.sum() <= 1:
				mask[:2] = True
			# randomly take only keep_ratio of the points outside the bounding box
			mask = mask | (torch.rand(mask.shape, device=mask.device) < keep_ratio)
			# mask = mask & keep_idx if keep_idx is not None else mask
			idx = mask.nonzero().squeeze()  # Indices of valid points
			# print("Number of points inside bbox: ", idx.size(0), " out of ", (K-1)*P, 
			# 		'\n, ratio: ', idx.size(0) / ((K-1)*P))
		else:
			idx = torch.arange(sampling_mean_fg_.size(0))

		# pos_emb_fg = self.pos_enc(sampling_mean_fg, sampling_var_fg)[0]  # ((K-1)xP)xDx(6*n_freq+3)
		# pos_emb_bg = self.pos_enc(sampling_mean_bg, sampling_var_bg)[0]  # PxDx(6*n_freq+3)

		pos_emb_fg = self.pos_enc(sampling_mean_fg, sampling_var_fg)[0]  # ((K-1)xP)xDx(6*n_freq+3)
		pos_emb_bg = self.pos_enc(sampling_mean_bg, sampling_var_bg)[0]  # PxDx(6*n_freq+3)

		if mask_ratio > 0.0:
			# mask the last ratio of the pos emb
			n_mask = int(pos_emb_fg.shape[-1] * (1 - mask_ratio)) // 3 * 3 + 3
			# pos_emb_fg[..., n_mask:] *= 0
			pos_emb_bg[..., n_mask:] *= 0

		pos_emb_fg, pos_emb_bg = pos_emb_fg.flatten(0, 1)[idx], pos_emb_bg.flatten(0, 1)  # Mx(6*n_freq+3), (P*D)x(6*n_freq+3)

		# 3. Concatenate the embeddings with z_fg and z_bg features
		# Assuming z_fg and z_bg are repeated for each query point
		# Also assuming K is the first dimension of z_fg and we need to repeat it for each query point
		
		z_fg = z_fg[:, None, :].expand(-1, P*D, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xPxD)xC
		z_fg = z_fg[idx]  # MxC

		input_fg = torch.cat([pos_emb_fg, z_fg], dim=-1)
		input_bg = torch.cat([pos_emb_bg, z_bg.repeat(P*D, 1)], dim=-1) # (P*D)x(6*n_freq+3+C)

		# 4. Return required tensors
		return input_fg, input_bg, idx

	def forward(self, mean, var, z_slots, fg_transform, fg_slot_position, dens_noise=0., 
		 			fg_object_size=None, keep_ratio=0.0, mask_ratio=0.0, view_dirs=None):
		"""
		1. pos emb by Fourier
		2. for each slot, decode all points from coord and slot feature
		input:
			mean: P*D*3, P = (N*H*W)
			var: P*D*3*3, P = (N*H*W)
			view_dirs: P*3, P = (N*H*W)
			z_slots: KxC, K: #slots, C: #feat_dim
			z_slots_texture: KxC', K: #slots, C: #texture_dim
			fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
			fg_slot_position: (K-1)x3 in nss space
			dens_noise: Noise added to density
		"""
		K, C = z_slots.shape
		P, D = mean.shape[0], mean.shape[1]

		# debug(sampling_coor_fg, fg_slot_position, save_name='before_transform')

		if self.locality:
			outsider_idx = torch.any(mean.flatten(0,1).abs() > self.locality_ratio, dim=-1).unsqueeze(0).expand(K-1, -1) # (K-1)x(P*D)

		z_bg = z_slots[0:1, :]  # 1xC
		z_fg = z_slots[1:, :]  # (K-1)xC

		# debug(sampling_coor_fg, save_name='after_transform')

		input_fg, input_bg, idx = self.processQueries(mean, var, fg_transform, fg_slot_position, z_fg, z_bg, 
						keep_ratio=keep_ratio, mask_ratio=mask_ratio, fg_object_size=fg_object_size)
		
		tmp = self.b_before(input_bg)
		bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P*D, self.out_ch])  # (P*D)x4 -> 1x(P*D)x4

		tmp = self.f_before(input_fg)
		tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # Mx64

		latent_fg = self.f_after_latent(tmp)  # Mx64
		fg_raw_rgb = self.f_color(latent_fg) # Mx3
		# put back the removed query points, for indices between idx[i] and idx[i+1], put fg_raw_rgb[i] at idx[i]
		fg_raw_rgb_full = torch.zeros((K-1)*P*D, 3, device=fg_raw_rgb.device, dtype=fg_raw_rgb.dtype) # ((K-1)xP*D)x3
		fg_raw_rgb_full[idx] = fg_raw_rgb
		fg_raw_rgb = fg_raw_rgb_full.view([K-1, P*D, 3])  # ((K-1)xP*D)x3 -> (K-1)x(P*D)x3

		fg_raw_shape = self.f_after_shape(tmp) # Mx1
		fg_raw_shape_full = torch.zeros((K-1)*P*D, 1, device=fg_raw_shape.device, dtype=fg_raw_shape.dtype) # ((K-1)xP*D)x1
		fg_raw_shape_full[idx] = fg_raw_shape
		fg_raw_shape = fg_raw_shape_full.view([K - 1, P*D])  # ((K-1)xP*D)x1 -> (K-1)x(P*D), density

		if self.locality:
			fg_raw_shape[outsider_idx] *= 0
		fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)x(P*D)x4

		all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # Kx(P*D)x4
		raw_masks = F.relu(all_raws[:, :, -1:], True)  # Kx(P*D)x1
		masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # Kx(P*D)x1
		raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
		raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

		unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # Kx(P*D)x4
		masked_raws = unmasked_raws * masks
		raws = masked_raws.sum(dim=0)

		return raws, masked_raws, unmasked_raws, masks

class DecoderIPEVD(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, n_layers=3, locality=True, 
		  			locality_ratio=4/7, fixed_locality=False, use_viewdirs=False, n_freq_viewdirs=3, n_freq_bg=None):
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
		assert self.fixed_locality == True
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

		if n_freq_bg is not None:
			before_skip[0] = nn.Linear(input_dim+6*(n_freq_bg-n_freq), z_dim)
			after_skip[0] = nn.Linear(z_dim+input_dim+6*(n_freq_bg-n_freq), z_dim)
			self.pos_enc_bg = PositionalEncoding(max_deg=n_freq_bg)
		else:
			self.pos_enc_bg = PositionalEncoding(max_deg=n_freq)

		self.b_before = nn.Sequential(*before_skip)
		self.b_after = nn.Sequential(*after_skip)

		self.b_color = nn.Linear(z_dim, 3)
		self.b_shape = nn.Linear(z_dim, 1)

		self.pos_enc = PositionalEncoding(max_deg=n_freq)

		self.use_viewdirs = use_viewdirs
		self.n_freq = n_freq
		self.n_freq_viewdirs = n_freq_viewdirs

		if self.use_viewdirs:
			self.viewdirs_encoding = PositionalEncoding(max_deg=n_freq_viewdirs)

			# self.fg_rgb_net0 = nn.Linear(z_dim, z_dim)

			# self.fg_rgb_net1 = nn.Sequential(
			# 	nn.Linear(z_dim+6*n_freq_viewdirs+3, z_dim),
			# 	nn.ReLU(True),
			# )

			self.bg_rgb_net0 = nn.Linear(z_dim, z_dim)
			# init as identity
			self.bg_rgb_net0.weight.data.copy_(torch.eye(z_dim))
			self.bg_rgb_net0.bias.data.copy_(torch.zeros(z_dim))

			self.bg_rgb_net1 = nn.Sequential(
				nn.Linear(z_dim+6*n_freq_viewdirs+3, z_dim),
				nn.ReLU(True),
			)
			# init as identity
			self.bg_rgb_net1[0].weight.data[:, :z_dim].copy_(torch.eye(z_dim))
			self.bg_rgb_net1[0].bias.data.copy_(torch.zeros(z_dim))

	def processQueries(self, mean, var, fg_transform, fg_slot_position, z_fg, z_bg, 
					keep_ratio=0.0, mask_ratio=0.0, fg_object_size=None):
		'''
		Process the query points and the slot features
		1. If self.fg_object_size is not None, do:
			Remove the query point that is too far away from the slot center, 
			the bouding box is defined as a cube with side length 2 * self.fg_object_size
			for the points outside the bounding box, keep only keep_ratio of them
			store the new sampling_coor_fg and the indices of the remaining points
		2. Do the pos emb by Fourier
		3. Concatenate the pos emb and the slot features
		4. If self.fg_object_size is not None, return the new sampling_coor_fg and their indices

		input: 	mean: PxDx3
				var: PxDx3
				fg_transform: 1x4x4
				fg_slot_position: (K-1)x3
				z_fg: (K-1)xC
				z_bg: 1xC
				ssize: supervision size (64)
				mask_ratio: frequency mask ratio to the pos emb
		return: input_fg: M * (60 + C) (M is the number of query points inside bbox), C is the slot feature dim, and 60 means increased-freq feat dim
				input_bg: Px(60+C)
				idx: M (indices of the query points inside bbox)
		'''
		P, D = mean.shape[0], mean.shape[1]
		K = z_fg.shape[0] + 1
		
		sampling_mean_fg = mean[None, ...].expand(K-1, -1, -1, -1).flatten(1, 2) # (K-1)*(P*D)*3
		sampling_mean_fg = torch.cat([sampling_mean_fg, torch.ones_like(sampling_mean_fg[:, :, 0:1])], dim=-1)  # (K-1)*(P*D)*4
		sampling_mean_fg = torch.matmul(fg_transform[None, ...], sampling_mean_fg[..., None]).squeeze(-1)  # (K-1)*(P*D)*4x1
		sampling_mean_fg = sampling_mean_fg[:, :, :3]  # (K-1)*(P*D)*3

		# transform fg_slot_position to the camera coordinate
		fg_slot_position = torch.cat([fg_slot_position, torch.ones_like(fg_slot_position[:, 0:1])], dim=-1)  # (K-1)x4
		fg_slot_position = torch.matmul(fg_transform.squeeze(0), fg_slot_position.t()).t() # (K-1)x4
		# print('Debug: ', fg_slot_position.shape, sampling_coor_fg.shape)
		fg_slot_position = fg_slot_position[:, :3]  # (K-1)x3
		sampling_mean_fg = sampling_mean_fg - fg_slot_position[:, None, :]  # (K-1)x(P*D)x3
		sampling_mean_fg = sampling_mean_fg.view([K-1, P, D, 3]).flatten(0, 1)  # ((K-1)xP)xDx3

		sampling_var_fg = var[None, ...].expand(K-1, -1, -1, -1).flatten(0, 1)  # ((K-1)xP)xDx3

		sampling_mean_bg, sampling_var_bg = mean, var

		sampling_mean_fg_ = sampling_mean_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xPxD)x3

		# 1. Remove the query points too far away from the slot center
		if fg_object_size is not None:
			# remove abs(x) > fg_object_size | abs(y) > fg_object_size | z > fg_object_size
			mask = torch.all(torch.abs(sampling_mean_fg_) < fg_object_size, dim=-1)  # ((K-1)xP)
			if mask.sum() <= 1:
				mask[:2] = True
			# randomly take only keep_ratio of the points outside the bounding box
			mask = mask | (torch.rand(mask.shape, device=mask.device) < keep_ratio)
			# mask = mask & keep_idx if keep_idx is not None else mask
			idx = mask.nonzero().squeeze()  # Indices of valid points
			# print("Number of points inside bbox: ", idx.size(0), " out of ", (K-1)*P, 
			# 		'\n, ratio: ', idx.size(0) / ((K-1)*P))
		else:
			idx = torch.arange(sampling_mean_fg_.size(0))

		# pos_emb_fg = self.pos_enc(sampling_mean_fg, sampling_var_fg)[0]  # ((K-1)xP)xDx(6*n_freq+3)
		# pos_emb_bg = self.pos_enc(sampling_mean_bg, sampling_var_bg)[0]  # PxDx(6*n_freq+3)

		pos_emb_fg = self.pos_enc(sampling_mean_fg, sampling_var_fg)[0]  # ((K-1)xP)xDx(6*n_freq+3)
		pos_emb_bg = self.pos_enc_bg(sampling_mean_bg, sampling_var_bg)[0]  # PxDx(6*n_freq+3)

		if mask_ratio > 0.0:
			# mask the last ratio of the pos emb
			n_mask = int(pos_emb_fg.shape[-1] * (1 - mask_ratio)) // 3 * 3 + 3
			# pos_emb_fg[..., n_mask:] *= 0
			pos_emb_bg[..., n_mask:] *= 0

		pos_emb_fg, pos_emb_bg = pos_emb_fg.flatten(0, 1)[idx], pos_emb_bg.flatten(0, 1)  # Mx(6*n_freq+3), (P*D)x(6*n_freq+3)

		# 3. Concatenate the embeddings with z_fg and z_bg features
		# Assuming z_fg and z_bg are repeated for each query point
		# Also assuming K is the first dimension of z_fg and we need to repeat it for each query point
		
		z_fg = z_fg[:, None, :].expand(-1, P*D, -1).flatten(start_dim=0, end_dim=1)  # ((K-1)xPxD)xC
		z_fg = z_fg[idx]  # MxC

		input_fg = torch.cat([pos_emb_fg, z_fg], dim=-1)
		input_bg = torch.cat([pos_emb_bg, z_bg.repeat(P*D, 1)], dim=-1) # (P*D)x(6*n_freq+3+C)

		# 4. Return required tensors
		return input_fg, input_bg, idx

	def forward(self, mean, var, z_slots, fg_transform, fg_slot_position, dens_noise=0., 
		 			fg_object_size=None, keep_ratio=0.0, mask_ratio=0.0, view_dirs=None):
		"""
		1. pos emb by Fourier
		2. for each slot, decode all points from coord and slot feature
		input:
			mean: P*D*3, P = (N*H*W)
			var: P*D*3*3, P = (N*H*W)
			view_dirs: P*3, P = (N*H*W)
			z_slots: KxC, K: #slots, C: #feat_dim
			z_slots_texture: KxC', K: #slots, C: #texture_dim
			fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0, otherwise it is 1x3x3 azimuth rotation of nss2cam0
			fg_slot_position: (K-1)x3 in nss space
			dens_noise: Noise added to density
		"""
		K, C = z_slots.shape
		P, D = mean.shape[0], mean.shape[1]

		# debug(sampling_coor_fg, fg_slot_position, save_name='before_transform')

		if self.locality:
			outsider_idx = torch.any(mean.flatten(0,1).abs() > self.locality_ratio, dim=-1).unsqueeze(0).expand(K-1, -1) # (K-1)x(P*D)

		z_bg = z_slots[0:1, :]  # 1xC
		z_fg = z_slots[1:, :]  # (K-1)xC

		# debug(sampling_coor_fg, save_name='after_transform')

		input_fg, input_bg, idx = self.processQueries(mean, var, fg_transform, fg_slot_position, z_fg, z_bg, 
						keep_ratio=keep_ratio, mask_ratio=mask_ratio, fg_object_size=fg_object_size)
		
		tmp = self.b_before(input_bg) # (P*D)x64
		tmp = self.b_after(torch.cat([input_bg, tmp], dim=1)) # (P*D)x64
		bg_shape = self.b_shape(tmp) # (P*D)x1

		if self.use_viewdirs:
			if view_dirs is not None:
				viewdirs_encoding_bg = self.viewdirs_encoding(view_dirs) # Px(6*n_freq_viewdirs+3)
				viewdirs_encoding_bg = viewdirs_encoding_bg.unsqueeze(1).expand(P, D, -1).flatten(0, 1) # (P*D)x(6*n_freq_viewdirs+3)
			else:# use dummy encodings
				viewdirs_encoding_bg = torch.zeros(P*D, 6*self.n_freq_viewdirs+3).to(tmp.device)
			tmp = self.bg_rgb_net0(tmp) # (P*D)x64
			tmp = torch.cat([tmp, viewdirs_encoding_bg], dim=-1) # (P*D)x(64+6*n_freq_viewdirs+3)
			tmp = self.bg_rgb_net1(tmp) # (P*D)x64

		bg_color = self.b_color(tmp) # (P*D)x3
		bg_raws = torch.cat([bg_color, bg_shape], dim=-1).view([1, P*D, self.out_ch])  # (P*D)x4 -> 1x(P*D)x4

		# FG
		tmp = self.f_before(input_fg)
		tmp = self.f_after(torch.cat([input_fg, tmp], dim=1))  # Mx64

		fg_raw_shape = self.f_after_shape(tmp) # Mx1
		fg_raw_shape_full = torch.zeros((K-1)*P*D, 1, device=fg_raw_shape.device, dtype=fg_raw_shape.dtype) # ((K-1)xP*D)x1
		fg_raw_shape_full[idx] = fg_raw_shape
		fg_raw_shape = fg_raw_shape_full.view([K - 1, P*D])  # ((K-1)xP*D)x1 -> (K-1)x(P*D), density

		# if self.use_viewdirs:
		# 	if view_dirs is not None:
		# 		view_dirs = torch.matmul(fg_transform.squeeze(0)[:3,:3], view_dirs.t()).t() # P*3
		# 	else: # use dummy encodings
		# 		view_dirs = torch.zeros(P, 3).to(tmp.device)
		# 	viewdirs_encoding = self.viewdirs_encoding(view_dirs) # Px(6*n_freq_viewdirs+3)
		# 	viewdirs_encoding = viewdirs_encoding.unsqueeze(0).unsqueeze(-2).expand(K-1, -1, D, -1).flatten(0, 2) # ((K-1)*P*D)x(6*n_freq_viewdirs+3)
		# 	viewdirs_encoding = viewdirs_encoding[idx] # Mx(6*n_freq_viewdirs+3)

		# 	tmp = self.fg_rgb_net0(tmp) # Mx64
		# 	tmp = torch.cat([tmp, viewdirs_encoding], dim=-1) # Mx(64+6*n_freq_viewdirs+3)
		# 	tmp = self.fg_rgb_net1(tmp) # Mx64

		latent_fg = self.f_after_latent(tmp)  # Mx64
		fg_raw_rgb = self.f_color(latent_fg) # Mx3
		# put back the removed query points, for indices between idx[i] and idx[i+1], put fg_raw_rgb[i] at idx[i]
		fg_raw_rgb_full = torch.zeros((K-1)*P*D, 3, device=fg_raw_rgb.device, dtype=fg_raw_shape.dtype) # ((K-1)xP*D)x3
		fg_raw_rgb_full[idx] = fg_raw_rgb
		fg_raw_rgb = fg_raw_rgb_full.view([K-1, P*D, 3])  # ((K-1)xP*D)x3 -> (K-1)x(P*D)x3

		if self.locality:
			fg_raw_shape[outsider_idx] *= 0
		fg_raws = torch.cat([fg_raw_rgb, fg_raw_shape[..., None]], dim=-1)  # (K-1)x(P*D)x4

		all_raws = torch.cat([bg_raws, fg_raws], dim=0)  # Kx(P*D)x4
		raw_masks = F.relu(all_raws[:, :, -1:], True)  # Kx(P*D)x1
		masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # Kx(P*D)x1
		raw_rgb = (all_raws[:, :, :3].tanh() + 1) / 2
		raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

		unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # Kx(P*D)x4
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
			fg_position = fg_position.clamp(-1, 1) # BxKx2

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
	
class SlotAttentionAblation(nn.Module):
	def __init__(self, num_slots, in_dim=64, slot_dim=64, iters=3, eps=1e-8, hidden_dim=128, color_dim=0,
		 		learnable_init=False):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5

		self.learnable_init = learnable_init
		if not self.learnable_init:
			self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
			self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
			init.xavier_uniform_(self.slots_logsigma)
			self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
			self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
			init.xavier_uniform_(self.slots_logsigma_bg)
		else:
			self.slots_init_fg = nn.Parameter(torch.randn(1, num_slots-1, slot_dim))
			self.slots_init_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
			init.xavier_uniform_(self.slots_init_fg)
			init.xavier_uniform_(self.slots_init_bg)

		self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
		self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
		self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
		self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

		self.gru = nn.GRUCell(slot_dim, slot_dim)
		self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

		hidden_dim = max(slot_dim, hidden_dim)

		self.to_res = nn.Sequential(
			nn.LayerNorm(slot_dim),
			nn.Linear(slot_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, slot_dim)
		)
		self.to_res_bg = nn.Sequential(
			nn.LayerNorm(slot_dim),
			nn.Linear(slot_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, slot_dim)
		)

		self.norm_feat = nn.LayerNorm(in_dim)
		if color_dim != 0:
			self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

	def forward(self, feat, num_slots=None, feat_color=None):
		"""
		input:
			feat: visual feature with position information, BxNxC
		output: slots: BxKxC, attn: BxKxN
		"""
		B, _, _ = feat.shape
		K = num_slots if num_slots is not None else self.num_slots

		if not self.learnable_init:
			mu = self.slots_mu.expand(B, K-1, -1)
			sigma = self.slots_logsigma.exp().expand(B, K-1, -1)
			slot_fg = mu + sigma * torch.randn_like(mu)
			mu_bg = self.slots_mu_bg.expand(B, 1, -1)
			sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
			slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)
		else:
			slot_fg = self.slots_init_fg.expand(B, K-1, -1)
			slot_bg = self.slots_init_bg.expand(B, 1, -1)

		feat = self.norm_feat(feat)
		k = self.to_k(feat)
		v = self.to_v(feat)

		if feat_color is not None:
			feat_color = self.norm_feat_color(feat_color)

		attn = None
		for _ in range(self.iters):
			slot_prev_bg = slot_bg
			slot_prev_fg = slot_fg
			q_fg = self.to_q(slot_fg)
			q_bg = self.to_q_bg(slot_bg)

			dots_fg = torch.einsum('bid,bjd->bij', q_fg, k) * self.scale
			dots_bg = torch.einsum('bid,bjd->bij', q_bg, k) * self.scale
			dots = torch.cat([dots_bg, dots_fg], dim=1)  # BxKxN
			attn = dots.softmax(dim=1) + self.eps  # BxKxN
			attn_bg, attn_fg = attn[:, 0:1, :], attn[:, 1:, :]  # Bx1xN, Bx(K-1)xN
			attn_weights_bg = attn_bg / attn_bg.sum(dim=-1, keepdim=True)  # Bx1xN
			attn_weights_fg = attn_fg / attn_fg.sum(dim=-1, keepdim=True)  # Bx(K-1)xN

			updates_bg = torch.einsum('bjd,bij->bid', v, attn_weights_bg)
			updates_fg = torch.einsum('bjd,bij->bid', v, attn_weights_fg)

			slot_bg = self.gru_bg(
				updates_bg.reshape(-1, self.slot_dim),
				slot_prev_bg.reshape(-1, self.slot_dim)
			)
			slot_bg = slot_bg.reshape(B, -1, self.slot_dim)
			slot_bg = slot_bg + self.to_res_bg(slot_bg)

			slot_fg = self.gru(
				updates_fg.reshape(-1, self.slot_dim),
				slot_prev_fg.reshape(-1, self.slot_dim)
			)
			slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
			slot_fg = slot_fg + self.to_res(slot_fg)
		
		if feat_color is not None: # (B,N,C)
			slot_fg_color = torch.einsum('bkn,bnd->bkd', attn_weights_fg, feat_color) # (B,K-1,C)
			slot_bg_color = torch.einsum('bkn,bnd->bkd', attn_weights_bg, feat_color) # (B,1,C)
			slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1)
			slot_bg = torch.cat([slot_bg, slot_bg_color], dim=-1)

		slots = torch.cat([slot_bg, slot_fg], dim=1)
		return slots, attn

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
			# w/ and w/o residual connection
			z_fg = z_fg + self.position_project(fg_slot_position[:, :2]) # KxC
			# slot_position = torch.cat([torch.zeros_like(fg_slot_position[0:1,]), fg_slot_position], dim=0)[:,:2] # Kx2
			# z_slots = self.position_project(slot_position) + z_slots # KxC

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

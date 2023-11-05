from turtle import position
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image as Image, ImageEnhance
import math

from models.model_general import MultiRouteEncoderSeparate, SlotAttention, build_grid, MultiDINOEncoder

class DecoderPosEmbedding(nn.Module):
	def __init__(self, hidden_dim=128, scale_factor=5):
		super().__init__()
		self.grid_embed = nn.Linear(4, hidden_dim, bias=True)
		self.scale_factor = scale_factor

	def apply_rel_position_scale(self, grid, position, scale):
		"""
		grid: (1, h, w, 2)
		position (batch*number_slots, 2)
		scale (batch*number_slots, 2)
		"""
		h, w = grid.shape[1:3]
		bns = position.shape[0]
		grid = grid.expand(bns, h, w, 2)
		position = position.unsqueeze(1).unsqueeze(1).expand(bns, h, w, 2) # bns, h, w, 2

		if scale is None:
			return grid - position
		else:
			scale = scale.unsqueeze(1).unsqueeze(1).expand(bns, h, w, 2)
			return ((grid - position) / (scale * self.scale_factor + 1e-8))

	def forward(self, x, position_latent, scale_latent=None):
		'''
		x: (B*K, h, w, d)
		position_latent: (B*K, 2)
		'''
		H, W = x.shape[1:3]
		grid = build_grid(H, W, device=x.device) # (1, h, w, 2)
		rel_grid = self.apply_rel_position_scale(grid, position_latent, scale_latent) # (bns, h, w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1) # (bns, h, w, 4)

		grid_embed = self.grid_embed(rel_grid) # (bns, h, w, d)
		return x + grid_embed

class Decoder(nn.Module):
	def __init__(self, z_dim, input_size=(16,16), output_size=(128,128)):
		super().__init__()

		self.decode_list = []
		for _ in range(int(math.log2(output_size[0]//input_size[0]))):
			self.decode_list.append(nn.ConvTranspose2d(z_dim, z_dim, 5, stride=(2, 2), padding=2, output_padding=1))
			self.decode_list.append(nn.ReLU())

		self.decode_list = nn.Sequential(*self.decode_list)

		self.conv5 = nn.ConvTranspose2d(z_dim, z_dim, 5, stride=(1, 1), padding=2)
		self.conv6 = nn.ConvTranspose2d(z_dim, 4, 3, stride=(1, 1), padding=1)
		self.input_size = input_size
		self.decoder_pos = DecoderPosEmbedding(hidden_dim=z_dim)

	def forward(self, x, position_latent, scale_latent=None):

		x = self.decoder_pos(x, position_latent, scale_latent)
		x = x.permute(0,3,1,2) # B, C, H, W

		x = self.decode_list(x)
		x = self.conv5(x)
		x = F.relu(x)
		x = self.conv6(x)

		x = x.permute(0,2,3,1)
		return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionDualAutoEncoder(nn.Module):
	def __init__(self, num_slots, num_iterations=3, input_nc=3, pos_emb=False, bottom=False,
				 shape_dim=48, color_dim=16, dino_dim=768, learnable_slot_init=True,
				 n_feat_layer=4,
				 decoder_input_size=(16,16), decoder_output_size=(128,128)):
		"""Builds the Slot Attention-based auto-encoder.
		Args:
		resolution: Tuple of integers specifying width and height of input image.
		num_slots: Number of slots in Slot Attention.
		num_iterations: Number of iterations in Slot Attention.
		"""
		super().__init__()
		# self.resolution = resolution
		# self.encoder_resolution = encoder_resolution
		self.decoder_input_size = decoder_input_size
		self.decoder_output_size = decoder_output_size
		self.num_slots = num_slots
		self.num_iterations = num_iterations
		self.z_dim = shape_dim + color_dim

		self.encoder = MultiRouteEncoderSeparate(input_nc=input_nc, pos_emb=pos_emb, 
												bottom=bottom, shape_dim=shape_dim, 
												n_feat_layer=n_feat_layer,
												color_dim=color_dim, input_dim=dino_dim)
		self.slot_attention = SlotAttention(
			num_slots=num_slots, in_dim=shape_dim, slot_dim=shape_dim, 
			color_dim=color_dim, iters=num_iterations, learnable_init=learnable_slot_init, learnable_pos=False)
		
		self.decoder = Decoder(z_dim=self.z_dim, input_size=decoder_input_size, output_size=self.decoder_output_size)

	def forward(self, x, feature_maps):
		'''
		input:
			x: B*3*H*W, in [-1, 1]
			feature_maps: list of tensors of B*C*H*W
		'''
		# Convolutional encoder with position embedding.
		B, _, H, W = feature_maps[0].shape
		feature_map_shape, feature_map_color = self.encoder(feature_maps, x)

		feat_shape = feature_map_shape.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC
		feat_color = feature_map_color.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC

		# Slot Attention module.
		z_slots, attn, fg_slot_position = self.slot_attention(feat=feat_shape, feat_color=feat_color)  # BxKxC, BxKxN, Bx(K-1)x2
		# `slots` has shape: [batch_size, num_slots, slot_size].
		# fg_slot_position has shape: [batch_size, num_slots-1, 2]
		
		# """Broadcast slot features to a 2D grid and collapse slot dimension.""".
		z_slots = z_slots.flatten(0, 1).unsqueeze(1).unsqueeze(2)
		z_slots = z_slots.repeat((1, self.decoder_input_size[0], self.decoder_input_size[1], 1)) # (B*K, H, W, C)
		# pad zero for background slot
		slot_position = torch.cat([torch.zeros((B, 1, 2)).to(x.device), fg_slot_position], dim=1) # (B, K, 2)
		slot_position = slot_position.flatten(0, 1) # (B*K, 2)

		# `slots` has shape: [batch_size*num_slots, h, w, slot_size].
		x_comp = self.decoder(z_slots, slot_position) # x_comp: [B*K, H, W, num_channels+1].

		# Undo combination of slot and batch dimension; split alpha masks.
		recons, masks = x_comp.reshape(B, self.num_slots, x.shape[2], x.shape[3], -1).split([3,1], dim=-1)
		# apply Tanh to recons
		recons = torch.tanh(recons) # (-1, 1)
		attn = attn.reshape(B, self.num_slots, H, W).unsqueeze(2) # (B, K, 1, H, W)
		# `recons` has shape: [batch_size, num_slots, width, height, num_channels].
		# `masks` has shape: [batch_size, num_slots, width, height, 1].

		# Normalize alpha masks over slots.
		masks = nn.Softmax(dim=1)(masks)
		recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
		recon_combined = recon_combined.permute(0,3,1,2) # B, H, W, C
		recons = recons.permute(0,1,4,2,3) # B, K, C, H, W
		masks = masks.permute(0,1,4,2,3) # B, K, 1, H, W

		return recon_combined, recons, masks, attn

class SlotAttentionSingleAutoEncoder(nn.Module):
	def __init__(self, num_slots, num_iterations=3, shape_dim=48, 
			  	dino_dim=768, learnable_slot_init=True, n_feat_layer=4,
				 decoder_input_size=(16,16), decoder_output_size=(128,128)):
		"""Builds the Slot Attention-based auto-encoder.
		Args:
		resolution: Tuple of integers specifying width and height of input image.
		num_slots: Number of slots in Slot Attention.
		num_iterations: Number of iterations in Slot Attention.
		"""
		super().__init__()
		# self.resolution = resolution
		# self.encoder_resolution = encoder_resolution
		self.decoder_input_size = decoder_input_size
		self.decoder_output_size = decoder_output_size
		self.num_slots = num_slots
		self.num_iterations = num_iterations
		self.z_dim = shape_dim

		self.encoder = MultiDINOEncoder(shape_dim=shape_dim, n_feat_layer=n_feat_layer, input_dim=dino_dim)
		self.slot_attention = SlotAttention(
			num_slots=num_slots, in_dim=shape_dim, slot_dim=shape_dim, color_dim=0,
			  iters=num_iterations, learnable_init=learnable_slot_init, learnable_pos=False)
		
		self.decoder = Decoder(z_dim=self.z_dim, input_size=decoder_input_size, output_size=self.decoder_output_size)

	def forward(self, feature_maps):
		'''
		input:
			x: B*3*H*W, in [-1, 1]
			feature_maps: list of tensors of B*C*H*W
		'''
		# Convolutional encoder with position embedding.
		B, _, H, W = feature_maps[0].shape
		feature_map_shape = self.encoder(feature_maps)

		feat_shape = feature_map_shape.permute([0, 2, 3, 1]).contiguous()  # BxHxWxC

		# Slot Attention module.
		z_slots, attn, fg_slot_position = self.slot_attention(feat=feat_shape, feat_color=None)  # BxKxC, BxKxN, Bx(K-1)x2
		# `slots` has shape: [batch_size, num_slots, slot_size].
		# fg_slot_position has shape: [batch_size, num_slots-1, 2]
		
		# """Broadcast slot features to a 2D grid and collapse slot dimension.""".
		z_slots = z_slots.flatten(0, 1).unsqueeze(1).unsqueeze(2)
		z_slots = z_slots.repeat((1, self.decoder_input_size[0], self.decoder_input_size[1], 1)) # (B*K, H, W, C)
		# pad zero for background slot
		slot_position = torch.cat([torch.zeros((B, 1, 2)).to(z_slots.device), fg_slot_position], dim=1) # (B, K, 2)
		slot_position = slot_position.flatten(0, 1) # (B*K, 2)

		# `slots` has shape: [batch_size*num_slots, h, w, slot_size].
		x_comp = self.decoder(z_slots, slot_position) # x_comp: [B*K, H, W, num_channels+1].

		# Undo combination of slot and batch dimension; split alpha masks.
		recons, masks = x_comp.reshape(B, self.num_slots, x_comp.shape[1], x_comp.shape[2], -1).split([3,1], dim=-1)
		# apply Tanh to recons
		recons = torch.tanh(recons) # (-1, 1)
		attn = attn.reshape(B, self.num_slots, H, W).unsqueeze(2) # (B, K, 1, H, W)
		# `recons` has shape: [batch_size, num_slots, width, height, num_channels].
		# `masks` has shape: [batch_size, num_slots, width, height, 1].

		# Normalize alpha masks over slots.
		masks = nn.Softmax(dim=1)(masks)
		recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
		recon_combined = recon_combined.permute(0,3,1,2) # B, H, W, C
		recons = recons.permute(0,1,4,2,3) # B, K, C, H, W
		masks = masks.permute(0,1,4,2,3) # B, K, 1, H, W

		return recon_combined, recons, masks, attn

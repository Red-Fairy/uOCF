import math
from os import X_OK

from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd
from models.model_general import InputPosEmbedding, build_grid

class EncoderPosEmbedding(nn.Module): # remove positional embedding on value
	def __init__(self, dim, slot_dim, hidden_dim=128):
		super().__init__()
		self.grid_embed = nn.Linear(4, dim, bias=True)
		self.input_to_k_fg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_fg = nn.Linear(dim, dim, bias=False)

		self.input_to_k_bg = nn.Linear(dim, dim, bias=False)
		self.input_to_v_bg = nn.Linear(dim, dim, bias=False)

		self.MLP_fg = nn.Linear(dim, slot_dim, bias=False)

		self.MLP_bg = nn.Linear(dim, slot_dim, bias=False)

		
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
		  random_init_pos=False, pos_no_grad=False, learnable_init=False, dropout=0.):
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

		self.learnable_pos = learnable_pos
		self.fg_position = nn.Parameter(torch.rand(1, num_slots-1, 2) * 1.5 - 0.75)

		if self.learnable_pos:
			self.attn_to_pos_bias = nn.Sequential(nn.Linear(n_feats, 2), nn.Tanh()) # range (-1, 1)
			self.attn_to_pos_bias[0].weight.data.zero_()
			self.attn_to_pos_bias[0].bias.data.zero_()

		self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)
		self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
		self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

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

		self.dropout = nn.Dropout(dropout)

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

				slot_bg = slot_prev_bg + self.dropout(updates_bg)

				slot_fg = slot_prev_fg + self.dropout(updates_fg)

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
	
class SlotAttentionAnchor(nn.Module):
	def __init__(self, num_slots, in_dim=64, slot_dim=64, color_dim=8, iters=4, eps=1e-8, hidden_dim=128,
		  learnable_pos=True, n_feats=64*64, global_feat=False, feat_dropout_dim=None,
		  dropout=0., momentum=0.5):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5
		self.pos_momentum = momentum

		self.anchors = torch.Tensor([[-0.5, -0.5], [-0.5, 0], [-0.5, 0.5], 
							   		[0, -0.5], [0, 0], [0, 0.5], 
									[0.5, -0.5], [0.5, 0], [0.5, 0.5]]) # (9, 2)

		self.query_init = nn.Linear(2, slot_dim) # init query with anchor position
		self.slots_init_bg = nn.Parameter((torch.randn(1, 1, slot_dim)))

		self.learnable_pos = learnable_pos

		if self.learnable_pos:
			self.attn_to_pos_bias = nn.Sequential(nn.Linear(n_feats, 2), nn.Tanh()) # range (-1, 1)
			self.attn_to_pos_bias[0].weight.data.zero_()
			self.attn_to_pos_bias[0].bias.data.zero_()

		self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)
		self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
		self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

		self.norm_feat = nn.LayerNorm(in_dim)
		if color_dim != 0:
			self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

		self.global_feat = global_feat

		self.dropout_shape_dim = feat_dropout_dim

		self.dropout = nn.Dropout(dropout)

	def forward(self, feat, feat_color=None, num_slots=None, dropout_shape_rate=None, 
			dropout_all_rate=None):
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

		# random take num_slots anchors
		fg_position = self.anchors[torch.randperm(self.anchors.shape[0])[:K-1]].unsqueeze(0).expand(B, -1, -1).to(feat.device) # (B, K-1, 2)
		slot_fg = self.query_init(fg_position) # (B, K-1, C)
		slot_bg = self.slots_init_bg.expand(B, 1, -1) # (B, 1, C)
		
		feat = self.norm_feat(feat)
		if feat_color is not None:
			feat_color = self.norm_feat_color(feat_color)

		k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2) # (1,N,2)

		# attn = None
		for it in range(self.iters):
			fg_position_prev = fg_position
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
			
			# momentum update slot position
			fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
			fg_position = fg_position * (1 - self.pos_momentum) + fg_position_prev * self.pos_momentum

			if it != self.iters - 1:
				updates_fg = torch.empty(B, K-1, self.slot_dim, device=k.device) # (B,K-1,C)
				for i in range(K-1):
					v_i = v[:, i] # (B,N,C)
					attn_i = attn_weights_fg[:, i] # (B,N)
					updates_fg[:, i] = torch.einsum('bn,bnd->bd', attn_i, v_i)

				updates_bg = torch.einsum('bn,bnd->bd',attn_weights_bg.squeeze(1), v_bg.squeeze(1)) # (B,N,C) * (B,N) -> (B,C)
				updates_bg = updates_bg.unsqueeze(1) # (B,1,C)

				slot_bg = slot_prev_bg + self.dropout(updates_bg)
				slot_fg = slot_prev_fg + self.dropout(updates_fg)

			else:
				if self.learnable_pos: # add a bias term
					fg_position = fg_position + self.attn_to_pos_bias(attn_weights_fg) / 5 # (B,K-1,2)
					fg_position = fg_position.clamp(-1, 1) # (B,K-1,2)
					
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
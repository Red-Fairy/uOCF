import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from models.model_general import InputPosEmbedding
from .utils import PositionalEncoding, build_grid
import itertools
from .transformer_attn import EncoderPosEmbedding, AdaLN


class OutputPositionEmbedding(nn.Module):
	def __init__(self, slot_dim, out_dim=768):
		super().__init__()
		self.grid_embed = nn.Linear(4, slot_dim, bias=True)
		self.mlp = nn.Linear(slot_dim, out_dim, bias=True)
		self.norm = nn.LayerNorm(out_dim)
		
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
		'''
		x: (b, n_slot-1, slot_dim)
		'''

		grid = build_grid(h, w, x.device) # (1, h, w, 2)
		if position_latent is not None:
			rel_grid = self.apply_rel_position_scale(grid, position_latent)
		else:
			rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)

		# rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
		rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, n_slot-1, h*w, 4)
		grid_embed = self.grid_embed(rel_grid) # (b, n_slot-1, h*w, d)

		x = x.unsqueeze(-2).repeat(1, 1, h*w, 1) # (B,K-1,h*w,slot_dim)
		x = self.mlp(x + grid_embed)
		x = self.norm(x)

		return x # (b, n_slot-1, h*w, d)


class SlotAttentionTransformer(nn.Module):
	def __init__(self, num_slots, in_dim=64, slot_dim=64, color_dim=8, iters=4, eps=1e-8,
		  learnable_pos=True, n_feats=64*64, 
		  momentum=0.5, pos_init='learnable', depth_scale_pred=False,
		  camera_dim=5, camera_modulation=False,
		  spatial_feat=False):
		super().__init__()
		self.num_slots = num_slots
		self.iters = iters
		self.eps = eps
		self.scale = slot_dim ** -0.5
		self.pos_momentum = momentum
		self.pos_init = pos_init

		if self.pos_init == 'learnable':
			self.fg_position = nn.Parameter(torch.rand(1, num_slots-1, 2) * 1.5 - 0.75)
		
		self.slots_init_fg = nn.Parameter((torch.randn(1, num_slots-1, slot_dim)))
		self.slots_init_bg = nn.Parameter((torch.randn(1, 1, slot_dim)))

		self.learnable_pos = learnable_pos
		if self.learnable_pos:
			self.attn_to_pos_bias = nn.Sequential(nn.Linear(n_feats, 2), nn.Tanh()) # range (-1, 1)
			self.attn_to_pos_bias[0].weight.data.zero_()
			self.attn_to_pos_bias[0].bias.data.zero_()
		
		self.depth_scale_pred = depth_scale_pred
		if depth_scale_pred:
			self.scale_bias = nn.Sequential(nn.Linear(2+camera_dim, 1), nn.Tanh()) # range (-1, 1)
			self.scale_bias[0].weight.data.zero_()
			self.scale_bias[0].bias.data.zero_()

		self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)

		self.to_q_fg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.to_q_fg =  nn.Linear(slot_dim, slot_dim, bias=False)
		self.to_q_bg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.to_q_bg =  nn.Linear(slot_dim, slot_dim, bias=False)

		self.norm_feat = nn.LayerNorm(in_dim)
		self.norm_feat_color = nn.LayerNorm(color_dim)
		self.slot_dim = slot_dim

		self.mlp_fg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.mlp_fg = nn.Sequential(nn.Linear(slot_dim, slot_dim), 
							  nn.GELU(), nn.Linear(slot_dim, slot_dim))
		self.mlp_bg_AdaLN = AdaLN(camera_dim, slot_dim, condition=camera_modulation)
		self.mlp_bg = nn.Sequential(nn.Linear(slot_dim, slot_dim),
							  nn.GELU(), nn.Linear(slot_dim, slot_dim))
		
		# self.norm_global_feat = nn.LayerNorm(768) # 768 is the dimension of the global feature (DINO)
		self.spatial_feat = spatial_feat
		DINO_dim = 768
		if self.spatial_feat:
			self.generate_cond = OutputPositionEmbedding(slot_dim+color_dim, out_dim=DINO_dim)
		else:
			self.generate_cond = nn.Sequential(nn.Linear(slot_dim+color_dim, DINO_dim), 
												nn.LayerNorm(DINO_dim))


	def forward(self, feat, feat_color, camera_modulation, 
				# global_feat, class_token, triplane_generator,
				num_slots=None, remove_duplicate=False, triplane_size=16
				):
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
		
		if self.pos_init == 'learnable':
			fg_position = self.fg_position.expand(B, -1, -1).to(feat.device)
		elif self.pos_init == 'random':
			fg_position = torch.rand(B, K-1, 2, device=feat.device) * 1.8 - 0.9 # (B, K-1, 2)
		else: # zero init
			fg_position = torch.zeros(B, K-1, 2, device=feat.device)

		slot_fg = self.slots_init_fg.expand(B, -1, -1) # (B, K-1, C)
		slot_bg = self.slots_init_bg.expand(B, 1, -1) # (B, 1, C)
		
		feat = self.norm_feat(feat)

		k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

		grid = build_grid(H, W, device=feat.device).flatten(1, 2) # (1,N,2)

		for it in range(self.iters):
			n_remove = 0
			if remove_duplicate and it == self.iters - 1:
				remove_idx = []
				with torch.no_grad():
					# calculate similarity matrix between slots
					slot_fg_norm = slot_fg / slot_fg.norm(dim=-1, keepdim=True) # (B,K-1,C)
					similarity_matrix = torch.matmul(slot_fg_norm, slot_fg_norm.transpose(1, 2)) # (B,K-1,K-1)
					pos_diff = fg_position.unsqueeze(2) - fg_position.unsqueeze(1) # (B,K-1,1,2) - (B,1,K-1,2) -> (B,K-1,K-1,2)
					pos_diff_norm = pos_diff.norm(dim=-1) # (B,K-1,K-1)
					for i in range(K-1): # if similarity_matrix[i,j] > 0.75 and pos_diff_norm[i,j] < 0.1, then remove slot j
						if i in remove_idx:
							continue
						for j in range(i+1, K-1):
							if similarity_matrix[:, i, j] > 0.75 and pos_diff_norm[:, i, j] < 0.15 and j not in remove_idx:
								remove_idx.append(j)
					# shift the index (remove the duplicate)
					remove_idx = sorted(remove_idx)
					shuffle_idx = [i for i in range(K-1) if i not in remove_idx]
					# shuffle_idx.extend(remove_idx)
					slot_fg = slot_fg[:, shuffle_idx]
					fg_position = fg_position[:, shuffle_idx]
					n_remove = len(remove_idx)

			q_fg = self.to_q_fg(self.to_q_fg_AdaLN(slot_fg, camera_modulation)) # (B,K-1,C)
			q_bg = self.to_q_bg(self.to_q_bg_AdaLN(slot_bg, camera_modulation)) # (B,1,C)
		
			attn = torch.empty(B, K-n_remove, N, device=feat.device)
			
			k, v = self.to_kv(feat, H, W, fg_position) # (B,K-1,N,C), (B,K-1,N,C)
			
			for i in range(K-n_remove):
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
			# fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
			fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) * (1 - self.pos_momentum) + fg_position * self.pos_momentum

			if it != self.iters - 1:
				updates_fg = torch.empty(B, K-1-n_remove, self.slot_dim, device=k.device) # (B,K-1,C)
				for i in range(K-1-n_remove):
					v_i = v[:, i] # (B,N,C)
					attn_i = attn_weights_fg[:, i] # (B,N)
					updates_fg[:, i] = torch.einsum('bn,bnd->bd', attn_i, v_i)

				updates_bg = torch.einsum('bn,bnd->bd',attn_weights_bg.squeeze(1), v_bg.squeeze(1)) # (B,N,C) * (B,N) -> (B,C)
				updates_bg = updates_bg.unsqueeze(1) # (B,1,C)

				slot_bg = slot_bg + updates_bg
				slot_fg = slot_fg + updates_fg

				slot_bg = slot_bg + self.mlp_bg(self.mlp_bg_AdaLN(slot_bg, camera_modulation))
				slot_fg = slot_fg + self.mlp_fg(self.mlp_fg_AdaLN(slot_fg, camera_modulation))

			else:
				if self.learnable_pos: # add a bias term
					fg_position = fg_position + self.attn_to_pos_bias(attn_weights_fg) * 0.1 # (B,K-1,2)
					# fg_position = fg_position.clamp(-1, 1) # (B,K-1,2)
				
				if self.depth_scale_pred:
					fg_depth_scale = self.scale_bias(torch.cat([fg_position, camera_modulation.unsqueeze(1).repeat(1, fg_position.shape[1], 1)], dim=-1)) / 4 + 1 # (B,K-1,1)
				else:
					fg_depth_scale = torch.ones(B, K-1-n_remove, 1, device=feat.device)
					
				# calculate slot color feature
				feat_color = self.norm_feat_color(feat_color)
				feat_color = feat_color.flatten(1, 2) # (B,N,C')
				slot_fg_color = torch.einsum('bkn,bnd->bkd', attn_weights_fg, feat_color) # (B,K-1,N) * (B,N,C') -> (B,K-1,C')
				slot_bg_color = torch.einsum('bn,bnd->bd', attn_weights_bg.squeeze(1), feat_color).unsqueeze(1) # (B,N) * (B,N,C') -> (B,C'), (B,1,C')

				# generate slot-specific triplane, global_feat: B*768*64*64
				# with torch.no_grad():
				# 	global_feat = global_feat.flatten(2, 3).unsqueeze(1).repeat(1, K-1, 1, 1) # (B, K-1, 768, N)
				# 	slot_global_feat = (attn_weights_fg.unsqueeze(2) * global_feat).flatten(0, 1) # (B*(K-1), 768, N)
				# 	slot_global_feat = F.interpolate(slot_global_feat.view(-1, 768, H, W), size=(16, 16), mode='bilinear', align_corners=True).flatten(-2, -1).permute(0, 2, 1) # (B*(K-1), 16*16, 768)
				# 	slot_global_feat = self.norm_global_feat(slot_global_feat) # (B*(K-1), 16*16, 768)
				# 	triplane_fg = triplane_generator(slot_global_feat,
				# 									torch.Tensor([[1.00,  0.00,  0.00,  0.00,  0.00,  0.00, -1.00, -2.00,
				# 													0.00,  1.00,  0.00,  0.00,  0.75,  0.75,  0.50,  0.50]]).to(global_feat.device).repeat(B*(K-1), 1))

		# generate slot-specific features
		slot_fg = torch.cat([slot_fg, slot_fg_color], dim=-1) # (B,K-1,C+C')
		if self.spatial_feat: # spatial broadcast and positional encoding
			feats_fg = self.generate_cond(slot_fg, triplane_size, triplane_size, fg_position) # (B,K-1,16*16,DINO_dim) 
		else:
			feats_fg = self.generate_cond(slot_fg).unsqueeze(-2) # (B, K-1, 1, DINO_dim)
		slot_bg = torch.cat([slot_bg, slot_bg_color], dim=-1) # (B,1,C+C')
		
		return slot_bg, attn, fg_position, fg_depth_scale, feats_fg
		# return slot_bg, attn, fg_position, fg_depth_scale, triplane_fg

class ModLN(nn.Module):
	"""
	Modulation with adaLN.
	
	References:
	DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
	"""
	def __init__(self, inner_dim: int, mod_dim: int, eps: float):
		super().__init__()
		self.norm = nn.LayerNorm(inner_dim, eps=eps)
		self.mlp = nn.Sequential(
			nn.SiLU(),
			nn.Linear(mod_dim, inner_dim * 2),
		)

	@staticmethod
	def modulate(x, shift, scale):
		# x: [N, L, D]
		# shift, scale: [N, D]
		return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

	def forward(self, x, cond):
		shift, scale = self.mlp(cond).chunk(2, dim=-1)  # [N, D]
		return self.modulate(self.norm(x), shift, scale)  # [N, L, D]


class ConditionModulationBlock(nn.Module):
	"""
	Transformer block that takes in a cross-attention condition and another modulation vector applied to sub-blocks.
	"""
	# use attention from torch.nn.MultiHeadAttention
	# Block contains a cross-attention layer, a self-attention layer, and a MLP
	def __init__(self, inner_dim: int, cond_dim: int, mod_dim: int, num_heads: int, eps: float,
				 attn_drop: float = 0., attn_bias: bool = False,
				 mlp_ratio: float = 4., mlp_drop: float = 0.):
		super().__init__()
		self.norm1 = ModLN(inner_dim, mod_dim, eps)
		self.cross_attn = nn.MultiheadAttention(
			embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
			dropout=attn_drop, bias=attn_bias, batch_first=True)
		self.norm2 = ModLN(inner_dim, mod_dim, eps)
		self.self_attn = nn.MultiheadAttention(
			embed_dim=inner_dim, num_heads=num_heads,
			dropout=attn_drop, bias=attn_bias, batch_first=True)
		self.norm3 = ModLN(inner_dim, mod_dim, eps)
		self.mlp = nn.Sequential(
			nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
			nn.GELU(),
			nn.Dropout(mlp_drop),
			nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
			nn.Dropout(mlp_drop),
		)

	def forward(self, x, cond, mod):
		# x: [N, L, D]
		# cond: [N, L_cond, D_cond]
		# mod: [N, D_mod]
		x = x + self.cross_attn(self.norm1(x, mod), cond, cond)[0]
		before_sa = self.norm2(x, mod)
		x = x + self.self_attn(before_sa, before_sa, before_sa)[0]
		x = x + self.mlp(self.norm3(x, mod))
		return x


class TriplaneTransformer(nn.Module):
	"""
	Transformer with condition and modulation that generates a triplane representation.
	
	Reference:
	Timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L486
	"""
	def __init__(self, inner_dim: int, image_feat_dim: int, camera_embed_dim: int,
				 triplane_low_res: int, triplane_high_res: int, triplane_dim: int,
				 num_layers: int, num_heads: int,
				 eps: float = 1e-6):
		super().__init__()

		# attributes
		self.triplane_low_res = triplane_low_res
		self.triplane_high_res = triplane_high_res
		self.triplane_dim = triplane_dim

		# modules
		# initialize pos_embed with 1/sqrt(dim) * N(0, 1)
		self.pos_embed = nn.Parameter(torch.randn(1, 3*triplane_low_res**2, inner_dim) * (1. / inner_dim) ** 0.5)
		self.layers = nn.ModuleList([
			ConditionModulationBlock(
				inner_dim=inner_dim, cond_dim=image_feat_dim, mod_dim=camera_embed_dim, num_heads=num_heads, eps=eps)
			for _ in range(num_layers)
		])
		self.norm = nn.LayerNorm(inner_dim, eps=eps)
		self.deconv = nn.ConvTranspose2d(inner_dim, triplane_dim, kernel_size=2, stride=2, padding=0)

	def forward(self, image_feats, camera_embeddings):
		# image_feats: [N, L_cond, D_cond]
		# camera_embeddings: [N, D_mod]

		assert image_feats.shape[0] == camera_embeddings.shape[0], \
			f"Mismatched batch size: {image_feats.shape[0]} vs {camera_embeddings.shape[0]}"

		N = image_feats.shape[0]
		H = W = self.triplane_low_res
		L = 3 * H * W

		x = self.pos_embed.repeat(N, 1, 1)  # [N, L, D]
		for layer in self.layers:
			x = layer(x, image_feats, camera_embeddings)
		x = self.norm(x)

		# separate each plane and apply deconv
		x = x.view(N, 3, H, W, -1)
		x = torch.einsum('nihwd->indhw', x)  # [3, N, D, H, W]
		x = x.contiguous().view(3*N, -1, H, W)  # [3*N, D, H, W]
		x = self.deconv(x)  # [3*N, D', H', W']
		x = x.view(3, N, *x.shape[-3:])  # [3, N, D', H', W']
		x = torch.einsum('indhw->nidhw', x)  # [N, 3, D', H', W']
		x = x.contiguous()

		assert self.triplane_high_res == x.shape[-2], \
			f"Output triplane resolution does not match with expected: {x.shape[-2]} vs {self.triplane_high_res}"
		assert self.triplane_dim == x.shape[-3], \
			f"Output triplane dimension does not match with expected: {x.shape[-3]} vs {self.triplane_dim}"

		return x

class CameraEmbedder(nn.Module):
	"""
	Embed camera features to a high-dimensional vector.
	
	Reference:
	DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L27
	"""
	def __init__(self, raw_dim: int, embed_dim: int):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(raw_dim, embed_dim),
			nn.SiLU(),
			nn.Linear(embed_dim, embed_dim),
		)

	def forward(self, x):
		return self.mlp(x)

class TriplaneSynthesizerTransformer(nn.Module):
	"""
	Transformer with condition and modulation that generates a triplane representation.
	"""
	def __init__(self, camera_embed_dim: int = 1024,
				 transformer_dim: int = 1024, transformer_layers: int = 12, transformer_heads: int = 16,
				 triplane_low_res: int = 32, triplane_high_res: int = 64, triplane_dim: int = 40,
				 encoder_feat_dim: int = 768, pretrained: bool = True):
		super().__init__()
		
		# attributes
		self.encoder_feat_dim = encoder_feat_dim
		self.camera_embed_dim = camera_embed_dim

		# modules
		self.camera_embedder = CameraEmbedder(
			raw_dim=12+4, embed_dim=camera_embed_dim,
		)
		self.transformer = TriplaneTransformer(
			inner_dim=transformer_dim, num_layers=transformer_layers, num_heads=transformer_heads,
			image_feat_dim=encoder_feat_dim,
			camera_embed_dim=camera_embed_dim,
			triplane_low_res=triplane_low_res, triplane_high_res=triplane_high_res, triplane_dim=triplane_dim,
		)

		if pretrained:
			self.load_pretrained()

	def load_pretrained(self): # load pretrained weights
		embedder_path = './checkpoints/camera_embedder.pth'
		self.camera_embedder.load_state_dict(torch.load(embedder_path))
		transformer_path = './checkpoints/transformer.pth'
		self.transformer.load_state_dict(torch.load(transformer_path))
		return

	def forward(self, image_feats, camera):
		# image_feats: [N, encoder_dim, H, W]
		# camera: [N, D_cam_raw]
		assert image_feats.shape[0] == camera.shape[0], "Batch size mismatch for image and camera"
		assert image_feats.shape[-1] == self.encoder_feat_dim, \
			f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

		# embed camera
		camera_embeddings = self.camera_embedder(camera)
		assert camera_embeddings.shape[-1] == self.camera_embed_dim, \
			f"Feature dimension mismatch: {camera_embeddings.shape[-1]} vs {self.camera_embed_dim}"

		# transformer generating planes
		planes = self.transformer(image_feats, camera_embeddings)
		assert planes.shape[0] == image_feats.shape[0], "Batch size mismatch for planes"
		assert planes.shape[1] == 3, "Planes should have 3 channels"

		return planes

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from models.model_general import InputPosEmbedding
from .utils import PositionalEncoding, build_grid
import itertools

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.

    Bugfix reference: https://github.com/NVlabs/eg3d/issues/67
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)

def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=2):
    assert padding_mode == 'zeros'
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.view(N*n_planes, C, H, W)

    coordinates = (2/box_warp) * coordinates # add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features

class SelfAttentionBlock(nn.Module):
	"""
	Transformer block with self-attention.
	"""
	# use attention from torch.nn.MultiHeadAttention
	# Block contains a cross-attention layer, a self-attention layer, and a MLP
	def __init__(self, inner_dim: int, num_heads: int, eps: float,
				 attn_drop: float = 0., attn_bias: bool = False,
				 mlp_ratio: float = 1., mlp_drop: float = 0.):
		super().__init__()
		self.self_attn = nn.MultiheadAttention(
			embed_dim=inner_dim, num_heads=num_heads,
			dropout=attn_drop, bias=attn_bias, batch_first=True)
		self.norm = nn.LayerNorm(inner_dim, eps=eps)
		self.mlp = nn.Sequential(
			nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
			nn.GELU(),
			nn.Dropout(mlp_drop),
			nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
			nn.Dropout(mlp_drop),
		)

	def forward(self, x):
		# x: [N, L, D]
		x = x + self.self_attn(x, x, x)[0]
		x = x + self.mlp(self.norm(x))
		return x

class TriplaneSynthesizer(nn.Module):
	'''
	generate triplane from a feature vector
	1. spatial broadcast (N, C) -> (N, H, W, C), N = batch, H, W: height and width of the triplane
	2. add positional encoding (sinusoidal) to the spatial broadcast (N, H, W, C) -> (N, H, W, C)
	3. self-attention blocks (N, H, W, C) -> (N, H, W, C)
	4. 1x1 conv to get the final triplane (N, H, W, C) -> (N, H, W, out_dim*3)
	'''
	def __init__(self, in_dim=96, out_dim=40, num_heads=4, num_layers=4, eps=1e-5, max_deg=5):
		super().__init__()
		self.in_dim = in_dim
		self.out_dim = out_dim
		self.num_heads = num_heads
		self.num_layers = num_layers
		self.eps = eps

		self.pos_enc = PositionalEncoding(max_deg=max_deg)
		self.pos_enc_proj = nn.Linear(4*max_deg+2, in_dim)
		init.zeros_(self.pos_enc_proj.weight)
		init.zeros_(self.pos_enc_proj.bias)

		self.self_attn_blocks = nn.ModuleList([SelfAttentionBlock(in_dim, num_heads, eps) 
												for _ in range(num_layers)])
		self.out_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim*3, kernel_size=1, stride=1, padding=0)

	def forward(self, x, H=64, W=64):
		# x: (N, C)
		N, C = x.shape
		# 1. spatial broadcast
		x = x.view([N, 1, 1, C]).expand(-1, H, W, -1)  # (N, H, W, C)
		# 2. add positional encoding
		grid = build_grid(H, W, device=x.device)  # (1, H, W, 2)
		x = x + self.pos_enc_proj(self.pos_enc(grid))  # (N, H, W, C)
		x = x.flatten(1, 2)  # (N, H*W, C)
		# 3. self-attention blocks
		for block in self.self_attn_blocks:
			x = block(x) # (N, H*W, C)
		# 4. 1x1 conv
		x = self.out_conv(x.view([N, H, W, -1]).permute(0, 3, 1, 2)).view([N, 3, self.out_dim, H, W])  # (N, 3, out_dim, H, W)
		return x

class OSGDecoder(nn.Module):
	"""
	Triplane decoder that gives RGB and sigma values from sampled features.
	Using ReLU here instead of Softplus in the original implementation.
	
	Reference:
	EG3D: https://github.com/NVlabs/eg3d/blob/main/eg3d/training/triplane.py#L112
	"""
	def __init__(self, n_features: int,
				 hidden_dim: int = 64, num_layers: int = 4, activation: nn.Module = nn.ReLU,
				 load_path='./checkpoints/decoder.pth'):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(3 * n_features, hidden_dim),
			activation(),
			*itertools.chain(*[[
				nn.Linear(hidden_dim, hidden_dim),
				activation(),
			] for _ in range(num_layers - 2)]),
			nn.Linear(hidden_dim, 1 + 3),
		)
		# init all bias to zero
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.zeros_(m.bias)
		# load pretrained weights
		if load_path is not None:
			self.load_state_dict(torch.load(load_path))

	def forward(self, sampled_features):
		# Aggregate features by mean
		# sampled_features = sampled_features.mean(1)
		# Aggregate features by concatenation
		_N, n_planes, _M, _C = sampled_features.shape
		sampled_features = sampled_features.permute(0, 2, 1, 3).reshape(_N, _M, n_planes*_C)
		x = sampled_features

		N, M, C = x.shape
		x = x.contiguous().view(N*M, C)

		x = self.net(x)
		x = x.view(N, M, -1)
		rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
		sigma = x[..., 0:1]

		return rgb, sigma

class DecoderTriplane(nn.Module):
	def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, triplane_dim=40, n_layers=3, locality=True, 
		  			locality_ratio=4/7, fixed_locality=False, pre_generate_triplane=False,
					mlp_act='relu', density_act='relu'):
		"""
		freq: raised frequency
		input_dim: pos emb dim + slot dim
		z_dim: network latent dim
		n_layers: #layers before/after skip connection.mlp_act
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

		self.plane_axes = generate_planes()  # 3x3x3
		if not pre_generate_triplane:
			self.triplane_synthesizer = TriplaneSynthesizer(in_dim=z_dim, out_dim=triplane_dim, 
									num_heads=4, num_layers=4, eps=1e-5, max_deg=n_freq)
		else:
			self.triplane_synthesizer = None

		activation_mlp = self._build_activation(mlp_act)
		self.fg_decoder = OSGDecoder(n_features=triplane_dim, hidden_dim=64, num_layers=4, activation=activation_mlp)

		before_skip = [nn.Linear(input_dim, z_dim), activation_mlp()]
		after_skip = [nn.Linear(z_dim + input_dim, z_dim), activation_mlp()]
		for _ in range(n_layers - 1):
			before_skip.append(nn.Linear(z_dim, z_dim))
			before_skip.append(activation_mlp())
			after_skip.append(nn.Linear(z_dim, z_dim))
			after_skip.append(activation_mlp())
		after_skip.append(nn.Linear(z_dim, self.out_ch))
		self.b_before = nn.Sequential(*before_skip)
		self.b_after = nn.Sequential(*after_skip)

		self.pos_enc = PositionalEncoding(max_deg=n_freq)

		if density_act == 'relu':
			self.density_act = torch.relu
		elif density_act == 'softplus':
			self.density_act = lambda x: F.softplus(x - 1) 
		else:
			assert False, 'density_act should be relu or softplus'

	def processQueries(self, mean, var, fg_transform, fg_slot_position, triplane_fg, z_bg, fg_object_size=None,
					rel_pos=True, bg_rotate=False):
		'''
		Process the query points and the slot features
		1. If self.fg_object_size is not None, do:
			Remove the query point that is too far away from the slot center, 
			the bouding box is defined as a cube with side length 2 * self.fg_object_size
			for the points outside the bounding box, keep only keep_ratio of them
			store the new sampling_coor_fg and the indices of the remaining points
		2. For background, do the pos emb by Fourier; for the foreground, sample the triplane features
		3. Concatenate the pos emb and the slot features
		4. If self.fg_object_size is not None, return the new sampling_coor_fg and their indices

		input: 	mean: PxDx3
				var: PxDx3
				fg_transform: 1x4x4
				fg_slot_position: (K-1)x3
				triplane_fg: (K-1)x3xCxHxW, 3: plane axes, C: triplane feature dim
				z_bg: 1xC
				ssize: supervision size (64)
				mask_ratio: frequency mask ratio to the pos emb
				rel_pos: use relative position to fg_slot_position or not
				bg_rotate: whether to rotate the background points to the camera coordinate
		return: feats_fg: list of (K-1)x3xMxC, M: number of points inside bbox, different for each slot
				input_bg: Px(60+C)
				idx: sum(M) (indices of the query points inside bbox)
		'''
		P, D = mean.shape[0], mean.shape[1]
		K = triplane_fg.shape[0] + 1

		# only keep the points that inside the cube, ((K-1)*P*D)
		# mask_locality = (torch.norm(mean.flatten(0,1), dim=-1) < self.locality_ratio).expand(K-1, -1).flatten(0, 1) if self.locality else torch.ones((K-1)*P*D, device=mean.device).bool()
		# mask_locality = torch.all(torch.abs(mean.flatten(0,1)) < self.locality_ratio, dim=-1).expand(K-1, -1).flatten(0, 1) if self.locality else torch.ones((K-1)*P*D, device=mean.device).bool()
		
		sampling_mean_fg = mean[None, ...].expand(K-1, -1, -1, -1).flatten(1, 2) # (K-1)*(P*D)*3

		if rel_pos:
			sampling_mean_fg = torch.cat([sampling_mean_fg, torch.ones_like(sampling_mean_fg[:, :, 0:1])], dim=-1)  # (K-1)*(P*D)*4
			sampling_mean_fg = torch.matmul(fg_transform[None, ...], sampling_mean_fg[..., None]).squeeze(-1)  # (K-1)*(P*D)*4
			sampling_mean_fg = sampling_mean_fg[:, :, :3]  # (K-1)*(P*D)*3
			
			fg_slot_position = torch.cat([fg_slot_position, torch.ones_like(fg_slot_position[:, 0:1])], dim=-1)  # (K-1)x4
			fg_slot_position = torch.matmul(fg_transform.squeeze(0), fg_slot_position.t()).t() # (K-1)x4
			fg_slot_position = fg_slot_position[:, :3]  # (K-1)x3

			sampling_mean_fg = sampling_mean_fg - fg_slot_position[:, None, :]  # (K-1)x(P*D)x3
	
		# sampling_mean_fg = sampling_mean_fg.view([K-1, P, D, 3]).flatten(0, 1)  # ((K-1)xP)xDx3
		# sampling_var_fg = var[None, ...].expand(K-1, -1, -1, -1).flatten(0, 1)  # ((K-1)xP)xDx3

		sampling_mean_bg, sampling_var_bg = mean, var

		if bg_rotate:
			sampling_mean_bg = torch.matmul(fg_transform[:, :3, :3], sampling_mean_bg[..., None]).squeeze(-1)  # PxDx3

		# 1. Remove the query points too far away from the slot center
		# 2-2. Sample the triplane features for the foreground points
		assert fg_object_size is not None # we should always remove the points outside the bounding box
		mask_idx = torch.empty((K-1)*P*D, device=mean.device).bool()
		feats_fg = []
		# iterate over each slot
		for i in range(K-1):
			sampling_fg_slot = sampling_mean_fg[i, :, :] # (P*D)x3
			mask_slot = torch.all(torch.abs(sampling_fg_slot) < fg_object_size, dim=-1)  # (P*D)
			mask_idx[i*P*D:(i+1)*P*D] = mask_slot
			if mask_slot.sum() == 0:
				feats_fg.append(torch.empty([1, 0, 3], device=mean.device))
			else:
				feats_fg.append(sample_from_planes(self.plane_axes.to(mean.device), 
										triplane_fg[i:i+1, ...], sampling_fg_slot[None, mask_slot, :], # 1x3xCxHxW, Mx3, M: number of points inside bbox
										mode='bilinear', padding_mode='zeros', box_warp=fg_object_size*2))  # 1x3xMxC
		
		# 2-1. Compute Fourier position embeddings for the background points
		pos_emb_bg = self.pos_enc(sampling_mean_bg, sampling_var_bg)[0].flatten(0, 1)  # PxDx(6*n_freq+3) -> (P*D)x(6*n_freq+3)

		# 3. Concatenate the embeddings with z_bg features
		input_bg = torch.cat([pos_emb_bg, z_bg.repeat(P*D, 1)], dim=-1) # (P*D)x(6*n_freq+3+C)

		# 4. Return required tensors
		return feats_fg, input_bg, mask_idx

	def forward(self, mean, var, z_slots, fg_transform, fg_slot_position, dens_noise=0., 
		 			fg_object_size=None, rel_pos=True, bg_rotate=False, triplane_fg=None):
		"""
		1. pos emb by Fourier
		2. for each slot, decode all points from coord and slot feature
		input:
			mean: P*D*3, P = (N*H*W)
			var: P*D*3, P = (N*H*W)
			view_dirs: P*3, P = (N*H*W)
			z_slots: KxC, K: #slots, C: #feat_dim
			fg_transform: If self.fixed_locality, it is 1x4x4 matrix nss2cam0 in nss space,
							otherwise it is 1x3x3 azimuth rotation of nss2cam0 (not used)
			fg_slot_position: (K-1)x3 in nss space
			dens_noise: Noise added to density

			if fg_slot_cam_position is not None, we should first project it world coordinates
			depth: K*1, depth of the slots
		"""
		_, C = z_slots.shape
		K = fg_slot_position.shape[0] + 1
		P, D = mean.shape[0], mean.shape[1]
		
		# process features
		if triplane_fg is None:
			assert self.triplane_synthesizer is not None
			z_fg = z_slots[1:, :]  # (K-1)xC
			z_bg = z_slots[0:1, :]  # 1xC
			triplane_fg = self.triplane_synthesizer(z_fg)  # (K-1)x3xCxHxW
		else:
			z_bg = z_slots  # 1xC

		feats_fg, input_bg, idx = self.processQueries(mean, var, fg_transform, fg_slot_position, triplane_fg, z_bg, 
						fg_object_size=fg_object_size, rel_pos=rel_pos, bg_rotate=bg_rotate)
		
		tmp = self.b_before(input_bg)
		bg_raws = self.b_after(torch.cat([input_bg, tmp], dim=1)).view([1, P*D, self.out_ch])  # (P*D)x4 -> 1x(P*D)x4
		bg_raws = torch.cat([(bg_raws[...,:-1].tanh() + 1) / 2, self.density_act(bg_raws[..., -1:])], dim=-1) # (P*D)x3

		fg_raw_rgb_part = torch.empty([0, 3], device=mean.device) # after the for loop, Mx3, M is the sum of all M_i
		fg_raw_shape_part = torch.empty([0, 1], device=mean.device) # after the for loop, Mx1, M is the sum of all M_i
		for feat_fg in feats_fg:
			if feat_fg.shape[1] == 0:
				continue
			fg_rgb_slot, fg_shape_slot = self.fg_decoder(feat_fg) # 1xMx3, 1xMx1
			fg_raw_rgb_part = torch.cat([fg_raw_rgb_part, fg_rgb_slot.squeeze(0)], dim=0)
			fg_raw_shape_part = torch.cat([fg_raw_shape_part, self.density_act(fg_shape_slot.squeeze(0))], dim=0) # concat along the first dim

		# put back the removed query points, for indices between idx[i] and idx[i+1], put fg_raw_rgb[i] at idx[i]
		fg_raw_rgb_full = torch.zeros((K-1)*P*D, 3, device=fg_raw_rgb_part.device, dtype=fg_raw_rgb_part.dtype) # ((K-1)xP*D)x3
		fg_raw_rgb_full[idx] = fg_raw_rgb_part # already activated
		fg_raw_rgb_full = fg_raw_rgb_full.view([K-1, P*D, 3])  # ((K-1)xP*D)x3 -> (K-1)x(P*D)x3

		fg_raw_shape_full = torch.zeros((K-1)*P*D, 1, device=fg_raw_shape_part.device, dtype=fg_raw_shape_part.dtype) # ((K-1)xP*D)x1
		fg_raw_shape_full[idx] = fg_raw_shape_part
		fg_raw_shape_full = fg_raw_shape_full.view([K - 1, P*D])  # ((K-1)xP*D)x1 -> (K-1)x(P*D), density

		fg_raws = torch.cat([fg_raw_rgb_full, fg_raw_shape_full[..., None]], dim=-1) # (K-1)x(P*D)x4

		raw_masks = torch.cat([bg_raws[..., -1:], fg_raws[..., -1:]], dim=0)  # Kx(P*D)x1
		masks = raw_masks / (raw_masks.sum(dim=0) + 1e-5)  # Kx(P*D)x1
		raw_sigma = raw_masks + dens_noise * torch.randn_like(raw_masks)

		raw_rgb = torch.cat([bg_raws[..., :-1], fg_raws[..., :-1]], dim=0)

		unmasked_raws = torch.cat([raw_rgb, raw_sigma], dim=2)  # Kx(P*D)x4
		masked_raws = unmasked_raws * masks
		raws = masked_raws.sum(dim=0)

		return raws, masked_raws, unmasked_raws, masks

	def _build_activation(self, options):
		if options == 'softplus':
			return nn.Softplus
		elif options == 'relu':
			return nn.ReLU
		elif options == 'silu':
			return nn.SiLU
		else:
			assert False, 'activation should be softplus or relu'


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
            # nn.Linear(dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, slot_dim)
        )

        self.MLP_bg = nn.Sequential(
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

    def forward_bg(self, x, h, w):
        grid = build_grid(h, w, x.device) # (1, h, w, 2)
        rel_grid = grid.unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1) # (b, 1, h, w, 2)
        # rel_grid = rel_grid.flatten(-3, -2) # (b, 1, h*w, 2)
        rel_grid = torch.cat([rel_grid, -rel_grid], dim=-1).flatten(-3, -2) # (b, 1, h*w, 4)
        grid_embed = self.grid_embed(rel_grid) # (b, 1, h*w, d)
        
        k_bg, v_bg = self.input_to_k_bg(x).unsqueeze(1), self.input_to_v_bg(x).unsqueeze(1) # (b, 1, h*w, d)
        k_bg, v_bg = k_bg + grid_embed, v_bg + grid_embed
        k_bg, v_bg = self.MLP_bg(k_bg), self.MLP_bg(v_bg)

        return k_bg, v_bg # (b, 1, h*w, d)


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
            # self.post_MLP = nn.Sequential(
            #         nn.LayerNorm(self.z_dim),
            #         nn.Linear(self.z_dim, self.z_dim),
            #         nn.ReLU(inplace=True),
            #         nn.Linear(self.z_dim, self.z_dim))
        else:
            self.position_project = None
        self.rel_pos = rel_pos
        self.fg_in_world = fg_in_world
        self.positional_encoding = PositionalEncoding(max_deg=n_freq)

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, fg_transform, fg_slot_position, dens_noise=0., invariant=True, cov=None):
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
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            sampling_coor_fg = torch.cat([sampling_coor_fg, torch.ones_like(sampling_coor_fg[:, :, 0:1])], dim=-1)  # (K-1)xPx4
            if not self.fg_in_world:
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx4x1
            sampling_coor_fg = sampling_coor_fg.squeeze(-1)[:, :, :3]  # (K-1)xPx3
        else:
            # relative position with fg slot position
            if self.rel_pos and invariant:
                sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :] # (K-1)xPx3

            sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # (K-1)xPx3x1
            outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1)  # (K-1)xP

        z_bg = z_slots[0:1, :]  # 1xC
        z_fg = z_slots[1:, :]  # (K-1)xC

        if self.position_project is not None and invariant:
            # w/ and w/o residual connection
            z_fg = z_fg + self.position_project(fg_slot_position[:, :2]) # (K-1)xC
            # slot_position = torch.cat([torch.zeros_like(fg_slot_position[0:1,]), fg_slot_position], dim=0)[:,:2] # Kx2
            # z_slots = self.position_project(slot_position) + z_slots # KxC

        query_bg = self.positional_encoding(sampling_coor_bg, cov)  # Px60, 60 means increased-freq feat dim
        # query_bg = sin_emb(sampling_coor_bg, n_freq=self.n_freq)  # Px60, 60 means increased-freq feat dim
        input_bg = torch.cat([query_bg, z_bg.expand(P, -1)], dim=1)  # Px(60+C)

        sampling_coor_fg_ = sampling_coor_fg.flatten(start_dim=0, end_dim=1)  # ((K-1)xP)x3
        if cov is not None: # (P*3) ---> ((K-1)xP)x3
            cov = cov.unsqueeze(0).expand(K-1, -1, -1).reshape(-1, 3)
        query_fg_ex = self.positional_encoding(sampling_coor_fg_, cov)  # ((K-1)xP)x60
        # query_fg_ex = sin_emb(sampling_coor_fg_, n_freq=self.n_freq)  # ((K-1)xP)x60
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


class SlotAttention(nn.Module):
    def __init__(self, in_dim=64, slot_dim=64, iters=4, eps=1e-8, hidden_dim=128):
        super().__init__()
        # self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = slot_dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma)
        self.slots_mu_bg = nn.Parameter(torch.randn(1, 1, slot_dim))
        self.slots_logsigma_bg = nn.Parameter(torch.zeros(1, 1, slot_dim))
        init.xavier_uniform_(self.slots_logsigma_bg)
        
        self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)

        # self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        # self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
        self.to_q = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))
        self.to_q_bg = nn.Sequential(nn.LayerNorm(slot_dim), nn.Linear(slot_dim, slot_dim, bias=False))

        self.gru_fg = nn.GRUCell(slot_dim, slot_dim)
        self.gru_bg = nn.GRUCell(slot_dim, slot_dim)

        hidden_dim = max(slot_dim, hidden_dim)

        self.to_res_fg = nn.Sequential(
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
        self.slot_dim = slot_dim

        self.to_res_fg = nn.Sequential(nn.LayerNorm(slot_dim),
                                           nn.Linear(slot_dim, slot_dim))
        
        self.to_res_bg = nn.Sequential(nn.LayerNorm(slot_dim),
                                           nn.Linear(slot_dim, slot_dim))

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

    def forward(self, feat, mask):
        """
        input:
            feat: visual feature with position information, BxHxWxC, C: shape_dim + color_dim
            mask: mask for foreground objects, KxHxW, K: number of foreground objects (exclude background)
        output:
            slot_feat: slot feature, BxKxC
        """
        B, H, W, _ = feat.shape
        N = H * W
        feat = feat.flatten(1, 2) # (B, N, C)
        K = mask.shape[0]

        mu = self.slots_mu.expand(B, K, -1)
        sigma = self.slots_logsigma.exp().expand(B, K, -1)
        slot_fg = mu + sigma * torch.randn_like(mu)
        
        fg_position = self.get_fg_position(mask) # Kx2
        fg_position = fg_position.expand(B, -1, -1).to(feat.device) # BxKx2
        
        mu_bg = self.slots_mu_bg.expand(B, 1, -1)
        sigma_bg = self.slots_logsigma_bg.exp().expand(B, 1, -1)
        slot_bg = mu_bg + sigma_bg * torch.randn_like(mu_bg)

        feat = self.norm_feat(feat)
        k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

        grid = build_grid(H, W, device=feat.device).flatten(1, 2).squeeze(0) # Nx2

        # attn = None
        for it in range(self.iters):
            slot_prev_bg = slot_bg
            slot_prev_fg = slot_fg
            q_fg = self.to_q(slot_fg)
            q_bg = self.to_q_bg(slot_bg) # (B,1,C)
            
            # attn = torch.empty(B, K, N, device=feat.device)
            
            k, v = self.to_kv(feat, H, W, fg_position) # (B,K,N,C)

            # compute background attention and updates
            attn_bg = torch.einsum('bd,bnd->bn', q_bg.squeeze(1), k_bg.squeeze(1)) * self.scale # BxN
            attn_bg = attn_bg.softmax(dim=1) + self.eps # BxN
            updates_bg = torch.einsum('bn,bnd->bd', attn_bg, v_bg.squeeze(1)) # BxC
            slot_bg = self.gru_bg(updates_bg, slot_bg.squeeze(1)).unsqueeze(1) # Bx1xC
            slot_bg = self.to_res_bg(slot_bg) + slot_prev_bg # Bx1xC

            # compute foreground attention and updates, each slot only compute attention on corresponding mask
            updates_fg = torch.empty(B, K, self.slot_dim, device=feat.device)
            for i in range(K):
                attn_this_slot = torch.einsum('bd,bnd->bn', q_fg[:, i, :], k[:, i, :, :]) * self.scale # BxN
                mask_this_slot = mask[i].flatten() # N
                # we will use softmax after masking, so we need to set the masked values to a very small value
                attn_this_slot = attn_this_slot.masked_fill(mask_this_slot.unsqueeze(0) == 0, -1e9)
                attn_this_slot = attn_this_slot.softmax(dim=1) # BxN
                updates_fg[:, i, :] = torch.einsum('bn,bnd->bd', attn_this_slot, v[:, i, :, :]) # BxC
                # update the position of this slot (weighted mean of the grid points, with attention as weights)
                fg_position[:, i, :] = torch.einsum('bn,nd->bd', attn_this_slot, grid) # Bx2

            slot_fg = self.gru_fg(updates_fg.reshape(-1, self.slot_dim), slot_fg.reshape(-1, self.slot_dim)).reshape(B, K, self.slot_dim) # BxKxC
            slot_fg = self.to_res_fg(slot_fg) + slot_prev_fg # BxKx2

        slots = torch.cat([slot_bg, slot_fg], dim=1) # (B,K+1,C)
                
        return slots, fg_position







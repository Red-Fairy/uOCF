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

class sam_encoder_v1(nn.Module):
    def __init__(self, sam_model, z_dim):
        super(sam_encoder_v1, self).__init__()

        self.sam = sam_model
        self.sam.requires_grad_(False)
        self.sam.eval()

        self.vit_dim = 256
        self.sam_dim = (z_dim // 4) * 3
        self.color_dim = z_dim - self.sam_dim
        self.down = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 128, 3, 2, 1),
                                    nn.ReLU(True)) # (128, 64, 64)
        self.input_0 = nn.Conv2d(128, 128, 3, 1, 1)
        self.input_1 = nn.Conv2d(256, 256, 3, 1, 1)

        self.input_2 = nn.Conv2d(256, self.color_dim, 3, 1, 1)

        self.conv1 = nn.Conv2d(self.vit_dim, self.vit_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.vit_dim, self.sam_dim, 3, 1, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x_sam, x):
        

        x = self.down(x) # (B, 128, 64, 64)
        x1 = self.input_0(x) # (B, 128, 64, 64)
        x = self.input_1(torch.cat([x, x1], dim=1)) # (B, 256, 64, 64)
        x = self.relu(x)
        x = self.input_2(x) # (B, color_dim, 64, 64)

        x_sam = self.sam.image_encoder(x_sam) # (B, 256, 64, 64)
        x_sam = self.conv1(x_sam)
        x_sam = self.relu(x_sam)
        x_sam = self.conv2(x_sam) # (B, sam_dim, 64, 64)

        return torch.cat([x, x_sam], dim=1)

class sam_encoder_v2(nn.Module):
    def __init__(self, sam_model, z_dim):
        super(sam_encoder_v2, self).__init__()

        self.sam = sam_model
        self.sam.requires_grad_(False)
        self.sam.eval()

        self.vit_dim = 256
        self.color_dim = 64
        self.total_dim = z_dim
        self.down = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 128, 3, 2, 1),
                                    nn.ReLU(True)) # (128, 64, 64)
        self.input_0 = nn.Conv2d(128, 128, 3, 1, 1)
        self.input_1 = nn.Conv2d(256, 256, 3, 1, 1)

        self.input_2 = nn.Conv2d(256, self.color_dim, 3, 1, 1)

        self.conv1 = nn.Conv2d(self.vit_dim+self.color_dim, self.vit_dim, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.vit_dim, self.total_dim, 3, 1, 1)
        self.relu = nn.ReLU(True)

    def forward(self, x_sam, x):

        x = self.down(x) # (B, 128, 64, 64)
        x1 = self.input_0(x) # (B, 128, 64, 64)
        x = self.input_1(torch.cat([x, x1], dim=1)) # (B, 256, 64, 64)
        x = self.relu(x)
        x = self.input_2(x) # (B, color_dim, 64, 64)

        x_sam = self.sam.image_encoder(x_sam) # (B, 256, 64, 64)

        x = self.conv1(torch.cat([x, x_sam], dim=1))
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        return x

class sam_encoder_v3(nn.Module):
    def __init__(self, sam_model, z_dim):
        super(sam_encoder_v3, self).__init__()

        self.sam = sam_model
        self.sam.requires_grad_(False)
        self.sam.eval()

        self.vit_dim = 256
        self.total_dim = z_dim
        self.down = nn.Sequential(nn.Conv2d(3, 64, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(64, 128, 3, 2, 1),
                                    nn.ReLU(True)) # (128, 64, 64)
        self.input_0 = nn.Conv2d(128, 128, 3, 1, 1)
        self.input_1 = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                    nn.ReLU(True))

        self.input_2 = nn.Conv2d(256, self.vit_dim, 3, 1, 1)

        self.MLP = nn.Sequential(nn.Conv2d(self.vit_dim, self.vit_dim, 1, 1, 1),
                                nn.ReLU(True),
                                nn.Conv2d(self.vit_dim, self.total_dim, 1, 1, 1))


    def forward(self, x_sam, x):

        x = self.down(x) # (B, 128, 64, 64)
        x1 = self.input_0(x) # (B, 128, 64, 64)
        x = self.input_1(torch.cat([x, x1], dim=1)) # (B, 256, 64, 64)
        x = self.input_2(x) # (B, 256, 64, 64)

        x_sam = self.sam.image_encoder(x_sam) # (B, 256, 64, 64)

        x = self.MLP(x_sam + x)

        return x

class sam_encoder_v0(nn.Module):
    def __init__(self, sam_model, z_dim):
        super(sam_encoder_v0, self).__init__()

        self.sam = sam_model
        self.sam.requires_grad_(False)
        self.sam.eval()

        self.vit_dim = 256
        self.total_dim = z_dim

        self.sam_conv = nn.Sequential(nn.Conv2d(self.vit_dim, self.vit_dim, 3, 1, 1),
                                    nn.ReLU(True),
                                    nn.Conv2d(self.vit_dim, self.total_dim, 3, 1, 1),
                                    nn.ReLU(True))

    def forward(self, x_sam):

        x_sam = self.sam.image_encoder(x_sam)
        x_sam = self.sam_conv(x_sam)

        return x_sam # (B, z_dim, 64, 64)

class sam_encoder_v00(nn.Module):
    def __init__(self, sam_model, z_dim):
        super(sam_encoder_v00, self).__init__()

        self.sam = sam_model.image_encoder
        self.sam.requires_grad_(False)
        self.sam.eval()

        self.vit_dim = 256
        self.total_dim = z_dim

        self.sam_conv = nn.Sequential(nn.Conv2d(self.vit_dim, self.vit_dim, 3, 1, 1),
                                    nn.Conv2d(self.vit_dim, self.total_dim, 3, 1, 1),)

    def forward(self, x_sam):
        
        x_sam = self.sam(x_sam)
        x_sam = self.sam_conv(x_sam)

        return x_sam # (B, z_dim, 64, 64)
    
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

        # a bug here, the MLP_fg & grid_embed was used twice
        # k, v = self.MLP_fg(k + grid_embed), self.MLP_fg(v + grid_embed) # (b, n_slot-1, h*w, d)

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


class Decoder(nn.Module):
    def __init__(self, n_freq=5, input_dim=33+64, z_dim=64, texture_dim=8, n_layers=3, locality=True, locality_ratio=4/7, fixed_locality=False, 
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
        z_dim += texture_dim
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

    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, z_slots_texture, fg_transform, fg_slot_position, dens_noise=0., invariant=True):
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
            sampling_coor_fg_temp = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None])  # (K-1)xPx3x1
            sampling_coor_fg_temp = sampling_coor_fg_temp.squeeze(-1)  # (K-1)xPx3
            outsider_idx = torch.any(sampling_coor_fg_temp.abs() > self.locality_ratio, dim=-1)  # (K-1)xP
            # relative position with fg slot position
            if self.rel_pos and invariant:
                sampling_coor_fg = sampling_coor_fg - fg_slot_position[:, None, :] # (K-1)xPx3
                sampling_coor_fg = torch.matmul(fg_transform[None, ...], sampling_coor_fg[..., None]).squeeze(-1)  # (K-1)xPx3x1
            else:
                sampling_coor_fg = sampling_coor_fg_temp

        z_bg = z_slots[0:1, :]  # 1xC
        z_fg = z_slots[1:, :]  # (K-1)xC

        z_bg_texture = z_slots_texture[0:1, :]  # 1xC'
        z_fg_texture = z_slots_texture[1:, :]  # (K-1)xC'

        if self.position_project is not None and invariant:
            # w/ and w/o residual connection
            # z_fg = z_fg + self.post_MLP(z_fg + self.position_project(fg_slot_position[:, :2])) # (K-1)xC
            z_fg = z_fg + self.position_project(fg_slot_position[:, :2]) # (K-1)xC
            # slot_position = torch.cat([torch.zeros_like(fg_slot_position[0:1,]), fg_slot_position], dim=0)[:,:2] # Kx2
            # z_slots = self.position_project(slot_position) + z_slots # KxC
        
        # concat feature
        z_fg = torch.cat([z_fg, z_fg_texture], dim=-1)  # (K-1)x(C+C')
        z_bg = torch.cat([z_bg, z_bg_texture], dim=-1)  # 1x(C+C')

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


class SlotAttention(nn.Module):
    def __init__(self, num_slots, in_dim=64, slot_dim=64, texture_dim=8, iters=4, eps=1e-8, hidden_dim=128, learnable_pos=True):
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

        if learnable_pos:
            self.fg_position = nn.Parameter(torch.rand(1, num_slots-1, 2) * 2 - 1)
        else:
            self.fg_position = None
        
        self.to_kv = EncoderPosEmbedding(in_dim, slot_dim)

        # self.to_k = nn.Linear(in_dim, slot_dim, bias=False)
        # self.to_v = nn.Linear(in_dim, slot_dim, bias=False)
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
        self.slot_dim = slot_dim

        self.to_texture_fg = nn.Sequential(nn.LayerNorm(texture_dim), 
                                        nn.Linear(texture_dim, texture_dim))
        
        self.to_texture_bg = nn.Sequential(nn.LayerNorm(texture_dim),
                                        nn.Linear(texture_dim, texture_dim))

    def forward(self, feat, feat_texture, num_slots=None):
        """
        input:
            feat: visual feature with position information, BxHxWxC
            feat_texture: texture feature with position information, BxHxWxC'
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
        k_bg, v_bg = self.to_kv.forward_bg(feat, H, W) # (B,1,N,C)

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
            grid = build_grid(H, W, feat.device).flatten(1, 2) # (B,N,2)
            fg_position = torch.einsum('bkn,bnd->bkd', attn_weights_fg, grid) # (B,K-1,N) * (B,N,2) -> (B,K-1,2)
            
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

                slot_fg = self.gru(
                    updates_fg.reshape(-1, self.slot_dim),
                    slot_prev_fg.reshape(-1, self.slot_dim)
                )
                slot_fg = slot_fg.reshape(B, -1, self.slot_dim)
                slot_fg = slot_fg + self.to_res(slot_fg)

            else:
                # calculate slot texture feature
                feat_texture = feat_texture.flatten(1, 2) # (B,N,C')
                feat_texture_fg = self.to_texture_fg(feat_texture) # (B,N,C')
                feat_texture_bg = self.to_texture_bg(feat_texture) # (B,N,C')
                slot_fg_texture = torch.einsum('bkn,bnd->bkd', attn_weights_fg, feat_texture_fg) # (B,K-1,N) * (B,N,C') -> (B,K-1,C')
                slot_bg_texture = torch.einsum('bn,bnd->bd', attn_weights_bg.squeeze(1), feat_texture_bg) # (B,N) * (B,N,C') -> (B,C')
                slot_bg_texture = slot_bg_texture.unsqueeze(1) # (B,1,C')


        slots = torch.cat([slot_bg, slot_fg], dim=1) # (B,K,C)
        slots_texture = torch.cat([slot_bg_texture, slot_fg_texture], dim=1) # (B,K,C')
                
        return slots, attn, fg_position, slots_texture


def sin_emb(x, n_freq=5, keep_ori=True):
    """
    create sin embedding for 3d coordinates
    input:
        x: Px3
        n_freq: number of raised frequency
    """
    embedded = []
    if keep_ori:
        embedded.append(x)
    emb_fns = [torch.sin, torch.cos]
    freqs = 2. ** torch.linspace(0., n_freq - 1, steps=n_freq)
    for freq in freqs:
        for emb_fn in emb_fns:
            embedded.append(emb_fn(freq * x))
    embedded_ = torch.cat(embedded, dim=1)
    return embedded_

def raw2outputs(raw, z_vals, rays_d, render_mask=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray in cam coor.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda x, y: 1. - torch.exp(-x * y)
    device = raw.device

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = raw[..., :3]

    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:,:-1]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True)
    depth_map = torch.sum(weights_norm * z_vals, -1)

    if render_mask:
        density = raw[..., 3]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=1)  # [N_rays,]
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights_norm


def get_perceptual_net(layer=4):
    assert layer > 0
    idx_set = [None, 4, 9, 16, 23, 30]
    idx = idx_set[layer]
    vgg = vgg16(pretrained=True)
    loss_network = nn.Sequential(*list(vgg.features)[:idx]).eval()
    for param in loss_network.parameters():
        param.requires_grad = False
    return loss_network


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean(), fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = F.leaky_relu(out, 0.2, inplace=True) * 1.4

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )



class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        stride=1,
        padding=1
    ):
        layers = []

        if downsample:
            layers.append(nn.AvgPool2d(kernel_size=2, stride=2))

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, stride=1, padding=1)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, stride=1, padding=1)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False, stride=1, padding=0
        )

    def forward(self, input):
        out = self.conv1(input) * 1.4
        out = self.conv2(out) * 1.4

        skip = self.skip(input) * 1.4
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, ndf, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: ndf*2,
            8: ndf*2,
            16: ndf,
            32: ndf,
            64: ndf//2,
            128: ndf//2
        }

        convs = [ConvLayer(3, channels[size], 1, stride=1, padding=1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, stride=1, padding=1)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input) * 1.4

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out) * 1.4

        out = out.view(batch, -1)
        out = self.final_linear(out)

class position_loss(nn.Module):
    def __init__(self, loss_weight=1.0, threshold=0.1):
        super().__init__()
        self.loss_weight = loss_weight
        self.threshold = threshold

    def forward(self, x):
        # x: K*2
        # compute the element-wise distance
        x = x.unsqueeze(0) - x.unsqueeze(1)
        x = x.norm(dim=-1)
        pos = (x != 0) * (x <= self.threshold)
        x = (self.threshold - x) * pos.float()
        # compute the loss
        loss = x.sum() / (x.shape[0] * x.shape[1])

        return loss * self.loss_weight

class set_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, loc1, loc2):
        """
        loc1: N * 2, position of N points on the XY plane in the first set
        loc2: N * 2, position of N points on the XY plane in the second set
        return the set-wise loss:
        for each point in the first set, find the closest point in the second set
        compute the distance between the two points
        aggregate the distances
        """
        N = loc1.size(0)
        assert N == loc2.size(0)
        # compute the distance matrix
        # dist_mat: N * N
        dist_mat = torch.pow(loc1, 2).sum(dim=1, keepdim=True).expand(N, N) + \
                torch.pow(loc2, 2).sum(dim=1, keepdim=True).expand(N, N).t()
        dist_mat.addmm_(1, -2, loc1, loc2.t())
        # find the closest point in the second set for each point in the first set
        # dist: N * 1
        dist, _ = torch.min(dist_mat, dim=1, keepdim=True)
        loss = dist.mean()
        return loss

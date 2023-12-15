import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from einops import rearrange
from functorch import jacrev, vmap

def contract(x):
    # x: [N, 3]
    return (2 - 1 / (torch.norm(x, dim=-1, keepdim=True))) * x / torch.norm(x, dim=-1, keepdim=True)

def parameterization(means, covs):
    '''
    means: [B, N, 3]
    covs: [B, N, 3, 3]
    '''
    B, N, _ = means.shape
    means = means.reshape([-1, 3])
    if len(covs.shape) == 4:
        covs = covs.reshape(-1, 3, 3)
    else:
        covs = covs.reshape(-1, 3)
    contr_mask = (torch.norm(means, dim=-1, keepdim=True) > 1).detach()
    with torch.no_grad():
        jac = vmap(jacrev(contract))(means)
        print('11', jac.shape, covs.shape)
    means = torch.where(contr_mask, contract(means), means)
    covs = torch.where(contr_mask.unsqueeze(-1).expand(jac.shape), jac, covs)
    return means.reshape([B, N, 3]), covs.reshape([B, N, 3, 3])

def expected_sin(x, x_var):
    """Estimates mean and variance of sin(z), z ~ N(x, var)."""
    # When the variance is wide, shrink sin towards zero.
    y = torch.exp(-0.5 * x_var) * torch.sin(x)  # [B, N, 2*3*L]
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y ** 2
    y_var = torch.maximum(torch.zeros_like(y_var), y_var)
    return y, y_var

def integrated_pos_enc_360(means_covs):
    P = torch.tensor([[0.8506508, 0, 0.5257311],
                      [0.809017, 0.5, 0.309017],
                      [0.5257311, 0.8506508, 0],
                      [1, 0, 0],
                      [0.809017, 0.5, -0.309017],
                      [0.8506508, 0, -0.5257311],
                      [0.309017, 0.809017, -0.5],
                      [0, 0.5257311, -0.8506508],
                      [0.5, 0.309017, -0.809017],
                      [0, 1, 0],
                      [-0.5257311, 0.8506508, 0],
                      [-0.309017, 0.809017, -0.5],
                      [0, 0.5257311, 0.8506508],
                      [-0.309017, 0.809017, 0.5],
                      [0.309017, 0.809017, 0.5],
                      [0.5, 0.309017, 0.809017],
                      [0.5, -0.309017, 0.809017],
                      [0, 0, 1],
                      [-0.5, 0.309017, 0.809017],
                      [-0.809017, 0.5, 0.309017],
                      [-0.809017, 0.5, -0.309017]]).T
    means, covs = means_covs
    P = P.to(means.device)
    means, x_cov = parameterization(means, covs)
    y = torch.matmul(means, P)
    y_var = torch.sum((torch.matmul(x_cov, P)) * P, -2)
    return expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]

def integrated_pos_enc(means, covs, min_deg, max_deg, diagonal=True):
    """Encode `means` with sinusoids scaled by 2^[min_deg:max_deg-1].
    Args:
        means_covs:[B, N, 3] a tuple containing: means, torch.Tensor, variables to be encoded.
        covs, [B, N, 3] torch.Tensor, covariance matrices.
        min_deg: int, the min degree of the encoding.
        max_deg: int, the max degree of the encoding.
        diagonal: bool, if true, expects input covariances to be diagonal (full otherwise).
    Returns:
        encoded: torch.Tensor, encoded variables.
    """
    if diagonal:
        scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)], device=means.device)  # [L]
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y = rearrange(torch.unsqueeze(means, dim=-2) * torch.unsqueeze(scales, dim=-1),
                      'batch sample scale_dim mean_dim -> batch sample (scale_dim mean_dim)')
        # [B, N, 1, 3] * [L, 1] = [B, N, L, 3]->[B, N, 3L]
        y_var = rearrange(torch.unsqueeze(covs, dim=-2) * torch.unsqueeze(scales, dim=-1) ** 2,
                          'batch sample scale_dim cov_dim -> batch sample (scale_dim cov_dim)')
    else:
        num_dims = means.shape[-1] # [3, L]
        basis = torch.cat([2 ** i * torch.eye(num_dims, device=means.device) for i in range(min_deg, max_deg)], 1)
        y = torch.matmul(means, basis)  # [B, N, 3] * [3, 3L] = [B, N, 3L]
        y_var = torch.sum((torch.matmul(covs, basis)) * basis, -2)

    ret = expected_sin(torch.cat([y, y + 0.5 * torch.tensor(np.pi)], dim=-1), torch.cat([y_var] * 2, dim=-1))[0]
    return torch.cat([ret, means], dim=-1)

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg=0, max_deg=5):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])

    def forward(self, x, y=None):
        x_ = x
        shape = list(x.shape[:-1]) + [-1]
        x_enc = (x[..., None, :] * self.scales[:, None].to(x.device)).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None].to(x.device)**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            
            x_ret = torch.cat([x_ret, x_], dim=-1) # N*(6*(max_deg-min_deg)+3)
            return x_ret, y_ret
        else:
            # PE
            x_ret = torch.sin(x_enc)
            x_ret = torch.cat([x_ret, x_], dim=-1) # N*(6*(max_deg-min_deg)+3)
            return x_ret
    
    def sin_emb(self, x, keep_ori=True):
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
        freqs = 2. ** torch.linspace(self.min_deg, self.max_deg-1, steps=self.max_deg - self.min_deg)
        for freq in freqs:
            for emb_fn in emb_fns:
                embedded.append(emb_fn(freq * x))
        embedded_ = torch.cat(embedded, dim=1)
        return embedded_
        

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

def lift_gaussian(directions, t_mean, t_var, r_var, diagonal):
    """Lift a Gaussian defined along a ray to 3D coordinates."""
    mean = torch.unsqueeze(directions, dim=-2) * torch.unsqueeze(t_mean, dim=-1)  # [B, 1, 3]*[B, N, 1] = [B, N, 3]
    d_norm_denominator = torch.sum(directions ** 2, dim=-1, keepdim=True) + 1e-10
    # min_denominator = torch.full_like(d_norm_denominator, 1e-10)
    # d_norm_denominator = torch.maximum(min_denominator, d_norm_denominator)

    if diagonal:
        d_outer_diag = directions ** 2  # eq (16)
        null_outer_diag = 1 - d_outer_diag / d_norm_denominator
        t_cov_diag = torch.unsqueeze(t_var, dim=-1) * torch.unsqueeze(d_outer_diag,
                                                                      dim=-2)  # [B, N, 1] * [B, 1, 3] = [B, N, 3]
        xy_cov_diag = torch.unsqueeze(r_var, dim=-1) * torch.unsqueeze(null_outer_diag, dim=-2)
        cov_diag = t_cov_diag + xy_cov_diag
        return mean, cov_diag
    else:
        d_outer = torch.unsqueeze(directions, dim=-1) * torch.unsqueeze(directions,
                                                                        dim=-2)  # [B, 3, 1] * [B, 1, 3] = [B, 3, 3]
        eye = torch.eye(directions.shape[-1], device=directions.device)  # [B, 3, 3]
        # [B, 3, 1] * ([B, 3] / [B, 1])[..., None, :] = [B, 3, 3]
        null_outer = eye - torch.unsqueeze(directions, dim=-1) * (directions / d_norm_denominator).unsqueeze(-2)
        t_cov = t_var.unsqueeze(-1).unsqueeze(-1) * d_outer.unsqueeze(-3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        xy_cov = t_var.unsqueeze(-1).unsqueeze(-1) * null_outer.unsqueeze(
            -3)  # [B, N, 1, 1] * [B, 1, 3, 3] = [B, N, 3, 3]
        cov = t_cov + xy_cov
        return mean, cov


def conical_frustum_to_gaussian(directions, t0, t1, base_radius, diagonal, stable=True):
    """Approximate a conical frustum as a Gaussian distribution (mean+cov).
    Assumes the ray is originating from the origin, and base_radius is the
    radius at dist=1. Doesn't assume `directions` is normalized.
    Args:
        directions: torch.tensor float32 3-vector, the axis of the cone
        t0: float, the starting distance of the frustum.
        t1: float, the ending distance of the frustum.
        base_radius: float, the scale of the radius as a function of distance.
        diagonal: boolean, whether or the Gaussian will be diagonal or full-covariance.
        stable: boolean, whether or not to use the stable computation described in
        the paper (setting this to False will cause catastrophic failure).
    Returns:
        a Gaussian (mean and covariance).
    """
    if stable:
        mu = (t0 + t1) / 2
        hw = (t1 - t0) / 2
        t_mean = mu + (2 * mu * hw ** 2) / (3 * mu ** 2 + hw ** 2)
        t_var = (hw ** 2) / 3 - (4 / 15) * ((hw ** 4 * (12 * mu ** 2 - hw ** 2)) /
                                            (3 * mu ** 2 + hw ** 2) ** 2)
        r_var = base_radius ** 2 * ((mu ** 2) / 4 + (5 / 12) * hw ** 2 - 4 / 15 *
                                    (hw ** 4) / (3 * mu ** 2 + hw ** 2))
    else:
        t_mean = (3 * (t1 ** 4 - t0 ** 4)) / (4 * (t1 ** 3 - t0 ** 3))
        r_var = base_radius ** 2 * (3 / 20 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3))
        t_mosq = 3 / 5 * (t1 ** 5 - t0 ** 5) / (t1 ** 3 - t0 ** 3)
        t_var = t_mosq - t_mean ** 2
    return lift_gaussian(directions, t_mean, t_var, r_var, diagonal)

def cylinder_to_gaussian(d, t0, t1, radius, diag):
    """Approximate a cylinder as a Gaussian distribution (mean+cov).

    Assumes the ray is originating from the origin, and radius is the
    radius. Does not renormalize `d`.

    Args:
      d: torch.float32 3-vector, the axis of the cylinder
      t0: float, the starting distance of the cylinder.
      t1: float, the ending distance of the cylinder.
      radius: float, the radius of the cylinder
      diag: boolean, whether or the Gaussian will be diagonal or full-covariance.

    Returns:
      a Gaussian (mean and covariance).
    """
    t_mean = (t0 + t1) / 2
    r_var = radius ** 2 / 4
    t_var = (t1 - t0) ** 2 / 12
    return lift_gaussian(d, t_mean, t_var, r_var, diag)

def raw2outputs(raw, z_vals, rays_d, render_mask=False, mip=False):
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
    if not mip:
        dists = torch.cat([dists, torch.tensor([1e-2], device=device).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = raw[..., :3]

    alpha = raw2alpha(raw[..., 3], dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:,:-1] # [N_rays, N_samples]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    # weights_norm = weights.detach() + 1e-5
    weights_norm = weights + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True) # [N_rays, N_samples]
    if not mip:
        depth_map = torch.sum(weights_norm * z_vals, -1) # [N_rays,]
    else:
        z_mids = 0.5 * (z_vals[..., :-1] + z_vals[..., 1:])
        depth_map = torch.sum(weights_norm * z_mids, -1) # [N_rays,]
    depth_map = torch.clamp(torch.nan_to_num(depth_map), z_vals[:, 0], z_vals[:, -1])

    if render_mask:
        density = raw[..., 3]  # [N_rays, N_samples]
        mask_map = torch.sum(weights * density, dim=1)  # [N_rays,]
        return rgb_map, depth_map, weights_norm, mask_map

    return rgb_map, depth_map, weights # un-normed weights falls in [0, 1], but may exceed 1


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

def build_grid(H, W, device, reverse=False):
    """
    Build a sampling grid for bilinear sampling
    """
    x = torch.linspace(-1+1/W, 1-1/W, W)
    y = torch.linspace(-1+1/H, 1-1/H, H)
    y, x = torch.meshgrid([y, x])
    if not reverse:
        grid = torch.stack([x, y], dim=2).to(device).unsqueeze(0) # (1, H, W, 2)
    else:
        grid = torch.stack([x, y, -x, -y], dim=2).to(device).unsqueeze(0)
    return grid

class surfaceLoss(nn.Module):
    def __init__(self):
        super(surfaceLoss, self).__init__()

    def forward(self, x):
        '''
        x: (N*H*W)*N_samples
        loss = -log(e**(-abs(x)) + e**(-abs(1-x)))
        '''
        loss = -torch.log(torch.exp(-torch.abs(x)) + torch.exp(-torch.abs(1-x))) + math.log(1+math.exp(-1)) # guarantee loss is greater than 0
        return loss.mean()

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, x1, x2, mask=None):
        '''
        x1, x2: N*3*H*W
        mask: N*1*H*W, determine the background (should be masked out)
        if mask is None, then all mask is determined by x1 (-1 is background)
        '''
        if mask is None:
            mask_fg = (x1 != -1).all(dim=1, keepdim=True).float()
        else:
            mask_fg = 1 - mask
        loss = torch.sum(mask_fg * (x1 - x2)**2, dim=(1,2,3)) / torch.sum(mask_fg, dim=(1,2,3))
        return loss.mean()
    
class SlotFeatureSlotLoss(nn.Module):
    '''
    Calculate the slot-feature-slot cycle consistency loss
    Input: 
        Representation for each slot (K*C)
        Representation for each spatial feature (N*C)
    Step 1: Normalize the representations and calculate the similarity matrix (K*N)
    Step 2: Softmax the similarity matrix to get A1(K*N) and A2(N*K)
    Step 3: Calculate the probability from slot i to slot j, p(i->j) = \sum_n A1(i,n)*A2(n,j) (n in [1,N])
    Step 4: Softmax the probability matrix P(i->j) = exp(p(i->j)) / \sum_n exp(p(i->n))
    Step 5: Calculate the loss: - sum_i log(P(i->i))
    '''
    def __init__(self, nabla_1=0.1, nabla_2=1):
        super(SlotFeatureSlotLoss, self).__init__()
        self.nabla_1 = nabla_1
        self.nabla_2 = nabla_2

    def forward(self, slot, feature):
        # Step 1
        # slot = nn.functional.normalize(slot, dim=1)
        # feature = nn.functional.normalize(feature, dim=1)
        # Step 2
        A1 = nn.functional.softmax(torch.matmul(slot, feature.transpose(0,1)) / self.nabla_1, dim=1) # K*N slot to feature
        A2 = nn.functional.softmax(torch.matmul(feature, slot.transpose(0,1)) / self.nabla_1, dim=1) # N*K feature to slot
        # Step 3
        P = torch.matmul(A1, A2) # K*K
        # Step 4
        P = nn.functional.softmax(P / self.nabla_2, dim=1)
        # Step 5
        loss = -torch.log(torch.diagonal(P).clamp(min=1e-8)).mean()
        return loss
    
class PositionSetLoss(nn.Module):
    '''
    Calculate the position set loss
    Input: 
        two sets of 2D position (N*2, N*2)
    For each position in set 1, find the nearest position in set 2
    and calculate the distance between them
    '''
    def __init__(self):
        super(PositionSetLoss, self).__init__()

    def forward(self, pos1, pos2):
        '''
        pos1, pos2: N*2
        '''
        pos1 = pos1.unsqueeze(1)
        pos2 = pos2.unsqueeze(0)
        dist = torch.sqrt(torch.sum((pos1 - pos2)**2, dim=2)) # N*N
        dist = torch.min(dist, dim=1)[0]
        return dist.mean()

class PseudoMaskLoss(nn.Module):
    '''
    Calculate the pseudo mask loss
    Input:
        pseudo mask of the image (N*L), integer entries, L = H*W, denoted by array B
        predicted soft mask of the image (K*N*L), float entries, denoted by A
    Calculation:
        flatten the pseudo mask and predicted soft mask
        calculate the similarity between each soft mask entry, obtaining matrix of size N*L*L
    '''
    def __init__(self):
        super(PseudoMaskLoss, self).__init__()
    
    def forward(self, pseudo_mask, soft_mask):
        loss = 0.
        N, L = pseudo_mask.shape
        for i in range(N):
            filter_mask = (pseudo_mask[i] != 0)
            pseudo_filtered = pseudo_mask[i][filter_mask] # L'
            soft_filtered = soft_mask[:, i][:, filter_mask] # K*L'
            unique_pseudo = torch.unique(pseudo_filtered)
            for j in unique_pseudo:
                soft_pseudo = soft_filtered[:, pseudo_filtered == j]
                if soft_pseudo.shape[1] > 1:
                    loss += torch.sum(1 - torch.matmul(soft_pseudo, soft_pseudo.transpose(0,1))) # K*K
        loss /= (N * L * L)
        return loss
    
def debug(coordinates, fg_position=None, cam2world=None, save_name='debug'):
    coordinates = coordinates.detach().cpu().numpy()
    fg_position = fg_position.detach().cpu().numpy() if fg_position is not None else None
    K = coordinates.shape[0]
    colors = cm.rainbow(np.linspace(0, 1, 10))
    fig = plt.figure(figsize=(40, 20))
    if cam2world is not None:
        cam2world = cam2world.cpu().numpy()
    for i in range(K):
        ax = plt.subplot(2, 4, i+1, projection='3d')
        # visualize the frustum
        ax.scatter(coordinates[i,:,0], coordinates[i,:,1], coordinates[i,:,2], c=colors[i], marker='o', s=3)
        # visualize the camera origin
        if cam2world is not None:
            ax.scatter(cam2world[0,0,3], cam2world[0,1,3], cam2world[0,2,3], c='r', marker='o', s=20)
        else:
            ax.scatter(0, 0, 0, c='r', marker='o', s=20)
        # visualize the foreground position
        if fg_position is not None:
            ax.scatter(fg_position[i, 0], fg_position[i, 1], fg_position[i, 2], c='r', marker='o', s=100)
    fig.savefig(f'{save_name}.png')



        

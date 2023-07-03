import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd

class PositionalEncoding(nn.Module):
    def __init__(self, min_deg=0, max_deg=5):
        super(PositionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.scales = torch.tensor([2 ** i for i in range(min_deg, max_deg)])

    def forward(self, x, y=None):
        # x: (P, 3), y: (P, 3)
        shape = list(x.shape[:-1]) + [-1]
        self.scales = self.scales.to(x.device)
        x_enc = (x[..., None, :] * self.scales[:, None]).reshape(shape)
        x_enc = torch.cat((x_enc, x_enc + 0.5 * torch.pi), -1)
        if y is not None:
            # IPE
            y_enc = (y[..., None, :] * self.scales[:, None]**2).reshape(shape)
            y_enc = torch.cat((y_enc, y_enc), -1)
            x_ret = torch.exp(-0.5 * y_enc) * torch.sin(x_enc)
            # y_ret = torch.maximum(torch.zeros_like(y_enc), 0.5 * (1 - torch.exp(-2 * y_enc) * torch.cos(2 * x_enc)) - x_ret ** 2)
            # concat the original coordinates
            return torch.cat((x_ret, x), -1)
        else:
            # PE
            x_ret = torch.sin(x_enc)
            return torch.cat((x_ret, x), -1)

def sin_emb(x, n_freq=5, keep_ori=True, cov=None):
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
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1. - alpha + 1e-10], -1), -1)[:,:-1] # [N_rays, N_samples]

    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    weights_norm = weights.detach() + 1e-5
    weights_norm /= weights_norm.sum(dim=-1, keepdim=True) # [N_rays, N_samples]
    depth_map = torch.sum(weights_norm * z_vals, -1)

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
    



        

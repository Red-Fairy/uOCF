import math
import torch
from torch import nn
from .utils import build_grid
import torch.nn.functional as F

def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32)

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
    projections = torch.bmm(coordinates, inv_planes) # shape N*n_planes, M, 3
    return projections[..., :2]

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', box_warp=None):
    """
    Samples from a batch of planes given a batch of coordinates.
    plane_axes: shape n_planes, 3, 3
    plane_features: shape N, n_planes, C, H, W; N is number of triplanes to be queried
    coordinates: shape N, M, 3; M is number of points
    """
    N, n_planes, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape
    plane_features = plane_features.reshape(N*n_planes, C, H, W)

    # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

    projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1) # shape N*n_planes, 1, M, 2
    # projected coordinates are in range [-1, 1], which is suitable for grid_sample
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode='zeros', align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
    return output_features # shape N, n_planes, M, C


class TriplaneGenerator(nn.Module):
    '''
    Generate Triplane given latent vector
    '''
    def __init__(self, z_dim=64, triplane_dim=32):
        '''
        z_dim: dimension of the latent space
        '''
        super(TriplaneGenerator, self).__init__()
        self.z_dim = z_dim
        self.triplane_dim = triplane_dim
        self.n_plane = 3

        self.positional_embedding = nn.Linear(4, z_dim)
        self.plane_generator = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, 1, 1),
                                             nn.ReLU(),
                                             nn.Conv2d(z_dim, z_dim, 3, 1, 1),
                                             nn.ReLU(),
                                             nn.Conv2d(z_dim, triplane_dim*3, 3, 1, 1))

    def forward(self, x, resolution=(128, 128)):
        '''
        x: latent vector of size (N, z_dim)
        broadcast the latent vector to H*W, then perform convolution on it to generate triplane
        '''
        H, W = resolution
        grid = build_grid(H, W, device=x.device, reverse=True) # (1, H, W, 4)
        feature_map = x.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W) # (N, z_dim, H, W)
        feature_map = feature_map + self.positional_embedding(grid).permute(0, 3, 1, 2) # (N, z_dim, H, W)
        # convert to triplane
        triplane = self.plane_generator(feature_map) # (N, triplane_dim*3, H, W)

        return triplane


class TriplaneDecoder(nn.Module):
    '''
    Decode batched sampled triplane feature to color and density
    '''
    def __init__(self, feat_dim, n_layer):
        super(TriplaneDecoder, self).__init__()
        self.out_ch = 4
        self.feat_dim = feat_dim
        decoder = []
        for i in range(n_layer):
            decoder.append(nn.Linear(self.feat_dim, self.feat_dim)),
            decoder.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder)

        self.post_density = nn.Linear(self.feat_dim, 1)
        self.post_color = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 4),
                                        nn.ReLU(),
                                        nn.Linear(self.feat_dim // 4, 3),)

    def forward(self, x):
        '''
        x: triplane feature of size (N, feat_dim), N=n_slots*n_points, feat_dim=triplane_dim*3
        '''
        x = self.decoder(x)
        density = self.post_density(x) # (N, 1)
        color = self.post_color(x) # (N, 3)
        return density, color
        

class TriplaneRendererFG(nn.Module):
    '''
    Triplane Decoder with foreground only
    '''
    def __init__(self, z_dim=64, triplane_dim=32, n_layer=2, locality_ratio=1, rel_pos=True, position_project=True):
        '''
        z_dim: dimension of the latent space
        n_layer: number of layers in the decoder
        locality_ratio: default 4/7, maximum value = 1
        '''
        super(TriplaneRendererFG, self).__init__()
        self.z_dim = z_dim
        self.n_layer = n_layer
        self.rel_pos = rel_pos
        self.locality_ratio = locality_ratio
        self.triplane_generator = TriplaneGenerator(z_dim=z_dim, triplane_dim=triplane_dim)
        self.decoder_fg = TriplaneDecoder(feat_dim=triplane_dim*3, n_layer=n_layer)

        if position_project:
            self.position_project = nn.Linear(2, self.z_dim)
        else:
            self.position_project = None

    def forward(self, sampling_coor_fg, z_slots, triplane_resolution=(128, 128), slot_pos=None):
        '''
        sampling_coor_fg: foreground sampling coordinates (N, M, 3); M is the number of foreground points, N is number of slots
        z_slots: latent vector of size (N, z_dim)
        slot_pos: position of the slots (N, 3) (in NSS space); must be provided if rel_pos is True
        '''
        if self.rel_pos:
            assert slot_pos is not None
            # compute query points' relative position with respect to the slots
            sampling_coor_fg = sampling_coor_fg - slot_pos.unsqueeze(1) # (N, M, 3)

        if self.position_project is not None:
            z_slots = z_slots + self.position_project(slot_pos[:, :2]) # (N, z_dim)

        triplane = self.triplane_generator(z_slots, resolution=triplane_resolution) # (N, triplane_dim*3, H, W)
        N, _, H, W = triplane.shape
        triplane = triplane.view(N, 3, -1, H, W) # (N, 3, triplane_dim, H, W)
        _, M, _ = sampling_coor_fg.shape
        plane_axes = generate_planes().to(sampling_coor_fg.device) # (3, 3)

        outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1) # (N, M)

        # sample from the triplane
        sampling_feat_fg = sample_from_planes(plane_axes, triplane, sampling_coor_fg) # (N, 3, M, triplane_dim)
        sampling_feat_fg = sampling_feat_fg.permute(0, 2, 1, 3).reshape(N*M, -1) # (N*M, 3*triplane_dim)
        sampling_density_fg, sampling_color_fg = self.decoder_fg(sampling_feat_fg) # (N*M, 1), (N*M, 3)
        sampling_density_fg = sampling_density_fg.view(N, M, 1) # (N, M, 1)
        # set the density of the points outside the locality to 0
        sampling_density_fg[outsider_idx] *= 0
        sampling_color_fg = sampling_color_fg.view(N, M, 3) # (N, M, 3)
        
        raw_masks = F.relu(sampling_density_fg, inplace=True) # (N, M, 1)
        # normalize the mask along the slot dimension
        masks = raw_masks / (torch.sum(raw_masks, dim=0, keepdim=True) + 1e-8) # (N, M, 1)

        raw_rgb = (sampling_color_fg.tanh() + 1) / 2 # (N, M, 3)
        unmasked_raws = torch.cat([raw_rgb, raw_masks], dim=-1) # (N, M, 4)
        masked_raws = unmasked_raws * masks # (N, M, 4)
        # sum up the masked raws along the slot dimension, obtain the composite color and density
        raws = torch.sum(masked_raws, dim=0) # (M, 4)

        return raws, masked_raws, unmasked_raws, masks
    
class TriplaneRenderer(nn.Module):
    '''
    Triplane Decoder with foreground and background
    '''
    def __init__(self, z_dim=64, triplane_dim=32, n_layer=2, locality_ratio=1, rel_pos=True, position_project=True):
        '''
        z_dim: dimension of the latent space
        n_layer: number of layers in the decoder
        locality_ratio: default 4/7, maximum value = 1
        '''
        super(TriplaneRenderer, self).__init__()
        self.z_dim = z_dim
        self.n_layer = n_layer
        self.rel_pos = rel_pos
        self.locality_ratio = locality_ratio
        self.triplane_generator_fg = TriplaneGenerator(z_dim=z_dim, triplane_dim=triplane_dim)
        self.triplane_generator_bg = TriplaneGenerator(z_dim=z_dim, triplane_dim=triplane_dim)
        self.decoder_fg = TriplaneDecoder(feat_dim=triplane_dim*3, n_layer=n_layer)
        self.decoder_bg = TriplaneDecoder(feat_dim=triplane_dim*3, n_layer=n_layer)

        if position_project:
            self.position_project = nn.Linear(2, self.z_dim)
        else:
            self.position_project = None


    def forward(self, sampling_coor_bg, sampling_coor_fg, z_slots, triplane_resolution=(128, 128), slot_pos=None):
        '''
        sampling_coor_fg: foreground sampling coordinates (N, M, 3); M is the number of foreground points, N is number of slots
        sampling_coor_bg: background sampling coordinates (1, M, 3)
        z_slots: latent vector of size (N+1, z_dim)
        slot_pos: position of the slots (N, 3) (in NSS space); must be provided if rel_pos is True
        '''
        if self.rel_pos:
            assert slot_pos is not None
            # compute query points' relative position with respect to the slots
            sampling_coor_fg = sampling_coor_fg - slot_pos.unsqueeze(1) # (N, M, 3)

        z_slots_fg, z_slots_bg = z_slots[:-1], z_slots[-1:]
        if self.position_project is not None:
            z_slots_fg = z_slots_fg + self.position_project(slot_pos[:, :2]) # (N, z_dim)
        triplane_fg, triplane_bg = self.triplane_generator_fg(z_slots_fg, resolution=triplane_resolution), self.triplane_generator_bg(z_slots_bg, resolution=triplane_resolution) # (N, triplane_dim*3, H, W), (1, triplane_dim*3, H, W)
        _, _, H, W = triplane_fg.shape
        N, M, _ = sampling_coor_fg.shape
        triplane_fg = triplane_fg.view(N, 3, -1, H, W) # (N, 3, triplane_dim, H, W)
        triplane_bg = triplane_bg.view(1, 3, -1, H, W) # (1, 3, triplane_dim, H, W)
        plane_axes = generate_planes().to(sampling_coor_fg.device) # (3, 3)

        outsider_idx = torch.any(sampling_coor_fg.abs() > self.locality_ratio, dim=-1) # (N, M)
        # print the ratio of points outside the locality

        # sample from the triplane
        sampling_feat_fg = sample_from_planes(plane_axes, triplane_fg, sampling_coor_fg) # (N, 3, M, triplane_dim)
        sampling_feat_fg = sampling_feat_fg.permute(0, 2, 1, 3).reshape(N*M, -1) # (N*M, 3*triplane_dim)
        sampling_density_fg, sampling_color_fg = self.decoder_fg(sampling_feat_fg) # (N*M, 1), (N*M, 3)
        sampling_density_fg = sampling_density_fg.view(N, M, 1) # (N, M, 1)
        # set the density of the points outside the locality to 0
        sampling_density_fg[outsider_idx] *= 0
        sampling_color_fg = sampling_color_fg.view(N, M, 3) # (N, M, 3)

        sampling_feat_bg = sample_from_planes(plane_axes, triplane_bg, sampling_coor_bg) # (1, 3, M, triplane_dim)
        sampling_feat_bg = sampling_feat_bg.permute(0, 2, 1, 3).reshape(1*M, -1) # (1*M, 3*triplane_dim)
        sampling_density_bg, sampling_color_bg = self.decoder_bg(sampling_feat_bg) # (1*M, 1), (1*M, 3)
        sampling_density_bg = sampling_density_bg.view(1, M, 1) # (1, M, 1)
        sampling_color_bg = sampling_color_bg.view(1, M, 3) # (1, M, 3)

        raw_masks = F.relu(torch.cat([sampling_density_bg, sampling_density_fg], dim=0), inplace=True) # (N+1, M, 1)
        # normalize the mask along the slot dimension
        masks = raw_masks / (torch.sum(raw_masks, dim=0, keepdim=True) + 1e-8) # (N+1, M, 1)

        raw_rgb = (torch.cat([sampling_color_bg, sampling_color_fg], dim=0).tanh() + 1) / 2 # (N+1, M, 3)

        unmasked_raws = torch.cat([raw_rgb, raw_masks], dim=-1) # (N+1, M, 4)
        masked_raws = unmasked_raws * masks # (N+1, M, 4)
        # sum up the masked raws along the slot dimension, obtain the composite color and density
        raws = torch.sum(masked_raws, dim=0) # (M, 4)

        return raws, masked_raws, unmasked_raws, masks

        



        





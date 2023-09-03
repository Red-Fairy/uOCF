import torch
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import collections
from .utils import conical_frustum_to_gaussian, cylinder_to_gaussian

Rays = collections.namedtuple('Rays', ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))

def pixel2world(slot_pixel_coord, cam2world, intrinsics=None, nss_scale=7.):
    '''
    slot_pixel_coord: (K-1) * 2 on the image plane, x and y coord are in range [-1, 1]
    cam2world: 4 * 4
    H, w: image height and width
    output: convert the slot pixel coord to world coord, then project to the XY plane in the world coord, 
            finally convert to NSS coord
    '''
    device = slot_pixel_coord.device
    if intrinsics is None:
        focal_ratio = (350. / 320., 350. / 240.)
        focal_x, focal_y = focal_ratio[0], focal_ratio[1]
        bias_x, bias_y = 1 / 2., 1 / 2.
        intrinsic = torch.tensor([[focal_x, 0, bias_x, 0],
                                [0, focal_y, bias_y, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).to(device)
    else:
        focal_x, focal_y = intrinsics[0, 0], intrinsics[1, 1]
        bias_x, bias_y = intrinsics[0, 2] / 2 + 1 / 2., intrinsics[1, 2] / 2 + 1 / 2. # convert to [0, 1]
        intrinsic = torch.tensor([[focal_x, 0, bias_x, 0],
                                [0, focal_y, bias_y, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]]).to(torch.float32).to(device)
    spixel2cam = intrinsic.inverse()
    world2nss = torch.tensor([[1/nss_scale, 0, 0],
                                [0, 1/nss_scale, 0],
                                [0, 0, 1/nss_scale]]).to(device)
    
    # convert to pixel coord [0, 1] and [0, 1]
    slot_pixel_coord = ((slot_pixel_coord + 1) / 2).to(device) # (K-1) * 2
    # append 1 to the end
    slot_pixel_coord = torch.cat([slot_pixel_coord, torch.ones_like(slot_pixel_coord[:, :1])], dim=1) # (K-1) * 3
    # convert to cam coord
    slot_cam_coord = torch.matmul(spixel2cam[:3, :3], slot_pixel_coord.t()).t() # (K-1) * 3
    # append 1 to the end, and covert to world coord
    slot_world_coord = torch.matmul(cam2world, torch.cat([slot_cam_coord, torch.ones_like(slot_cam_coord[:, :1])], dim=1).t()).t() # (K-1) * 4
    # normalize
    slot_world_coord = slot_world_coord / slot_world_coord[:, 3:]
    # project to the XY plane
    ray = slot_world_coord[:, :3] - cam2world[:3, 3:].view(1, 3) # (K-1) * 3
    XY_pos = slot_world_coord[:, :3] - ray * (slot_world_coord[:, 2:3] / ray[:, 2:]) # (K-1) * 3
    return torch.matmul(world2nss, XY_pos.t()).t() # (K-1) * 3

class Projection(object):
    def __init__(self, focal_ratio=(350. / 320., 350. / 240.),
                 near=1, far=4, frustum_size=[128, 128, 128], device='cpu',
                 nss_scale=7, render_size=(64, 64), intrinsics=None):
        self.render_size = render_size
        self.device = device
        self.focal_ratio = focal_ratio
        self.near = near
        self.far = far
        self.frustum_size = frustum_size

        self.nss_scale = nss_scale
        self.world2nss = torch.tensor([[1/nss_scale, 0, 0, 0],
                                        [0, 1/nss_scale, 0, 0],
                                        [0, 0, 1/nss_scale, 0],
                                        [0, 0, 0, 1]]).unsqueeze(0).to(device)
        self.construct_intrinsic(intrinsics)

        # radii is the 2/sqrt(12) width of the pixel in nss world coordinates
        # calculate the distance in world coord between point (0,0) and (0,1) of camera plane
        ps = torch.tensor([[0, 0, 0, 1], [0, 1, 0, 1]]).to(torch.float32).to(device)
        ps = torch.matmul(self.spixel2cam, ps.t()).t() # 2x4
        ps = ps / ps[:, 3:] # 2x4
        ps = ps[:, :3] # 2x3
        self.radii = torch.norm(ps[1] - ps[0]) / torch.sqrt(torch.tensor(3.)).to(device) / self.nss_scale

    def construct_intrinsic(self, intrinsics=None):
        if intrinsics is None:
            self.focal_x = self.focal_ratio[0] * self.frustum_size[0]
            self.focal_y = self.focal_ratio[1] * self.frustum_size[1]
            bias_x = (self.frustum_size[0] - 1.) / 2.
            bias_y = (self.frustum_size[1] - 1.) / 2.
        else: # intrinsics stores focal_ratio and principal point
            self.focal_x = intrinsics[0, 0] * self.frustum_size[0]
            self.focal_y = intrinsics[1, 1] * self.frustum_size[1]
            bias_x = ((intrinsics[0, 2] + 1) * self.frustum_size[0] - 1.) / 2.
            bias_y = ((intrinsics[1, 2] + 1) * self.frustum_size[1] - 1.) / 2.
        intrinsic_mat = torch.tensor([[self.focal_x, 0, bias_x, 0],
                                        [0, self.focal_y, bias_y, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]).to(torch.float32)
        self.cam2spixel = intrinsic_mat.to(self.device)
        self.spixel2cam = intrinsic_mat.inverse().to(self.device)
        
    def construct_frus_coor(self, stratified=False):
        x = torch.arange(self.frustum_size[0])
        y = torch.arange(self.frustum_size[1])
        z = torch.arange(self.frustum_size[2])
        x, y, z = torch.meshgrid([x, y, z])
        x_frus = x.flatten().to(self.device)
        y_frus = y.flatten().to(self.device)
        z_frus = z.flatten().to(self.device)
        # project frustum points to vol coord
        depth_range = torch.linspace(self.near, self.far, self.frustum_size[2]).to(self.device) # D
        z_cam = depth_range[z_frus].to(self.device) # (WxHxD)

        # stratified sampling
        if stratified:
            z_cam = z_cam.view(-1, self.frustum_size[2]) # (WxH)xD
            z_cam = z_cam + torch.rand_like(z_cam) * (self.far - self.near) / self.frustum_size[2] / 2 # (WxH)xD
            z_cam = z_cam.flatten() # (WxHxD)

        # print('z_cam', z_cam.shape, x_frus.shape)
        x_unnorm_pix = x_frus * z_cam
        y_unnorm_pix = y_frus * z_cam
        z_unnorm_pix = z_cam
        pixel_coor = torch.stack([x_unnorm_pix, y_unnorm_pix, z_unnorm_pix, torch.ones_like(x_unnorm_pix)])
        return pixel_coor # 4x(WxHxD)

    def construct_sampling_coor(self, cam2world, partitioned=False, intrinsics=None, frustum_size=None, stratified=False):
        """
        construct a sampling frustum coor in NSS space, and generate z_vals/ray_dir
        input:
            cam2world: Nx4x4, N: #images to render
        output:
            frus_nss_coor: (NxDxHxW)x3
            z_vals: (NxHxW)xD
            ray_dir: (NxHxW)x3
        """
        if intrinsics is not None: # overwrite intrinsics
            self.construct_intrinsic(intrinsics)
        if frustum_size is not None: # overwrite frustum_size
            self.frustum_size = frustum_size
        N = cam2world.shape[0]
        W, H, D = self.frustum_size
        pixel_coor = self.construct_frus_coor(stratified=stratified) # 4x(WxHxD)
        frus_cam_coor = torch.matmul(self.spixel2cam, pixel_coor.float())  # 4x(WxHxD)
        # debug(frus_cam_coor, save_name='frus_cam_coor')
        frus_world_coor = torch.matmul(cam2world, frus_cam_coor)  # Nx4x(WxHxD)
        # debug(frus_world_coor, save_name='frus_world_coor', cam2world=cam2world)
        frus_nss_coor = torch.matmul(self.world2nss, frus_world_coor)  # Nx4x(WxHxD)
        # debug(frus_nss_coor, save_name='frus_nss_coor', cam2world=cam2world/self.nss_scale)
        frus_nss_coor = frus_nss_coor.view(N, 4, W, H, D).permute([0, 4, 3, 2, 1])  # NxDxHxWx4
        frus_nss_coor = frus_nss_coor[..., :3] / frus_nss_coor[..., 3:]  # NxDxHxWx3
        scale = H // self.render_size[0]
        if partitioned:
            frus_nss_coor_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                frus_nss_coor_.append(frus_nss_coor[:, :, h::scale, w::scale, :])
            frus_nss_coor = torch.stack(frus_nss_coor_, dim=0)  # 4xNxDx(H/s)x(W/s)x3
            frus_nss_coor = frus_nss_coor.flatten(start_dim=1, end_dim=4)  # 4x(NxDx(H/s)x(W/s))x3
        else:
            frus_nss_coor = frus_nss_coor.flatten(start_dim=0, end_dim=3)  # (NxDxHxW)x3

        z_vals = (frus_cam_coor[2] - self.near) / (self.far - self.near)  # (WxHxD) range=[0,1]
        z_vals = z_vals.expand(N, W * H * D)  # Nx(WxHxD)

        if partitioned:
            z_vals = z_vals.view(N, W, H, D).permute([0, 2, 1, 3])  # NxHxWxD
            z_vals_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                z_vals_.append(z_vals[:, h::scale, w::scale, :])
            z_vals = torch.stack(z_vals_, dim=0)  # 4xNx(H/s)x(W/s)xD
            z_vals = z_vals.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))xD
        else:
            z_vals = z_vals.view(N, W, H, D).permute([0, 2, 1, 3]).flatten(start_dim=0, end_dim=2)  # (NxHxW)xD

        # infer ray_origin from cam2world
        # ray_origin = cam2world[:, :3, 3]  # Nx3
        # ray_origin = ray_origin.unsqueeze(1).unsqueeze(1).expand(N, H, W, 3).flatten(start_dim=0, end_dim=2)  # (NxHxW)x3

        # construct cam coord for ray_dir
        x = torch.arange(self.frustum_size[0])
        y = torch.arange(self.frustum_size[1])
        X, Y = torch.meshgrid([x, y])
        Z = torch.ones_like(X)
        pix_coor = torch.stack([Y, X, Z]).to(self.device)  # 3xHxW, 3=xyz
        cam_coor = torch.matmul(self.spixel2cam[:3, :3], pix_coor.flatten(start_dim=1).float())  # 3x(HxW)
        ray_dir = cam_coor.permute([1, 0])  # (HxW)x3
        ray_dir = ray_dir.view(H, W, 3)

        if partitioned:
            ray_dir = ray_dir.expand(N, H, W, 3)
            ray_dir_ = []
            for i in range(scale ** 2):
                h, w = divmod(i, scale)
                ray_dir_.append(ray_dir[:, h::scale, w::scale, :])
            ray_dir = torch.stack(ray_dir_, dim=0)  # 4xNx(H/s)x(W/s)x3
            ray_dir = ray_dir.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))x3
        else:
            ray_dir = ray_dir.expand(N, H, W, 3).flatten(start_dim=0, end_dim=2)  # (NxHxW)x3

        return frus_nss_coor, z_vals, ray_dir
    
    def construct_origin_dir(self, cam2world):
        '''
        construct ray origin and direction for each pixel in the frustum
        ray_origin: (NxHxW)x3, ray_dir: (NxHxW)x3
        both are in world coord
        '''
        N, W, H = cam2world.shape[0], self.frustum_size[0], self.frustum_size[1]
        x = torch.arange(self.frustum_size[0])
        y = torch.arange(self.frustum_size[1])
        X, Y = torch.meshgrid([x, y])
        Z = torch.ones_like(X)
        pix_coor = torch.stack([Y, X, Z]).to(self.device)  # 3xHxW, 3=xyz
        cam_coor = torch.matmul(self.spixel2cam[:3, :3], pix_coor.flatten(start_dim=1).float())  # 3x(HxW)
        ray_dir = cam_coor.permute([1, 0])  # (HxW)x3
        ray_dir = ray_dir.view(H, W, 3).expand(N, H, W, 3).flatten(1, 2)  # Nx(HxW)x3
        # convert to world coord, cam2world[:, :3, :3] is Nx3x3
        ray_dir = torch.matmul(cam2world[:, :3, :3].unsqueeze(1).expand(-1, H * W, -1, -1), ray_dir.unsqueeze(-1)).squeeze(-1)  # Nx(HxW)x3
        ray_dir = ray_dir.flatten(0, 1)  # (NxHxW)x3
        # ray_dir = F.normalize(ray_dir, dim=-1).flatten(0, 1)  # (NxHxW)x3

        ray_origin = cam2world[:, :3, 3]  # Nx3
        ray_origin = ray_origin / self.nss_scale
        ray_origin = ray_origin.unsqueeze(1).unsqueeze(1).expand(N, H, W, 3).flatten(start_dim=0, end_dim=2)  # (NxHxW)x3

        near, far = self.near / self.nss_scale, self.far / self.nss_scale

        return ray_origin, ray_dir, near, far
    
    def construct_sampling_coor_new(self, cam2world, partitioned=False, intrinsics=None, frustum_size=None, stratified=False):
        """
        construct a sampling frustum coor in NSS space, and generate z_vals/ray_dir
        input:
            cam2world: Nx4x4, N: #images to render
        output:
            frus_nss_coor: (NxDxHxW)x3
            z_vals: (NxHxW)xD
            ray_dir: (NxHxW)x3
        """
        if intrinsics is not None: # overwrite intrinsics
            self.construct_intrinsic(intrinsics)
        if frustum_size is not None: # overwrite frustum_size
            self.frustum_size = frustum_size
        N = cam2world.shape[0]
        W, H, D = self.frustum_size

        ray_origin, ray_dir, near_nss, far_nss = self.construct_origin_dir(cam2world)
        # sample frustum_size[2] points along each ray
        z_vals = torch.linspace(near_nss, far_nss, D).to(self.device) # D
        z_vals = z_vals.unsqueeze(0).expand(N*H*W, D) # (NxHxW)xD

        if stratified:
            z_vals = z_vals.view(N, H*W, D) # NxHxWxD
            z_vals = z_vals + torch.rand_like(z_vals) * (far_nss - near_nss) / D / 2
            z_vals = z_vals.flatten(start_dim=0, end_dim=1)

        # construct sampling points
        ray_dir_ = ray_dir.unsqueeze(-2).expand(N*H*W, D, 3) # (NxHxW)xDx3
        ray_origin_ = ray_origin.unsqueeze(-2).expand(N*H*W, D, 3) # (NxHxW)xDx3
        z_vals_ = z_vals.view(N*H*W, D, 1)
        # print(ray_origin_.shape, ray_dir_.shape, z_vals_.shape)
        sampling_points = ray_origin_ + ray_dir_ * z_vals_.view(N*H*W, D, 1) # (NxHxW)xDx3
        sampling_points = sampling_points.view(N, H, W, D, 3).permute([0, 3, 1, 2, 4]).flatten(0, 3) # (NxDxHxW)x3

        if partitioned:
            scale = H // self.render_size[0]

            sampling_points = sampling_points.view(N, D, H, W, 3)
            sampling_points_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                sampling_points_.append(sampling_points[:, :, h::scale, w::scale, :])
            sampling_points = torch.stack(sampling_points_, dim=0)  # 4xNxDx(H/s)x(W/s)x3
            sampling_points = sampling_points.flatten(start_dim=1, end_dim=4)  # 4x(NxDx(H/s)x(W/s))x3

            z_vals = z_vals.view(N, H, W, D)
            z_vals_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                z_vals_.append(z_vals[:, h::scale, w::scale, :])
            z_vals = torch.stack(z_vals_, dim=0)  # 4xNx(H/s)x(W/s)xD
            z_vals = z_vals.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))xD

            ray_dir = ray_dir.view(N, H, W, 3)
            ray_dir_ = []
            for i in range(scale ** 2):
                h, w = divmod(i, scale)
                ray_dir_.append(ray_dir[:, h::scale, w::scale, :])
            ray_dir = torch.stack(ray_dir_, dim=0)  # 4xNx(H/s)x(W/s)x3
            ray_dir = ray_dir.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))x3

        # z_vals = (z_vals - near_nss) / (far_nss - near_nss) # (NxHxW)xD, put to range [0, 1] TODO: used in uORF, but is not correct

        return sampling_points, z_vals, ray_dir

    def sample_along_rays(self, cam2world, partitioned=False, intrinsics=None, frustum_size=None, stratified=True, ray_shape='cone'):
        """Stratified sampling along the rays.

        Args:
        batch_size: N*H*W
        origins: torch.tensor(float32), [batch_size, 3], ray origins.
        directions: torch.tensor(float32), [batch_size, 3], ray directions.
        radii: torch.tensor(float32), [batch_size, 3], ray radii.
        num_samples: int.
        near: torch.tensor, [batch_size, 1], near clip.
        far: torch.tensor, [batch_size, 1], far clip.
        stratified: bool, use randomized stratified sampling.
        lindisp: bool, sampling linearly in disparity rather than depth.

        Returns:
        t_vals: torch.tensor, [batch_size, num_samples], sampled z values.
        means: torch.tensor, [batch_size, num_samples, 3], sampled means.
        covs: torch.tensor, [batch_size, num_samples, 3], sampled covariances.
        ray_dir: torch.tensor, [batch_size, 3], ray directions.
        """

        if intrinsics is not None: # overwrite intrinsics
            self.construct_intrinsic(intrinsics)
        if frustum_size is not None: # overwrite frustum_size
            self.frustum_size = frustum_size
        N = cam2world.shape[0]
        W, H, D = self.frustum_size # D: num_samples

        ray_origin, ray_dir, near_nss, far_nss = self.construct_origin_dir(cam2world) # (N*H*W)*3, (N*H*W)*3

        batch_size = N * H * W
        device = ray_origin.device

        t_vals = torch.linspace(0., 1., D + 1,  device=device)
        t_vals = near_nss * (1. - t_vals) + far_nss * t_vals

        if stratified:
            mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
            upper = torch.cat([mids, t_vals[..., -1:]], -1)
            lower = torch.cat([t_vals[..., :1], mids], -1)
            t_rand = torch.rand(batch_size, D + 1, device=device)
            t_vals = lower + (upper - lower) * t_rand
        else:
            # Broadcast t_vals to make the returned shape consistent.
            t_vals = torch.broadcast_to(t_vals, [batch_size, D + 1])

        radii = self.radii.unsqueeze(0).expand(batch_size, 1) # (NxHxW)x1
        
        means, covs = self.cast_rays(t_vals, ray_origin, ray_dir, radii, ray_shape) # (N*H*W)*D*3, (N*H*W)*D*3

        if partitioned:
            scale = H // self.render_size[0]
            means = means.view(N, H, W, D, 3)
            means_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                means_.append(means[:, h::scale, w::scale, ...])
            means = torch.stack(means_, dim=0)  # 4xNx(H/s)x(W/s)xDx3
            means = means.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))xDx3

            covs = covs.view(N, H, W, D, 3)
            covs_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                covs_.append(covs[:, h::scale, w::scale, ...])
            covs = torch.stack(covs_, dim=0)  # 4xNx(H/s)x(W/s)xDx3
            covs = covs.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))xDx3

            t_vals = t_vals.view(N, H, W, D+1)
            t_vals_ = []
            for i in range(scale**2):
                h, w = divmod(i, scale)
                t_vals_.append(t_vals[:, h::scale, w::scale, :])
            t_vals = torch.stack(t_vals_, dim=0)  # 4xNx(H/s)x(W/s)x(D+1)
            t_vals = t_vals.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))x(D+1)

            ray_dir = ray_dir.view(N, H, W, 3)
            ray_dir_ = []
            for i in range(scale ** 2):
                h, w = divmod(i, scale)
                ray_dir_.append(ray_dir[:, h::scale, w::scale, :])
            ray_dir = torch.stack(ray_dir_, dim=0)  # 4xNx(H/s)x(W/s)x3
            ray_dir = ray_dir.flatten(start_dim=1, end_dim=3)  # 4x(Nx(H/s)x(W/s))x3
        
        return (means, covs), t_vals[..., :D], ray_dir

    def cast_rays(self, t_samples, origins, directions, radii, ray_shape, diagonal=True):
        """Cast rays (cone- or cylinder-shaped) and featurize sections of it.
        Args:
            t_samples: float array [B, n_sample+1], the "fencepost" distances along the ray.
            origins: float array [B, 3], the ray origin coordinates.
            directions [B, 3]: float array, the ray direction vectors.
            radii[B, 1]: float array, the radii (base radii for cones) of the rays.
            ray_shape: string, the shape of the ray, must be 'cone' or 'cylinder'.
            diagonal: boolean, whether or not the covariance matrices should be diagonal. 
            if true, cov will only have 3 values, not 3*3.
        Returns:
            a tuple of arrays of means and covariances.
        """
        t0 = t_samples[..., :-1]  # [B, n_samples]
        t1 = t_samples[..., 1:]
        if ray_shape == 'cone':
            gaussian_fn = conical_frustum_to_gaussian
        elif ray_shape == 'cylinder':
            raise NotImplementedError
        else:
            assert False
        means, covs = gaussian_fn(directions, t0, t1, radii, diagonal)
        means = means + torch.unsqueeze(origins, dim=-2)
        return means, covs
    
    def sample_along_rays_360(self, cam2world, partitioned=False, intrinsics=None, frustum_size=None, stratified=True, ray_shape='cone'):

        if intrinsics is not None: # overwrite intrinsics
            self.construct_intrinsic(intrinsics)
        if frustum_size is not None: # overwrite frustum_size
            self.frustum_size = frustum_size
        N = cam2world.shape[0]
        W, H, D = self.frustum_size # D: num_samples

        ray_origin, ray_dir, near_nss, far_nss = self.construct_origin_dir(cam2world) # (N*H*W)*3, (N*H*W)*3

        batch_size = N * H * W
        device = ray_origin.device

        t_vals = torch.linspace(0., 1., D + 1,  device=device)
        far_inv = 1 / far_nss
        near_inv = 1 / near_nss
        t_vals = near_inv * (1. - t_vals) + far_inv * t_vals

        if stratified:
            mids = 0.5 * (t_inv[..., 1:] + t_inv[..., :-1])
            upper = torch.cat([mids, t_inv[..., -1:]], -1)
            lower = torch.cat([t_inv[..., :1], mids], -1)
            t_rand = torch.rand(batch_size, D + 1, device=device)
            t_inv = lower + (upper - lower) * t_rand
        else:
            # Broadcast t_inv to make the returned shape consistent.
            t_inv = torch.broadcast_to(t_inv, [batch_size, D + 1])

        t_vals = 1 / t_inv
        radii = self.radii.unsqueeze(0).expand(batch_size, 1) # (NxHxW)x1

        means, covs = self.cast_rays(t_vals, ray_origin, ray_dir, radii, ray_shape, False)
        return t_inv, (means, covs)

if __name__ == '__main__':
    pass

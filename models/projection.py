import torch
import numpy as np
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from matplotlib import cm

def pixel2world(slot_pixel_coord, cam2world):
    '''
    slot_pixel_coord: (K-1) * 2 on the image plane, x and y coord are in range [-1, 1]
    cam2world: 4 * 4
    H, w: image height and width
    output: convert the slot pixel coord to world coord, then project to the XY plane in the world coord, 
            finally convert to NSS coord
    '''
    device = slot_pixel_coord.device
    focal_ratio = (350. / 320., 350. / 240.)
    focal_x, focal_y = focal_ratio[0], focal_ratio[1]
    bias_x, bias_y = 1 / 2., 1 / 2.
    intrinsic = torch.tensor([[focal_x, 0, bias_x, 0],
                              [0, focal_y, bias_y, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]]).to(device)
    spixel2cam = intrinsic.inverse()
    nss_scale = 7
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
                 nss_scale=7, render_size=(64, 64)):
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
        self.focal_x = self.focal_ratio[0] * self.frustum_size[0]
        self.focal_y = self.focal_ratio[1] * self.frustum_size[1]
        bias_x = (self.frustum_size[0] - 1.) / 2.
        bias_y = (self.frustum_size[1] - 1.) / 2.
        intrinsic_mat = torch.tensor([[self.focal_x, 0, bias_x, 0],
                                      [0, self.focal_y, bias_y, 0],
                                      [0, 0, 1, 0],
                                      [0, 0, 0, 1]])
        self.cam2spixel = intrinsic_mat.to(self.device)
        self.spixel2cam = intrinsic_mat.inverse().to(self.device) 
        
    def construct_frus_coor(self, z_vals=None):
        x = torch.arange(self.frustum_size[0])
        y = torch.arange(self.frustum_size[1])
        z = torch.arange(self.frustum_size[2])
        x, y, z = torch.meshgrid([x, y, z])
        x_frus = x.flatten().to(self.device)
        y_frus = y.flatten().to(self.device)
        z_frus = z.flatten().to(self.device)
        # project frustum points to vol coord
        depth_range = torch.linspace(self.near, self.far, self.frustum_size[2]).to(self.device)
        z_cam = depth_range[z_frus].to(self.device)

        # print('z_cam', z_cam.shape, x_frus.shape)
        x_unnorm_pix = x_frus * z_cam
        y_unnorm_pix = y_frus * z_cam
        z_unnorm_pix = z_cam
        pixel_coor = torch.stack([x_unnorm_pix, y_unnorm_pix, z_unnorm_pix, torch.ones_like(x_unnorm_pix)])
        return pixel_coor # 4x(WxHxD)

    def construct_sampling_coor(self, cam2world, partitioned=False):
        """
        construct a sampling frustum coor in NSS space, and generate z_vals/ray_dir
        input:
            cam2world: Nx4x4, N: #images to render
        output:
            frus_nss_coor: (NxDxHxW)x3
            z_vals: (NxHxW)xD
            ray_dir: (NxHxW)x3
        """
        N = cam2world.shape[0]
        W, H, D = self.frustum_size
        pixel_coor = self.construct_frus_coor()
        frus_cam_coor = torch.matmul(self.spixel2cam, pixel_coor.float())  # 4x(WxHxD)
        # debug
        # frus_cam_coor_debug = frus_cam_coor.reshape(1, 4, -1).permute(0, 2, 1).cpu().numpy()
        # colors = cm.rainbow(np.linspace(0, 1, 1))
        # cam2world_debug = cam2world.cpu().numpy()
        # fig = plt.figure(figsize=(40, 20))
        # for i in range(1):
        #     ax = plt.subplot(2, 4, i+1, projection='3d')
        #     # visualize the frustum
        #     ax.scatter(frus_cam_coor_debug[i,:,0], frus_cam_coor_debug[i,:,1], frus_cam_coor_debug[i,:,2], c=colors[i], marker='o', s=3)
        #     # visualize the camera origin
        #     ax.scatter(0, 0, 0, c='r', marker='o', s=20)
        # fig.savefig('frus_cam_coor.png')
        frus_world_coor = torch.matmul(cam2world, frus_cam_coor)  # Nx4x(WxHxD)
        frus_nss_coor = torch.matmul(self.world2nss, frus_world_coor)  # Nx4x(WxHxD)
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

    def sample_pdf(self, bins, weights, N_samples, det=True):
        # Get pdf
        # bins: [N_rays, N_samples_coarse-1]
        # weights: [N_rays, N_samples_coarse-2]
        weights = weights + 1e-5 # prevent nans
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

        # Take uniform samples
        if det:
            u = torch.linspace(0., 1., steps=N_samples)
            u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        else:
            u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

        # Invert CDF
        u = u.contiguous().to(cdf.device)
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

        # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

        denom = (cdf_g[...,1]-cdf_g[...,0])
        denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
        t = (u-cdf_g[...,0])/denom
        samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

        return samples

    def sample_importance(self, cam2world, z_vals, ray_dir, weights, N_inportance, size=(64, 64)):
        '''
        :param cam2world: (N, 4, 4)
        :param z_vals: (NxHxW)xN_corase in range [0, 1]
        :param weights: (NxHxW)xN_corase
        :param N_samples: int
        :param ray_dir: (NxHxW)x3, in camera space
        :param size: (H, W), supervision size
        :output z_vals: (NxHxW)x(N_inportance+N_coarse)
        :output pixel_coor: (NxHxW)x(N_inportance+N_coarse)x3 in world space
        '''

        N = cam2world.shape[0]
        N_coarse = z_vals.shape[-1]
        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1]) # (NxHxW)x(N_coarse-1)
        z_samples = self.sample_pdf(z_vals_mid, weights[...,1:-1], N_inportance)  # (NxHxW)xN_inportance
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)  # (NxHxW)x(N_inportance+N_coarse)

        z_pos = z_vals * (self.far - self.near) + self.near  # (NxHxW)x(N_inportance+N_coarse)

        # constuct pixel coor, first in camera space, camera origin is (0, 0, 0)
        pixel_coor = torch.einsum('ij,ikj->ikj', ray_dir, z_pos.unsqueeze(-1).expand(-1, -1, 3)) # (NxHxW)x(N_inportance+N_coarse)x3
        # convert to world space, first reshape to (N, 3, ((N_inportance+N_coarse)*H*W), then multiply cam2world
        pixel_coor = pixel_coor.reshape(N, *size, N_inportance+N_coarse, 3).permute(0, 4, 3, 1, 2).reshape(N, 3, -1) # Nx3x((N_inportance+N_coarse)*H*W)
        pixel_coor = torch.cat([pixel_coor, torch.ones(N, 1, pixel_coor.shape[-1]).to(pixel_coor.device)], dim=1) # Nx4x((N_inportance+N_coarse)*H*W)
        pixel_coor = torch.bmm(cam2world, pixel_coor) # Nx4x((N_inportance+N_coarse)*H*W)
        pixel_coor = pixel_coor[:, :3, :] # Nx3x((N_inportance+N_coarse)*H*W)
        pixel_coor = pixel_coor.permute(0, 2, 1).flatten(0, 1) # ((NxHxW)x(N_inportance+N_coarse))x3

        # transform to nss space
        pixel_coor = pixel_coor / self.nss_scale

        return pixel_coor, z_vals
    

if __name__ == '__main__':
    pass
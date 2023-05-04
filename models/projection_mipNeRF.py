import torch
import numpy as np
import torch.nn.functional as F

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
                 near=5, far=16, frustum_size=[128, 128, 128], device='cpu',
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

        # calculate the pixel width in world coord (identical to cam coord)
        pt1, pt2 = torch.Tensor([0, 0, 1, 1]), torch.Tensor([1, 1, 1, 1])
        pt1, pt2 = pt1.to(self.device), pt2.to(self.device)
        pt1_cam, pt2_cam = torch.matmul(self.spixel2cam, pt1) / pt1[-1], torch.matmul(self.spixel2cam, pt2) / pt2[-1]
        self.width = torch.norm(pt1_cam - pt2_cam, dim=0) / torch.sqrt(torch.tensor(6.)).to(self.device)

    def construct_frus_coor(self):
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

        x_unnorm_pix = x_frus * z_cam
        y_unnorm_pix = y_frus * z_cam
        z_unnorm_pix = z_cam
        pixel_coor = torch.stack([x_unnorm_pix, y_unnorm_pix, z_unnorm_pix, torch.ones_like(x_unnorm_pix)])
        return pixel_coor

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

        frus_world_coor = torch.matmul(cam2world, frus_cam_coor)  # Nx4x(WxHxD)
        frus_nss_coor = torch.matmul(self.world2nss, frus_world_coor)  # Nx4x(WxHxD)
        frus_nss_coor = frus_nss_coor.view(N, 4, W, H, D).permute([0, 4, 3, 2, 1])  # NxDxHxWx4
        frus_nss_coor = frus_nss_coor[..., :3]  # NxDxHxWx3
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

    def construct_sampling_coor_MipNeRF(self, cam2world):
        """
        cam2world: Nx4x4
        nss_space: world coordinate / nss_scale
        Construct a sampling frustum coor in NSS space with MipNeRF sampling strategy (require both mean and cov)
        return frus_nss_coor (mean) and frus_nss_cov (covariance), both in NSS space, and generate z_vals/ray_dir
        """
        # step 1, generate ray origin, ray direction, and ray radius (the distance between adjacent pixels in world space)
        N = cam2world.shape[0]
        x, y = torch.meshgrid(torch.arange(self.frustum_size[0]), torch.arange(self.frustum_size[1]))
        x, y = x.to(self.device), y.to(self.device)
        camera_directions = torch.stack([(x - self.frustum_size[0] * 0.5) / self.focal_x,
                                            -(y - self.frustum_size[1] * 0.5) / self.focal_y,
                                            -torch.ones_like(x)], axis=-1)
        camera_directions = camera_directions.reshape(-1, 3)  # (HxW)x3
        camera_directions = camera_directions.expand(N, self.frustum_size[0] * self.frustum_size[1], 3)  # Nx(HxW)x3
        # convert to world space, cam2world: Nx4x4
        ray_dir = torch.matmul(cam2world[:, :3, :3], camera_directions.permute([0, 2, 1])).permute([0, 2, 1]) # N*(HxW)*3
        ray_dir = F.normalize(ray_dir, dim=-1)  # Nx(HxW)x3
        ray_origins = cam2world[:, :3, 3:].expand(N, 3, self.frustum_size[0] * self.frustum_size[1])  # Nx3x(HxW)
        radius = self.width.expand(N, self.frustum_size[0] * self.frustum_size[1])  # Nx(HxW)

        # step 2, call sample_along_rays
        from .ray_utils import sample_along_rays
        ray_origins, ray_dir, radius = ray_origins.reshape(-1, 3), ray_dir.reshape(-1, 3), radius.reshape(-1, 1)
        z_vals, (frus_coor, frus_cov) = sample_along_rays(origins=ray_origins, directions=ray_dir, radii=radius, 
                                                 num_samples=self.frustum_size[2],near=self.near, far=self.far)
        # we only need diagonal entries of frus_conv
        z_vals = z_vals[:,:-1].reshape(N * self.frustum_size[0] * self.frustum_size[1], self.frustum_size[2])  # Nx(HxW)xD
        frus_coor = frus_coor.reshape(-1, 3)
        frus_cov = frus_cov.reshape(-1, 3) # output: (Nx(HxW))x3

        # step 3, convert to NSS space
        frus_nss_coor = frus_coor / self.nss_scale
        frus_nss_cov = frus_cov / self.nss_scale ** 2

        return (frus_nss_coor, frus_nss_cov), z_vals, ray_dir

if __name__ == '__main__':
    pass
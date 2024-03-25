import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from .utils import PositionalEncoding, focal2fov, fibonacci_sphere, quaternion_raw_multiply, matrix_to_quaternion
import numpy as np

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class GuassianMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, n_layers=3):
        super(GuassianMLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.MLP = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.MLP.append(nn.Linear(in_dim, hidden_dim))
                self.MLP.append(nn.ReLU())
            elif i == n_layers-1:
                self.MLP.append(nn.Linear(hidden_dim, out_dim))
                self.MLP.append(nn.ReLU())
            else:
                self.MLP.append(nn.Linear(hidden_dim, hidden_dim))

        self.MLP = nn.Sequential(*self.MLP)

    def forward(self, x):
        return self.MLP(x)

class GaussianDecoder(nn.Module):
    '''

    '''

    def __init__(self, n_freq=5, n_samples_fg=1000, n_samples_bg=1000, fg_boundary=3, bg_boundary=10, z_dim=64):

        self.register_buffer('n_samples_fg', n_samples_fg)
        self.register_buffer('n_samples_bg', n_samples_bg)

        self.points_theta_phi_fg = fibonacci_sphere(self.n_samples_fg)
        self.points_theta_phi_bg = fibonacci_sphere(self.n_samples_bg)

        self.fg_boundary = fg_boundary
        self.bg_boundary = bg_boundary

        self.depth_act = nn.Sigmoid()
        self.rotation_bias_activation = torch.nn.Tanh()
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        self.positional_encoding = PositionalEncoding(max_deg=n_freq)

        self.split_dimensions = [1, 2, 1, 3, 4, 3] # depth, rotation_bias, opacity, scaling, rotation (quaternion), color

        self.fg_mlp = GuassianMLP(in_dim=z_dim+4*n_freq+2, out_dim=sum(self.split_dimensions))
        self.bg_mlp = GuassianMLP(in_dim=z_dim+4*n_freq+2, out_dim=sum(self.split_dimensions))

    def processQueries(self, z_fg, z_bg):
        '''
        z_fg: K*C
        z_bg: 1*C
        return the input_fg and input_bg rotate to the camera coordinate, 
                (K, n_samples, C+4*n_freq+2) and (1, n_samples, C+4*n_freq+2)
        '''
        K = z_fg.shape[0]
        points_enc_fg = self.positional_encoding(self.points_theta_phi_fg) # (n_samples, 4*n_freq+2)
        input_fg = z_fg.unsqueeze(1).expand(-1, self.n_samples_fg, -1)
        input_fg = torch.cat([input_fg, points_enc_fg.unsqueeze(0).expand(K, -1, -1)], dim=-1) # (K, n_samples_fg, C+4*n_freq+2)

        points_enc_bg = self.positional_encoding(self.points_theta_phi_bg) # (n_samples, 4*n_freq+2)
        input_bg = z_bg.unsqueeze(1).expand(-1, self.n_samples_bg, -1)
        input_bg = torch.cat([input_bg, points_enc_bg.unsqueeze(0).expand(1, -1, -1)], dim=-1) # (1, n_samples_bg, C+4*n_freq+2)

        return input_fg, input_bg

    def transform_rotations(self, rotations, source_cv2wT_quat):
        """
        Applies a transform that rotates the predicted rotations from 
        camera space to world space.
        Args:
            rotations: predicted in-camera rotation quaternions (B x N x 4)
            source_cameras_to_world: transformation quaternions from 
                camera-to-world matrices transposed(B x 4)
        Retures:
            rotations with appropriately applied transform to world space
        """

        Mq = source_cv2wT_quat.unsqueeze(1).expand(*rotations.shape)

        rotations = quaternion_raw_multiply(Mq, rotations) 
        
        return rotations

    def get_pos(self, depth, rotation_offset, rotation, size, center_position=None):
        '''
        depth: (K, n_samples, 1)
        rotation_offset: (K, n_samples, 2), rotation offset
        rotation: rotation matrix from camera to world (K, 3, 3)
        center_position: (K, 3) in world coordinate
        return: gaussian splatting position (K, n_samples, 3) in world coordinate
        '''
        K, n_samples, _ = depth.shape
        center_position = center_position if center_position is not None else torch.zeros(K, 3).to(depth.device)
        depth = self.depth_act(depth) * size
        ray_theta_phi = self.points_theta_phi.unsqueeze(0).expand(K, -1, -1) +\
                     self.rotation_bias_activation(rotation_offset) / math.sqrt(math.pi/n_samples/4)
        ray_y = torch.sin(ray_theta_phi[:, :, 1])
        ray_x = torch.cos(ray_theta_phi[:, :, 1]) * torch.sin(ray_theta_phi[:, :, 0])
        ray_z = torch.cos(ray_theta_phi[:, :, 1]) * torch.cos(ray_theta_phi[:, :, 0])
        ray_direction = torch.stack([ray_x, ray_y, ray_z], dim=-1) # (K, n_samples, 3)
        ray_direction = torch.matmul(ray_direction, rotation)

        pos = center_position.unsqueeze(1) + ray_direction * depth

        return pos

    def forward(self, fg_transform, fg_slot_position, z_fg, z_bg):
        '''
        fg_transform: 1*4*4
        '''
        K = z_fg.shape[0]

        input_fg, input_bg = self.processQueries(z_fg, z_bg)
        fg_output = self.fg_mlp(input_fg) # (K, n_samples, sum(self.split_dimensions))
        bg_output = self.bg_mlp(input_bg)

        quat_transform = matrix_to_quaternion(fg_transform.squeeze(0)).unsqueeze(0) # (1, 4)

        fg_splits = torch.split(fg_output, self.split_dimensions, dim=-1)
        depth_fg, rotation_bias_fg, opacity_fg, scaling_fg, rotation_fg, color_fg = fg_splits
        pos_fg = self.get_pos(depth_fg, rotation_bias_fg, rotation_fg, self.fg_boundary, fg_slot_position).flatten(0, 1).unsqueeze(0) # (1, K*n_samples, 3)
        rotation_fg = self.transform_rotations(rotation_fg.flatten(0, 1).unsqueeze(0), quat_transform) # (1, K*n_samples, 4)
        opacity_fg, scaling_fg, color_fg = opacity_fg.flatten(0, 1).unsqueeze(0), scaling_fg.flatten(0, 1).unsqueeze(0), color_fg.flatten(0, 1).unsqueeze(0)

        bg_splits = torch.split(bg_output, self.split_dimensions, dim=-1)
        depth_bg, rotation_bias_bg, opacity_bg, scaling_bg, rotation_bg, color_bg = bg_splits
        pos_bg = self.get_pos(depth_bg, rotation_bias_bg, rotation_bg, self.bg_boundary).flatten(0, 1).unsqueeze(0) # (1, n_samples, 3)
        rotation_bg = self.transform_rotations(rotation_bg.flatten(0, 1).unsqueeze(0), quat_transform) # (1, n_samples, 4)

        out_dict = {
            'xyz': torch.cat([pos_fg, pos_bg], dim=1), # (1, K*n_samples+n_samples, 3)
            'opacity': self.opacity_activation(torch.cat([opacity_fg, opacity_bg], dim=1)),
            'scaling': self.scaling_activation(torch.cat([scaling_fg, scaling_bg], dim=1)),
            'rotation': self.rotation_activation(torch.cat([rotation_fg, rotation_bg], dim=1)),
            'color': torch.cat([color_fg, color_bg], dim=1),
        }

        return out_dict


def render(args, world_view_transform,
                 full_proj_transform, 
                 camera_center, 
                 pc : dict,
                 bg_color=torch.tensor([0,0,0], dtype=torch.float32, device='cuda'),
                 scaling_modifier = 1.0, 
                 override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc['xyz'], dtype=pc['xyz'].dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(2*math.atan(1/2/args.focal_ratioX))
    tanfovy = math.tan(2*math.atan(1/2/args.focal_ratioY))

    raster_settings = GaussianRasterizationSettings(
        image_height=args.supervision_size,
        image_width=args.supervision_size,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc['xyz']
    means2D = screenspace_points
    opacity = pc['opacity']

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    scales = pc["scaling"]
    rotations = pc["rotation"]

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    assert override_color is not None
    colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

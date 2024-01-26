import copy
import os
import torchvision.transforms.functional as TF

from data.base_dataset import BaseDataset
from PIL import Image
import torch
import glob
import numpy as np
import random
import torchvision
import pickle
import cv2

def gen_cam2world(cam_pos, origin):
	'''
	cam_pos: 3D position of the camera
	origin: 3D position of the origin
	'''
	x, y, z = cam_pos

	# Calculte the cam2world matrix for the camera positioned at (x, y, z), pointing at (origin[0], origin[1], origin[2])
	forward_direction = torch.tensor([origin[0]-x, origin[1]-y, origin[2]-z], dtype=torch.float32)
	forward_direction = forward_direction / torch.norm(forward_direction)

	# world up vector is (0, 1, 0)
	# up_direction = torch.tensor([x*z/(x**2 + y**2), y*z/(x**2 + y**2), -1], dtype=torch.float32)
	up_direction = torch.tensor([(x-origin[0])*(z-origin[2])/((x-origin[0])**2 + (y-origin[1])**2), (y-origin[1])*(z-origin[2])/((x-origin[0])**2 + (y-origin[1])**2), -1], dtype=torch.float32)
	up_direction = up_direction / torch.norm(up_direction)
	right_direction = torch.cross(up_direction, forward_direction)

	# Construct the cam2world matrix
	rotation_matrix = torch.stack([right_direction, up_direction, forward_direction], dim=1)
	# flip the y axis (we use left hand coordinate system)
	# rotation_matrix[:, 1] = -rotation_matrix[:, 1]
	translation_vector = torch.tensor([x, y, z], dtype=torch.float32).view(3, 1)
	cam2world_matrix = torch.cat([rotation_matrix, translation_vector], dim=1)
	cam2world_matrix = torch.cat([cam2world_matrix, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], dim=0) # [4, 4]

	return cam2world_matrix

class MultiscenesSingleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.add_argument('--start_scene_idx', type=int, default=0, help='start scene index')
        parser.add_argument('--n_scenes', type=int, default=1000, help='dataset length is #scenes')
        parser.add_argument('--n_img_each_scene', type=int, default=10, help='for each scene, how many images to load in a batch')
        parser.add_argument('--no_shuffle', action='store_true')
        parser.add_argument('--transparent', action='store_true')
        parser.add_argument('--bg_color', type=float, default=-1, help='background color')
        parser.add_argument('--encoder_size', type=int, default=256, help='encoder size')
        parser.add_argument('--camera_normalize', action='store_true', help='normalize the camera pose to (0, dist, 0)')
        parser.add_argument('--camera_normalize_mode', choices=['left', 'right'], default='right', 
                        help='which hand to normalize the camera pose to, right hand (0, dist, 0), or left hand (0, -dist, 0)')
        parser.add_argument('--fixed_dist', default=None, type=float, help='fixed distance when camera_normalize')
        parser.add_argument('--jitter_pose', action='store_true', help='jitter the camera pose')
        parser.add_argument('--jitter_strength', default=1, type=float, help='jitter strength') # elevation: np.pi/12 * strength, azimuth: np.pi/4 * strength

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.n_scenes = opt.n_scenes
        self.n_img_each_scene = opt.n_img_each_scene
        self.jitter_strength = opt.jitter_strength

        self.scenes = []
        for i in range(opt.start_scene_idx, opt.start_scene_idx + self.n_scenes):
            self.scenes.append([])

        for filename in sorted(glob.glob(os.path.join(opt.dataroot, '*_sc????_az??.png'))) + sorted(glob.glob(os.path.join(opt.dataroot, '*_sc????_az??_dist?.png'))):
            scene_idx = int(filename.split('/')[-1].split('_')[1][2:])
            if scene_idx >= opt.start_scene_idx and scene_idx < opt.start_scene_idx + self.n_scenes:
                self.scenes[scene_idx-opt.start_scene_idx].append(filename)
    
        for i in range(len(self.scenes)):
            self.scenes[i] = sorted(self.scenes[i])

        self.bg_color = opt.bg_color
        # assert self.opt.isTrain
        assert self.opt.camera_normalize
        assert self.opt.fixed_dist is not None

    def _transform(self, img, size=None):
        size = self.opt.load_size if size is None else size
        img = img.resize((size, size), Image.BILINEAR)
        # img = TF.resize(img, (size, size), interpolation=Image.BILINEAR)
        img = TF.to_tensor(img)
        img = TF.normalize(img, [0.5] * img.shape[0], [0.5] * img.shape[0])  # [0,1] -> [-1,1]
        return img

    def _transform_encoder(self, img, normalize=True):
        img = img.resize((self.opt.encoder_size, self.opt.encoder_size), Image.BILINEAR)
        # img = TF.resize(img, (self.opt.encoder_size, self.opt.encoder_size), interpolation=Image.BILINEAR)
        img = TF.to_tensor(img)
        if normalize:
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    def _transform_mask(self, img, normalize=True):
        img = TF.resize(img, (self.opt.load_size, self.opt.load_size), Image.NEAREST)
        img = TF.to_tensor(img)
        if normalize:
            img = TF.normalize(img, [0.5] * img.shape[0], [0.5] * img.shape[0])  # [0,1] -> [-1,1]
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing, here it is scene_idx
        """
        scene_idx = index
        scene_filenames = self.scenes[scene_idx]

        rets = []
        input_rot = torch.zeros((4,4))
        
        path = scene_filenames[0]
        img = Image.open(path).convert('RGB')
        img_data = self._transform(img)

        pose_path = path.replace('.png', '_RT.txt')
        input_pose = torch.tensor(np.loadtxt(pose_path), dtype=torch.float32)
        cam2world = torch.Tensor([[-1, 0, 0, 0],
                                [0, 0, -1, self.opt.fixed_dist],
                                [0, -1, 0, 0],
                                [0, 0, 0, 1]]) if self.opt.camera_normalize_mode == 'right' else \
                    torch.Tensor([[1, 0, 0, 0],
                                [0, 0, -1, -self.opt.fixed_dist],
                                [0, 1, 0, 0],
                                [0, 0, 0, 1]])
        input_rot = torch.matmul(cam2world, torch.inverse(input_pose))
        pose = copy.deepcopy(cam2world)
    
        # support two types of depth maps
        if (self.opt.isTrain and self.opt.depth_supervision) \
                    or (not self.opt.isTrain and self.opt.vis_disparity):
            depth_path_pfm = path.replace('.png', '_depth.pfm')
            depth_path_png = path.replace('.png', '_depth.png')
            if os.path.isfile(depth_path_pfm):
                depth = cv2.imread(depth_path_pfm, -1)
                depth = cv2.resize(depth, (self.opt.load_size, self.opt.load_size), interpolation=Image.BILINEAR).astype(np.float32)
                depth = torch.from_numpy(depth).unsqueeze(0)  # 1xHxW
            elif os.path.isfile(depth_path_png):
                depth = Image.open(depth_path_png)
                depth.resize((self.opt.load_size, self.opt.load_size), Image.BILINEAR)
                depth = np.array(depth).astype(np.float32)
                depth = torch.from_numpy(depth).unsqueeze(0)  # 1xHxW
            else:
                assert False
            ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'depth': depth}
        else:
            ret = {'img_data': img_data, 'path': path, 'cam2world': pose,}

        normalize = False if self.opt.encoder_type == 'SD' else True
        ret['img_data_large'] = self._transform_encoder(img, normalize=normalize)
            
        if os.path.isfile(path.replace('.png', '_intrinsics.txt')):
            intrinsics_path = path.replace('.png', '_intrinsics.txt')
            intrinsics = np.loadtxt(intrinsics_path)
            intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
            intrinsics[0, 2] = intrinsics[1, 2] = 0 # remove translation
            ret['intrinsics'] = intrinsics

        rets.append(ret)

        if self.opt.isTrain:
            if not self.opt.jitter_pose: # define a distribution for sampling, first sample the first view, then nearby views have higher probability
                first_view_idx = random.randint(0, len(scene_filenames)-1)
                sigma = 0.1
                idx = np.minimum(np.abs(np.arange(0, len(scene_filenames)) - first_view_idx), 1 - np.abs(np.arange(0, len(scene_filenames)) - first_view_idx)) / len(scene_filenames)
                p = np.exp(-idx**2 / (2 * sigma**2))
                p[first_view_idx] = 0
                p = p / np.sum(p)
                filenames = list(np.random.choice(scene_filenames, self.n_img_each_scene - 1, replace=False, p=p))

                for path in filenames:
                    img = Image.open(path).convert('RGB')
                    img_data = self._transform(img)
                    pose_path = path.replace('.png', '_RT.txt')
                    pose = np.loadtxt(pose_path)
                    pose = torch.tensor(pose, dtype=torch.float32)
                    cam2world = torch.matmul(input_rot, pose)
                    ret = {'img_data': img_data, 'path': path, 'cam2world': cam2world,}
                    rets.append(ret)

            else:
                radius = torch.sqrt(torch.sum(input_pose[:3, 3] ** 2))
                x, y = input_pose[0, 3], input_pose[1, 3]
                elevation = torch.asin(input_pose[2, 3] / radius)
                azimuth = torch.atan2(y, x)
                elevation_jitter = np.pi/6 * self.jitter_strength * (random.random() - 0.5) * 2 + elevation
                azimuth_jitter = np.pi * self.jitter_strength * (random.random() - 0.5) * 2 + azimuth
                x_, y_, z_ = radius * torch.cos(elevation_jitter) * torch.cos(azimuth_jitter), radius * torch.cos(elevation_jitter) * torch.sin(azimuth_jitter), radius * torch.sin(elevation_jitter)
                pose_jitter = gen_cam2world((x_, y_, z_), (0, 0, 0))
                cam2world_jitter = torch.matmul(input_rot, pose_jitter)
                ret = {'cam2world': cam2world_jitter,}
                rets.append(ret)

        return rets

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_scenes

    def set_epoch(self, epoch):
        self.jitter_strength = self.opt.jitter_strength * min(epoch / self.opt.coarse_epoch, 1) 

def collate_fn(batch):
    # "batch" is a list (len=batch_size) of list (len=n_img_each_scene) of dict
    flat_batch = [item for sublist in batch for item in sublist]
    img_data = torch.stack([x['img_data'] for x in flat_batch if 'img_data' in x])
    paths = [x['path'] for x in flat_batch if 'path' in x]
    cam2world = torch.stack([x['cam2world'] for x in flat_batch])
    ret = {
        'img_data': img_data,
        'paths': paths,
        'cam2world': cam2world,
    }
    if 'depth' in flat_batch[0]:
        ret['depth'] = torch.stack([x['depth'] for x in flat_batch if 'depth' in x])

    if 'img_data_large' in flat_batch[0]:
        ret['img_data_large'] = torch.stack([x['img_data_large'] for x in flat_batch if 'img_data_large' in x]) # 1x3xHxW

    if 'intrinsics' in flat_batch[0]:
        ret['intrinsics'] = torch.stack([x['intrinsics'] for x in flat_batch if 'intrinsics' in x])


    return ret
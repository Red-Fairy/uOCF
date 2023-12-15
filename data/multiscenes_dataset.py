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


class MultiscenesDataset(BaseDataset):
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
        parser.add_argument('--diff_intrinsic', action='store_true', help='different intrinsics for each view')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.n_scenes = opt.n_scenes
        self.n_img_each_scene = opt.n_img_each_scene
        # image_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*.png')))  # root/00000_sc000_az00_el00.png
        # mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask.png')))
        # fg_mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask_for_moving.png')))
        # moved_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_moved.png')))
        # bg_mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask_for_bg.png')))
        # bg_in_mask_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_mask_for_providing_bg.png')))
        # changed_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_changed.png')))
        # bg_in_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_providing_bg.png')))
        # depth_filenames = sorted(glob.glob(os.path.join(opt.dataroot, '*_depth.png')))
        # changed_filenames_set, bg_in_filenames_set = set(changed_filenames), set(bg_in_filenames)
        # bg_mask_filenames_set, bg_in_mask_filenames_set = set(bg_mask_filenames), set(bg_in_mask_filenames)
        # image_filenames_set, mask_filenames_set = set(image_filenames), set(mask_filenames)
        # fg_mask_filenames_set, moved_filenames_set = set(fg_mask_filenames), set(moved_filenames)
        # filenames_set = image_filenames_set - mask_filenames_set - fg_mask_filenames_set - moved_filenames_set - changed_filenames_set - bg_in_filenames_set - bg_mask_filenames_set - bg_in_mask_filenames_set - set(depth_filenames)
        # filenames = sorted(list(filenames_set))

        # for i in range(opt.start_scene_idx, opt.start_scene_idx + self.n_scenes):
        #     scene_filenames = [x for x in filenames if 'sc{:04d}'.format(i) in x]
        #     self.scenes.append(scene_filenames)

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

        if self.opt.isTrain:
            if self.opt.no_shuffle:
                filenames = scene_filenames[:self.n_img_each_scene]
            else:
                filenames = random.sample(scene_filenames, self.n_img_each_scene)
        else:
            if self.opt.video:
                filenames = scene_filenames[self.opt.visual_idx:self.opt.visual_idx + 1]
            else:
                filenames = scene_filenames[:self.n_img_each_scene]

        rets = []
        input_rot = torch.zeros((4,4))
        for rd, path in enumerate(filenames):
            img = Image.open(path).convert('RGB')
            img_data = self._transform(img)
            pose_path = path.replace('.png', '_RT.txt')
            try:
                pose = np.loadtxt(pose_path)
            except FileNotFoundError:
                print('filenotfound error: {}'.format(pose_path))
                assert False
            pose = torch.tensor(pose, dtype=torch.float32)
            azi_path = pose_path.replace('_RT.txt', '_azi_rot.txt')
            if self.opt.fixed_locality:
                azi_rot = np.eye(3)  # not used; placeholder
            else:
                azi_rot = np.loadtxt(azi_path)
            azi_rot = torch.tensor(azi_rot, dtype=torch.float32)
            if self.opt.camera_normalize:
                if rd == 0: # rotate the camera to (0, dist, 0), z points to the center of the scene, y down, x right (opencv/COLMAP convention)
                    cam2world = torch.zeros((4, 4))
                    cam2world[0, 0] = -1
                    cam2world[1, 2] = -1
                    cam2world[2, 1] = -1
                    cam2world[3, 3] = 1
                    cam2world[1, 3] = torch.norm(pose[:3, 3])
                    input_rot = torch.matmul(cam2world, torch.inverse(pose))
                    pose = copy.deepcopy(cam2world)
                else:
                    pose = torch.matmul(input_rot, pose)
            
            # support two types of depth maps
            if (self.opt.isTrain and (self.opt.depth_supervision or self.opt.scaled_depth_map)) \
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
                    ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot}
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot, 'depth': depth}
            else:
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'azi_rot': azi_rot}

            if (rd == 0 or (self.opt.isTrain and self.opt.position_loss)) and self.opt.encoder_type != 'CNN':
                normalize = False if self.opt.encoder_type == 'SD' else True
                ret['img_data_large'] = self._transform_encoder(img, normalize=normalize)
            if os.path.isfile(path.replace('.png', '_intrinsics.txt')):
                intrinsics_path = path.replace('.png', '_intrinsics.txt')
                intrinsics = np.loadtxt(intrinsics_path)
                intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
                ret['intrinsics'] = intrinsics
            mask_path = path.replace('.png', '_mask.png')
            if os.path.isfile(mask_path):
                mask = Image.open(mask_path).convert('RGB')
                mask_l = mask.convert('L')
                mask = self._transform_mask(mask)
                ret['mask'] = mask
                mask_l = self._transform_mask(mask_l)
                mask_flat = mask_l.flatten(start_dim=0)  # HW,
                greyscale_dict = mask_flat.unique(sorted=True)  # 8,
                onehot_labels = mask_flat[:, None] == greyscale_dict  # HWx8, one-hot
                onehot_labels = onehot_labels.type(torch.uint8)
                mask_idx = onehot_labels.argmax(dim=1)  # HW
                bg_color_idx = torch.argmin(torch.abs(greyscale_dict - self.bg_color))
                bg_color = greyscale_dict[bg_color_idx]
                fg_idx = mask_flat != bg_color  # HW
                ret['mask_idx'] = mask_idx
                ret['fg_idx'] = fg_idx
                obj_idxs = []
                obj_idxs_test = []
                for i in range(len(greyscale_dict)):
                    if i == bg_color_idx and self.opt.isTrain:
                        bg_mask = mask_l == greyscale_dict[i]  # 1xHxW
                        ret['bg_mask'] = bg_mask
                        continue
                    obj_idx = mask_l == greyscale_dict[i]  # 1xHxW
                    obj_idxs.append(obj_idx)
                    if (not self.opt.isTrain) and i != bg_color_idx:
                        obj_idxs_test.append(obj_idx)
                obj_idxs = torch.stack(obj_idxs)  # Kx1xHxW
                ret['obj_idxs'] = obj_idxs  # Kx1xHxW
                if not self.opt.isTrain:
                    obj_idxs_test = torch.stack(obj_idxs_test)  # Kx1xHxW
                    ret['obj_idxs_fg'] = obj_idxs_test  # Kx1xHxW
            
            # pseudo mask from SAMM
            pseudo_mask_path = path.replace('.png', '_mask.png').replance('img', 'mask')
            if os.path.isfile(pseudo_mask_path):
                mask = Image.open(pseudo_mask_path).convert('L')
                mask = self._transform_mask(mask, normalize=False)
                ret['pseudo_mask'] = mask

            rets.append(ret)
        return rets

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_scenes


def collate_fn(batch):
    # "batch" is a list (len=batch_size) of list (len=n_img_each_scene) of dict
    flat_batch = [item for sublist in batch for item in sublist]
    img_data = torch.stack([x['img_data'] for x in flat_batch])
    paths = [x['path'] for x in flat_batch]
    cam2world = torch.stack([x['cam2world'] for x in flat_batch])
    azi_rot = torch.stack([x['azi_rot'] for x in flat_batch])
    if 'depth' in flat_batch[0]:
        depths = torch.stack([x['depth'] for x in flat_batch])  # Bx1xHxW
    else:
        depths = None
    ret = {
        'img_data': img_data,
        'paths': paths,
        'cam2world': cam2world,
        'azi_rot': azi_rot,
        'depth': depths,
    }
    if 'img_data_large' in flat_batch[0]:
        ret['img_data_large'] = torch.stack([x['img_data_large'] for x in flat_batch if 'img_data_large' in x]) # 1x3xHxW

    if 'intrinsics' in flat_batch[0]:
        ret['intrinsics'] = torch.stack([x['intrinsics'] for x in flat_batch if 'intrinsics' in x])

    if 'mask' in flat_batch[0]:
        masks = torch.stack([x['mask'] for x in flat_batch])
        ret['masks'] = masks
        mask_idx = torch.stack([x['mask_idx'] for x in flat_batch])
        ret['mask_idx'] = mask_idx
        fg_idx = torch.stack([x['fg_idx'] for x in flat_batch])
        ret['fg_idx'] = fg_idx
        obj_idxs = flat_batch[0]['obj_idxs']  # Kx1xHxW
        ret['obj_idxs'] = obj_idxs
        if 'bg_mask' in flat_batch[0]:
            bg_mask = torch.stack([x['bg_mask'] for x in flat_batch])
            ret['bg_mask'] = bg_mask # Bx1xHxW
        if 'obj_idxs_fg' in flat_batch[0]:
            ret['obj_idxs_fg'] = flat_batch[0]['obj_idxs_fg']
    
    if 'pseudo_mask' in flat_batch[0]:
        pseudo_masks = torch.stack([x['pseudo_mask'] for x in flat_batch])
        ret['pseudo_masks'] = pseudo_masks # Nx1xHxW

    return ret
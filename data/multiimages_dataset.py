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


class MultiimagesDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(input_nc=3, output_nc=3)
        parser.add_argument('--start_image_idx', type=int, default=0, help='start image index')
        parser.add_argument('--n_images', type=int, default=1000, help='number of images to load')
        parser.add_argument('--n_images_per_epoch', type=int, default=None, help='dataset length is #images')
        parser.add_argument('--n_img_each_scene', type=int, default=60, help='for each image, how many images to load in a batch')
        parser.add_argument('--no_shuffle', action='store_true')
        parser.add_argument('--transparent', action='store_true')
        parser.add_argument('--bg_color', type=float, default=-1, help='background color')
        parser.add_argument('--encoder_size', type=int, default=256, help='encoder size')
        parser.add_argument('--camera_normalize', action='store_true', help='normalize the camera pose to (0, dist, 0)')
        parser.add_argument('--fixed_dist', default=None, type=float, help='fixed distance when camera_normalize')
        parser.add_argument('--n_pseudo_view', type=int, default=0, help='number of pseudo views')
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.n_images = opt.n_images
        self.n_images_per_epoch = opt.n_images_per_epoch

        self.images = []
        for i in range(opt.start_image_idx, opt.start_image_idx + self.n_images):
            self.images.append([])

        for filename in os.listdir(opt.dataroot):
            if not filename.endswith('.png') or 'depth' in filename or 'mask' in filename or 'intrinsics' in filename:
                continue
            image_idx = int(filename.split('/')[-1].split('_')[0])
            if image_idx >= opt.start_image_idx and image_idx < opt.start_image_idx + self.n_images:
                self.images[image_idx-opt.start_image_idx].append(os.path.join(opt.dataroot, filename))
    
        for i in range(len(self.images)):
            # print('image %d: %d images' % (i, len(self.images[i])))
            assert len(self.images[i]) == 1  # only support n_img_each_image == 1

        self.bg_color = opt.bg_color

        assert self.opt.encoder_type == 'DINO'
        # assert self.opt.isTrain  # only support training
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
            index - - a random integer for data indexing, here it is image_idx
        """
        image_idx = index if self.n_images_per_epoch is None else random.randint(0, self.n_images - 1)
        filenames = self.images[image_idx]

        rets = []

        path = filenames[0]

        img = Image.open(path).convert('RGB')
        img_data = self._transform(img)

        pose = torch.zeros((4, 4))
        pose[0, 0] = -1
        pose[1, 2] = -1
        pose[2, 1] = -1
        pose[3, 3] = 1
        pose[1, 3] = self.opt.fixed_dist

        # support two types of depth maps
        if (self.opt.isTrain and self.opt.depth_supervision) \
                    or (not self.opt.isTrain and self.opt.vis_disparity):
            depth_path_pfm = path.replace('.png', '_depth.pfm').replace('img60', 'depth60')
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
                ret = {'img_data': img_data, 'path': path, 'cam2world': pose}
            ret = {'img_data': img_data, 'path': path, 'cam2world': pose, 'depth': depth}
        else:
            ret = {'img_data': img_data, 'path': path, 'cam2world': pose}

        ret['img_data_large'] = self._transform_encoder(img, normalize=True)
        if os.path.isfile(path.replace('.png', '_intrinsics.txt')):
            intrinsics_path = path.replace('.png', '_intrinsics.txt')
            intrinsics = np.loadtxt(intrinsics_path)
            intrinsics = torch.tensor(intrinsics, dtype=torch.float32)
            intrinsics[0, 2] = intrinsics[1, 2] = 0 # remove translation
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
        
        # # pseudo mask from SAM
        # pseudo_mask_path = path.replace('.png', '_mask.png').replace('img', 'mask')
        # if os.path.isfile(pseudo_mask_path) and self.opt.isTrain and (self.opt.pseudo_mask_loss or self.opt.vis_mask):
        #     mask = Image.open(pseudo_mask_path).convert('L')
        #     mask = self._transform_mask(mask, normalize=False)
        #     ret['pseudo_mask'] = mask

        rets.append(ret)


        return rets

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_images_per_epoch if self.n_images_per_epoch is not None else self.n_images
    
    def set_epoch(self, epoch):
        pass


def collate_fn(batch):
    # "batch" is a list (len=batch_size) of list (len=n_img_each_image) of dict
    flat_batch = [item for sublist in batch for item in sublist]
    img_data = torch.stack([x['img_data'] for x in flat_batch if 'img_data' in x])
    paths = [x['path'] for x in flat_batch]
    cam2world = torch.stack([x['cam2world'] for x in flat_batch if 'cam2world' in x])
    if 'depth' in flat_batch[0]:
        depths = torch.stack([x['depth'] for x in flat_batch])  # Bx1xHxW
    else:
        depths = None
    ret = {
        'img_data': img_data,
        'paths': paths,
        'cam2world': cam2world,
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
    
    # if 'pseudo_mask' in flat_batch[0]:
    #     pseudo_masks = torch.stack([x['pseudo_mask'] for x in flat_batch])
    #     ret['pseudo_mask'] = pseudo_masks # Nx1xHxW

    return ret
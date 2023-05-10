import torch
import torchvision.transforms.functional as TF
import glob
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from segment_anything import sam_model_registry
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_scene', type=int, default=0)
parser.add_argument('--n_scenes', type=int, default=1000)

args = parser.parse_args()

class MultiscenesDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, start_scene=0, n_scenes=5000, input_size=14*64):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.input_size = input_size
        self.scenes = []
        image_filenames = sorted(glob.glob(os.path.join(dataroot, '*.png')))  # root/00000_sc000_az00_el00.png
        mask_filenames = sorted(glob.glob(os.path.join(dataroot, '*_mask.png')))
        fg_mask_filenames = sorted(glob.glob(os.path.join(dataroot, '*_mask_for_moving.png')))
        moved_filenames = sorted(glob.glob(os.path.join(dataroot, '*_moved.png')))
        bg_mask_filenames = sorted(glob.glob(os.path.join(dataroot, '*_mask_for_bg.png')))
        bg_in_mask_filenames = sorted(glob.glob(os.path.join(dataroot, '*_mask_for_providing_bg.png')))
        changed_filenames = sorted(glob.glob(os.path.join(dataroot, '*_changed.png')))
        bg_in_filenames = sorted(glob.glob(os.path.join(dataroot, '*_providing_bg.png')))
        changed_filenames_set, bg_in_filenames_set = set(changed_filenames), set(bg_in_filenames)
        bg_mask_filenames_set, bg_in_mask_filenames_set = set(bg_mask_filenames), set(bg_in_mask_filenames)
        image_filenames_set, mask_filenames_set = set(image_filenames), set(mask_filenames)
        fg_mask_filenames_set, moved_filenames_set = set(fg_mask_filenames), set(moved_filenames)
        filenames_set = image_filenames_set - mask_filenames_set - fg_mask_filenames_set - moved_filenames_set - changed_filenames_set - bg_in_filenames_set - bg_mask_filenames_set - bg_in_mask_filenames_set
        filenames = sorted(list(filenames_set))
        self.start_scene = start_scene
        self.n_scenes = n_scenes
        self.n_img_each_scene = 4
        for i in range(self.start_scene, self.n_scenes+self.start_scene):
            scene_filenames = [x for x in filenames if 'sc{:04d}'.format(i) in x]
            self.scenes.append(scene_filenames)

    def _transform_encoder(self, img): # for ImageNet encoder
        img = TF.resize(img, (self.input_size, self.input_size))
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing, here it is scene_idx
        """
        scene_idx = index
        scene_filenames = self.scenes[scene_idx]
        filenames = scene_filenames[:self.n_img_each_scene]
        rets = []
        for rd, path in enumerate(filenames):
            img = Image.open(path).convert('RGB')
            img_data = self._transform_encoder(img)
            rets.append((img_data, path))
        paths = [x[1] for x in rets]
        imgs = torch.stack([x[0] for x in rets])
        return imgs, paths
            
    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.n_scenes

sam_model = sam_model_registry['vit_l'](checkpoint='/viscam/u/redfairy/pretrained_weights/SAM/sam_vit_l_0b3195.pth')
sam_encoder = sam_model.image_encoder.cuda().eval()

feat_size = 64
input_size = 16*feat_size
dataset = MultiscenesDataset('/viscam/projects/uorf-extension/datasets/room_diverse_nobg/train-1obj-manysize-trans-orange', input_size=input_size, start_scene=args.start_scene, n_scenes=args.n_scenes)
print("Dataset size: ", len(dataset))

out_channel = 1024
for imgs, paths in tqdm(dataset):
    imgs = imgs.cuda()
    with torch.no_grad():
        feats = sam_encoder(imgs)  # Bx256x64x64
        feats = feats.cpu().numpy()
    # save features
    for rd, path in enumerate(paths):
        path = path.replace('.png', '-SAM.npy')
        np.save(path, feats[rd])
        # print(feats[rd].shape, path)

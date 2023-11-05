import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from torch.utils.data import DataLoader

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_size=1e5):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname) and 'mask' not in fname:
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_size, len(images))]

class SingleImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, max_dataset_size, load_size=None, large_size=None):
        self.root = root
        self.load_size = load_size
        self.large_size = large_size
        self.paths = make_dataset(root, max_dataset_size)

    def __getitem__(self, index):

        img = Image.open(self.paths[index]).convert('RGB')
        ret = {}

        if self.large_size:
            img_large = img.resize((self.large_size, self.large_size), Image.BILINEAR)
            img_large = transforms.ToTensor()(img_large)
            img_large = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])(img_large)
            ret['x_large'] = img_large
    
        
        if self.load_size:
            # normalize to [-1, 1]
            img = img.resize((self.load_size, self.load_size), Image.BILINEAR)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])(img)
            ret['x'] = img

        return ret

    def __len__(self):
        return len(self.paths)
    
def collate_fn(batch):
    ret = {}
    if 'x_large' in batch[0]:
        ret['x_large'] = torch.stack([b['x_large'] for b in batch])
    if 'x' in batch[0]:
        ret['x'] = torch.stack([b['x'] for b in batch])
    return ret

def create_dataloader(root, batch_size, load_size=128, large_size=None, num_workers=4, max_dataset_size=1e5):
    dataset = SingleImageDataset(root, max_dataset_size=max_dataset_size, load_size=load_size, large_size=large_size)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers,
                                             collate_fn=collate_fn)
    return dataloader
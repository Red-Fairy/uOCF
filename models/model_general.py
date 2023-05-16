import math
from .op import conv2d_gradfix
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models import vgg16
from torch import autograd

from models.resnet import resnet34, resnet18
from .utils import PositionalEncoding, sin_emb

class Encoder(nn.Module):
    def __init__(self, input_nc=3, z_dim=64, bottom=False, pos_emb=False):

        super().__init__()

        self.bottom = bottom

        input_nc = input_nc + 4 if pos_emb else input_nc
        self.pos_emb = pos_emb

        if self.bottom:
            self.enc_down_0 = nn.Sequential(nn.Conv2d(input_nc, z_dim, 3, stride=1, padding=1),
                                            nn.ReLU(True))
        self.enc_down_1 = nn.Sequential(nn.Conv2d(z_dim if bottom else input_nc, z_dim, 3, stride=2 if bottom else 1, padding=1),
                                        nn.ReLU(True))
        self.enc_down_2 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_down_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=2, padding=1),
                                        nn.ReLU(True))
        self.enc_up_3 = nn.Sequential(nn.Conv2d(z_dim, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_2 = nn.Sequential(nn.Conv2d(z_dim*2, z_dim, 3, stride=1, padding=1),
                                      nn.ReLU(True),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.enc_up_1 = nn.Sequential(nn.Conv2d(z_dim * 2, z_dim, 3, stride=1, padding=1),
                                    #   nn.ReLU(True)
                                      )
        
        

    def forward(self, x):
        """
        input:
            x: input image, Bx3xHxW
        output:
            feature_map: BxCxHxW
        """

        if self.pos_emb:
            W, H = x.shape[3], x.shape[2]
            X = torch.linspace(-1, 1, W)
            Y = torch.linspace(-1, 1, H)
            y1_m, x1_m = torch.meshgrid([Y, X])
            x2_m, y2_m = -x1_m, -y1_m  # Normalized distance in the four direction
            pixel_emb = torch.stack([x1_m, x2_m, y1_m, y2_m]).to(x.device).unsqueeze(0)  # 1x4xHxW
            x_ = torch.cat([x, pixel_emb], dim=1)
        else:
            x_ = x

        if self.bottom:
            x_down_0 = self.enc_down_0(x_)
            x_down_1 = self.enc_down_1(x_down_0)
        else:
            x_down_1 = self.enc_down_1(x_)
        x_down_2 = self.enc_down_2(x_down_1)
        x_down_3 = self.enc_down_3(x_down_2)
        x_up_3 = self.enc_up_3(x_down_3)
        x_up_2 = self.enc_up_2(torch.cat([x_up_3, x_down_2], dim=1))
        feature_map = self.enc_up_1(torch.cat([x_up_2, x_down_1], dim=1))  # BxCxHxW
        
        feature_map = feature_map
        return feature_map

class sam_encoder(nn.Module):
    def __init__(self, sam_model):
        super(sam_encoder, self).__init__()

        self.sam = sam_model.image_encoder
        self.sam.requires_grad_(False)
        self.sam.eval()

    def forward(self, x_sam):
        
        return self.sam(x_sam) # (B, 256, 64, 64)

class dualRouteEncoder(nn.Module):
    def __init__(self, bottom=False, pos_emb=False, input_nc=3, shape_dim=48, color_dim=16):
        super().__init__()

        self.Encoder = Encoder(bottom=bottom, z_dim=color_dim, pos_emb=pos_emb, input_nc=input_nc)

        vit_dim = 256
        self.shallow_encoder = nn.Sequential(nn.Conv2d(vit_dim, vit_dim, 3, stride=1, padding=1),
                                            nn.ReLU(True),
                                            nn.Conv2d(vit_dim, shape_dim, 3, stride=1, padding=1))

    def forward(self, sam_feature, x):
        '''
        input:
            sam_feature: (B, 256, 64, 64)
            x: input images of size (B, 3, 64, 64) or (B, 3, 128, 128) if bottom is True
        output:
            spatial feature (B, shape_dim+color_dim, 64, 64)
        '''
        feat_color = self.Encoder(x)
        feat_shape = self.shallow_encoder(sam_feature)

        return torch.cat([feat_shape, feat_color], dim=1)

class DinoEncoder(nn.Module):
    def __init__(self, dino_dim=768, z_dim=64, hidden_dim=128):
        super().__init__()

        self.shallow_encoder = nn.Sequential(nn.Conv2d(dino_dim, hidden_dim, 3, stride=1, padding=1),
                                            nn.ReLU(True),
                                            nn.Conv2d(hidden_dim, z_dim, 3, stride=1, padding=1))

    def forward(self, dino_feats):
        '''
        input:
            dino_feature: (B, dino_dim, 64, 64)
        output:
            spatial feature (B, z_dim, 64, 64)
        '''
        return self.shallow_encoder(dino_feats)

class SDEncoder(nn.Module):
    def __init__(self, z_dim=64):
        super().__init__()
        '''
        input: list of Tensors: B*512*32*32, B*640*32*32, B*512*64*64
        output: B*z_dim*64*64
        '''

        self.conv1 = nn.Sequential(nn.Conv2d(512, 128, 1, stride=1, padding=0),
                                   nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.conv2 = nn.Sequential(nn.Conv2d(640, 128, 1, stride=1, padding=0),
                                      nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.conv3 = nn.Conv2d(512, 128, 1, stride=1, padding=0)

        self.out = nn.Sequential(nn.Conv2d(128*3, 128, 3, stride=1, padding=1),
                                    nn.ReLU(True),
                                nn.Conv2d(128, z_dim, 3, stride=1, padding=1))

    def forward(self, x):
        x1 = self.conv1(x[0])
        x2 = self.conv2(x[1])
        x3 = self.conv3(x[2])
        x = torch.cat([x1, x2, x3], dim=1)
        return self.out(x)

                        
    
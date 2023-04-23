from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed, write_location, tensor2im

import torch
from util.util import get_spherical_cam2world
import torchvision
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image


opt = TestOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
model = create_model(opt)      # create a model given opt.model and other options
model.setup(opt)               # regular setup: load and print networks; create schedulers
visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
meters_tst = {stat: AverageMeter() for stat in model.loss_names}

set_seed(opt.seed)

web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
                        '{}_{}'.format(opt.testset_name, opt.epoch))  # define the website directory
print('creating web directory', web_dir)
webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

manipulation = True
# wanted idx
idx = 0
for id, data in enumerate(dataset):
    if id < idx:
        continue
    visualizer.reset()
    model.set_input(data)  # unpack data from data loader

    with torch.no_grad():
        model.forward()
        img_path = model.get_image_paths()
        # model.compute_visuals()
        # visuals = model.get_current_visuals()
        # visualizer.display_current_results(visuals, epoch=None, save_result=False)
        # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)

        if manipulation:
            offset1 = 0.4
            # offset2 = offset1
            num_slots = opt.num_slots if opt.n_objects_eval is None else opt.n_objects_eval
            fg_slot_position = torch.zeros((num_slots-1, 2))
            fg_slot_position[0] = torch.tensor([offset1, offset1])
            fg_slot_position[1] = torch.tensor([0, offset1])
            fg_slot_position[2] = torch.tensor([0, 0])
            fg_slot_position[3] = torch.tensor([0, offset1*2])
            fg_slot_position[4] = torch.tensor([-offset1, offset1])
            fg_slot_position[5] = torch.tensor([-offset1, offset1 * 2])
            fg_slot_position[6] = torch.tensor([offset1, offset1 * 2])
            fg_slot_position[7] = torch.tensor([offset1, 0])
            fg_slot_position[8] = torch.tensor([-offset1, 0])
            z_slots = torch.cat([model.z_slots[0:1], model.z_slots[1:2].repeat(num_slots-1, 1)], dim=0)
            z_slots_texture = torch.cat([model.z_slots_texture[0:1], model.z_slots_texture[1:2].repeat(num_slots-1, 1)], dim=0)
            model.forward_position(fg_slot_nss_position=fg_slot_position, z_slots=z_slots, z_slots_texture=z_slots_texture)

        cam2world_input = model.cam2world[0:1].cpu()
        # print(cam2world_input)
        radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
        z = (cam2world_input[:, 2, 3]) / radius
        theta = torch.acos(z)
        radius, theta = radius.item(), theta.item()

        cam2worlds = get_spherical_cam2world(radius, theta, 30)
        cam2worlds = torch.from_numpy(cam2worlds).float()

        # video writer in .avi format
        # video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
        video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))

        for i in tqdm(range(cam2worlds.shape[0])):
            cam2world = cam2worlds[i:i+1]
            # print(cam2world)
            model.visual_cam2world(cam2world)
            model.compute_visuals()
            visuals = model.get_current_visuals()
            save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size, suffix=f'_{i}')
            visual_name = 'x_rec0'
            img = tensor2im(visuals[visual_name])
            # img_pil = Image.fromarray(img)
            # img_pil.save(os.path.join(web_dir, 'images_debug' , 'rendered_{}.png'.format(i)))
            # write to video, transform to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img)

        video_writer.release()

        # congregate the images into a video for each slot
        # for j in tqdm(range(1, opt.num_slots)):
        #     video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered_slot{}.mp4'.format(j)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
        #     for i in range(cam2worlds.shape[0]):
        #         path = os.path.join(web_dir, 'images' , 'rendered_slot{}_{}.png'.format(j, i))
        #         img = cv2.imread(path)
        #         video_writer.write(img)
        #     video_writer.release()
                
            # visualizer.display_current_results(visuals, epoch=None, save_result=False)
            # img_path = model.get_image_paths()
            # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)
            # rendered = (model.visual_cam2world(cam2world) + 1) / 2
            # path = os.path.join(web_dir, 'images' ,'rendered_{}.png'.format(i))
            # torchvision.utils.save_image(rendered[0], path)
            # torchvision.utils.save_image(rendered, path)

        # congregate the images into a video
        # video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
        # for i in range(cam2worlds.shape[0]):
        #     path = os.path.join(web_dir, 'images' , 'rendered_{}.png'.format(i))
        #     img = cv2.imread(path)
        #     video_writer.write(img)
        # video_writer.release()


    # visuals = model.get_current_visuals()
    # visualizer.display_current_results(visuals, epoch=None, save_result=False)
    # img_path = model.get_image_paths()
    # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)
    # print('process image... %s' % img_path)
    break

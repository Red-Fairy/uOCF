from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed, write_location

import torch
from util.util import get_spherical_cam2world, tensor2im
import torchvision
import cv2
from tqdm import tqdm
import numpy as np


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

# wanted idx
idx = 0
for data in dataset:
    for id in range(idx):
        continue
    visualizer.reset()
    model.set_input(data)  # unpack data from data loader
    visual_names = [f'slot{i}_view0_unmasked' for i in range(1, opt.num_slots)] + ['x_rec0']

    with torch.no_grad():
        model.forward()
        model.compute_visuals()
        visuals = model.get_current_visuals()
        visualizer.display_current_results(visuals, epoch=None, save_result=False)
        img_path = model.get_image_paths()
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)

        # modify the position of the slots
        fg_slot_position = torch.zeros((opt.num_slots-1, 2))
        fg_slot_position[0] = torch.tensor([0, 0])
        fg_slot_position[1] = torch.tensor([0, 0])
        fg_slot_position[2] = torch.tensor([0, 0])
        fg_slot_position[3] = torch.tensor([0, 0])
        model.forward_position(fg_slot_nss_position=fg_slot_position)

        cam2world_input = model.cam2world[0:1].cpu()
        # print(cam2world_input)
        radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
        z = (cam2world_input[:, 2, 3]) / radius
        theta = torch.acos(z)
        radius, theta = radius.item(), theta.item()

        cam2worlds = get_spherical_cam2world(radius, theta, 48)
        cam2worlds = torch.from_numpy(cam2worlds).float()

        video_writers = []
        for j in range(1, opt.num_slots):
            video_writers.append(cv2.VideoWriter(os.path.join(web_dir, 'rendered_slot{}.mp4'.format(j)), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128)))
        video_writers.append(cv2.VideoWriter(os.path.join(web_dir, 'rendered_rec.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128)))

        for i in tqdm(range(cam2worlds.shape[0])):
            cam2world = cam2worlds[i:i+1]
            # print(cam2world)
            model.visual_cam2world(cam2world)
            model.compute_visuals()
            visuals = model.get_current_visuals()
            # print(len(visuals))
            for j, visual_name in enumerate(visual_names):
                # img = visuals[visual_name].detach().cpu()
                img = tensor2im(visuals[visual_name])
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video_writers[j].write(img)
                # path = os.path.join(web_dir, 'images' ,'rendered_slot{}_{}.png'.format(j, i))
                # torchvision.utils.save_image(img, path)

        for j in range(1, opt.num_slots):
            video_writers[j-1].release()

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

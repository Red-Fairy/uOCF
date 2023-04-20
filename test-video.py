from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed, write_location

import torch
from util.util import get_spherical_cam2world
import torchvision
import cv2


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

for i, data in enumerate(dataset):
    visualizer.reset()
    model.set_input(data)  # unpack data from data loader

    with torch.no_grad():
        model.forward()
        # model.compute_visuals()

        cam2world_input = model.cam2world[0:1].cpu()
        # print(cam2world_input)
        radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
        z = (cam2world_input[:, 2, 3]) / radius
        theta = torch.acos(z)
        radius, theta = radius.item(), theta.item()
        print(radius, theta)

        cam2worlds = get_spherical_cam2world(radius, theta, 64)
        cam2worlds = torch.from_numpy(cam2worlds).float()

        # for i in range(cam2worlds.shape[0]):
        #     cam2world = cam2worlds[i:i+1]
        #     # print(cam2world)
        #     rendered = (model.visual_cam2world(cam2world) + 1) / 2
        #     path = os.path.join(web_dir, 'images' ,'rendered_{}.png'.format(i))
        #     # torchvision.utils.save_image(rendered[0], path)
        #     torchvision.utils.save_image(rendered, path)

        # congregate the images into a video
        video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
        for i in range(cam2worlds.shape[0]):
            path = os.path.join(web_dir, 'images' , 'rendered_{}.png'.format(i))
            img = cv2.imread(path)
            video_writer.write(img)
        video_writer.release()


    # visuals = model.get_current_visuals()
    # visualizer.display_current_results(visuals, epoch=None, save_result=False)
    # img_path = model.get_image_paths()
    # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)
    # print('process image... %s' % img_path)

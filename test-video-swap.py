from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, get_spiral_cam2world, set_seed, write_location, tensor2im

import torch
from util.util import get_spherical_cam2world, parse_wanted_indice
import torchvision
import cv2
from tqdm import tqdm
import numpy as np
from PIL import Image
from glob import glob


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

manipulation = True

obj_to_swap = [(2, 3), (0, 1)]
suffix = f'swap_{"_".join([str(idx[0])+str(idx[1]) for idx in obj_to_swap])}'

n_frames = 15

wanted_indices = parse_wanted_indice(opt.wanted_indices)

for j, data in enumerate(dataset):

	if not wanted_indices is None and j not in wanted_indices:
		continue

	print('Visualizing scene No.: ', j)

	web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
							f'{opt.testset_name}/scene{j}_{opt.video_mode}{suffix}')  # define the website directory
	print('creating web directory', web_dir)
	webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

	visualizer.reset()
	model.set_input(data)  # unpack data from data loader
	model.set_visual_names(add_attn=True)

	with torch.no_grad():
		model.forward()
		img_path = model.get_image_paths()

		model.compute_visuals()
		visuals = model.get_current_visuals()
		# visualizer.display_current_results(visuals, epoch=None, save_result=False)
		save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)

		if manipulation:
			for i in range(len(obj_to_swap)):
				obj1, obj2 = obj_to_swap[i]
				pos1, pos2 = model.fg_slot_nss_position[obj1].clone(), model.fg_slot_nss_position[obj2].clone()
				model.fg_slot_nss_position[obj1], model.fg_slot_nss_position[obj2] = pos2, pos1
			model.forward_position()

		cam2world_input = model.cam2world[0:1].cpu()
		radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
		x, y = cam2world_input[:, 0, 3], cam2world_input[:, 1, 3]
		radius_xy, angle_xy = torch.sqrt(x ** 2 + y ** 2).item(), torch.atan2(y, x).item()
		theta = torch.acos((cam2world_input[:, 2, 3]) / radius)
		radius, theta, z = radius.item(), theta.item(), cam2world_input[:, 2, 3].item()

		if opt.video_mode == 'spherical':
			cam2worlds = get_spherical_cam2world(radius, theta, 45)
		elif opt.video_mode == 'spiral':
			cam2worlds = get_spiral_cam2world(radius_xy, z, (angle_xy, angle_xy + np.pi / 4), 60, height_range=(0.95, 1.15))
		else:
			assert False

		for i in tqdm(range(cam2worlds.shape[0])):
			cam2world = cam2worlds[i:i+1]
			# print(cam2world)
			model.visual_cam2world(cam2world)
			# model.visual_names = list(filter(lambda x: 'input' not in x and 'attn' not in x, model.visual_names))
			model.visual_names = list(filter(lambda x: 'rec' in x, model.visual_names))
			# model.compute_visuals(cam2world=cam2world)
			visuals = model.get_current_visuals()
			save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size, suffix=f'_{i:03d}')





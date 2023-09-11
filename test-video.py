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

manipulation = False

remove_obj_idx = [0, 3]
# suffix = f'_remove_obj_{"_".join([str(idx) for idx in remove_obj_idx])}'
suffix = ''

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
	# model.set_visual_names()

	with torch.no_grad():
		model.forward()
		img_path = model.get_image_paths()

		if manipulation:
			num_slots = opt.num_slots if opt.n_objects_eval is None else opt.n_objects_eval
			for idx in remove_obj_idx:
				model.fg_slot_nss_position[idx] = torch.tensor([100, 100, 0]).to(model.device)
			model.forward_position()

		model.compute_visuals()
		visuals = model.get_current_visuals()
		save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)

		cam2world_input = model.cam2world[0:1].cpu()
		radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
		x, y = cam2world_input[:, 0, 3], cam2world_input[:, 1, 3]
		radius_xy, angle_xy = torch.sqrt(x ** 2 + y ** 2).item(), torch.atan2(y, x).item()
		theta = torch.acos((cam2world_input[:, 2, 3]) / radius)
		radius, theta, z = radius.item(), theta.item(), cam2world_input[:, 2, 3].item()

		if opt.video_mode == 'spherical':
			cam2worlds = get_spherical_cam2world(radius, theta, 45)
		elif opt.video_mode == 'spiral':
			cam2worlds = get_spiral_cam2world(radius_xy, z, (angle_xy, angle_xy + np.pi / 4), 30, height_range=(0.9, 1.1), radius_range=(0.6, 0.8), origin=(0, -1.5))
			# cam2worlds = get_spiral_cam2world(radius_xy, z, (angle_xy - np.pi / 12, angle_xy + np.pi / 4), 20)
		else:
			assert False

		# cam2worlds = torch.from_numpy(cam2worlds).float()

		# video writer in .avi format
		# video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
		
		for i in tqdm(range(cam2worlds.shape[0])):
			cam2world = cam2worlds[i:i+1]
			# print(cam2world)
			model.visual_cam2world(cam2world)
			# model.visual_names = list(filter(lambda x: 'input' not in x and 'attn' not in x, model.visual_names))
			model.visual_names = list(filter(lambda x: 'rec' in x, model.visual_names))
			# model.compute_visuals(cam2world=cam2world)
			visuals = model.get_current_visuals()
			save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size, suffix=f'_{i:03d}')

		resolution = (opt.frustum_size, opt.frustum_size)

		video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 60, resolution)
		visual_image_paths = list(filter(lambda x: 'rec0' in x and 'disparity' not in x, glob(os.path.join(web_dir, 'images', '*.png'))))
		visual_image_paths.sort()
		for visual_image_path in visual_image_paths:
			img = cv2.imread(visual_image_path)
			video_writer.write(img)

		# render disparity map if opt.vis_disparity is True
		if opt.vis_disparity:
			visual_image_paths = list(filter(lambda x: 'disparity_rec0' in x, glob(os.path.join(web_dir, 'images', '*.png'))))
			visual_image_paths.sort()
			for visual_image_path in visual_image_paths:
				img = cv2.imread(visual_image_path)
				video_writer.write(img)

		video_writer.release()

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, get_spiral_cam2world, set_seed, write_location, tensor2im

import torch
from util.util import get_spherical_cam2world
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

spherical = False # False for spiral
suffix = 'spherical' if spherical else 'spiral'

wanted_indices = [37, 42]

for j, data in enumerate(dataset):

	if j not in wanted_indices:
		continue

	print('Visualizing scene No.: ', j)

	web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
							f'{opt.testset_name}/scene{j}_{suffix}')  # define the website directory
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
			offset1 = 0.4
			# offset2 = offset1
			num_slots = opt.num_slots if opt.n_objects_eval is None else opt.n_objects_eval
			inferred_nss_position = model.fg_slot_nss_position[0:1].cpu()
			inferred_image_position = model.fg_slot_image_position[0:1].cpu()
			# save the inferred position
			with open(os.path.join(web_dir, 'inferred_position.txt'), 'w') as f:
				write_location(f, inferred_image_position, 0, 'image')
				write_location(f, inferred_nss_position, 0, 'nss')
			fg_slot_position = torch.zeros((num_slots-1, 2))
			fg_slot_position[0] = torch.tensor([0 ,0])
			model.forward_position(fg_slot_nss_position=fg_slot_position)

		cam2world_input = model.cam2world[0:1].cpu()
		radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
		x, y = cam2world_input[:, 0, 3], cam2world_input[:, 1, 3]
		radius_xy, angle_xy = torch.sqrt(x ** 2 + y ** 2).item(), torch.atan2(y, x).item()
		theta = torch.acos((cam2world_input[:, 2, 3]) / radius)
		radius, theta, z = radius.item(), theta.item(), cam2world_input[:, 2, 3].item()

		if spherical:
			cam2worlds = get_spherical_cam2world(radius, theta, 30)
		else:
			cam2worlds = get_spiral_cam2world(radius_xy, z, (angle_xy - np.pi / 6, angle_xy + np.pi / 6), 30)

		# cam2worlds = torch.from_numpy(cam2worlds).float()

		# video writer in .avi format
		# video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.avi'), cv2.VideoWriter_fourcc(*'XVID'), 30, (128, 128))
		
		for i in tqdm(range(cam2worlds.shape[0])):
			cam2world = cam2worlds[i:i+1]
			# print(cam2world)
			model.visual_cam2world(cam2world)
			# model.visual_names = list(filter(lambda x: 'input' not in x and 'attn' not in x, model.visual_names))
			model.visual_names = list(filter(lambda x: 'rec' in x, model.visual_names))
			model.compute_visuals()
			visuals = model.get_current_visuals()
			save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size, suffix=f'_{i}')
			# visual_name = 'x_rec0'
			# img = tensor2im(visuals[visual_name])
			# img_pil = Image.fromarray(img)
			# img_pil.save(os.path.join(web_dir, 'images_debug' , 'rendered_{}.png'.format(i)))
			# write to video, transform to BGR
			# img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			# video_writer.write(img)

		video_writer = cv2.VideoWriter(os.path.join(web_dir, 'rendered.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
		visual_image_paths = list(filter(lambda x: 'rec0' in x, glob(os.path.join(web_dir, 'images', '*.png'))))
		visual_image_paths.sort()
		for visual_image_path in visual_image_paths:
			img = cv2.imread(visual_image_path)
			video_writer.write(img)

		video_writer.release()

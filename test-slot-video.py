from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed, write_location

import torch
from util.util import get_spherical_cam2world, tensor2im, get_spiral_cam2world, parse_wanted_indice, get_predefined_cam2world, get_cam2world_from_path
import torchvision
import cv2
from tqdm import tqdm
import numpy as np
import imageio


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

# web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
# 						'{}_{}'.format(opt.testset_name, opt.epoch))  # define the website directory
# print('creating web directory', web_dir)
# webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

# wanted idx
# wanted_indices = [x for x in range(140) if x % 3 == 0]
wanted_indices = parse_wanted_indice(opt.wanted_indices)

center = opt.move2center # if true, move all object to the scene center

suffix = '_center' if center else ''

for idx, data in enumerate(dataset):

	if not wanted_indices is None and idx not in wanted_indices:
		continue

	web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
							f'{opt.testset_name}_{opt.epoch}/scene{idx}_{opt.video_mode}{suffix}')  # define the website directory
	print('creating web directory', web_dir)
	webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

	visualizer.reset()
	model.set_input(data)  # unpack data from data loader
	if not center:
		visual_names = ['slot0_view0_unmasked'] + [f'slot{i}_view0' for i in range(1, opt.num_slots)] + ['x_rec0']
	else:
		visual_names = [f'slot{i}_view0_unmasked' for i in range(0, opt.num_slots)] + ['x_rec0']


	with torch.no_grad():
		model.forward()

		model.compute_visuals()
		visuals = model.get_current_visuals()
		visualizer.display_current_results(visuals, epoch=None, save_result=False)
		img_path = model.get_image_paths()
		save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)

		# modify the position of the slots
		if center:
			fg_slot_position = torch.zeros((opt.num_slots-1, 2))
			fg_slot_position[0] = torch.tensor([0, 0])
			fg_slot_position[1] = torch.tensor([0, 0])
			fg_slot_position[2] = torch.tensor([0, 0])
			fg_slot_position[3] = torch.tensor([0, 0])
			model.forward_position(fg_slot_nss_position=fg_slot_position)

		# cam2world_input = torch.from_numpy(np.loadtxt('/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-cabinet-test/00008_sc0004_az00_RT.txt')).unsqueeze(0)
		cam2world_input = model.cam2world[0:1].cpu()
		radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
		x, y = cam2world_input[:, 0, 3], cam2world_input[:, 1, 3]
		radius_xy, angle_xy = torch.sqrt(x ** 2 + y ** 2).item(), torch.atan2(y, x).item()
		theta = torch.asin((cam2world_input[:, 2, 3]) / radius)
		radius, theta, z = radius.item(), theta.item(), cam2world_input[:, 2, 3].item()

		if opt.video_mode == 'spherical':
			cam2worlds = get_spherical_cam2world(radius, theta, 45)
			# cam2worlds = get_spherical_cam2world(radius, theta, n_views=24, radians=False, camera_normalize=True, origin=(0, 0, 0), around_table=True)
			# cam2worlds = get_spherical_cam2world(radius, 20, n_views=24, radians=False, camera_normalize=True)
		elif opt.video_mode == 'spiral':
			cam2worlds = get_spiral_cam2world(radius_xy, z, (angle_xy, angle_xy + np.pi * 1 / 4), 60, height_range=(0.95, 1.15))
		elif opt.video_mode == 'predefined':
			cam2worlds = get_cam2world_from_path(data['paths'][0].replace('.png', '_RT.txt'), 60)
			# cam2worlds = get_predefined_cam2world(model.opt.dataroot, idx, 60, model.opt.fixed_dist)
		else:
			assert False

		resolution = (256, 256)
		video_writers = []
		for j in range(0, opt.num_slots):
			# video_writers.append(imageio.get_writer(os.path.join(web_dir, 'rendered_slot{}.gif'.format(j)), mode='I', duration=33))
			video_writers.append(cv2.VideoWriter(os.path.join(web_dir, 'rendered_slot{}.mp4'.format(j)), cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution))
		# video_writers.append(imageio.get_writer(os.path.join(web_dir, 'rendered_rec.gif'), mode='I', duration=33))
		video_writers.append(cv2.VideoWriter(os.path.join(web_dir, 'rendered_rec.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution))

		for i in tqdm(range(cam2worlds.shape[0])):
			cam2world = cam2worlds[i:i+1]
			print(cam2world)
			model.visual_cam2world(cam2world)
			model.compute_visuals(cam2world=cam2world)
			visuals = model.get_current_visuals()
			# print(len(visuals))
			for j, visual_name in enumerate(visual_names):
				img = visuals[visual_name]
				path = os.path.join(web_dir, 'images' ,'rendered_slot{}_{}.png'.format(j, i))
				torchvision.utils.save_image(img.detach().cpu() / 2 + 0.5, path)

				img = tensor2im(img)
				img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
				video_writers[j].write(img)


		for video_writer in video_writers:
			# video_writer.close()
			video_writer.release()

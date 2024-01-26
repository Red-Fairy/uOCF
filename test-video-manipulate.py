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

suffix = ''
n_frames = 30

# obj_to_move = 1
# suffix = f'moving_obj_{obj_to_move}'
# dst = torch.tensor([0.4, 0.4, 0]).to(model.device)

swap = False
obj_to_swap = [(0, 2), (1, 3)]
if swap:
	suffix += f'swap_{"_".join([str(idx[0])+str(idx[1]) for idx in obj_to_swap])}'

translate1, translate2, translate3 = False, False, True
trans3_stage = 1
r = 0.36

bias = torch.tensor([-0.15, -0.1, 0]).to(model.device)
translate_dst = torch.tensor([[0, r, 0], [0, -r, 0], [r, 0, 0], [-r, 0, 0]]).to(model.device) + bias # kitchen-easy, scene 97

# bias = torch.tensor([-0.1, 0, 0]).to(model.device)
# translate_dst = torch.tensor([[0, r, 0], [r, 0, 0], [0, -r, 0], [-r, 0, 0]]).to(model.device) + bias # kitchen-hard, scene 6

if translate1:
	suffix += '_translate1'
elif translate2:
	suffix += '_translate2'
elif translate3:
	suffix += '_translate3_' + str(trans3_stage)

wanted_indices = parse_wanted_indice(opt.wanted_indices)

for j, data in enumerate(dataset):

	if not wanted_indices is None and j not in wanted_indices:
		continue

	print('Visualizing scene No.: ', j)

	web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
							f'{opt.testset_name}_{opt.epoch}/scene{j}_{opt.video_mode}{suffix}')  # define the website directory
	print('creating web directory', web_dir)
	webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

	visualizer.reset()
	model.set_input(data)  # unpack data from data loader
	model.set_visual_names()

	with torch.no_grad():
		model.forward()
		img_path = model.get_image_paths()

		model.compute_visuals()
		visuals = model.get_current_visuals()
		# visualizer.display_current_results(visuals, epoch=None, save_result=False)
		save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)

		cam2world_input = model.cam2world[0:1].cpu()
		radius = torch.sqrt(torch.sum(cam2world_input[:, :3, 3] ** 2, dim=1))
		x, y = cam2world_input[:, 0, 3], cam2world_input[:, 1, 3]
		radius_xy, angle_xy = torch.sqrt(x ** 2 + y ** 2).item(), torch.atan2(y, x).item()
		theta = torch.acos((cam2world_input[:, 2, 3]) / radius)
		radius, theta, z = radius.item(), theta.item(), cam2world_input[:, 2, 3].item()

		# kiteasy
		cam2world = get_spiral_cam2world(radius_xy, z, (angle_xy, angle_xy), 1, radius_range=(0.5, 0.7), origin=(-1., -1.5, 0))

		# kithard
		# cam2world = get_spiral_cam2world(radius_xy, z, (angle_xy, angle_xy), 1, radius_range=(0.8, 1.0), origin=(-1., -1, 0))

		ori_pos = model.fg_slot_nss_position.clone()
		# print(ori_pos)

		for i in tqdm(range(n_frames+1)):
			if translate1:
				model.fg_slot_nss_position = ori_pos + (translate_dst - ori_pos) * i / n_frames
			if translate2:
				theta = 2 * np.pi * i / n_frames
				# translate_dst = torch.tensor([[r*np.sin(theta+np.pi), r*np.cos(theta+np.pi), 0],
				# 				  			[r*np.sin(theta), r*np.cos(theta), 0],
				# 							 [r*np.sin(theta+np.pi/2), r*np.cos(theta+np.pi/2), 0],
				# 							 [r*np.sin(theta-np.pi/2), r*np.cos(theta-np.pi/2), 0]]).to(model.device) + bias
				# translate_dst = torch.tensor([[r*np.sin(theta-np.pi/2), r*np.cos(theta-np.pi/2), 0],
				# 							[r*np.sin(theta+np.pi/2), r*np.cos(theta+np.pi/2), 0],
				# 							[r*np.sin(theta+np.pi), r*np.cos(theta+np.pi), 0],
				# 				  			[r*np.sin(theta), r*np.cos(theta), 0]]).to(model.device) + bias
				# translate_dst = torch.tensor([[r*np.cos(theta+np.pi/2), r*np.sin(theta+np.pi/2), 0],
				# 							[r*np.cos(theta), r*np.sin(theta), 0],
				# 							[r*np.cos(theta+np.pi*3/2), r*np.sin(theta+np.pi*3/2), 0],
				# 				  			[r*np.cos(theta+np.pi), r*np.sin(theta+np.pi), 0]]).to(model.device) + bias # kithard
				translate_dst = torch.tensor([[r*np.cos(theta+np.pi/2), r*np.sin(theta+np.pi/2), 0],
											[r*np.cos(theta+np.pi*3/2), r*np.sin(theta+np.pi*3/2), 0],
											[r*np.cos(theta), r*np.sin(theta), 0],
								  			[r*np.cos(theta+np.pi), r*np.sin(theta+np.pi), 0]]).to(model.device) + bias # kiteasy
				model.fg_slot_nss_position = translate_dst.float()
			if translate3:
				model.fg_slot_nss_position += torch.tensor([[100, 100, 0]]).to(model.device)
				''' for kitchen-hard dataset scene 5'''
				# if trans3_stage == 1: # move obj 1 to the scene center, remove other objects
				# 	model.fg_slot_nss_position = ori_pos + (bias - ori_pos) * i / n_frames
				# elif trans3_stage == 2: # obj 1 at the scene center, obj 2 land on the top of obj 1
				# 	model.fg_slot_nss_position[0] = bias
				# 	model.fg_slot_nss_position[3] = torch.tensor([-0.02, -0.02, 0.5]).to(model.device) - torch.tensor([-0.02, -0.02, 0.42]).to(model.device) * i / n_frames + bias
				# 	# rest obj not shown
				# 	model.fg_slot_nss_position[1:3] = torch.tensor([100, 100, 0]).to(model.device)
				# elif trans3_stage == 3: # obj 1 at the scene center, obj 2 land on the top of obj 1, obj 3 land on the top of obj 2
				# 	model.fg_slot_nss_position[0] = bias
				# 	model.fg_slot_nss_position[3] = torch.tensor([-0.02, -0.02, 0.08]).to(model.device) + bias
				# 	# obj2, from top ([-0.02, -0.02, 0.5]) to bottom ([-0.02, -0.02, 0.14])
				# 	model.fg_slot_nss_position[2] = torch.tensor([-0.02, -0.02, 0.5]).to(model.device) - torch.tensor([-0.02, -0.02, 0.36]).to(model.device) * i / n_frames + bias
				# 	# rest obj not shown
				# 	model.fg_slot_nss_position[1] = torch.tensor([100, 100, 0]).to(model.device)
				# elif trans3_stage == 4: # obj 1 at the scene center, obj 2 land on the top of obj 1, obj 3 land on the top of obj 2, obj 4 land on the top of obj 3
				# 	model.fg_slot_nss_position[0] = bias
				# 	model.fg_slot_nss_position[3] = torch.tensor([-0.02, -0.02, 0.08]).to(model.device) + bias
				# 	model.fg_slot_nss_position[2] = torch.tensor([-0.02, -0.02, 0.14]).to(model.device) + bias
				# 	# obj3, from top ([-0.025, -0.015, 0.5]) to bottom ([-0.025, -0.015, 0.2])
				# 	model.fg_slot_nss_position[1] = torch.tensor([-0.025, -0.015, 0.5]).to(model.device) - torch.tensor([-0.025, -0.015, 0.3]).to(model.device) * i / n_frames + bias
				''' for kitchen-easy dataset, scene 97'''
				if trans3_stage == 1:
					model.fg_slot_nss_position[0] = torch.tensor([-0.15, 0.15, 0.8]).to(model.device) - torch.tensor([0, 0, 0.8]).to(model.device) * i / n_frames + bias
				elif trans3_stage == 2:
					model.fg_slot_nss_position[0] = torch.tensor([-0.15, 0.15, 0]).to(model.device) + bias
					model.fg_slot_nss_position[2] = torch.tensor([-0.15, 0.11, 0.8]).to(model.device) - torch.tensor([0, 0, 0.68]).to(model.device) * i / n_frames + bias
				elif trans3_stage == 3:
					model.fg_slot_nss_position[0] = torch.tensor([-0.15, 0.15, 0]).to(model.device) + bias
					model.fg_slot_nss_position[2] = torch.tensor([-0.15, 0.11, 0.12]).to(model.device) + bias
					model.fg_slot_nss_position[1] = torch.tensor([0.15, -0.15, 0.8]).to(model.device) - torch.tensor([0, 0, 0.8]).to(model.device) * i / n_frames + bias
				elif trans3_stage == 4:
					model.fg_slot_nss_position[0] = torch.tensor([-0.15, 0.15, 0]).to(model.device) + bias
					model.fg_slot_nss_position[2] = torch.tensor([-0.15, 0.11, 0.12]).to(model.device) + bias
					model.fg_slot_nss_position[1] = torch.tensor([0.15, -0.15, 0]).to(model.device) + bias
					model.fg_slot_nss_position[3] = torch.tensor([0.14, -0.20, 0.8]).to(model.device) - torch.tensor([0, 0, 0.72]).to(model.device) * i / n_frames + bias
				''' for kitchen-hard dataset scene 6'''
				# if trans3_stage == 1:
				# 	model.fg_slot_nss_position[3] = torch.tensor([0, 0, 0.8]).to(model.device) - torch.tensor([0, 0, 0.8]).to(model.device) * i / n_frames + bias
				# elif trans3_stage == 2:
				# 	model.fg_slot_nss_position[3] = torch.tensor([0, 0, 0]).to(model.device) + bias
				# 	model.fg_slot_nss_position[1] = torch.tensor([0, 0, 0.8]).to(model.device) - torch.tensor([0, 0, 0.75]).to(model.device) * i / n_frames + bias
				# elif trans3_stage == 3:
				# 	model.fg_slot_nss_position[3] = torch.tensor([0, 0., 0]).to(model.device) + bias
				# 	model.fg_slot_nss_position[1] = torch.tensor([0, 0., 0.05]).to(model.device) + bias
				# 	model.fg_slot_nss_position[0] = torch.tensor([-0.01, 0., 0.8]).to(model.device) - torch.tensor([0, 0, 0.63]).to(model.device) * i / n_frames + bias
				# elif trans3_stage == 4:
				# 	model.fg_slot_nss_position[3] = torch.tensor([0, 0., 0]).to(model.device) + bias
				# 	model.fg_slot_nss_position[1] = torch.tensor([0, 0., 0.05]).to(model.device) + bias
				# 	model.fg_slot_nss_position[0] = torch.tensor([-0.01, 0., 0.13]).to(model.device) + bias
				# 	model.fg_slot_nss_position[2] = torch.tensor([0.033, 0.045, 0.8]).to(model.device) - torch.tensor([0, 0, 0.59]).to(model.device) * i / n_frames + bias


			model.forward_position()
			model.visual_cam2world(cam2world)
			model.visual_names = list(filter(lambda x: 'rec' in x, model.visual_names))
			visuals = model.get_current_visuals()
			save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size, suffix=f'_{i:03d}')



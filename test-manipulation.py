from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed, write_location, parse_wanted_indice

import torch


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

wanted_indices = parse_wanted_indice(opt.wanted_indices)

# file = open(os.path.join(opt.results_dir, opt.name, opt.exp_id, 'slot_location.txt'), 'w+')

suffix = 'swap_02'

for j, data in enumerate(dataset):

	if not wanted_indices is None and j not in wanted_indices:
		continue
	  
	visualizer.reset()
	model.set_input(data)  # unpack data from data loader
	# model.test()           # run inference: forward + compute_visuals

	web_dir = os.path.join(opt.results_dir, opt.name, opt.exp_id,
							f'{opt.testset_name}/scene{j}_{suffix}')  # define the website directory
	print('creating web directory', web_dir)
	webpage = HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

	with torch.no_grad():
		model.forward()

		fg_positions = model.fg_slot_nss_position.clone()
		model.fg_slot_nss_position[0] = fg_positions[2]
		model.fg_slot_nss_position[2] = fg_positions[0]
		model.fg_slot_nss_position[1] = fg_positions[1]
		model.fg_slot_nss_position[3] = fg_positions[3]
		
		# model.fg_slot_nss_position[0] = torch.tensor([0.4, 0.4, 0])
		# model.fg_slot_nss_position[2] = torch.tensor([0.2, 0.2, 0])

		model.forward_position()
		model.compute_visuals()

	visuals = model.get_current_visuals()
	visualizer.display_current_results(visuals, epoch=None, save_result=False)
	img_path = model.get_image_paths()
	save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.load_size)


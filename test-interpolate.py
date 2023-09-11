from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer, save_images
from util.html import HTML
import os
from util.util import AverageMeter, set_seed, write_location
import numpy as np
import torch
from tqdm import tqdm

n_steps = 25

manipulation = True

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(
        opt
    )  # create a visualizer that display/save images and plots
    meters_tst = {stat: AverageMeter() for stat in model.loss_names}

    set_seed(opt.seed)

    web_dir = os.path.join(
        opt.results_dir,
        opt.name,
        opt.exp_id,
        "{}_{}".format(opt.testset_name, opt.epoch),
    )  # define the website directory
    print("creating web directory", web_dir)
    webpage = HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    file = open(
        os.path.join(
            opt.results_dir,
            opt.name,
            opt.exp_id,
            "{}_{}".format(opt.testset_name, opt.epoch),
            "slot_location.txt",
        ),
        "w+",
    )
    
    for i, data in enumerate(dataset):
        visualizer.reset()
        model.set_input(data)  # unpack data from data loader
        # model.test()           # run inference: forward + compute_visuals

        with torch.no_grad():
            model.forward()
            img_path = model.get_image_paths()
            z_slot_1 = np.loadtxt(
                os.path.join(
                    opt.results_dir,
                    opt.name,
                    opt.exp_id,
                    "{}_{}".format(opt.testset_name, opt.epoch),
                    "z_slots_0.txt",
                )
            )
            z_slot_5 = np.loadtxt(
                os.path.join(
                    opt.results_dir,
                    opt.name,
                    opt.exp_id,
                    "{}_{}".format(opt.testset_name, opt.epoch),
                    "z_slots_1.txt",
                )
            )
            for i in tqdm(range(n_steps)):
                z_slot_interpolate = z_slot_1 + (z_slot_5 - z_slot_1) * i / (n_steps - 1)
                z_slots = model.z_slots
                z_slots[1] = torch.tensor(z_slot_interpolate).cuda()
                fg_slot_position = torch.zeros((opt.num_slots - 1, 2))
                if manipulation:
                    fg_slot_position[0] = torch.tensor([1/7, 1/7])
                else:
                    fg_slot_position[0] = torch.tensor([0, 0])
                model.forward_position(
                    fg_slot_nss_position=fg_slot_position, z_slots=z_slots
                )

                model.compute_visuals()
                model.visual_names = list(filter(lambda x: 'rec' in x, model.visual_names))
                visuals = model.get_current_visuals()

                save_images(
                    webpage,
                    visuals,
                    img_path,
                    aspect_ratio=opt.aspect_ratio,
                    width=opt.load_size,
                    suffix=f"_{i}",
                )

    webpage.save()
    file.close()

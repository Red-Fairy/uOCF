from pyexpat import model
import time
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.util import AverageMeter, set_seed
from PIL import Image

def main():
    opt = TestOptions().parse()   # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options

    model = create_model(opt)
    model.setup(opt) 

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots

    for data in dataset:

        model.set_input(data)         # unpack data from dataset and apply preprocessing

        model.forward()

        print(model.fg_slot_position)
        
        model.compute_visuals()

        model.get_current_visuals()

        visualizer.display_current_results(model.get_current_visuals(), epoch=0, save_result=False)

        break


if __name__ == '__main__':
    main()


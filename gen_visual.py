from math import floor
import os
import matplotlib.pyplot as plt
from PIL import Image
from math import ceil


root = '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/0801-real/4obj-load4obj-CIT-ttt-potvase-520/660ep_test4obj_660/images'

n_scenes = 95

n_scene_per_img = 20

# Get a list of all the image files in the folder
image_files = [f for f in os.listdir(root) if f.endswith('.png')]

wanted_visual_names = ['input_image', 'x_rec0', 'x_rec1', 'x_rec2']

img_paths = list(filter(lambda x: any([name in x for name in wanted_visual_names]), image_files))

# Calculate the number of rows and columns based on the number of images


# Set up the plot
for vis_idx in range(ceil(n_scenes / n_scene_per_img)):

    num_cols = n_scene_per_img if vis_idx < floor(n_scenes / n_scene_per_img) else n_scenes % n_scene_per_img
    num_rows = len(wanted_visual_names)

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(num_cols*2, num_rows*2))

    for i in range(num_cols):
        img_path_scene = list(filter(lambda x: f'sc{i+vis_idx*n_scene_per_img:04d}' in x, img_paths))
        img_path_scene = [os.path.join(root, img_path) for img_path in img_path_scene]
        for idx, name in enumerate(wanted_visual_names):
            image_path = list(filter(lambda x: name in x, img_path_scene))[0]
            image = Image.open(image_path)
            axs[idx, i].imshow(image)
            axs[idx, i].axis('off')

    plt.subplots_adjust(wspace=0.03, hspace=0.03)

    plt.savefig(f'vis_{vis_idx}.png', bbox_inches='tight', pad_inches=0)
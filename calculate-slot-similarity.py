import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch
from tqdm import tqdm

root = '/viscam/projects/uorf-extension/uOCF/checkpoints/OCTScenes/1212-modNorm/load-default-noMask/regular_saveLatent_latest'
slot_latent_dir = os.path.join(root, 'slot_latent')
save_dir = os.path.join(root, 'slot_latent_vis')
os.makedirs(save_dir, exist_ok=True)

n_scenes = 40
n_slots = 7

for i in tqdm(range(n_scenes)):
    latent = []
    position = []
    for j in range(n_slots):
        latent.append(np.loadtxt(os.path.join(slot_latent_dir, 'sc{}_latent{}.txt'.format(i, j))))
        position.append(np.loadtxt(os.path.join(slot_latent_dir, 'sc{}_position{}.txt'.format(i, j))))
    latent = np.array(latent)
    # calculate similarity matrix
    similarity = np.zeros((n_slots, n_slots))
    distance = np.zeros((n_slots, n_slots))
    for j in range(n_slots):
        for k in range(n_slots):
            similarity[j, k] = np.dot(latent[j][:48], latent[k][:48]) / (np.linalg.norm(latent[j][:48]) * np.linalg.norm(latent[k][:48]))
            distance[j, k] = np.linalg.norm(position[j] - position[k])
    # plot similarity matrix, show in grid and heatmap, print values
    plt.figure()
    plt.imshow(similarity)
    # print the value of each cell
    for j in range(n_slots):
        for k in range(n_slots):
            text = plt.text(k, j, 'sim: {:.2f}\ndist: {:.2f}'.format(similarity[j, k], distance[j, k]),
                ha="center", va="center", color="w", fontsize=8)
    # print the position of each slot, on the top of each column
    plt.colorbar()
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, 'sc{}_similarity.png'.format(i)))
    plt.close()
    

    
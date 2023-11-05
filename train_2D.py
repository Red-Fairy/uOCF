from json import decoder
import os
import argparse
from data.singleimgs_dataset import create_dataloader
from models.model_2D import *
from tqdm import tqdm
import time
import datetime
import torch.optim as optim
import torch
from util.util import logger
import math
from torchvision.transforms.functional import to_tensor
from torchvision import transforms
from PIL import Image
from torch import nn
from models.networks import init_net
from util.util import tensor2im, save_image
from torch.utils.tensorboard import SummaryWriter
from models.networks import get_exp_decay_schedule_with_warmup

parser = argparse.ArgumentParser()

# /svl/u/redfairy/datasets/room-real/chairs/train-4obj

parser.add_argument
parser.add_argument('--model_dir', default='./test', type=str, help='where to save models' )
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--num_slots', default=5, type=int, help='Number of slots in Slot Attention.')
parser.add_argument('--num_iterations', default=3, type=int, help='Number of attention iterations.')
parser.add_argument('--shape_dim', default=48, type=int, help='shape dimension size')
parser.add_argument('--color_dim', default=16, type=int, help='color dimension size')
parser.add_argument('--n_feat_layer', default=4, type=int, help='number of feature layers to extract from DINO')
parser.add_argument('--bottom', action='store_true', help='bottom attention layer')
parser.add_argument('--single_route', action='store_true', help='single route encoder')

parser.add_argument('--load_size', default=128, type=int, help='size for image to load')
parser.add_argument('--decoder_input_size', default=16, type=int, help='size for image to load')

parser.add_argument('--learning_rate', default=0.0004, type=float)
parser.add_argument('--warmup_steps', default=1000, type=int, help='Number of warmup steps for the learning rate.')
parser.add_argument('--num_steps', default=1000000, type=int, help='Number of total step.')
parser.add_argument('--num_decay_steps', default=100000, type=int, help='Number of decay steps for the learning rate.')

parser.add_argument('--num_workers', default=4, type=int, help='number of workers for loading data')
parser.add_argument('--num_epochs', default=1000, type=int, help='number of workers for loading data')

parser.add_argument('--data_root', required=True, type=str, help='path to data')

parser.add_argument('--gpu_ids', default=[0], type=int, nargs='+', help='gpu ids: e.g. 0 1 2 3')
parser.add_argument('--visual_freq', default=5, type=int, help='frequency of visualization')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.model_dir = os.path.join('checkpoints', args.model_dir)

os.makedirs(args.model_dir, exist_ok=True)
os.makedirs(os.path.join(args.model_dir,'weights'), exist_ok=True)
os.makedirs(os.path.join(args.model_dir,'visuals'), exist_ok=True)

log = logger(args.model_dir)
log.info(str(args))

writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'runs'))

def main():
	# prepare datasets
	large_size = 14 * 64
	data_loader = create_dataloader(args.data_root, args.batch_size, load_size=args.load_size, large_size=large_size, num_workers=args.num_workers)

	# create model
	if not args.single_route:
		model = SlotAttentionDualAutoEncoder(args.num_slots, args.num_iterations, bottom=args.bottom,
										shape_dim = args.shape_dim, color_dim = args.color_dim,
										n_feat_layer=args.n_feat_layer,
										decoder_input_size=(args.decoder_input_size, args.decoder_input_size),
										decoder_output_size=(args.load_size, args.load_size))
	else:
		model = SlotAttentionSingleAutoEncoder(args.num_slots, args.num_iterations, shape_dim = args.shape_dim,
										n_feat_layer=args.n_feat_layer,
										decoder_input_size=(args.decoder_input_size, args.decoder_input_size),
										decoder_output_size=(args.load_size, args.load_size))
	model = init_net(model, init_type='normal', init_gain=0.02, gpu_ids=args.gpu_ids)
	DINO_encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device).eval()

	criterion = nn.MSELoss()

	params = [{'params': model.parameters()}]
	optimizer = optim.Adam(params, lr=args.learning_rate)
	scheduler = get_exp_decay_schedule_with_warmup(optimizer, args.warmup_steps // args.batch_size, args.num_decay_steps // args.batch_size)

	start = time.time()
	i = 0
	epoch = 0

	while True:
		epoch += 1
		model.train()
		total_loss = 0

		for batch_data in tqdm(data_loader):
			with torch.no_grad(): # B*C*H*W
				feature_maps = DINO_encoder.get_intermediate_layers(batch_data['x_large'].to(device), n=args.n_feat_layer, reshape=True)

			x = batch_data['x']

			if not args.single_route:
				x = x.to(device)
				recons, recons_slot, masks, attn = model(x, feature_maps)
			else:
				recons, recons_slot, masks, attn = model(feature_maps)

			loss = criterion(recons, x)
			total_loss += loss.item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			scheduler.step()
			# print(x.shape, recons.shape, recons_slot.shape, masks.shape, attn.shape)

			if i % args.visual_freq == 0:
				save_visuals(epoch, x[0], recons[0], recons_slot[0], masks[0], attn[0])
			i += 1

		total_loss /= len(data_loader)
		writer.add_scalar('Loss/train', total_loss, epoch)

		log.info("Epoch: {}, Loss: {}, Time: {}".format(epoch, total_loss,
			datetime.timedelta(seconds=time.time() - start)))

		torch.save({
			'model_state_dict': model.state_dict(),
			}, os.path.join(args.model_dir, 'weights/model_{}.ckpt'.format(epoch)))
			
		if i > args.num_steps // args.batch_size:
			break


def save_visuals(epoch, img, recon=None, recons_slot=None, masks_slot=None, attn_slot=None):
	'''
	img: C*H*W (input)
	recon: C*H*W
	recons_slot: K*C*H*W
	masks_slot: K*1*H*W
	attn_slot: K*1*H*W (attention map)
	'''
	# save images
	name = 'epoch_{}'.format(epoch)

	img = tensor2im(img) # numpy, H*W*C
	save_path = os.path.join(args.model_dir, 'visuals', name + '_input.png')
	save_image(img, save_path)

	if recon is not None:
		recon = tensor2im(recon)
		save_path = os.path.join(args.model_dir, 'visuals', name + '_recon.png')
		save_image(recon, save_path)
	
	if recons_slot is not None:
		for i in range(recons_slot.shape[0]):
			recon = tensor2im(recons_slot[i])
			save_path = os.path.join(args.model_dir, 'visuals', name + '_recon_slot_{}.png'.format(i))
			save_image(recon, save_path)
		
	if masks_slot is not None:
		for i in range(masks_slot.shape[0]):
			mask = tensor2im(masks_slot[i])
			save_path = os.path.join(args.model_dir, 'visuals', name + '_mask_slot_{}.png'.format(i))
			save_image(mask, save_path)

	if attn_slot is not None:
		for i in range(attn_slot.shape[0]):
			attn = tensor2im(attn_slot[i])
			save_path = os.path.join(args.model_dir, 'visuals', name + '_attn_slot_{}.png'.format(i))
			save_image(attn, save_path)

if __name__ == '__main__':
	main()



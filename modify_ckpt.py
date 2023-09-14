import copy
import torch
root = '/viscam/projects/uorf-extension/'
filepath = 'I-uORF/checkpoints/kitchen-hard/0828/4obj-loadchairs-fine256-depth-FT-viewdirs/2000_net_Decoder.pth'
ckpt = torch.load(root + filepath)
# print(ckpt.keys())

ckpt_new = copy.deepcopy(ckpt)
del ckpt_new['b_after.6.weight']
del ckpt_new['b_after.6.bias']

ckpt_new['b_color.weight'] = ckpt['b_after.6.weight'][:3]
ckpt_new['b_color.bias'] = ckpt['b_after.6.bias'][:3]

ckpt_new['b_shape.weight'] = ckpt['b_after.6.weight'][3:]
ckpt_new['b_shape.bias'] = ckpt['b_after.6.bias'][3:]

torch.save(ckpt_new, root + filepath)

# print(ckpt_new.keys())
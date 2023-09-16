import copy
import torch
root = '/viscam/projects/uorf-extension/'
filepath = 'I-uORF/checkpoints/kitchen-easy/dataset-0817-0828/fit-single-scene-2/1000_net_Decoder.pth'
ckpt = torch.load(root + filepath)
print(ckpt.keys())
print(ckpt['b_before.0.weight'].shape, ckpt['b_before.0.bias'].shape)
print(ckpt['b_after.0.weight'].shape, ckpt['b_after.0.bias'].shape)
exit()

ckpt_new = copy.deepcopy(ckpt)
del ckpt_new['b_after.6.weight']
del ckpt_new['b_after.6.bias']

ckpt_new['b_color.weight'] = ckpt['b_after.6.weight'][:3]
ckpt_new['b_color.bias'] = ckpt['b_after.6.bias'][:3]

ckpt_new['b_shape.weight'] = ckpt['b_after.6.weight'][3:]
ckpt_new['b_shape.bias'] = ckpt['b_after.6.bias'][3:]

del ckpt_new['b_before.0.weight']
# del ckpt_new['b_before.0.bias']

ckpt_new['b_before.0.weight'] = torch.zeros(ckpt['b_before.0.weight'].shape[0], 129+30)
ckpt_new['b_before.0.weight'][:, :126] = ckpt['b_before.0.weight'][:, :126]
ckpt_new['b_before.0.weight'][:, -3:] = ckpt['b_before.0.weight'][:, -3:]

ckpt_new['b_after.0.weight'] = torch.zeros(ckpt['b_after.0.weight'].shape[0], 225+30)
ckpt_new['b_after.0.weight'][:, :222] = ckpt['b_after.0.weight'][:, :222]
ckpt_new['b_after.0.weight'][:, -3:] = ckpt['b_after.0.weight'][:, -3:]

torch.save(ckpt_new, root + filepath)

# print(ckpt_new.keys())
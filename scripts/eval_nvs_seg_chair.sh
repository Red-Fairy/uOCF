#!/bin/bash
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_chair/test_ood_2.5'}
PORT=${2:-12783}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_chair' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --exp_id 'uORF-pretrained' \
    --model 'uorf_eval'
echo "Done"

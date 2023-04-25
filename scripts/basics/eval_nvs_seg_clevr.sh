#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/unseen'}
CHECKPOINT=${2:-'./'}
PORT=8077
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' --exp_id 'uORF-pretrained' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 40 --num_slots 8 \
    --model 'uorf_eval' \
    --testset_name 'unseen_test'
echo "Done"

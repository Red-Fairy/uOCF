#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/test'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 36 --num_slots 8 \
    --model 'uorf_eval_T' --project \
    --exp_id 'run-2023-03-22-07-24-52' \
# done
echo "Done"
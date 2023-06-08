#!/bin/bash
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_bg/train-3obj-manysize-orange'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_bg' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --model 'uorf_eval' \
    --pos_emb --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_diverse_bg/uORF-3obj' \
    --attn_iter 3 --testset_name 'regular_test' \
# done
echo "Done"
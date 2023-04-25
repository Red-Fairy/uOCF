#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/room_diverse_generation/image_generation/datasets/1200shape_nobg_test'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --model 'uorf_eval_T' \
    --sam_encoder --encoder_size 1024 \
    --pos_emb --exp_id '0409-sam' \
    --z_dim 64 \
    --project --attn_iter 4 --testset_name trail_test \
# done
echo "Done"
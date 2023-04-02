#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/unseen'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 40 --num_slots 8 \
    --model 'uorf_eval_T' \
    --exp_id '0328-project-loadEncoder' --pos_emb \
    --project --attn_iter 4 --testset_name unseen_test \
# done
echo "Done"
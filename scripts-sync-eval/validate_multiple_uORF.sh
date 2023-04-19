#!/bin/bash
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_multiple_bg/test'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_multiple' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --model 'uorf_eval' \
    --pos_emb --exp_id '0416-uORF-bg' \
    --attn_iter 4 --testset_name test_ood_143epoch \
# done
echo "Done"
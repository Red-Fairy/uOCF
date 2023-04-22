#!/bin/bash
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_chair/test'}
PORT=${2:-12783}
python test-video.py --dataroot $DATAROOT --n_scenes 10 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_chair' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --num_slots 5 \
    --model 'uorf_eval_T_sam' \
    --sam_encoder --encoder_size 1024 \
    --exp_id '0416-sam-texture-48-16-4view' \
    --z_dim 48 --texture_dim 16 \
    --project --attn_iter 4 --no_loss --testset_name test_video \
# done
echo "Done"
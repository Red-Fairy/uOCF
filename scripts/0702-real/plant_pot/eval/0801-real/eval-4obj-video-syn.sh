#!/bin/bash
DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/plant_pots/test-white-4obj-nofoot-viewrange-large-4050'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=0 python test-video.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 \
    --n_samp 256 --num_slots 6 \
    --model 'uorf_general_eval' \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/0724-new/4obj-load-freezeBG-4848-6slot-4050' \
    --shape_dim 48 --color_dim 48 --bottom --color_in_attn --fixed_locality \
    --attn_iter 4 --no_loss --recon_only --video --visual_idx 0 --testset_name test_video/idx0 \
# done
echo "Done"
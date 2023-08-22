#!/bin/bash
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-easy/4obj-all-test-0817'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python test-slot-video.py --dataroot $DATAROOT \
    --n_scenes 107 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 32 --frustum_size 128 \
    --n_samp 256 --num_slots 6 \
    --model 'uorf_general_eval' \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/kitchen-easy/dataset-0817/4obj-load4obj-ttt-objtop-r2' \
    --shape_dim 48 --color_dim 48 --bottom --color_in_attn --fixed_locality --video_mode 'spiral' \
    --attn_iter 4 --no_loss --recon_only --video --testset_name test_video_slot_360 \
# done
echo "Done"
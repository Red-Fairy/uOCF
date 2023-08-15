#!/bin/bash
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/4obj-kitchen-cupbowlplate-occlusion-test_multiview'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python test-video.py --dataroot $DATAROOT --n_scenes 15 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 32 --frustum_size 128 \
    --n_samp 256 --num_slots 6 \
    --model 'uorf_general_eval' \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/0801-real/4obj-load4obj-CIT-ttt-cupbowlplate-all-613-distort' \
    --shape_dim 48 --color_dim 48 --bottom --color_in_attn --fixed_locality \
    --attn_iter 4 --no_loss --recon_only --video --testset_name test_video_occlusion_378 \
# done
echo "Done"
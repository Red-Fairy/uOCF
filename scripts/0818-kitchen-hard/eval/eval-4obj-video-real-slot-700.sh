#!/bin/bash
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-all-test'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python test-video.py --dataroot $DATAROOT --n_scenes 95 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 256 --input_size 128 --render_size 32 \
    --frustum_size 256 \
    --n_samp 256 --num_slots 6 \
    --model 'uorf_general_eval_IPE' \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/kitchen-hard/0828/4obj-loadchairs-objtop-bg-fine128-load30' \
    --shape_dim 72 --color_dim 24 --bottom --fixed_locality --video_mode 'spiral' \
    --wanted_indices '2' --epoch 450 \
    --attn_iter 4 --no_loss --recon_only --video --testset_name test_video_slot \
# done
echo "Done"
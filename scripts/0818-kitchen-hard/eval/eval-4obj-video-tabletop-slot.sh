#!/bin/bash
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-all-test'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot $DATAROOT \
    --n_scenes 15 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'kitchen-hard' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 32 --frustum_size 128 \
    --n_samp 256 --num_slots 6 \
    --model 'uorf_general_eval' \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/kitchen-hard/dataset-tabletop/4obj-load4obj-fastDecay' \
    --shape_dim 48 --color_dim 48 --bottom --color_in_attn --fixed_locality \
    --attn_iter 4 --recon_only --testset_name regular_test \
# done
echo "Done"
#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/room_diverse_generation/image_generation/datasets/1200shape_50bg'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 1  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --coarse_epoch 400  --niter 800 \
    --z_dim 64 --num_slots 5 --attn_iter 4 \
    --exp_id 0405-project \
    --project \
    --pos_emb --emb_path /viscam/u/redfairy/I-uORF/checkpoints/room_diverse/uORF-pretrained/latest_net_Encoder.pth \
    --model 'uorf_nogan_T' --bottom \
    --lr 3e-4 --lr_encoder 6e-5 \
# done
echo "Done"

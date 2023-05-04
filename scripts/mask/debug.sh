DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_nobg/train-1obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_mask' \
    --display_port $PORT --display_ncols 4 --print_freq 100 --display_freq 100 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_sam_mask' \
    --num_slots 2 --attn_iter 3 \
    --shape_dim 48 --color_dim 16 \
    --bottom \
    --sam_encoder --encoder_size 1024 \
    --project \
    --coarse_epoch 40 --niter 80 --percept_in 10 \
    --exp_id '0503-mask-1obj-scale3.5-light' \
    --is_train --obj_scale 3.5 \
    --save_epoch_freq 10 \
    --dummy_info 'mask, scale=3.5, light weight model' \
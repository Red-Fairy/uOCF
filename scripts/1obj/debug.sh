DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_nobg/1200shape_nobg-1obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_sam' \
    --num_slots 2 --attn_iter 4 \
    --z_dim 48 --texture_dim 16 \
    --bottom \
    --sam_encoder --encoder_size 1024 \
    --project \
    --coarse_epoch 25 --niter 50 --invariant_in 10 --percept_in 10 \
    --exp_id '0425-1obj' \
    --no_learnable_pos \
    --dummy_info 'sam encoder v00, remove convs' \
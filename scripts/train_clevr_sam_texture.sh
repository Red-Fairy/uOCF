DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 100 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --num_slots 8 \
    --coarse_epoch 500 --niter 1000 --init_n_epoch 0 --percept_in 100 \
    --model 'uorf_nogan_T_sam' \
    --exp_id '0416-sam-texture-16-16' \
    --z_dim 16 --texture_dim 16 \
    --sam_encoder --encoder_size 1024 \
    --save_epoch_freq 25 \
    --attn_iter 4 \
    --seed 2013 --lr 3e-4 \
    --project \
    --dummy_info 'sam v0 texture' \
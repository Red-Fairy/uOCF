DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --num_slots 8 \
    --coarse_epoch 500 --niter 1000 \
    --model 'uorf_nogan_T_sam' \
    --exp_id '0414-sam-texture-4view' \
    --z_dim 30 --texture_dim 6 \
    --sam_encoder --encoder_size 1024 \
    --attn_iter 4 \
    --seed 2023 --lr 3e-4 \
    --project \
    --dummy_info 'seed 2023 sam texture' \
DATAROOT=${1:-'/viscam/u/redfairy/room_diverse_generation/image_generation/datasets/1200shape_nobg-5000'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_sam' \
    --num_slots 5 --attn_iter 4 \
    --z_dim 48 --texture_dim 16 \
    --sam_encoder --encoder_size 1024 \
    --project \
    --lr 3e-4 --coarse_epoch 120  --niter 240 \
    --bottom \
    --exp_id '0409-sam-texture-diverse' \
    --dummy_info 'frozen sam encoder v0, disentangle texture, share grid embed projection, correct deduct operation (before azi transform), move deduction after locality, add decoder MLP to z-slots projection (w/ residual), 4 round attn, use ImageNet ResNet18 encoder, pyramid upsample' \

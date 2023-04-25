DATAROOT=${1:-'/viscam/u/redfairy/datasets/room_chair/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_chair' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --z_dim 64 --num_slots 5 \
    --model 'uorf_nogan_T' \
    --exp_id '0401-project-loadEncoder' --attn_iter 4 \
    --project \
    --pos_emb --emb_path /viscam/u/redfairy/I-uORF/checkpoints/room_chair/uORF-pretrained/latest_net_Encoder.pth \
    --lr 3e-4 --lr_encoder 3e-5 \
    --niter 800 --coarse_epoch 400 \
    --dummy_info 'share grid embed projection, correct deduct operation (before azi transform), move deduction after locality, add decoder MLP to z-slots projection (w/o residual), 4 round attn, pos embedding on encoder (+4), load pretrained encoder with lr *= 0.1, w/o no bg MLP encoding' \

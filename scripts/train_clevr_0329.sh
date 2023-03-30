DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 --num_slots 8 \
    --model 'uorf_nogan_T' \
    --exp_id '0329-project-loadEncoder-positionLoss' --attn_iter 4 \
    --project \
    --pos_emb --emb_path /viscam/u/redfairy/I-uORF/checkpoints/clevr_567/clevr_567_models/latest_net_Encoder.pth \
    --position_loss --position_loss_weight 0.1 \
    --dummy_info 'share grid embed projection, correct deduct operation (before azi transform), move deduction after locality, add decoder MLP to z-slots projection (w/ residual), 4 round attn, lr 3e-4, pos embedding on encoder (+4), load pretrained encoder with lr /=5' \
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --z_dim 40 --num_slots 8 \
    --model 'uorf_nogan_T_multiview' \
    --exp_id '0403-multiview' --attn_iter 4 \
    --project \
    --pos_emb --emb_path /viscam/u/redfairy/I-uORF/checkpoints/clevr_567/uORF-pretrained/latest_net_Encoder.pth \
    --lr 3e-4 --lr_encoder 3e-5 \
    --niter 800 --coarse_epoch 400 \
    --continue_train --epoch_count 100 --epoch 100 \
    --multiview_loss \
    --dummy_info 'use multiview loss, share grid embed projection, correct deduct operation (before azi transform), move deduction after locality, add decoder MLP to z-slots projection (w/ residual), 4 round attn, pos embedding on encoder (+4), load pretrained encoder with lr *= 0.1, w/ bg MLP encoding' \

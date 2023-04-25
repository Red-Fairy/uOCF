DATAROOT=${1:-'/viscam/projects/uorf/datasets/3600shape_50bg'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'room_multiple' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --z_dim 64 --num_slots 5 \
    --model 'uorf_nogan_T' \
    --exp_id '0411-sam' --attn_iter 4 \
    --sam_encoder --encoder_size 1024 \
    --project \
    --lr 3e-4 --coarse_epoch 80  --niter 160 --percept_in 24 \
    --save_epoch_freq 10 \
    --dummy_info 'w/ bg MLP encoding, input positional encoding w/o 2-, apply invariant in 100' \

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a6000:1

#SBATCH --job-name="T_uORF"
#SBATCH --output=logs/%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
DATAROOT=${1:-'/svl/u/redfairy/datasets/bridgeV2/put_sushi_on_plate_formatted/scene_00'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 170 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'bridge' \
    --display_port $PORT --display_ncols 4 --print_freq 85 --display_freq 85 --save_epoch_freq 100 \
    --load_size 128 --n_samp 128 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uocf_dual_DINO_trans' \
    --attn_decay_steps 100000 --freezeInit_steps 5000 --lr 0.0002 --load_intrinsics \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 8 --attn_iter 6 --shape_dim 48 --color_dim 48 \
    --coarse_epoch 500 --niter 1000 --percept_in 75 --no_locality_epoch 150 --seed 2024 \
    --stratified --fixed_locality --fg_object_size 3 --dense_sample_epoch 150 --n_feat_layers 1 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/room_ABO_multiple/1224-2-7obj/load-extrinsicDepth' \
    --load_encoder 'load_freeze' --load_slotattention 'load_freeze' --load_decoder 'load_train' \
    --bg_density_loss --depth_supervision --depth_in 0 \
    --camera_normalize --camera_modulation --bg_rotate --scaled_depth --depth_scale 12.2 --fixed_dist 12.2 \
    --vis_mask \
    --exp_id '1228-debut/7obj-load-single-table' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

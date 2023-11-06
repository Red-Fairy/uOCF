#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:3090:1

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
DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/chairs/train-1obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1296 --n_img_each_scene 3 \
    --checkpoints_dir 'checkpoints' --name 'room_real_chairs' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 5 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uocf_dual_trans' \
    --attn_decay_steps 100000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 2 --attn_iter 6 --shape_dim 48 --color_dim 16 \
    --coarse_epoch 200 --niter 200 --percept_in 20 --no_locality_epoch 40 --seed 2025 \
    --stratified --fixed_locality --fg_object_size 3 --n_feat_layers 4 \
    --exp_id '1105/1obj-dual' \
    --dummy_info 'DINO from scratch 1 obj with BG and position loss (150 epoch), dense sampling at 50' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=16
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
DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/plant_pots/train-white-1obj-nofoot-viewrange-large-trans-4050'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 970 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50  --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uorf_general_mask' \
    --attn_decay_steps 200000 \
    --num_slots 1 --attn_iter 3 \
    --shape_dim 72 --color_dim 24 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --coarse_epoch 100 --niter 100 --percept_in 25 --no_locality_epoch 50 --centered \
    --world_obj_scale 3 --obj_scale 3 --fixed_locality --seed 2025 \
    --exp_id '0719/1obj-mask-fixed-4050-r2' \
    --dummy_info '1obj mask' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

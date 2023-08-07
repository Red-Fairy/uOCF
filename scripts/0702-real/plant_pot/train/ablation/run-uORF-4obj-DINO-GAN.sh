#!/bin/bash

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
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

DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/plant_pots/train-white-4obj-nofoot-viewrange-large-4050'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_with_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' \
    --display_port $PORT --display_ncols 4 --print_freq 50  --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --coarse_epoch 120 --no_locality_epoch 60 --z_dim 96 --num_slots 5 --near 6 --far 20 \
    --model 'uorf_gan_DINO' --bottom  --encoder_type 'DINO' --encoder_size 896 \
    --fixed_locality \
    --continue_train --epoch_count 20 \
    --exp_id 'uORF-4obj-fixed-GAN-DINO' \
# done
echo "Done"

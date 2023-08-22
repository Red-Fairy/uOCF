#!/bin/bash

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a5000:1

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

DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room-real/chairs/train-4obj'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'room_real_chairs' \
    --display_port $PORT --display_ncols 4 --print_freq 50 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --coarse_epoch 120 --no_locality_epoch 60 --z_dim 96 --num_slots 5 --near 8 --far 18 \
    --model 'uorf_nogan' --bottom \
    --exp_id 'ablation/uORF-4obj-96' \
# done
echo "Done"

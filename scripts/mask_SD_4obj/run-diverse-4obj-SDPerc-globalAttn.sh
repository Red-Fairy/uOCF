#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_nobg/train-1obj-manysize-trans-orange'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_bg_SD' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 \
    --load_size 128 --n_samp 72 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_SD' --dataset_mode 'multiscenesSD' \
    --num_slots 4 --attn_iter 4 --z_dim 64 --encoder_size 256 \
    --project \
    --coarse_epoch 60 --niter 120 --percept_in 10 \
    --attn_decay_steps 150000 --save_epoch_freq 2 --use_SD_percept \
    --exp_id '0513-bg-SD/attn-scratch-SDPercept' \
    --dummy_info 'slot attention, train from scratch' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

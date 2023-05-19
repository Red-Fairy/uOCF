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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_bg/train-4obj-manysize-orange'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4 --save_epoch_freq 5 \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_bg_SAM' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --seed 123123123 \
    --load_size 256 --n_samp 64 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_sam_new' --dataset_mode 'multiscenes' \
    --attn_decay_steps 200000 --project --bottom \
    --num_slots 5 --attn_iter 4 --shape_dim 48 --color_dim 16 --encoder_size 1024 \
    --coarse_epoch 80 --niter 160 --percept_in 10 --locality_in 10 --locality_full 30 --surface_loss --surface_in 10 \
    --exp_id '0517-bg-SAM/scratch' \
    --dummy_info 'slot attention, train from scratch, locality decay' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

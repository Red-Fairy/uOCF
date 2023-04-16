#!/bin/bash
#SBATCH --account=viscam --partition=viscam --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:3090:1

#SBATCH --job-name="T_uORF"
#SBATCH --output=logs/T_uORF_clevr567_%j.out

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
DATAROOT=${1:-'/viscam/u/redfairy/datasets/room_chair/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_chair' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --num_slots 5 \
    --coarse_epoch 400 --niter 800 \
    --model 'uorf_nogan_T_sam' \
    --exp_id '0416-sam-texture' \
    --z_dim 56 --texture_dim 8 \
    --sam_encoder --encoder_size 1024 \
    --save_epoch_freq 25 \
    --attn_iter 4 \
    --project \
    --seed 2023 \
    --dummy_info 'sam v0 56+8' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

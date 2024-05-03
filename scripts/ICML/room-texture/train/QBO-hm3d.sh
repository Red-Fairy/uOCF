#!/bin/bash

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

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

DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/uorf_real_dataset'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=0 python train_without_gan.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 1 --no_shuffle  \
    --checkpoints_dir 'checkpoints' --name 'ICML' \
    --display_port $PORT --display_ncols 4 --print_freq 50 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --coarse_epoch 5000 --niter 5000 --no_locality_epoch 0 --z_dim 96 --num_slots 5 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/room_real_chairs/ablation/uORF-4obj-GAN-QBO-inPaperResults' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' \
    --near 1.97 --far 6.56 --nss_scale 2.30 \
    --model 'uorf_nogan' --bottom --learnable_slot_init --fixed_locality \
    --exp_id 'room-texture/QBO-real-rebuttal' \
# done
echo "Done"

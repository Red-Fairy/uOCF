#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

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

# sample process (list hostnames of the nodes you've requested)
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/clevr_bg/train-1obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'clevr_bg' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --bottom \
    --model 'uorf_general' \
    --attn_decay_steps 100000 --freezeInit_ratio 1 --freezeInit_steps 50000 \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 2 --attn_iter 4 --shape_dim 24 --color_dim 8 \
    --coarse_epoch 500 --niter 500 --percept_in 50 --no_locality_epoch 100 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/clevr_bg/1obj-mask' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_freeze' \
    --exp_id '1obj-bg-load-DINO' \
    --dummy_info 'from scratch locality 50epoch' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

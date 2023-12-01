#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a40:1

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/CLEVR/stack'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'CLEVR-stack' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uorf_general_IPE' \
    --attn_decay_steps 50000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 8 --attn_iter 4 --shape_dim 36 --color_dim 12 \
    --coarse_epoch 600 --niter 1200 --percept_in 20 --no_locality_epoch 400 --seed 2027 \
    --learnable_slot_init \
    --stratified --fixed_locality --fg_object_size 3 --dense_sample_epoch 400 --n_dense_samp 256 \
    --exp_id '1130/5-7obj-stack-v2' \
    --dummy_info 'DINO from scratch 1 obj with BG, position loss in the nss space, weight position = 0.1, strat from 100' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

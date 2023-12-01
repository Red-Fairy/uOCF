#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

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

# sample process (list hostnames of the nodes you've requested)
DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/chairs/train-2-4obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'rebuttal' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 5 \
    --load_size 128 --n_samp 64 --input_size 128 --frustum_size_fine 128 \
    --supervision_size 64 --frustum_size 64 \
    --model 'uorf_general_IPE' \
    --attn_decay_steps 100000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 6 --attn_iter 4 --shape_dim 72 --color_dim 24 \
    --freezeInit_steps 50000 \
    --coarse_epoch 8 --niter 16 --percept_in 2 --no_locality_epoch 0 --seed 2025 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/room_real_chairs/0824/4obj-load-IPE-r4' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_freeze' \
    --stratified --fixed_locality --fg_object_size 3 --dense_sample_epoch 2 --n_dense_samp 256 --one2four \
    --learnable_slot_init \
    --continue_train --epoch_count 2 \
    --exp_id '5slot/chairs' \
    --dummy_info 'DINO from scratch 1 obj with BG and position loss (156 epoch), dense sampling at 50' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

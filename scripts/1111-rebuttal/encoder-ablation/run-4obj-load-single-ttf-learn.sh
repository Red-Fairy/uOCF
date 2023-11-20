#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

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
DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/chairs/train-4obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'rebuttal' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 5 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uocf_single' \
    --attn_decay_steps 100000 --freezeInit_steps 20000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 4 --shape_dim 64 --color_dim 0 \
    --coarse_epoch 40 --niter 80 --percept_in 10 --no_locality_epoch 0 --seed 2025 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/rebuttal/encoder/1obj-single' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_freeze' \
    --stratified --fixed_locality --fg_object_size 3 --dense_sample_epoch 10 --one2four --n_feat_layer 1 \
    --learnable_slot_init \
    --continue_train --epoch_count 40 --n_dense_samp 224 \
    --exp_id 'encoder/4obj-load-single-ttf-learn' \
    --dummy_info 'DINO from scratch 1 obj with BG and position loss (120 epoch), dense sampling at 50' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

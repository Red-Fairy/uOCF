#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
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
DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/plant_pots/train-white-4obj-nofoot-viewrange-large-4550'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 3 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 20 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uorf_general' \
    --attn_decay_steps 100000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 6 --attn_iter 4 --shape_dim 96 --color_dim 32 --near 6 --far 20 \
    --freezeInit_steps 100000 \
    --coarse_epoch 80 --niter 160 --percept_in 20 --no_locality_epoch 0 --seed 2023 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/legacy/1obj-scratch-nofoot-fixed-large-range4055-r2' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_freeze' \
    --fixed_locality \
    --exp_id '0726-checkout/4obj-loadScratch-noloc-ttf-6slot' \
    --dummy_info 'DINO load from 1 obj scratch, 300 epoch' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

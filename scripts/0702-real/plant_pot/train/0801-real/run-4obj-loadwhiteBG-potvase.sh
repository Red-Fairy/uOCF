#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G

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
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/4obj-potvase-bg_multiview'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 470 --n_img_each_scene 3 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 20 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uorf_general' \
    --attn_decay_steps 5000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 6 --attn_iter 4 --shape_dim 48 --color_dim 48 --near 6 --far 20 \
    --freezeInit_steps 100000 --lr 0.0003 \
    --coarse_epoch 600 --niter 1200 --percept_in 250 --no_locality_epoch 0 --seed 2023 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/0731-real/4obj-load4obj-CIT-pot-vase' \
    --load_encoder 'load_freeze' --load_slotattention 'load_freeze' --load_decoder 'load_freeze' \
    --fixed_locality --color_in_attn --freeze_fg_only \
    --exp_id '0801-real/4obj-loadwhiteBG-CIT-freezeFG-potvase-grey' \
    --dummy_info 'DINO load from 4 obj real white BG, 700 ep' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

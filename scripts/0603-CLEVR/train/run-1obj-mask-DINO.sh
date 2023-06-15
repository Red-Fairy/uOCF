#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanrtx:1

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/clevr_nobg/train-1obj-centered'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 8  \
    --checkpoints_dir 'checkpoints' --name 'DEBUG' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 \
    --load_size 128 --n_samp 16 --input_size 128 --supervision_size 10 --frustum_size 10 \
    --model 'uorf_general_mask' \
    --num_slots 1 --attn_iter 3 \
    --shape_dim 24 --color_dim 8 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --coarse_epoch 200 --niter 200 --percept_in 40 --no_locality_epoch 60 --centered --nss_scale 1 \
    --world_obj_scale 3 --obj_scale 3 \
    --attn_decay_steps 100000 \
    --bg_color '-1' \
    --exp_id '1obj-mask' \
    --save_epoch_freq 5 \
    --dummy_info 'scale 3, near 6, far 20' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

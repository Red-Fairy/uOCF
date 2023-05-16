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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_multiple_nobg/train-1obj-each'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1500 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_multiple_mask' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 5 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uorf_nogan_T_SD_fgmask' --dataset_mode 'multiscenesSD' \
    --num_slots 1 --attn_iter 3 --z_dim 64 --encoder_size 256 \
    --project \
    --coarse_epoch 600 --niter 600 --percept_in 50 \
    --attn_decay_steps 100000 \
    --transparent  --mask_in 0 --bg_color -1 --surface_loss \
    --exp_id '0515-SD/maskfg-1obj-globalattn-each-movingFG-surface' \
    --dummy_info 'mask fg scale-4, light, moving FG position, global attention' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

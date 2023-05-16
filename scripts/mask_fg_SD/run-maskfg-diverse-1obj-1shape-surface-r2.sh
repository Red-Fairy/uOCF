#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_nobg/train-1obj-1shape'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 200 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_mask' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uorf_nogan_T_SD_fgmask' --dataset_mode 'multiscenesSD' \
    --num_slots 1 --attn_iter 3 --z_dim 64 --encoder_size 256 \
    --project \
    --coarse_epoch 600 --niter 600 --percept_in 50 \
    --attn_decay_steps 100000 --save_epoch_freq 10 \
    --transparent  --mask_in 0 --bg_color -1 --surface_loss --obj_scale 3 \
    --exp_id '0516-SD/maskfg-1obj-1shape-obj3' \
    --dummy_info 'mask fg scale=3, moving FG position, global attention, locality out' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

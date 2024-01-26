#!/bin/bash

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

DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/hm3d_img'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'ICML' \
    --display_port $PORT --display_ncols 4  \
    --input_size 128 --load_size 128 --n_samp 256 --render_size 32 --frustum_size 128 \
    --model 'uorf_eval' --bottom \
    --z_dim 96 --num_slots 5 \
    --near 1.97 --far 6.56 --nss_scale 2.30 \
    --vis_attn --recon_only --fixed_locality --learnable_slot_init \
    --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/ICML/room-texture/QBO-hm3d' \
# done
echo "Done"

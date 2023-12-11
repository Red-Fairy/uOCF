#!/bin/bash

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G

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

DATAROOT=${1:-'/svl/u/redfairy/datasets/OSTScene/train-A-4-new'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 3000 --n_img_each_scene 2  \
    --checkpoints_dir 'checkpoints' --name 'OCTScenes' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --save_epoch_freq 10000 \
    --load_size 128 --n_samp 4 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --coarse_epoch 50 --niter 100 --z_dim 96 --bottom \
    --no_locality_epoch 40 --percept_in 20 \
    --num_slots 7 --fixed_locality \
    --near 1.0 --far 10.0 --nss_scale 2.5 --obj_scale 0 \
    --model 'uorf_OCT' --diff_intrinsic \
    --exp_id 'uORT/6obj' \
# done
echo "Done"

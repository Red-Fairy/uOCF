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

DATAROOT=${1:-'/svl/u/redfairy/datasets/room-real/plant_pots/test-4obj-large-4050'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --bottom \
    --n_samp 256 --z_dim 96 --num_slots 5 \
    --model 'uorf_eval' \
    --fixed_locality --near_plane 6 --far_plane 20 \
    --pos_emb --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/ablation/uORF-4obj-GAN-fixed-new' \
    --attn_iter 3 --testset_name 'regular_test_218'  \
# done
echo "Done"
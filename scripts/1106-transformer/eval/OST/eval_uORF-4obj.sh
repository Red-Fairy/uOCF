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

DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/OSTScene/test-1scene-8view'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 8 \
    --checkpoints_dir 'checkpoints' --name '' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 256 --render_size 8 --frustum_size 128 --bottom \
    --num_slots 2 --fixed_locality \
    --attn_iter 3 --z_dim 64 \
    --near 1.0 --far 10.0 --nss_scale 2.5 --obj_scale 0 \
    --model 'uorf_eval' --recon_only \
    --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/OSTScenes/uORF-1obj-3' \
   --testset_name 'regular_test_fg' --epoch 40000 \
# done
echo "Done"
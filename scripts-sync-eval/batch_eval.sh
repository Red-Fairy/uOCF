#!/bin/bash
#SBATCH --account=viscam --partition=viscam --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:2080ti:1

#SBATCH --job-name="T_uORF"
#SBATCH --output=logs/eval_%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_multiple_bg/test_ood_2.4'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_multiple' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 \
    --n_samp 256 --num_slots 5 \
    --model 'uorf_eval_T_sam' \
    --sam_encoder --encoder_size 1024 \
    --exp_id '0417-sam-texture-bg-48-16-4view' \
    --z_dim 48 --texture_dim 16 --bottom \
    --project --attn_iter 4 --testset_name 'test_ood_2.4_179' \
# done
echo "Done"
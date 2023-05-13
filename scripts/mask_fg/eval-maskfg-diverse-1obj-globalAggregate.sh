#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=10
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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_nobg/train-1obj-manysize-trans-orange-test'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_mask' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 --n_samp 256 \
    --num_slots 1 --attn_iter 3 \
    --model 'uorf_eval_T_sam_fgmask' \
    --shape_dim 48 --color_dim 16 \
    --bottom \
    --sam_encoder --encoder_size 1024 --project \
    --bg_color '-1' \
    --exp_id '0508-SAM/maskfg-1obj-globalAggregate' \
    --feature_aggregate \
    --testset_name test_regular \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

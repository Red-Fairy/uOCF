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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room-real/chairs/test-4obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 5 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_real_chairs' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --n_samp 128 --input_size 128 --render_size 32 --frustum_size 128 \
    --model 'uorf_general_eval' \
    --num_slots 5 --attn_iter 4 \
    --shape_dim 72 --color_dim 24 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --world_obj_scale 4.5 --obj_scale 4.5 --near_plane 6 --far_plane 20 \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_chairs/0709/4obj-loadPos-ttf-r3' \
    --dummy_info 'regular test' --testset_name 'regular_160end_near6far20' --epoch 160 \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

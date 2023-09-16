#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
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
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-easy/4obj-all-test-0817'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot $DATAROOT --n_scenes 20 --start_scene_idx 80 --n_img_each_scene 2  \
    --checkpoints_dir 'checkpoints' --name 'kitchen-easy' \
    --display_port $PORT --display_ncols 4 \
    --load_size 256 --n_samp 256 --input_size 128 --render_size 32 --frustum_size 256 \
    --model 'uorf_general_eval' \
    --num_slots 5 --attn_iter 4 \
    --shape_dim 72 --color_dim 24 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --world_obj_scale 4.5 --obj_scale 4.5 --near_plane 6 --far_plane 20 \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/kitchen-easy/dataset-0817-new/4obj-loadchairs-fine256' \
    --fixed_locality --recon_only --no_shuffle --fg_object_size 3 --show_recon_stats \
    --nss_scale 7 --epoch 1000 --vis_disparity \
    --dummy_info 'test_real' --testset_name 'test4obj_cabinet_load256' \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

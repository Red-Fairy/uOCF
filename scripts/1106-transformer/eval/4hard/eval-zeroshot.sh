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
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-train-10-largeFOV'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot $DATAROOT --n_scenes 1 --start_scene_idx 0 --n_img_each_scene 2  \
    --checkpoints_dir 'checkpoints' --name '' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --n_samp 256 --input_size 128 --render_size 32 --frustum_size 128 \
    --model 'uocf_dual_DINO_trans_eval' --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 8 --attn_iter 6 --shape_dim 48 --color_dim 48 \
    --fixed_locality --fg_object_size 3 --n_feat_layers 1 \
    --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/kitchen-hard/1217-loadOCT/zeroshot' \
    --attn_dropout 0 --attn_momentum 0.5 --pos_init 'zero' \
    --camera_modulation --camera_normalize --scaled_depth --depth_scale 12.2 --bg_rotate \
    --vis_attn --vis_render_mask --recon_only \
    --dummy_info '' --testset_name 'test' \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

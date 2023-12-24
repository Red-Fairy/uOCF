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
DATAROOT=${1:-'/svl/u/redfairy/datasets/OCTScene/test-A-img4-res256'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot $DATAROOT --n_scenes 100 --start_scene_idx 3100 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name '' \
    --display_port $PORT --display_ncols 4 \
    --model 'uocf_dual_DINO_OCT_eval' --diff_intrinsic \
    --load_size 128 --n_samp 256 --render_size 32 --frustum_size 128 \
    --num_slots 8 --attn_iter 6 --shape_dim 48 --color_dim 48 --n_feat_layers 1  \
    --near 1.0 --far 10.0 --nss_scale 2.5 --fg_object_size 1 --obj_scale 2.5 \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/OCTScenes/1212-modNorm/load-default-maskDepth' \
    --fixed_locality \
    --camera_normalize --camera_modulation --bg_rotate --scaled_depth --depth_scale 4 \
    --attn_dropout 0 --attn_momentum 0.5 --pos_init 'zero' \
    --vis_attn --vis_mask --no_shuffle --show_recon_stats \
    --dummy_info 'test_real' --testset_name 'regular_test' \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

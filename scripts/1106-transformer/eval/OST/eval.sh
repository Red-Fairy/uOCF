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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/OSTScene/test-1scene'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=0 python test.py --dataroot $DATAROOT --n_scenes 5 --start_scene_idx 0 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'kitchen-easy' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --n_samp 256 --input_size 128 --render_size 32 --frustum_size 128 \
    --model 'uocf_dual_trans_eval' \
    --num_slots 5 --attn_iter 6 \
    --shape_dim 72 --color_dim 24 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --n_feat_layers 4 \
    --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/OSTScenes/1202/debug' \
    --fixed_locality --recon_only --no_shuffle --show_recon_stats \
    --attn_dropout 0 --attn_momentum 0.5 --random_init_pos \
    --near 2.0 --far 12.0 --nss_scale 4.0 --fg_object_size 1.5 --obj_scale 2.5 \
    --load_intrinsics --scaled_depth --depth_scale 4.0 --vis_attn \
    --epoch 1000 \
    --dummy_info 'test_real' --testset_name 'debug_load128' \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

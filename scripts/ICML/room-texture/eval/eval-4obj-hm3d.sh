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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/hm3d_img'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python -m visdom.server -p $PORT &>/dev/null &
python test.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 2  \
    --checkpoints_dir 'checkpoints' --name '' \
    --display_port $PORT --display_ncols 4 \
    --input_size 128 --load_size 128 --n_samp 256 --render_size 32 --frustum_size 128 \
    --model 'uocf_dual_DINO_trans_eval' --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 6 --shape_dim 48 --color_dim 48 \
    --fixed_locality --fg_object_size 1 --n_feat_layers 1 \
    --near 1.97 --far 6.56 --nss_scale 2.30 \
    --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/ICML/room-texture/4obj-hm3d' \
    --recon_only --vis_render_mask --vis_attn --epoch 1000 \
    --remove_duplicate \
    --dummy_info 'regular test' --testset_name 'hm3d' \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

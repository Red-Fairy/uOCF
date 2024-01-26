#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a40:1

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
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-all-train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 324 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'ICML' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 10 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 --frustum_size_fine 128 \
    --model 'uocf_dual_DINO_trans' \
    --attn_decay_steps 50000 --lr 0.00002 --freezeInit_steps 30000 \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 6 --shape_dim 48 --color_dim 48 \
    --coarse_epoch 160 --niter 500 --percept_in 0 --no_locality_epoch 0 --seed 2024 \
    --stratified --fixed_locality --fg_object_size 3 --dense_sample_epoch 0 --n_feat_layers 1 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/ICML/kitchen-hard/4obj-load' \
    --load_epoch 500 --load_encoder 'load_freeze' --load_slotattention 'load_freeze' --load_decoder 'load_train' \
    --bg_density_loss --depth_supervision \
    --remove_duplicate --remove_duplicate_in 0 \
    --vis_mask \
    --scaled_depth --depth_scale_pred --depth_scale_param 2 \
    --continue_train --epoch 180 --epoch_count 181 \
    --exp_id 'kitchen-hard/4obj-load-noPlane-freeze' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

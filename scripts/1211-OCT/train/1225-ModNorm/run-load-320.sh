#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a6000:1

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
DATAROOT=${1:-'/svl/u/redfairy/datasets/OCTScene/train-A-img60-res320-resize256'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 3000 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'OCTScenes' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 2 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uocf_dual_DINO_trans' --load_intrinsics \
    --attn_decay_steps 100000 --bottom --lr 0.0003 \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 8 --attn_iter 6 --shape_dim 48 --color_dim 48 --n_feat_layers 1  \
    --coarse_epoch 100 --niter 200 --percept_in 10 --no_locality_epoch 30 --seed 2027 \
    --stratified --fixed_locality --dense_sample_epoch 30 \
    --near 2.0 --far 10.0 --nss_scale 2.5 --fg_object_size 1.25 --obj_scale 2.5 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/ICML/room-multiple/uOCF-load-2-4obj-r2' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' \
    --attn_dropout 0 --attn_momentum 0.5 --pos_init 'zero' --one2four \
    --bg_density_loss --bg_penalize_plane 4.0 --depth_supervision --weight_depth_ranking 3 \
    --camera_normalize --camera_modulation --bg_rotate \
    --scaled_depth --depth_scale 4 --fixed_dist 4 \
    --remove_duplicate --remove_duplicate_in 10 \
    --exp_id '1225-modNorm/load-fixedDist-320' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

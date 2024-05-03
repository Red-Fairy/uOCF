#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a5000:1

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/ABO-multiple/train-2-4obj-4050'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'ICML' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 20 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uocf_dual_DINO_trans' \
    --attn_decay_steps 100000 --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 6 --shape_dim 48 --color_dim 48 \
    --coarse_epoch 40 --niter 80 --percept_in 5 --no_locality_epoch 20 --seed 2027 \
    --stratified --fixed_locality --fg_object_size 3 --dense_sample_epoch 20 --n_feat_layers 1 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/uOCF/checkpoints/ICML/room-multiple/uOCF-1obj' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' --one2four \
    --bg_density_loss \
    --remove_duplicate --remove_duplicate_in 10 \
    --attn_dropout 0 --attn_momentum 0.5 --pos_init 'zero' \
    --exp_id 'room-multiple-objverse/uOCF-4obj-v3-r3' \
    --continue_train --epoch_count 11 \
    --dummy_info '' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

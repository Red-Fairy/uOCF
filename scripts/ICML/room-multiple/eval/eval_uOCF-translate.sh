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

DATAROOT=${1:-'/svl/u/redfairy/datasets/ABO-multiple/test-4obj-move'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=1 python test.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_real_chairs' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --model 'uocf_dual_DINO_trans_manip' --dataset_mode 'multiscenes_manip' --bottom \
    --load_size 128 --input_size 128 --render_size 32 --frustum_size 128 --n_samp 256 \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 6 --shape_dim 48 --color_dim 48 \
    --fixed_locality --fg_object_size 3 --n_feat_layers 1 \
    --attn_dropout 0 --attn_momentum 0.5 --pos_init 'zero' \
    --remove_duplicate \
    --manipulate_mode 'translation' --epoch 160  \
    --pos_emb --exp_id '/viscam/projects/uorf-extension/uOCF/checkpoints/ICML/room-multiple/uOCF-load-2-4obj' \
    --testset_name 'manip_translatation'  \

echo "Done"

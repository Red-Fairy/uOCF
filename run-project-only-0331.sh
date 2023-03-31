#!/bin/bash
#SBATCH --account=viscam --partition=viscam --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:3090:1

#SBATCH --job-name="T_uORF"
#SBATCH --output=logs/T_uORF_clevr567_%j.out

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
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --z_dim 40 --num_slots 8 \
    --model 'uorf_nogan_T' \
    --exp_id '0331-project-only-loadEncoder' --attn_iter 4 \
    --project --relative_position \
    --pos_emb --emb_path /viscam/u/redfairy/I-uORF/checkpoints/clevr_567/uORF-pretrained/latest_net_Encoder.pth \
    --lr 2e-4 --lr_encoder 2e-5 \
    --niter 800 --coarse_epoch 400 \
    --dummy_info 'share grid embed projection, correct deduct operation (before azi transform), add decoder MLP to z-slots projection (w/o residual), 4 round attn, pos embedding on encoder (+4), load pretrained encoder with lr *= 0.1' \

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

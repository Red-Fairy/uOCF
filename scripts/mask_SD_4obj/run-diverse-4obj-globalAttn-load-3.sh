#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_bg/train-4obj-manysize-orange'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_bg_SD' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 2 \
    --load_size 128 --n_samp 72 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_SD' --dataset_mode 'multiscenesSD' \
    --attn_decay_steps 150000 --freezeInit_ratio 0.2 \
    --num_slots 5 --attn_iter 4 --z_dim 64 --encoder_size 256 \
    --coarse_epoch 60 --niter 120 \
    --project --percept_in 10 --surface_loss --surface_in 10 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_diverse_mask/0513-SD/maskfg-1obj-globalattn-each-movingFG-surface-debug3' \
    --exp_id '0515-bg-SD/attn-scratch-4obj-loadDecoder' --only_decoder \
    --dummy_info 'slot attention, train from scratch, light to_res_fg and to_res_bg, scale=4, post update fg pos, only load decoder, freeze init ratio 0.2' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"

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

#!/bin/bash
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-easy/4obj-all-test-0817'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=1 python test-video-manipulate.py --dataroot $DATAROOT \
    --n_scenes 107 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 256 --input_size 128 --render_size 32 --frustum_size 256 \
    --n_samp 256 --num_slots 5 \
    --model 'uorf_general_eval' \
    --encoder_size 896 --encoder_type 'DINO' --bottom --near 6 --far 20 \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/kitchen-easy/dataset-0817-new/4obj-loadchairs-ttt-box-stratified' \
    --shape_dim 72 --color_dim 24 --fixed_locality --video_mode 'spiral' --fg_object_size 3 \
    --wanted_indices '95' --epoch 720 \
    --attn_iter 4 --no_loss --recon_only --video --testset_name test_manipulation_256 \
# done
echo "Done"
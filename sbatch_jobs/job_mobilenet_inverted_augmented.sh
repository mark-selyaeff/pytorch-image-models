#!/bin/bash
#SBATCH --job-name=inverted_images
#SBATCH --account=Project_2004728
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:500
#SBATCH --cpus-per-task=12
#SBATCH -o mobilenet_inverted_waugs.out
##SBATCH --mail-type=BEGIN #uncomment to enable mail

# module load python-data
source /scratch/project_2004728/envs/adv_env/bin/activate
cp /scratch/project_2004728/imagenet_files.tar $LOCAL_SCRATCH
echo "Copying done;"
cd $LOCAL_SCRATCH
ls -la --block-size=G
tar xf imagenet_files.tar
rm imagenet_files.tar
echo "UNZIPPING DONE;"


IMAGENET_FOLDER="$LOCAL_SCRATCH/imagenet_files"

cd /scratch/project_2004728/pytorch-image-models

srun python train.py $IMAGENET_FOLDER --model mobilenetv3_large_100 -b 1024 --sched step --epochs 300 --decay-epochs 2.4 --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 12 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --aa rand-m9-mstd0.5 --remode pixel --reprob 0.2 --native-amp --lr .064 --experiment mobilenet_inverted_waugs --invert-images --no-prefetcher --log-wandb --lr-noise 0.42 0.9

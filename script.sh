#!/bin/bash -l

echo "running script"
#$ -P synthdata       # Specify the SCC project name you want to use
#$ -N secondjob  # Project name.  unique every time 
#$ -l h_rt=24:00:00   # Specify the hard time limit for the job
#$ -o std_out_26 # standard out file
#$ -e err_26 # error file
#$ -l gpu_type=Titan
#$ -l gpus=2
#$ -pe omp 4
#$ -V
#$ -m b

# https://www.bu.edu/tech/support/research/system-usage/running-jobs/submitting-jobs/
# setup modules 

cd /projectnb/synthdata/tejas/Image_To_Video
# source scc_setup_26.sh

module load python3/3.10.12
# pip install torch==2.1.0 --target /projectnb/textconv/dgordon/packages --upgrade
module load gcc/8.3.0
module load cuda/11.6
source venv/bin/activate

# export PATH="$PATH:/usr4/dl523/dgordon/.local/bin"
# echo "PATH is now set to: $PATH"

# run job

python train.py --dataset-root hmdb51 --annotation-path annotations -t epoch=0-step=221.ckpt --batch-size 8

deactivate
# some other useful options:
# optimizer.lr=_
# loader.batch_size=_
# trainer.max_epochs=_ you need to increase this in tandem with the checkpoint.
# train.ckpt=

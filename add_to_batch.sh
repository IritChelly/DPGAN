#!/bin/bash

### sbatch config parameters must start with #SBATCH, to ignore just add another # - like ##SBATCH

#SBATCH --partition short					### specify partition name where to run a job

##SBATCH --time 0-3:00:00						### limit the time of job running, partition limit can override this

#SBATCH --job-name python_test						### name of the job
#SBATCH --output /home/tohamy/BNP/GAN_DP/code_18_E_on_G_dpgan/log_test.log			### output log for running job - %J for job number
##SBATCH --mail-user=tohamy@post.bgu.ac.il				### users email for sending job status
#SBATCH --mail-type=ALL							### conditions when to send the email

#SBATCH --gres=gpus:1							### number of GPUs, ask for more than 1 only if you can parallelize your code for multi GPU

### Start you code below ####

module load anaconda							### load anaconda module
conda activate condirit							### activating environment, environment must be configured before running the job (conda)
python train.py configs/cifar/selfcondgan.yaml	### execute jupyter lab command

#!/bin/bash -l
#SBATCH --account=intro_vsc37115 
#SBATCH --cluster=wice
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=18
#SBATCH --gpus-per-node=1
#SBATCH --time=16:00:00
#SBATCH --job-name=weakSegmentation
#SBATCH --output=%x.log
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=wout.seynaeve1@student.kuleuven.be

module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 

echo Job starting at $(date "+%Y.%m.%d-%H:%M:%S")
echo Job is running on $(hostname)
echo Job is running on ${SLURM_CLUSTER_NAME} cluster
echo The node architecture is $VSC_ARCH_LOCAL
echo Current path is $(pwd)


python trainWeaklySupervised.py -l 1e-9

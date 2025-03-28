#!/bin/bash -l
#SBATCH --account=intro_vsc37115 
#SBATCH --cluster=wice
#SBATCH --partition=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=8
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


python trainWeaklySupervisedCLEVR.py --configuration 0 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 1 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 2 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 3 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 4 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 5 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 6 -e 160
python trainWeaklySupervisedCLEVR.py --configuration 7 -e 160





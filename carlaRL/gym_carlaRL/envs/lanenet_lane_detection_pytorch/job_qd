#!/bin/bash

#BSUB -J debug_lanenet
#BSUB -o my_job.%J.out
#BSUB -N
#BSUB -R "rusage[mem=40]"

module load python/3.7

virtualenv myenv --python=python3.7
source myenv/bin/activate

pip install -r requirements.txt

python train.py

deactivate
#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/Dist
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/Dist/configs" 

for i in 5
do
	for j in {2..20}
	do
		$PYTHON -m differential_privacy --config=$CONFIG_FOLDER/Bot/ffnn/${i}nodes.json
		mv ./results/Bot/ffnn/${i}nodes/dist_${i}nodes.csv ./results/Bot/ffnn/${i}nodes/dist_${i}nodes_it${j}.csv
	done
done

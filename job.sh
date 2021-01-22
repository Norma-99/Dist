#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/Dist
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/Dist/configs" 

for i in 3 5 7 9
do
	for j in {1..50}
	do
		$PYTHON -m differential_privacy --config=$CONFIG_FOLDER/${i}nodes.json
		mv ./results/${i}nodes/dist_${i}nodes.csv ./results/${i}nodes/dist_${i}nodes_it${j}.csv
	done
done

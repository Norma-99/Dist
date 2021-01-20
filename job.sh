#!/bin/bash

#SBATCH --chdir=/scratch/nas/4/norma/Dist
#SBATCH --output=/scratch/nas/4/norma/.log/stdout-%j.out
#SBATCH --error=/scratch/nas/4/norma/.log/stderr-%j.out

PYTHON="/scratch/nas/4/norma/venv/bin/python"
CONFIG_FOLDER="/scratch/nas/4/norma/Dist/configs" 

for i in 1 3 5 7
do
	for j in 1
	do
		$PYTHON -m fog_embedded --config=$CONFIG_FOLDER/${i}nodes.json
		mv ./results/${i}nodes/dist_${i}nodes.csv ./results/${i}nodes/dist_${i}nodes_it${j}.csv
	done
done

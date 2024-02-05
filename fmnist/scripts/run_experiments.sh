#!/bin/bash

for experiment_type in "performance" "attack" "fairness"; do
	LENGTH=$(python scripts/count_experiments.py $experiment_type)

	for i in $(seq $LENGTH); do
	    python "$experiment_type"_eval.py -i $i
	done
done

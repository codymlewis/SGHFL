#!/bin/bash


for dataset in "apartment" "solar_home"; do
    for experiment_type in "performance" "attack" "fairness"; do
        length=$(python scripts/count_experiments.py $experiment_type)

        for i in $(seq $length); do
            python main.py -d $dataset -i $i --$experiment_type
        done
    done
done

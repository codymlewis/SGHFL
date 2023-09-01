#!/bin/bash

experiment_type=$1

length=$(python scripts/count_experiments.py solar_home_$experiment_type)

for i in $(seq $length); do
    python solar_home_eval.py -i $i --$experiment_type
done
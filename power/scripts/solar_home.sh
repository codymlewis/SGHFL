#!/bin/bash

experiment_type=$1

length=$(python scripts/count_experiments.py solar_home/$experiment_type)

for i in $(seq $length); do
    python main.py -d solar_home -i $i --$experiment_type
done

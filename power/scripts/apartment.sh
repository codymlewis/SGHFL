#!/bin/bash

experiment_type=$1

length=$(python scripts/count_experiments.py apartment/$experiment_type)

for i in $(seq $length); do
    python main.py -d apartment -i $i --$experiment_type
done

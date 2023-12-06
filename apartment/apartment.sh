#!/bin/bash

experiment_type=$1

length=$(python scripts/count_experiments.py $experiment_type)

for i in $(seq $length); do
    python main.py -i $i --$experiment_type
done

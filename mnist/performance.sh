#!/bin/bash

LENGTH=$(python count_experiments.py performance)

for i in $(seq $LENGTH); do
    python performance_eval.py -i $i
done

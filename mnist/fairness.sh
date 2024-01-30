#!/bin/bash

LENGTH=$(python count_experiments.py fairness)

for i in $(seq $LENGTH); do
    python fairness_eval.py -i $i
done

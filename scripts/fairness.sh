#!/bin/bash

LENGTH=$(python scripts/count_experiments.py fairness)

for i in $(seq $LENGTH); do
    python fairness_eval.py -i $i
done
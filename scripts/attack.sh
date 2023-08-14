#!/bin/bash

LENGTH=$(python scripts/count_experiments.py attack)

for i in $(seq $LENGTH); do
    python attack_eval.py -i $i
done
#!/bin/bash

LENGTH=$(python scripts/count_experiments.py solar_home_performance)

for i in $(seq $LENGTH); do
    python solar_home_eval.py -i $i
done
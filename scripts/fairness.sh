#!/bin/bash

LENGTH=$(python -c "import json; f = open('configs/fairness.json', 'r'); print(len(json.load(f))); f.close()")

for i in $(seq $LENGTH); do
    python fairness_eval.py -i $i
done
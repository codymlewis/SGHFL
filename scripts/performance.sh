#!/bin/bash

LENGTH=$(python -c "import json; f = open('configs/performance.json', 'r'); print(len(json.load(f))); f.close()")

for i in $(seq $LENGTH); do
    python performance_eval.py -i $i
done
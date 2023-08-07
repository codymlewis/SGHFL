#!/bin/bash

for i in {1..4}; do
    python fairness_eval.py -i $i
done
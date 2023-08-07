#!/bin/bash

for i in {1..8}; do
    python fairness_eval.py -i $i
done
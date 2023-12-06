#!/bin/bash


run_many_seeds () {
    for ((i = 1; i <= $2; i++)); do
        $1 -s $i
    done
}

for extra in {'','--fl-middle-server-km'}' '{'','--intermediate-finetuning 5'}; do
    run_many_seeds "python main.py $extra" 10
done

for attack in "empty" "lie" "ipm"; do
    for aggregator in "fedavg" "median" "centre"; do
        run_many_seeds "python main.py --attack $attack --fl-server-aggregator $aggregator --num-middle-servers 0" 10
    done
done

for extra in {'','--fl-middle-server-aggregator topk'}' '{'','--fl-middle-server-km'}' '{'','--intermediate-finetuning 5'}; do
    run_many_seeds "python main.py --fairness $extra" 10
done

#!/bin/bash


run_many_seeds () {
    for ((i = 1; i <= $2; i++)); do
        $1 -s $i
    done
}

for extra in {'','--middle-server-km','--middle-server-fp'}' '{'','--intermediate-finetuning 5'}; do
    run_many_seeds "python main.py $extra" 10
done

for attack in "empty" "lie" "ipm"; do
    for aggregator in "fedavg" "median" "centre" "krum" "trimmed_mean"; do
        run_many_seeds "python main.py --attack $attack --server-aggregator $aggregator --num-middle-servers 0" 10
    done
done

for extra in {'','--middle-server-aggregator topk'}' '{'','--middle-server-km','--middle-server-fp','--middle-server-mrcs'}' '{'','--intermediate-finetuning 5'}; do
    run_many_seeds "python main.py --fairness $extra" 10
done

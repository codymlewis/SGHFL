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
        for sat in $(seq 0 0.1 1); do
            run_many_seeds "python main.py --attack $attack --server-aggregator $aggregator --pct-saturation $sat --pct-adversaries 1.0" 10
        done
        for adv in $(seq 0 0.1 1); do
            run_many_seeds "python main.py --attack $attack --server-aggregator $aggregator --pct-saturation 1.0 --pct-adversaries $adv" 10

            if [ $adv -gt 0.4 ]; then
                sat="$((0.5  / $adv))"
                run_many_seeds "python main.py --attack $attack --server-aggregator $aggregator --pct-saturation $sat --pct-adversaries $adv" 10
            fi
        done
    done
done

for extra in {'','--middle-server-aggregator topk'}' '{'','--middle-server-km','--middle-server-fp','--middle-server-mrcs'}' '{'','--intermediate-finetuning 5'}; do
    run_many_seeds "python main.py --fairness $extra" 10
done

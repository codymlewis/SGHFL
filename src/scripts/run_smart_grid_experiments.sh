#!/bin/bash


run_many_seeds () {
    for ((i = 1; i <= $2; i++)); do
        $1 -s $i
    done
}

for dataset in "l2rpn" "apartment" "solar_home"; do
    if [ $dataset == 'l2rpn' ]; then
        rounds="50"
    else
        rounds="10"
    fi
    
    for attack in "none" "empty" "lie" "ipm"; do
        for aggregator in "fedavg" "duttagupta" "li" "ssfgm"; do
            server_aggregator=$aggregator
            ms_aggregator=$aggregator
            if [ $aggregator == "duttagupta" ]; then
                ms_aggregator="fedavg"
            elif [ $aggregator == "li" ]; then
                server_aggregator="fedavg"
            fi

            for drop_point in 0.4 1.1; do
                if [ $attack == 'none' ]; then
                    run_many_seeds "python main.py --dataset $dataset --rounds $rounds --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point" 5
                else
                    run_many_seeds "python main.py --dataset $dataset --rounds $rounds --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 0.5 --pct-adversaries 1.0" 5
                    run_many_seeds "python main.py --dataset $dataset --rounds $rounds --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 1.0 --pct-adversaries 0.5" 5
                fi
            done
        done
    done
done

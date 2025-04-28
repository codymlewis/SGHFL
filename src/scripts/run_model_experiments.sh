#!/bin/bash


run_many_seeds () {
    for ((i = 1; i <= $2; i++)); do
        $1 -s $i
    done
}

for model in "FFN" "CNN" "LSTM" "Attention"; do
    rounds="50"

    for attack in "none" "empty" "lie" "ipm"; do
        for aggregator in "fedavg" "phocas:ssfgm"; do
            server_aggregator=$aggregator
            ms_aggregator=$aggregator

            if [ $aggregator == "phocas:ssfgm" ]; then
                server_aggregator="phocas"
                ms_aggregator="ssfgm"
            elif [ $aggregator == "phocas:lissfgm" ]; then
                server_aggregator="phocas"
                ms_aggregator="lissfgm"
            fi

            for drop_point in 0.4 1.1; do
                if [ $attack == 'none' ]; then
                    run_many_seeds "python main.py --model $model --dataset l2rpn --rounds $rounds --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point" 5
                else
                    run_many_seeds "python main.py  --model $model --dataset l2rpn --rounds $rounds --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 0.5 --pct-adversaries 1.0" 5
                    run_many_seeds "python main.py  --model $model --dataset l2rpn --rounds $rounds --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 1.0 --pct-adversaries 0.5" 5
                fi
            done
        done
    done
done

mv results/results.csv results/model_results.csv

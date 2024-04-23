#!/bin/bash


run_many_seeds () {
    for ((i = 1; i <= $2; i++)); do
        $1 -s $i
    done
}

for dataset in "l2rpn" "apartment" "solar_home"; do
    for attack in "none" "empty" "lie" "ipm"; do
        for server_aggregator in "fedavg" "median" "krum" "trimmed_mean" "phocas" "geomedian" "fedprox" "ssfgm" "space_sample_mean"; do
            for ms_aggregator in "fedavg" "topk" "geomedian" "kickback_momentum" "fedprox" "mrcs" "ssfgm" "space_sample_mean"; do
                for drop_point in 0.4 1.1; do
                    if [ $attack == 'none' ]; then
                        run_many_seeds "python main.py --dataset $dataset --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point" 5
                    else
                        run_many_seeds "python main.py --dataset $dataset --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 0.5 --pct-adversaries 1.0" 5
                        run_many_seeds "python main.py --dataset $dataset --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 1.0 --pct-adversaries 0.5" 5
                    fi
                done
            done
        done
    done
done

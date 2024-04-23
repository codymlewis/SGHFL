#!/bin/bash


run_many_seeds () {
    for ((i = 1; i <= $2; i++)); do
        $1 -s $i
    done
}

for dataset in "l2rpn" "apartment" "solar_home"; do
    for attack in "none" "empty" "lie" "ipm"; do
        for server_aggregator in "fedavg" "median" "topk" "krum" "trimmed_mean" "phocas" "geomedian" "kickback_momentum" "fedprox" "mrcs" "ssfgm" "space_sample_mean"; do
            for ms_aggregator in "fedavg" "median" "topk" "krum" "trimmed_mean" "phocas" "geomedian" "kickback_momentum" "fedprox" "mrcs" "ssfgm" "space_sample_mean"; do
                for drop_point in 0.4 0.8 1.1; do
                    if [ $attack == 'none' ]; then
                        run_many_seeds "python main.py --dataset $dataset --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point" 10
                    else
                        run_many_seeds "python main.py --dataset $dataset --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 0.5 --pct-adversaries 1.0" 10
                        run_many_seeds "python main.py --dataset $dataset --attack $attack --server-aggregator $server_aggregator --middle-server-aggregator $ms_aggregator --drop-point $drop_point --pct-saturation 1.0 --pct-adversaries 0.5" 10
                    fi
                done
            done
        done
    done
done

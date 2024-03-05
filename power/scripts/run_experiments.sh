#!/bin/bash


for dataset in "apartment" "solar_home"; do
    for experiment_type in "performance" "attack" "fairness"; do
        length=$(python scripts/count_experiments.py $experiment_type)

        for i in $(seq $length); do
            if [[ $experiment_type -eq "attack" ]]; then
                for sat in $(seq 0 0.1 1); do
                    python main.py -d $dataset -i $i --$experiment_type --pct-saturation $sat --pct-dc-adversaries 1.0
                done
                for dcadv in $(seq 0 0.1 1); do
                    python main.py -d $dataset -i $i --$experiment_type --pct-saturation 1.0 --pct-dc-adversaries $dcadv

                    if (( $(echo "$dcadv > 0.4" | bc -l) )); then
                        sat="$(echo 0.5 / $dcadv | bc -l)"
                        python main.py -d $dataset -i $i --$experiment_type --pct-saturation $sat --pct-dc-adversaries $dcadv
                    fi
                done
            else
                python main.py -d $dataset -i $i --$experiment_type
            fi
        done
    done
done

#!/usr/bin/env bash


# We state by evaluating the sensitivity of the parameters of the algorithm
for padversaries in $(seq 0.0 0.1 0.5); do
  # First we evaluate the sensitivity of r in mitigating LIE
  for r in 0.0001 0.001 0.01 0.1 0.2 0.3; do
    python sensitivity.py --r "$r" --attack lie --padversaries "$padversaries"
  done

  # Next, we evaluate c against number of shifted random adversaries
  for c in $(seq 0.1 0.1 0.9); do
    python sensitivity.py --c "$c" --padversaries "$padversaries" --attack shifted_random
  done
done

# Then we ablate the overall algorithm
for attack in "no_attack" "lie" "shifted_random"; do
  for flags in {'','--space-sampling'}' '{'','--fractional-geomedian'}; do
    for padversaries in $(seq 0.0 0.1 0.5); do
      python ablation.py --attack "$attack" --padversaries "$padversaries" $flags
    done
  done
done

# Then we conclude by comparing topomean to other aggregation rules
for aggregator in "mean" "median" "geomedian" "krum" "trmean" "phocas" "ssfgm"; do
  for padversaries in $(seq 0.0 0.05 0.5); do
    python comparison.py --aggregator "$aggregator" --padversaries "$padversaries"
  done
done

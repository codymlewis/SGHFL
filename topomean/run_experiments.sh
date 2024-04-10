#!/usr/bin/env bash


# We state by evaluating the sensitivity of the parameters of the algorithm
for padversaries in $(seq 0.0 0.1 0.5); do
  # First we evaluate the sensitivity of e1 in mitigating LIE
  for e1 in 0.0001 0.001 0.01 0.1 0.2 0.3; do
    python sensitivity.py --e1 "$e1" --attack lie --padversaries "$padversaries"
  done

  # Next we evaluate the relationship between e2 and shifted random adversaries
  for e2 in 0.001 0.01 0.05 0.1; do
    python sensitivity.py --e2 "$e2" --attack shifted_random --padversaries "$padversaries"
  done

  # Next, we evaluate c against number of shifted random adversaries
  for c in $(seq 0.1 0.1 0.9); do
    python sensitivity.py --c "$c" --padversaries "$padversaries" --attack shifted_random
  done

  # Finally, we evaluate various overlap scaling functions
  for osf in "non-overlap" "overlap" "chi-overlap" "chi-non-overlap" "distances" "similarities" "density" "none"; do
    python sensitivity.py --overlap-scaling-function "$osf" --padversaries "$padversaries" --attack lie
    python sensitivity.py --overlap-scaling-function "$osf" --padversaries "$padversaries" --attack shifted_random
  done
done

# Then we ablate the overall algorithm
for attack in "no_attack" "lie" "shifted_random"; do
  for flags in {'','--eliminate-close'}' '{'','--take-topomap'}' '{'','--scale-by-overlap'}; do
    for padversaries in 0.1 0.4; do
      python ablation.py --attack "$attack" --padversaries "$padversaries" $flags
    done
  done
done

# Then we conclude by comparing topomean to other aggregation rules
for aggregator in "mean" "median" "geomedian" "krum" "trmean" "phocas" "topomean"; do
  for padversaries in $(seq 0.0 0.05 0.5); do
    python comparison.py --aggregator "$aggregator" --padversaries "$padversaries"
  done
done

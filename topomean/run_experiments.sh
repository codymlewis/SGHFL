#!/usr/bin/env bash


# First we evaluate the sensitivity of e1 in mitigating LIE
for e1 in 0.0001 0.001 0.01 0.1 0.2 0.3; do
  for padversaries in $(seq 0.1 0.1 0.5); do
    python sensitivity.py --e1 "$e1" --attack lie --padversaries "$padversaries"
  done
done

# Next we evaluate the relationship between e2 and {dimensionality, number of adversaries}
for e2 in $(seq 1.0 0.5 10.0); do
  for padversaries in $(seq 0.1 0.1 0.5); do
    python sensitivity.py --e2 "$e2" --padversaries "$padversaries"
  done

  for dimensions in $(seq 2 10); do
    python sensitivity.py --e2 "$e2" --dimensions "$dimensions"
  done
done

# Finally, we evaluate K against {dimensionality, number of points, number of adversaries}
for K in $(seq 2 10); do
  for padversaries in $(seq 0.1 0.1 0.5); do
    python sensitivity.py --K "$K" --padversaries "$padversaries"
  done

  for npoints in $(seq 100 100 1000); do
    python sensitivity.py --K "$K" --npoints "$npoints"
  done

  for dimensions in $(seq 2 10); do
    python sensitivity.py --K "$K" --dimensions "$dimensions"
  done
done


# Then we ablate the overall algorithm
for attack in "lie" "shifted_random"; do
  for flags in {'','--eliminate-close'}' '{'','--take-dense-spheres'}' '{'','--scale-by-overlap'}; do
    python ablation.py --attack "$attack" $flags
  done
done

# Then we conclude by comparing topomean to other aggregation rules
for aggregator in "mean" "median" "geomedian" "krum" "trmean" "phocas" "topomean"; do
  for padversaries in $(seq 0.1 0.05 0.5); do
    python comparison.py --aggregator "$aggregator" --padversaries "$padversaries"
  done
done

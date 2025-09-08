#!/bin/bash

path=../results/color/256_orthogonal_8_20240610-061353
nodes=1

for i in $(seq 1 $nodes); do
  python run_bfgs.py --basedir $path --nodes $nodes --batch $((i-1))
done

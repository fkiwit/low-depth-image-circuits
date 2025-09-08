#!/bin/bash

n_jobs=21
timestamp=$(date +%Y%m%d-%H%M%S)

sbatch --array=0-$((n_jobs-1)) run_sweeping_array_job.sbatch -j $n_jobs -t $timestamp

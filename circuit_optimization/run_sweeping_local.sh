#!/bin/bash

n_jobs=1
timestamp=$(date +%Y%m%d-%H%M%S)
# resolution=256
# circuit=sparse
# circuit=unitary
circuit=orthogonal
# color=rgb
layers=4
# dataset="cifar10"
# dataset="fashion_mnist"
# dataset="imagenette_128"
dataset="mnist"

iters=100
samples_per_class=11

# basedir=${dataset}_${circuit}_${layers}_${timestamp}
basedir=${circuit}_${layers}_${timestamp}

# Check if Ray is running. Then connect to it, otherwise start a new one.
if echo $(ray status 2>&1) | grep -q "Active"; then
    echo "Connecting to existing Ray cluster."
    ray status
else
    echo "No existing Ray cluster found. Starting a new one."

    head_node_ip=$(hostname --ip-address)
    ray start --head --port=6379 --num-cpus=0 --num-gpus=0 --include-dashboard=true
    ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=10002 --max-worker-port=10999
    ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=11000 --max-worker-port=11999
    ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=12000 --max-worker-port=12999
    ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=13000 --max-worker-port=13999
fi

for ((i=0; i<n_jobs; i++)); do
    echo Processing batch $(($i + 1)) of $n_jobs

# python processing.py \
python run_sweeping.py \
        --batch $i \
        --nodes $n_jobs \
        --basedir $basedir \
        --circuit $circuit \
        --layers $layers \
        --dataset $dataset \
        --iters $iters \
        --use_ray \
        --samples_per_class $samples_per_class \
        # --res $resolution \
        # --color $color \

done

echo "Sweeping is done"
echo "Starting to collect data"

python data_collector_sweep.py --basedir ../data/${dataset}/$basedir

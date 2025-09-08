#!/bin/bash
#SBATCH --job-name=compression
#SBATCH --output=../slurm_logs/run_%A/output_%a.out
#SBATCH --error=../slurm_logs/run_%A/error_%a.err
#SBATCH --nodes=1
#SBATCH --account=m4408
#SBATCH --qos=regular
#SBATCH --constraint=cpu
#SBATCH --time=24:00:00

while getopts ":j:t:" opt; do
  case $opt in
    j)
      n_jobs=$OPTARG
      ;;
    t)
      timestamp=$OPTARG
      ;;
  esac
done

if [[ -z $n_jobs || -z $timestamp ]]; then
  echo "Error: All attributes (-j and -t) must be provided."
  exit 1
fi

# Define the dataset
dataset="imagenette_128"
# dataset="cifar10"
# dataset="fashion_mnist"
# dataset="mnist"
# resolution=32
# color=rgb

# Define the circuit layout
circuit="orthogonal"
layers=8

iters=300
# samples_per_class=20
basedir=${resolution}_${circuit}_${layers}_${timestamp}

conda activate /pscratch/sd/f/fkiwit/conda/general

head_node_ip=$(hostname --ip-address)
ray start --head --port=6379 --num-cpus=0
ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=10002 --max-worker-port=10999
ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=11000 --max-worker-port=11999
ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=12000 --max-worker-port=12999
ray start --address=${head_node_ip}:6379 --num-cpus=64 --min-worker-port=13000 --max-worker-port=13999

python run_sweeping.py \
  --batch $SLURM_ARRAY_TASK_ID \
  --basedir $basedir \
  --nodes $n_jobs \
  --jobid $SLURM_ARRAY_JOB_ID \
  --dataset $dataset \
  --circuit $circuit \
  --layers $layers \
  --iters $iters \
  --use_ray \
  # --samples_per_class $samples_per_class \
  # --res $resolution \
  # --color $color \

mkdir -p ../data/$dataset/$basedir/slurm_logs
cp ../slurm_logs/run_${SLURM_ARRAY_JOB_ID}/output_${SLURM_ARRAY_TASK_ID}.out ../data/$dataset/$basedir/slurm_logs
cp ../slurm_logs/logs_sweep/run_${SLURM_ARRAY_JOB_ID}/error_${SLURM_ARRAY_TASK_ID}.err ../data/$dataset/$basedir/slurm_logs

exit

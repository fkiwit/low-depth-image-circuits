import numpy as np

from utils.sequential_circuit_optimizer import sweeping_two_cnot_sequential_circuit
from utils.image_compression import calc_MPS
import ray

def optimize_circuit(n_qubits, i):
    # sample from gaussian
    random_state = np.random.normal(0, 1, 2**n_qubits)
    random_state /= np.linalg.norm(random_state)
    A_tensors, Lambda_tensors = calc_MPS(random_state, chi_max=512)
    sc = sweeping_two_cnot_sequential_circuit(L=n_qubits, layers=0, batchsize=1)
    overlaps_complete = []
    for layer in range(2):
        sc.add_one_layer(A_tensors)
        overlaps, time, Bnew = sc.optimize_circuit(A_tensors, iters=200)
        overlaps_complete.append(overlaps)
    np.save(f'../data/randoms/overlaps_random_{n_qubits}_{i}.npy', overlaps_complete)

ray.init()

@ray.remote
def optimize_circuit_parallel(n_qubits, i):
    optimize_circuit(n_qubits, i)

for n_qubits in range(7, 20):
    futures = [optimize_circuit_parallel.remote(n_qubits, i) for i in range(100)]
    ray.get(futures)

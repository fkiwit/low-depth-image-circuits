import argparse
import yaml
from tqdm import tqdm
from time import perf_counter

from threading import Thread

import numpy as np
import jax
from jax import numpy as jnp
from scipy.optimize import minimize

from utils.unitary_to_pennylane import UnitaryToPennylane

jax.config.update("jax_compilation_cache_dir", "_jit_compiled")
jax.config.update('jax_platform_name', 'gpu')
jax.config.update("jax_enable_x64", True)

def loss_fn(params, target):
    return 1 - jnp.sum(jnp.abs(target.conj() * circuit(params)))

grad_fn = jax.grad(loss_fn, argnums=0)

def run_program(loss_fn, grad_fn, params, targets, paths, gpu_id, iters):

    gpu_id = gpu_id % 4

    loss_fn = jax.jit(loss_fn, device=jax.devices()[gpu_id])
    grad_fn = jax.jit(grad_fn, device=jax.devices()[gpu_id])

    start = perf_counter()
    loss_fn(params[0], targets[0])
    print(f"Warm up loss {gpu_id}: {perf_counter() - start:.2f} s")
    start = perf_counter()
    grad_fn(params[0], targets[0])
    print(f"Warm up grad {gpu_id}: {perf_counter() - start:.2f} s")

    jax.device_put(params, jax.devices()[gpu_id])
    jax.device_put(targets, jax.devices()[gpu_id])

    class Callback :
        def __init__(self, gpu_id, sample_id):
            self.iteration = 0
            self.losses = []
            self.gpu_id = gpu_id
            self.sample_id = sample_id

        def optimization_callback(self, xk):
            self.iteration += 1
            self.losses.append(loss_fn(xk, target))
            pbar.update(1)
            pbar.set_postfix(loss=str(self.losses[-1])[:10], gpu=self.gpu_id, sample=self.sample_id)

    for i, (param, target, path) in enumerate(zip(params, targets, paths)):
        cb = Callback(gpu_id=gpu_id, sample_id=i)
        with tqdm(total=iters, leave=False) as pbar:
            res = minimize(
                fun=lambda param: loss_fn(param, target),
                callback=cb.optimization_callback,
                method='BFGS',
                x0=param,
                options={"return_all": True, 'maxiter': iters, "xrtol": 0, "gtol": 0},
                jac=lambda param: grad_fn(param, target)
                )

        np.save(f"{path}_bfgs_params.npy", res.x)
        np.save(f"{path}_bfgs_loss.npy", np.asarray(cb.losses))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="_TEST")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--devices", type=int, default=4)
    parser.add_argument("--iters", type=int, default=2000)

    args = parser.parse_args()

    config = {
        "basedir": args.basedir,
        "batch": args.batch,
        "nodes": args.nodes,
        "depth": args.depth,
        "devices": args.devices,
        "iters": args.iters
        }

    base_path = f"{config['basedir']}/"
    data = np.load(f"{base_path}compressed.npy", allow_pickle=True)

    config_sweeep = yaml.load(open(f"{base_path}config.yml", "r"), Loader=yaml.FullLoader)

    batch_size = int(np.ceil(len(data) / config["nodes"]))
    start_index = config["batch"] * batch_size
    end_index = start_index + batch_size
    batch_data = data[start_index:end_index]

    staircase_circuit = data[f"staircase_circuit_{config['depth']-1}"][0]
    staircase_circuit = np.asarray(staircase_circuit)
    unitary_to_pennylane = UnitaryToPennylane(staircase_circuit)
    circuit, _ = unitary_to_pennylane.get_circuit(RY=False)

    params = batch_data[f"pennylane_params_{config['depth']-1}"]
    params = jnp.asarray(np.vstack(params))
    params = jnp.array_split(params, config["devices"])

    targets = batch_data[f"frqi_target"]
    targets = jnp.asarray(np.vstack(targets))
    targets = jnp.array_split(targets, config["devices"])

    filepath = batch_data["filepath"]
    filename = batch_data["filename"]

    merged_paths = ["/".join(path_parts) for path_parts in zip(filepath, filename)]
    merged_paths = np.array_split(merged_paths, config["devices"])

    processes = []
    for gpu_id in range(config["devices"]):
        p = Thread(target=run_program, args=(loss_fn, grad_fn, params[gpu_id], targets[gpu_id], merged_paths[gpu_id], gpu_id, config["iters"]))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

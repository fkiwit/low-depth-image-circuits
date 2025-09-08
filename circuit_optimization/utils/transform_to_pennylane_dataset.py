import os
import numpy as np
import ray
import pennylane as qml
from tqdm import tqdm
import re

from utils.unitary_to_pennylane import UnitaryToPennylane

@ray.remote
def process_sample_ray(file_idx, data_dir):
    """
    Load data from results_{file_idx}.npz, build the circuits, and compute:
      - local_ops (operations for depth 4 and 8)
      - target (the FRQI target state)
      - label (the class label)
      - params_dict (the learned parameters for each circuit)
      - states (the final quantum states)
    """
    file_path = os.path.join(data_dir, f"result_{file_idx}.npz")
    sample = np.load(file_path, allow_pickle=True)

    local_ops = {}
    params_dict = {}
    states = {}

    for i in [3, 7]:
        unitary_to_pennylane = UnitaryToPennylane(sample[f"staircase_circuit_{i}"])
        circuit, params = unitary_to_pennylane.get_circuit(RY=True)
        state = circuit(params)

        local_ops[f"operations_d{i+1}"] = circuit.tape.operations
        params_dict[f"params_d{i+1}"] = params
        states[f"state_d{i+1}"] = state

    target = sample["frqi_target"].astype(np.float32)
    label = int(sample["class_int"])

    return local_ops, target, label, params_dict, states

def main_ray(file_indices, data_dir):
    """
    1) Launch Ray tasks to process each file in parallel.
    2) Track progress by repeatedly calling ray.wait(...) and ray.get(...).
    3) Collect results, compute fidelities, and store everything in dictionaries/lists.

    Returns:
        operations, target_state, class_label, params, fidelities_d4, fidelities_d8
    """

    # Launch all tasks in parallel
    futures = [process_sample_ray.remote(idx, data_dir) for idx in file_indices]

    # Prepare final storage
    operations = {f"operations_d{i+1}": [] for i in [3, 7]}
    params = {f"params_d{i+1}": [] for i in [3, 7]}
    target_state = []
    class_label = []
    fidelities_d4 = []
    fidelities_d8 = []

    # We'll track the set of pending tasks
    pending = set(futures)

    # Use a tqdm progress bar
    pbar = tqdm(total=len(futures), desc="Processing files")

    # While there are still pending tasks:
    while pending:
        # Wait for at least one task to finish
        done, pending = ray.wait(list(pending), num_returns=1, timeout=None)

        # 'done' is a list of object refs that completed
        for ref in done:
            local_ops, t, lbl, params_dict, states = ray.get(ref)

            # Append to our result arrays
            for i in [3, 7]:
                operations[f"operations_d{i+1}"].append(local_ops[f"operations_d{i+1}"])
                params[f"params_d{i+1}"].append(params_dict[f"params_d{i+1}"])

            target_state.append(t)
            class_label.append(lbl)

            fidelity_d4 = np.abs(np.vdot(states["state_d4"], t))**2
            fidelity_d8 = np.abs(np.vdot(states["state_d8"], t))**2
            fidelities_d4.append(fidelity_d4)
            fidelities_d8.append(fidelity_d8)

            # Update the progress bar for each completed file
            pbar.update(1)

    pbar.close()
    return operations, target_state, class_label, params, fidelities_d4, fidelities_d8

if __name__ == "__main__":
    ######################################
    # 1) Initialize Ray
    ######################################
    ray.init()  # If on a cluster, configure as needed (e.g. ray.init(address="auto"))

    ######################################
    # 2) Define data dir and file indices
    ######################################
    # basedir = '/global/homes/f/fkiwit/dev/data_compression/data/imagenette_128/_orthogonal_8_20250312-115249'
    basedir = '/global/homes/f/fkiwit/dev/data_compression/data/imagenette_128/orthogonal_8_20250415-235928'
    files = os.listdir(basedir)
    file_indices = [int(re.search(r'result_(\d+)\.npz', f).group(1)) for f in files if f.endswith('.npz') and re.search(r'result_(\d+)\.npz', f)]

    ######################################
    # 3) Run the main parallel processing
    ######################################
    (
        operations,
        target_state,
        class_label,
        params,
        fidelities_d4,
        fidelities_d8
    ) = main_ray(file_indices, basedir)

    ######################################
    # 4) Build a PennyLane Dataset
    ######################################
    # Example of grabbing layout for the first sample
    ops_d4 = operations["operations_d4"][0]
    circuit_layout_d4 = [(op.name, list(op.wires)) for op in ops_d4]
    ops_d8 = operations["operations_d8"][0]
    circuit_layout_d8 = [(op.name, list(op.wires)) for op in ops_d8]

    dataset_params = qml.data.Dataset(
        data_name="imagenette",
        exact_state=qml.data.attribute(
            target_state,
            doc="The exact state that the corresponding circuit should prepare.",
        ),
        labels=qml.data.attribute(
            class_label,
            doc="The correct labels classifying the corresponding images.",
        ),
        circuit_layout_d4=qml.data.attribute(
            circuit_layout_d4,
            doc="The layout of the depth 4 circuit.",
        ),
        circuit_layout_d8=qml.data.attribute(
            circuit_layout_d8,
            doc="The layout of the depth 8 circuit.",
        ),
        params_d4=qml.data.attribute(
            params["params_d4"],
            doc="Parameters for the depth 4 circuit.",
        ),
        params_d8=qml.data.attribute(
            params["params_d8"],
            doc="Parameters for the depth 8 circuit.",
        ),
        fidelities_d4=qml.data.attribute(
            fidelities_d4,
            doc="Fidelities between the depth 4 state and the exact state.",
        ),
        fidelities_d8=qml.data.attribute(
            fidelities_d8,
            doc="Fidelities between the depth 8 state and the exact state.",
        ),
    )

    ######################################
    # 5) Save the dataset
    ######################################
    output_file = "/pscratch/sd/f/fkiwit/pennylane_data/imagenette_26.h5"
    dataset_params.write(output_file, mode="w")
    print(f"Dataset written to: {output_file}")

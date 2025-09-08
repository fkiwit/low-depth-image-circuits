import numpy as np
import os
import logging
import argparse
import yaml
import ray
from time import perf_counter

from utils.sequential_circuit_optimizer import sweeping_two_cnot_sequential_circuit, sweeping_three_cnot_sequential_circuit, sweeping_single_cnot_sequential_circuit

from utils.image_encodings import FRQI_encoding, FRQI_RGBa_encoding, move_qubits_right
from utils.image_compression import calc_MPS, calc_state
from utils.unitary_to_pennylane import UnitaryToPennylane

from ray.experimental.tqdm_ray import tqdm

logging.getLogger("ray").setLevel(logging.CRITICAL)

class Orchestrator:
    def __init__(self, config: dict):
        """
        Init method for the orchestrator
        1) Creates the results directory
        2) Sets up the logging
        3) Saves the config
        """

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(config["results_dir"], f"logging_{args.batch}.log"))
            ]
        )

        logging.info("Processing config:")
        for key, value in config.items():
            logging.info(f"{key}: {value}")

    @ray.remote(max_retries=5, retry_exceptions=True, num_cpus=1)
    def process_image_remote(process_image, image_batch, label_batch, index_batch, config) -> bool:
        """
        This function is ray wrapper for the process_image function

        :param process_image: Function to compress a batch of images
        :param image_batch: Batch of images to compress
        :param results_batch: Batched information about the images (e.g. filepaths and class)
        :param config: Configuration of the compression
        :type process_image: function
        :type image_batch: pd.DataFrame
        :type results_batch: pd.DataFrame
        :type config: dict

        :return: Boolen indicating successfull compression
        :rtype: bool
        """

        success = process_image(image_batch, label_batch, index_batch, config)
        return success

    @staticmethod
    def process_image(image_batch, label_batch, index_batch, config) -> bool:
        """
        This function takes a badge of images, fits the MPS circuit,
        and transforms the circuit to pennylane parameters

        :param image_batch: Batch of images to compress
        :param results_batch: Batched information about the images (e.g. filepaths and class)
        :param config: Configuration of the compression
        :type image_batch: pd.DataFrame
        :type results_batch: pd.DataFrame
        :type config: dict

        :return: Boolen indicating successfull compression
        :rtype: bool
        """

        sweeping_batch_size_true = image_batch.shape[0]
        if config["color"] == "rgb":
            frqi_target = FRQI_RGBa_encoding(images=image_batch, indexing="hierarchical")
        elif config["color"] == "gray":
            frqi_target = FRQI_encoding(images=image_batch, indexing="hierarchical")
            frqi_target = move_qubits_right(frqi_target)
        else:
            raise ValueError("Unknown color encoding")
        A_tensors, Lambda_tensors = calc_MPS(np.asarray(frqi_target), chi_max=128)
        n_qubits = len(A_tensors)

        if config["circuit"] == "unitary":
            sc = sweeping_three_cnot_sequential_circuit(L=n_qubits, layers=0, batchsize=sweeping_batch_size_true)
        elif config["circuit"] == "orthogonal":
            sc = sweeping_two_cnot_sequential_circuit(L=n_qubits, layers=0, batchsize=sweeping_batch_size_true)
        elif config["circuit"] == "sparse":
            sc = sweeping_single_cnot_sequential_circuit(L=n_qubits, layers=0, batchsize=sweeping_batch_size_true)
        else:
            raise ValueError("Unknown circuit type")
        sc.chi_max = 128

        overlaps_list = []
        results_sweeping = {}
        for l in tqdm(range(config["layers"])):
            results_sweeping[f"pennylane_params_{l}"] = []
            results_sweeping[f"staircase_circuit_{l}"] = []
            results_sweeping[f"state_{l}"] = []
            if config["circuit"] == "sparse":
                results_sweeping[f"first_gates_{l}"] = []


            sc.add_one_layer(A_tensors)
            overlaps, time, Bnew = sc.optimize_circuit(A_tensors, iters=config["iters"])
            overlaps_list += overlaps

            state = calc_state(Bnew)

            for i in range(len(image_batch)):
                unitary_to_pennylane = UnitaryToPennylane(sc.gates[i])
                circuit, params = unitary_to_pennylane.get_circuit(RY=True)

                results_sweeping[f"pennylane_params_{l}"].append(params)
                results_sweeping[f"staircase_circuit_{l}"].append(sc.gates[i])
                results_sweeping[f"state_{l}"].append(state[i])
                if config["circuit"] == "sparse":
                    results_sweeping[f"first_gates_{l}"].append(sc.first_gates[i])


        overlaps = np.asarray(overlaps_list).T
        if overlaps.ndim == 1:
            overlaps = overlaps[None]

        # Unroll the batch and save separately
        for i in range(len(image_batch)):
            results = {}
            results["loss_sweep"] = overlaps
            results["frqi_target"] = frqi_target[i]
            results["class_int"] = label_batch[i]
            results["index"] = index_batch[i]

            for key, value in results_sweeping.items():
                results[key] = value[i]
            results["time_sweep"] = time / sweeping_batch_size_true

            result_path = os.path.join(config["results_dir"], f"result_{index_batch[i]}.npz")
            np.savez_compressed(result_path, **results)

        return True

    def pipeline(self, config) -> None:
        """
        Function to process the images

        :return: None
        """
        logging.info("Start processing")
        time_start = perf_counter()

        labels = np.load(f"../data/{config['dataset']}/labels.npy")
        samples_per_class = config["samples_per_class"]
        indices = np.concatenate([
            np.where(labels == label)[0][:samples_per_class] for label in range(10)
        ])
        indices = np.array(indices)
        images = np.load(f"../data/{config['dataset']}/images_p1.npy", mmap_mode="r")

        images = images[0][indices]
        labels = labels[indices]

        print(len(indices))

        n_processed_files = len(indices)
        batch_size = int(np.ceil(n_processed_files / config["nodes"]))
        start = config["batch"] * batch_size
        stop = (config["batch"] + 1) * batch_size
        images = images[start:stop]
        labels = labels[start:stop]
        indices = indices[start:stop]

        logging.info(f"Total number of files to process: {len(indices)}")
        print("number of files to process: ", len(images), len(labels), len(indices))
        if images.ndim == 4 and images.shape[-1] == 3:
            config["color"] = "rgb"
            config["shape"] = list(images.shape[1:3])
        elif images.ndim == 3:
            config["color"] = "gray"
            config["shape"] = list(images.shape[1:3])
        with open(os.path.join(config["results_dir"], "config.yml"), "w") as filehandler:
            yaml.dump(config, filehandler)

        logging.info(f"Number of files to process in this job: {len(images)}")

        logging.info(f"Time preparing: {perf_counter() - time_start} s")
        time_start = perf_counter()
        logging.info("Start sweeping")

        times_dictionary = {}
        num_splits = int(np.ceil(len(images) / config["sweeping_batch_size"]))
        images_batches = np.array_split(images, num_splits)
        labels_batches = np.array_split(labels, num_splits)
        indices_batches = np.array_split(indices, num_splits)

        time_start = perf_counter()
        logging.info("Start init")

        # Compress the images
        times_dictionary["connect_cluster"] = perf_counter() - time_start
        time_start = perf_counter()
        if config["use_ray"]:
            ray.init()
            logging.info(ray.cluster_resources())
            success = ray.get([self.process_image_remote.remote(
                self.process_image,
                image_batch,
                label_batch,
                index_batch,
                config) for (image_batch, label_batch, index_batch) in zip(images_batches, labels_batches, indices_batches)])
            logging.info("Shutting down ray cluster")
            ray.shutdown()
        else:
            results_sweep = []
            for i, (image_batch, label_batch, index_batch) in enumerate(zip(images_batches, labels_batches, indices_batches)):
                logging.info(f"Processing batch {i+1}/{len(images_batches)}")
                results_sweep.append(self.process_image(image_batch, label_batch, index_batch, config))

        times_dictionary["time_compression"] = perf_counter() - time_start
        logging.info(f"Time compression: {times_dictionary['time_compression']} s")
        time_start = perf_counter()

        with open(f"{config['results_dir']}/times_{config['batch']}.yml", "w") as filehandler:
            yaml.dump(times_dictionary, filehandler)

        logging.info(f"Time postprocessing: {perf_counter() - time_start} s")
        logging.info("Sweeping successful")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--basedir", type=str, default="_TEST")
    parser.add_argument("--batch", type=int, default=0)
    parser.add_argument("--jobid", type=int, default=0)
    parser.add_argument("--use_ray", action="store_true")

    # parser.add_argument("--res", type=int, default=32)
    parser.add_argument("--circuit", type=str, default="orthogonal")
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mnist")

    parser.add_argument("--iters", type=int, default=10)
    # parser.add_argument("--samples_per_class", type=int, default=10)
    parser.add_argument("--samples_per_class", type=int, default=None)
    parser.add_argument("--sweeping_batch_size", type=int, default=1)

    args = parser.parse_args()

    config = {
        "basedir": args.basedir,
        "batch": args.batch,
        "nodes": args.nodes,
        "jobid": args.jobid,
        "use_ray": args.use_ray,

        "dataset": args.dataset,
        "circuit": args.circuit,
        "layers": args.layers,

        "iters": args.iters,
        "samples_per_class": args.samples_per_class,

        "sweeping_batch_size": args.sweeping_batch_size
        }

    results_dir = f"../data/{config['dataset']}/{config['basedir']}"
    config["results_dir"] = results_dir
    os.makedirs(config["results_dir"], exist_ok=True)

    orchestrator = Orchestrator(config=config)
    orchestrator.pipeline(config)

# Models
from utils.tensor_network_training import main as main_tn
from utils.vqc_training import main as main_vqc

import argparse
from typing import Dict, Union
import os

# Ray should be imported at the end
import ray
from ray import tune
from ray.tune.schedulers import FIFOScheduler, ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter

from ray_helper import GCTuneCallback

class RaySubmission:

    def __init__(
         self,
         dataset: Union[str, tune.grid_search],
         fold: Union[int, tune.grid_search],
         compression_depth: Union[int, bool, tune.grid_search],
         scheduler_config: Dict = {"name": "asha"},
         gpus: float = 0.5,
         ):
        """
        :param dataset: Name of the dataset, e.g., 'mnist', or a grid_search().
        :param fold: Which fold to use or a grid_search for tuning multiple folds.
        :param compression_depth: Depth for compression or False if none.
        :param scheduler_config: Dictionary specifying which scheduler to use and its params.
           Example:
                 {
                    "name": "fifo",  # or "asha"
                    "use_stopper": True,  # only relevant if 'name' == "fifo"
                    "time_attr": "epoch", # if using "asha"
                    "metric": "validation_accuracy",
                    "mode": "max",
                    "grace_period": 10,
                    ...
                 }
        :param gpus: Number of GPUs to allocate per trial.
        """
        self.choices_dataset = ["mnist", "cifar10", "fashion_mnist", "imagenette_128"]

        self.gpus = gpus
        self.scheduler_config = scheduler_config

        self.config_dataset = {
            "dataset_name": dataset,
            "compression_depth": compression_depth,
            "batch_size": 100,
            "fold": fold,
        }

    def run(self):
        self.basepath = f"{os.getcwd()}/_results/{self.config_model['model_name']}"
        main = self._get_main()

        print("Ray status:")
        print(ray.cluster_resources())
        self.config_model.update({"basepath": self.basepath})
        self.config_model.update(self.config_dataset)

        if self.scheduler_config["name"] == "fifo":
            scheduler = FIFOScheduler()
            search_alg = None
        elif self.scheduler_config["name"] == "asha":
            hyperopt_search = HyperOptSearch(
                metric="validation_accuracy",
                mode="max"
            )
            hyperopt_search = ConcurrencyLimiter(hyperopt_search, max_concurrent=16)

            scheduler = ASHAScheduler(
                time_attr="epoch",               # Or "epoch" depending on what your trainable reports.
                metric="validation_accuracy",    # This is the metric that your trainable reports.
                mode="max",                      # Use "min" if you are minimizing the metric.
                max_t=200,                       # Max number of epochs to run.
                grace_period=20,                 # Let trials run for at least 10 epochs.
                reduction_factor=2,              # How aggressively to downsample.
                brackets=1                       # Number of brackets.
            )
            search_alg = hyperopt_search
        else:
            raise ValueError(f"Scheduler {self.scheduler_config['name']} not recognized.")

        stopper = self.scheduler_config.get("stopper", None)

        run_kwargs = {
            "config": self.config_model,
            "trial_dirname_creator": lambda trial: (
                f"{trial.trial_id}_"
                f"{trial.config['model_name']}_"
                f"{trial.config['dataset_name']}"
            ),
            "scheduler": scheduler,
            "stop": stopper,
            "num_samples": 200,
            "storage_path": self.basepath,
            "resources_per_trial": {"cpu": 10, "gpu": self.gpus},
            "log_to_file": True,
            "raise_on_failed_trial": True,
            "max_concurrent_trials": 16,
            "callbacks": [GCTuneCallback()],
        }
        # Include the search algorithm if defined.
        if search_alg is not None:
            run_kwargs["search_alg"] = search_alg

        analysis = tune.run(main, **run_kwargs)

class RaySubmissionTN(RaySubmission):
    def __init__(
         self,
         model_name,
         patched,
         n_factors,
         warmstart,
         dataset,
         compression_depth,
         fold,
         n_samples_warm_start,
         batch_size,
         learning_rate,
         epochs,
         chi_final
         ):
        super().__init__(dataset=dataset, fold=fold, compression_depth=compression_depth)
        self.choices_model = ["mps", "mpo"]
        self.choices_patched = [True, False]
        self.choices_warmstart = [True, False]

        if patched:
            self.choices_n_factors = [1, 4, 16]
        else:
            self.choices_n_factors = [1, 2, 3]

        self.config_model = {
            "model_name": model_name,
            "patched": patched,
            "n_factors": n_factors,
            "warmstart": warmstart,
            "compression_depth": compression_depth,
            "n_samples_warm_start": n_samples_warm_start,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "chi_final": chi_final
        }

    @staticmethod
    def _get_main():
        return main_tn

class RaySubmissionVQC(RaySubmission):
    def __init__(
         self,
         model_name,
         n_qubits,
         depth,
         building_block_tag,
         temperature,
         optimizer,
         dataset,
         fold,
         compression_depth,
         batch_size,
         learning_rate,
         epochs
         ):
        super().__init__(dataset=dataset, fold=fold, compression_depth=compression_depth)

        self.config_model = {
            "model_name": model_name,  # "LinearVQC", "NonLinearVQC", "NonLinearVQC_shadow"
            "n_qubits": n_qubits,
            "depth": depth,
            "building_block_tag": building_block_tag,
            "temperature": temperature,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs
        }

    @staticmethod
    def _get_main():
        return main_vqc

class RaySubmissionCNN(RaySubmission):
    def __init__(
         self,
         model_name,
         depth,
         output_channels,
         nodes_linear,
         dropout_prob,
         dataset,
         learning_rate,
         weight_decay,
         epochs,
         fold,
         compression_depth,
         batch_size,
         scheduler_config,
         ):
        super().__init__(dataset=dataset, fold=fold, compression_depth=compression_depth, gpus=0.5, scheduler_config=scheduler_config)

        self.config_model = {
            "model_name": model_name,
            "output_channels": output_channels,
            "depth": depth,
            "nodes_linear": nodes_linear,
            "dropout_prob": dropout_prob,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "epochs": epochs,
            "fold": 0,
            "batch_size": batch_size
        }

    @staticmethod
    def _get_main():
        return main_cnn

if __name__ == "__main__":
    ray.init()
    print(ray.cluster_resources())

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="cnn", help="Name of the config to run (e.g., 'cnn', 'mps', 'NonLinearVQC_shadow')")
    args = parser.parse_args()

    config_name = args.config_name

    if config_name == "NonLinearVQC_shadow":
        submission = RaySubmissionVQC(
        model_name="NonLinearVQC_shadow",
        n_qubits=11,
        depth=4,
        building_block_tag="su4",
        temperature=128,
        optimizer="adam",
        dataset="imagenette_128",
        fold=tune.grid_search(list(range(5))),
        compression_depth=tune.grid_search(list(range(9))),
        batch_size=100,
        learning_rate=8e-4,
        epochs=100
        )

    elif config_name == "mpo":
        submission = RaySubmissionTN(
           model_name="mps",
           patched=False,
           n_factors=tune.grid_search([1, 4, 16]),
           warmstart=tune.grid_search([True, False]),
           fold=tune.grid_search(list(range(5))),
           dataset=tune.grid_search(["mnist", "cifar10", "fashion_mnist", "imagenette_128"]),
           compression_depth=0,
           n_samples_warm_start=1000,
           batch_size=100,
           learning_rate=1e-4,
           epochs=50,
           chi_final=32
        )
        submission.run()

    elif config_name == "cnn":
        from utils.cnn_training import main as main_cnn
        submission = RaySubmissionCNN(
        model_name="cnn",
        depth=5,
        output_channels=32,
        nodes_linear=64,
        dropout_prob=tune.quniform(0, 0.6, 0.05),
        dataset="cifar10",
        learning_rate=tune.qloguniform(1e-3, 1, 5e-4),
        epochs=200,
        weight_decay=tune.qloguniform(1e-6, 1e-1, 5e-7),
        fold=0,
        compression_depth=0,
        batch_size=100,
        scheduler_config={"name": "asha"}
        )

    submission.run()

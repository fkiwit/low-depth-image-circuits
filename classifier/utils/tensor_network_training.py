import os
import sys
import pickle
import numpy as np
import jax
import optax
from jax import numpy as jnp
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import json
import gc

from torch.utils.tensorboard import SummaryWriter
from ray.air import session

from utils.tensor_network_utils import select_first_n_samples_per_class, create_balanced_batches, MPS_pretraining, MPO_pretraining

# -------------------------------------------------------------------------
# Base class for MPS/MPO classifiers
# -------------------------------------------------------------------------

class TensorNetworkClassifier:
    """
    Provides a generic training loop using JAX & Optax.
    Subclasses must implement:
      - init_classifier()
      - _predict() logica
    """
    def __init__(self, chi_final=32, num_classes=10, learning_rate=1e-4,
                 batch_size=100, epochs=20, trial_dir=None):
        self.chi_final = chi_final
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.params = None
        self.predict_fn = None
        self.optimizer = None
        self.opt_state = None
        self.Lc = None
        self.trial_dir = trial_dir

    def init_classifier(self, *args, **kwargs):
        raise NotImplementedError("Subclass must implement init_classifier.")

    def _predict(self, params, A_tensors):
        raise NotImplementedError("Subclass must implement _predict.")

    def predict(self, A_tensors):
        return self.predict_fn(self.params, A_tensors)

    def loss_fn(self, params, A_tensors, labels):
        logits = self.predict_fn(params, A_tensors)
        targets = jnp.eye(self.num_classes)[labels]
        return jnp.mean(optax.softmax_cross_entropy(logits, targets))

    def fit(self, A_tensors_train, labels_train, A_tensors_val, labels_val, training_metrics):
        n_samples = len(labels_train)
        grad_fn = jax.jit(jax.grad(self.loss_fn))
        self.optimizer = optax.adam(self.learning_rate)
        self.opt_state = self.optimizer.init(self.params)

        @jax.jit
        def update(params, opt_state, A_batch, y_batch):
            grads = grad_fn(params, A_batch, y_batch)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state

        writer = SummaryWriter(self.trial_dir)
        for epoch in range(self.epochs):
            # Shuffle
            idxs = np.arange(n_samples)
            np.random.shuffle(idxs)
            # Mini-batch update
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = start_idx + self.batch_size
                batch_indices = idxs[start_idx:end_idx]

                A_batch = [A[batch_indices] for A in A_tensors_train]
                y_batch = labels_train[batch_indices]

                self.params, self.opt_state = update(self.params, self.opt_state, A_batch, y_batch)

            train_acc, train_loss = self.compute_batched_accuracy_and_loss(A_tensors_train, labels_train)
            val_acc, val_loss = self.compute_batched_accuracy_and_loss(A_tensors_val, labels_val)

            training_metrics["train_loss"].append(float(train_loss))
            training_metrics["train_acc"].append(float(train_acc))
            training_metrics["val_loss"].append(float(val_loss))
            training_metrics["val_acc"].append(float(val_acc))

            writer.add_scalar('Loss/train', float(train_loss), epoch)
            writer.add_scalar('Accuracy/train', float(train_acc), epoch)
            writer.add_scalar('Loss/val', float(val_loss), epoch)
            writer.add_scalar('Accuracy/val', float(val_acc), epoch)

            session.report({
                "epoch": epoch,
                "training_loss": float(train_loss),
                "training_accuracy": float(train_acc),
                "validation_loss": float(val_loss),
                "validation_accuracy": float(val_acc)
            })
            gc.collect()

        writer.close()

        return training_metrics

    def evaluate(self, A_tensors, labels):
        logits = self.predict_fn(self.params, A_tensors)
        preds = jnp.argmax(logits, axis=1)
        acc = jnp.mean(preds == labels)
        loss = self.loss_fn(self.params, A_tensors, labels)
        return acc, loss

    def compute_batched_accuracy_and_loss(self, A_tensors, labels):
        n_samples = len(labels)
        acc_total = 0
        loss_total = 0
        batch_size = 100
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            A_batch = [A[start_idx:end_idx] for A in A_tensors]
            y_batch = labels[start_idx:end_idx]
            acc_batch, loss_batch = self.evaluate(A_batch, y_batch)
            acc_total += acc_batch * len(y_batch)
            loss_total += loss_batch * len(y_batch)
        return acc_total / n_samples, loss_total / n_samples

# -------------------------------------------------------------------------
# MPS Classifier
# -------------------------------------------------------------------------

class MPSClassifier(TensorNetworkClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_classifier(self, n_qubits, Lc, warmstart=False, dataset=None):
        self.Lc = Lc
        if warmstart:
            mps_classifier = MPS_pretraining(dataset, chi_final=self.chi_final)
            self.params = [m / np.sqrt(2) for m in mps_classifier]
        else:
            self.params = self._init_random(Lc, n_qubits)

        self.predict_fn = jax.jit(lambda p, A: self._predict(p, A))

    def _init_random(self, Lc, n_qubits):
        chi_left = 1
        Ms_left, Ms_right = [], []
        for _ in range(Lc - 1):
            chi_right = min(chi_left * 2, self.chi_final)
            matrix_1 = jnp.asarray(np.random.normal(size=(chi_left, 2, chi_right), scale=np.sqrt(1 / chi_left)))
            matrix_2 = jnp.asarray(np.random.normal(size=(chi_right, 2, chi_left), scale=np.sqrt(1 / chi_left)))
            Ms_left.append(matrix_1)
            Ms_right.append(matrix_2)
            chi_left = chi_right
        if n_qubits % 2 == 0:
            chi_right_odd = min(chi_left * 2, self.chi_final)
            additional_tensor = jnp.asarray(np.random.normal(size=(chi_left, 2, chi_right_odd),
                                                            scale=np.sqrt(1 / chi_left)))
            Ms_left.append(additional_tensor)
            classification_node = jnp.asarray(np.random.normal(
                size=(self.num_classes, chi_left, 2, chi_right_odd),
                scale=1 / np.sqrt(chi_right_odd * chi_right_odd)))
                # Change to scale=1))
        else:
            chi_right = min(chi_left * 2, self.chi_final)
            matrix_1 = jnp.asarray(np.random.normal(size=(chi_left, 2, chi_right), scale=np.sqrt(1 / chi_left)))
            matrix_2 = jnp.asarray(np.random.normal(size=(chi_right, 2, chi_left), scale=np.sqrt(1 / chi_left)))
            Ms_left.append(matrix_1)
            Ms_right.append(matrix_2)
            chi_left = chi_right
            classification_node = jnp.asarray(np.random.normal(
                size=(self.num_classes, chi_left, 2, chi_right),
                scale=1/chi_right))
                # Change to scale=1))
        return Ms_left + [classification_node] + Ms_right[::-1]

    def _predict(self, params, A_tensors):
        left = jnp.ones((A_tensors[0].shape[0], 1, 1))
        # Contract left side
        for mps, inp in zip(params[:self.Lc], A_tensors[:self.Lc]):
            left = jnp.einsum('bij,ikl,bjkm -> blm', left, mps, inp)

        right = jnp.ones((A_tensors[0].shape[0], 1, 1))
        # Contract right side (reverse)
        for mps, inp in zip(params[:self.Lc:-1], A_tensors[:self.Lc:-1]):
            right = jnp.einsum('ijk,bljm,bkm -> bil', mps, inp, right)

        # Classification node
        logits = jnp.einsum('ijkl,bmkn,bjm,bln -> bi',
                            params[self.Lc],
                            A_tensors[self.Lc],
                            left, right)
        return logits

# -------------------------------------------------------------------------
# MPO Classifier
# -------------------------------------------------------------------------

class MPOClassifier(TensorNetworkClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_classifier(self, n_qubits, Lc, warmstart=False, dataset=None):
        self.Lc = Lc
        if warmstart:
            mps_classifier = MPO_pretraining(dataset, chi_final=self.chi_final)
            self.params = [m / 2 for m in mps_classifier]
        else:
            self.params = self._init_random(Lc, n_qubits, self.chi_final, self.num_classes)

        self.predict_fn = jax.jit(lambda p, A: self._predict(p, A))
        # self.predict_fn = lambda p, A: self._predict(p, A)

    @staticmethod
    def _init_random(Lc, n_qubits, chi_final, num_classes):
        chi_left = 1
        Ms_left, Ms_right = [], []
        for _ in range(Lc - 1):
            chi_right = min(chi_left * 4, chi_final)
            Ms_left.append(jnp.asarray(np.random.normal(size=(chi_left, 2, 2, chi_right), scale=np.sqrt(1 / chi_left))))
            Ms_right.append(jnp.asarray(np.random.normal(size=(chi_right, 2, 2, chi_left), scale=np.sqrt(1 / chi_left))))
            chi_left = chi_right
        if n_qubits % 2 == 0:
            chi_right_odd = min(chi_left * 4, chi_final)
            additional_tensor = jnp.asarray(np.random.normal(size=(chi_left, 2, 2, chi_right_odd),
                                                            scale=np.sqrt(1 / chi_left)))
            Ms_left.append(additional_tensor)
            classification_node = jnp.asarray(np.random.normal(
                size=(num_classes, chi_left, 2, 2, chi_right_odd),
                scale=1/np.sqrt(chi_right_odd * chi_right_odd)))
        else:
            chi_right = min(chi_left * 4, chi_final)
            Ms_left.append(jnp.asarray(np.random.normal(size=(chi_left, 2, 2, chi_right), scale=np.sqrt(1 / chi_left))))
            Ms_right.append(jnp.asarray(np.random.normal(size=(chi_right, 2, 2, chi_left), scale=np.sqrt(1 / chi_left))))
            chi_left = chi_right
            classification_node = jnp.asarray(np.random.normal(
                size=(num_classes, chi_left, 2, 2, chi_right),
                scale=1/chi_right))
        return Ms_left + [classification_node] + Ms_right[::-1]

    def _predict(self, params, A_tensors):
        left = jnp.ones((A_tensors[0].shape[0], 1, 1, 1))
        for mps, inp in zip(params[:self.Lc], A_tensors[:self.Lc]):
            left = jnp.einsum('bijk,jlmn,bilo,bkmp -> bonp', left, mps, inp, inp)

        right = jnp.ones((A_tensors[0].shape[0], 1, 1, 1))
        for mps, inp in zip(params[:self.Lc:-1], A_tensors[:self.Lc:-1]):
            right = jnp.einsum('ijkl,bmjn,bokp,bnlp -> bmio', mps, inp, inp, right)

        logits = jnp.einsum('ijklm,bnko,bplq,bnjp,bomq -> bi',
                            params[self.Lc],
                            A_tensors[self.Lc],
                            A_tensors[self.Lc],
                            left,
                            right)
        return logits

def main(ray_config, use_ray=True):
    print("starting")
    if use_ray:
        from ray.air import session
        trial_dir = session.get_trial_dir()
    else:
        trial_dir = ray_config["basepath"]

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    dataset_name = ray_config["dataset_name"]
    n_factors = ray_config["n_factors"]
    warmstart = ray_config["warmstart"]
    fold = ray_config["fold"]
    patched = ray_config["patched"]
    mode = ray_config["model_name"]
    compression_depth = ray_config["compression_depth"]

    hparams = {
        "patched": patched,
        "dataset_name": dataset_name,
        "n_factors": n_factors,
        "warmstart": warmstart,
        "compression_depth": compression_depth,
        "fold": fold,
        "chi_final": ray_config["chi_final"],
        "n_samples_warm_start": ray_config["n_samples_warm_start"],
        "batch_size": ray_config["batch_size"],
        "learning_rate": ray_config["learning_rate"],
        "epochs": ray_config["epochs"],
        "model_name": ray_config["model_name"]
    }

    data_path = f"{os.path.dirname(os.path.dirname(ray_config['basepath']))}/../data/{dataset_name}/"
    os.makedirs(trial_dir, exist_ok=True)

    # Load A_tensors and labels
    if not hparams["compression_depth"]:
        labels = np.load(f"{data_path}labels.npy")
        if patched:
            with open(f"{data_path}mps_p{n_factors}.pkl", "rb") as f:
                A_tensors = pickle.load(f)
        else:
            with open(f"{data_path}mps_p1.pkl", "rb") as f:
                A_tensors = pickle.load(f)
            A_tensors = A_tensors * n_factors

    else:
        assert not patched, "Training on conpressed states only supported for non-patched MPS"
        labels = np.load(f"{data_path}compressed/labels.npy")
        with open(f"{data_path}compressed/mps_p1_c{hparams['compression_depth']}.pkl", "rb") as f:
            A_tensors = pickle.load(f)
        A_tensors = A_tensors * n_factors
    print("tensors loaded")

    # Split data via StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_index, test_index = list(skf.split(np.zeros_like(labels), labels))[fold]

    A_tensors_val = []
    A_tensors_train = []
    for i in range(len(A_tensors)):
        A = A_tensors.pop(0)
        A_tensors_val.append(A[test_index])
        A_tensors_train.append(A[train_index])


    labels_val = labels[test_index]
    labels_train = labels[train_index]
 
    n_qubits = len(A_tensors_train)
    Lc = n_qubits // 2

    # Prepare warmstart data if needed
    A_warm, y_warm = select_first_n_samples_per_class(
        A_tensors_train, labels_train, n_samples_per_class=100
    )
    dataset = create_balanced_batches(A_warm, y_warm, batch_size=100, shuffle=True)

    # Initialize a classifier (MPS or MPO). 
    if mode == "mps":
        classifier = MPSClassifier(
            chi_final=hparams["chi_final"],
            num_classes=10,
            learning_rate=hparams["learning_rate"],
            batch_size=hparams["batch_size"],
            epochs=hparams["epochs"],
            trial_dir=trial_dir
        )

    elif mode == "mpo":
        classifier = MPOClassifier(
            chi_final=hparams["chi_final"],
            num_classes=10,
            learning_rate=hparams["learning_rate"],
            batch_size=hparams["batch_size"],
            epochs=hparams["epochs"],
            trial_dir=trial_dir
        )
    else:
        raise ValueError("Invalid mode. Choose 'mps' or 'mpo'.")

    print(f"Training {mode} classifier with the following hyperparameters:")
    # Classifier init
    classifier.init_classifier(n_qubits=n_qubits,
                               Lc=Lc,
                               warmstart=warmstart,
                               dataset=dataset)
    print("Classifier initialized.")

    acc_warm, loss_warm = classifier.compute_batched_accuracy_and_loss(A_warm, y_warm)
    acc_train, loss_train = classifier.compute_batched_accuracy_and_loss(A_tensors_train, labels_train)
    acc_val, loss_val = classifier.compute_batched_accuracy_and_loss(A_tensors_val, labels_val)

    training_metrics = {
        "train_loss": [loss_train],
        "train_acc": [acc_train],
        "val_loss": [loss_val],
        "val_acc": [acc_val]
    }

    print(f"Initialization -> "
          f"Warm Acc: {acc_warm*100:.2f}%, Train Loss: {loss_warm:.4f}, "
          f"Val Acc: {acc_val*100:.2f}%, Val Loss: {loss_val:.4f}")
    print("Classifier evaluated.")

    hparams["acc_warm"] = float(acc_warm)
    hparams["acc_val"] = float(acc_val)
    # Save final parameters and hparams
    with open(os.path.join(trial_dir, 'initial_params.pkl'), 'wb') as f:
        pickle.dump(classifier.params, f)
    with open(os.path.join(trial_dir, 'hparams.json'), 'w') as f:
        json.dump(hparams, f, indent=4)

    # Train
    training_metrics = classifier.fit(A_tensors_train, labels_train, A_tensors_val, labels_val, training_metrics)

    if not use_ray:
        # Save training metrics as a pandas dataframe
        metrics_df = pd.DataFrame(training_metrics)
        metrics_df.to_csv(os.path.join(trial_dir, 'training_metrics.csv'), index=False)

if __name__ == "__main__":
    from datetime import datetime

    basepath = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{basepath}/results_ray_tmp/{timestamp}"

    ray_config = {
        "model_name": "mpo",
        "basepath": result_dir,
        "patched": True,
        "dataset_name": "mnist",
        "n_factors": 1,
        "warmstart": True,
        "compression_depth": 0,
        "fold": 0,
        "chi_final": 32,
        "n_samples_warm_start": 1000,
        "batch_size": 100,
        "learning_rate": 1e-4,
        "epochs": 100
        }

    main(ray_config, use_ray=False)

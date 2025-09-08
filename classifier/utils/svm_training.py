import os
import numpy as np
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import sys

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import csv

def load_dataset(data_set, patches, basepath="../data", compression_level=False):
    try:
        path = os.path.join(basepath, data_set)
        if not compression_level:
            labels = np.load(os.path.join(path, "labels.npy"))
            states = np.load(os.path.join(path, f"states_p{patches}.npy"))
        else:
            labels = np.load(os.path.join(path, "compressed/labels.npy"))
            states = np.load(os.path.join(path, f"compressed/states_p{patches}_c{compression_level}.npy"))
            states = states[None]
    except FileNotFoundError:
        raise ValueError("States file not found")

    return states, labels

def compute_kernel(train_images_patches, val_images_patches=None, degree=2, coef0=1, gamma=None):
    # K(X, Y) = (gamma * X.dot(Y.T) + coef0) ** degree
    # K is computed for every patch (for loop) and the final kernel is the product of all the kernels (<Psi_1|<Psi_0|Phi_0>|Phi_1>)
    if val_images_patches is None:
        val_images_patches = train_images_patches

    kernels = []
    counter = 0
    for patch_1, patch_2 in zip(val_images_patches, train_images_patches):
        counter += 1
        K = np.dot(patch_1, patch_2.T)
        kernels.append(K)

    K = np.prod(kernels, axis=0)
    K = gamma * K + coef0
    K = np.power(K, degree)
    return K

def compute_kernel_multicopy(copies, train_images_patches, val_images_patches=None, degree=2, coef0=1, gamma=None):
    # K(X, Y) = (gamma * X.dot(Y.T) + coef0) ** degree
    # K is computed for every patch (for loop) and the final kernel is the product of all the kernels (<Psi_1|<Psi_0|Phi_0>|Phi_1>)
    if val_images_patches is None:
        val_images_patches = train_images_patches

    K = np.dot(val_images_patches[0], train_images_patches[0].T)

    K = K * copies
    K = gamma * K + coef0
    K = np.power(K, degree)
    return K

def compute_kernels(train_images_patches, val_images_patches):
    train_kernels = [np.dot(patch, patch.T) for patch in train_images_patches]
    train_kernel = np.prod(train_kernels, axis=0)
    train_kernel = np.abs(train_kernel) ** 2
    val_kernels = [np.dot(patch, train_images_patches[i].T) for i, patch in enumerate(val_images_patches)]
    val_kernel = np.prod(val_kernels, axis=0)
    val_kernel = np.abs(val_kernel) ** 2
    return train_kernel, val_kernel

def train_and_evaluate(train_kernel, train_labels, val_kernel, val_labels):
    svc = SVC(kernel='precomputed')
    ovr_classifier = OneVsRestClassifier(svc, n_jobs=10)
    ovr_classifier.fit(train_kernel, train_labels)

    train_predictions = ovr_classifier.predict(train_kernel)
    val_predictions = ovr_classifier.predict(val_kernel)

    train_accuracy = accuracy_score(train_labels, train_predictions)
    val_accuracy = accuracy_score(val_labels, val_predictions)

    return train_predictions, val_predictions, train_accuracy, val_accuracy

def main(config):
    # Hyerparameters
    data_set = config["dataset"]
    n_splits = 5

    if config["factoring"] == "patched":
        states, labels = load_dataset(data_set, config["factors"])
    elif config["factoring"] == "multicopy":
        states, labels = load_dataset(data_set, 1, compression_level=config["compression_level"])

    # train_accuracies = []
    # val_accuracies = []
    fold = config["foldindex"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # TODO: fix splitting, check shape, especially for multi patch images
    train_index, val_index = list(skf.split(states[0], labels))[fold]
    train_states_fold = states[:, train_index, :]
    val_states_fold = states[:, val_index, :]
    train_labels_fold = labels[train_index]
    val_labels_fold = labels[val_index]

    # train_kernel, val_kernel = compute_kernels(train_images_patches_fold, val_images_patches_fold)
    # gamma = 1 / (train_states_fold.shape[-1] * train_states_fold[0].var())
    # TODO: check the hyperparameters
    gamma = 1
    coef0 = 0
    if config["factoring"] == "patched":
        train_kernel = compute_kernel(train_states_fold, train_states_fold, degree=2, coef0=coef0, gamma=gamma)
        val_kernel = compute_kernel(train_states_fold, val_states_fold, degree=2, coef0=coef0, gamma=gamma)
    elif config["factoring"] == "multicopy":
        train_kernel = compute_kernel_multicopy(config["factors"], train_states_fold, train_states_fold, degree=2, coef0=coef0, gamma=gamma)
        val_kernel = compute_kernel_multicopy(config["factors"], train_states_fold, val_states_fold, degree=2, coef0=coef0, gamma=gamma)

    train_predictions, val_predictions, train_accuracy, val_accuracy = train_and_evaluate(train_kernel, train_labels_fold, val_kernel, val_labels_fold)
    print("Training set accuracy:", train_accuracy)
    print("Validation set accuracy:", val_accuracy)
    config["train_accuracy"] = train_accuracy
    config["val_accuracy"] = val_accuracy
    config["block_shape_1d"] = config["factors"]

    resutls_dir = f"_results/svm/{config['dataset']}_{config['timestamp']}"
    os.makedirs(resutls_dir, exist_ok=True)
    csv_file = os.path.join(resutls_dir, f"{config['factoring']}_f{config['factors']}_c{config['compression_level']}.csv")

    # Check if the file exists
    file_exists = os.path.isfile(csv_file)
    # Write the config to a CSV file
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(config.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some arguments.')

    parser.add_argument('--timestamp', type=str, default='_test')
    parser.add_argument('--foldindex', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--factors', type=int, default=1)
    parser.add_argument('--compression_level', type=int, default=0)
    parser.add_argument('--factoring', type=str, choices=['patched', 'multicopy'], default='multicopy')

    args = parser.parse_args()

    config = {
        "timestamp": args.timestamp,
        "foldindex": args.foldindex,
        "dataset": args.dataset,
        "factors": args.factors,
        "compression_level": args.compression_level,
        "factoring": args.factoring,
        
    }

    main(config)

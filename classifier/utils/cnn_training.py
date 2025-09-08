import os
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from jax import numpy as jnp
from tqdm import tqdm
import dill

import numpy as np
from sklearn.model_selection import StratifiedKFold
from datetime import datetime

from utils.vqc_training import Callback

def move_qubits_left(states, num=1):
    """
    Permute qubits cyclically to the left:
    (q_1, q_2, ..., q_n) -> (q_num+1, q_num+2, ..., q_num)
    
    states : ndarray
        The batch of n-qubit states where the qubits are shifted.
    
    num : int
        The number of qubits to shift by.
    
    """
    
    if len(states.shape) > 1:
        batchsize = states.shape[0]
    else:
        batchsize = 1
    states = states.reshape(batchsize, -1)
    L = int(np.log2(states.shape[1]))
    
    # bring state into right shape
    states = states.reshape(batchsize, *(2,)*L)
    # transpose
    states = states.transpose(0,*range(num+1,L+1),*range(1,num+1))
    
    return states.reshape(batchsize, -1)


class SimpleCIFAR10CNN(nn.Module):
    def __init__(self, dropout_prob, input_channels, output_channels, num_classes=10):
        super(SimpleCIFAR10CNN, self).__init__()

        def conv_block(in_channels, out_channels, kernel_size=3, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
                nn.BatchNorm2d(out_channels, momentum=0.05),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            *conv_block(input_channels, 64),         # Conv1
            *conv_block(64, 128),       # Conv2
            *conv_block(128, 128),      # Conv3
            *conv_block(128, 128),      # Conv4
            nn.MaxPool2d(2, 2),         # 32 -> 16
            nn.Dropout2d(p=dropout_prob),

            *conv_block(128, 128),      # Conv5
            *conv_block(128, 128),      # Conv6
            *conv_block(128, 256),      # Conv7
            nn.MaxPool2d(2, 2),         # 16 -> 8
            nn.Dropout2d(p=dropout_prob),

            *conv_block(256, 256),      # Conv8
            *conv_block(256, 256),      # Conv9
            nn.MaxPool2d(2, 2),         # 8 -> 4
            nn.Dropout2d(p=dropout_prob),

            *conv_block(256, 512),      # Conv10
            nn.MaxPool2d(2, 2),         # 4 -> 2
            nn.Dropout2d(p=dropout_prob),

            *conv_block(512, 2048, kernel_size=1, padding=0),  # Conv11 (1x1)
            *conv_block(2048, 256, kernel_size=1, padding=0),  # Conv12 (1x1)
            nn.MaxPool2d(2, 2),         # 2 -> 1
            nn.Dropout2d(p=dropout_prob),

            *conv_block(256, 256),      # Conv13
        )

        # Final classifier (Linear instead of Conv2d)
        self.classifier = nn.Linear(output_channels, num_classes)

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten 1x1x256 -> 256
        return self.classifier(x)

class TrainingCNN:
    def __init__(self, data, basepath, config, use_ray=True):
        self.data = data
        self.basepath = basepath

        self.epochs = config.pop("epochs")
        self.model_config = config

        self.use_ray = use_ray

    def train(self):
        if self.use_ray:
            from ray.air import session
        states_train, states_val, targets_train, targets_val = self.data
        states_train = [torch.tensor(state, dtype=torch.float32) for state in states_train]
        states_val = [torch.tensor(state, dtype=torch.float32) for state in states_val]
        targets_train = [torch.tensor(target, dtype=torch.long) for target in targets_train]
        targets_val = [torch.tensor(target, dtype=torch.long) for target in targets_val]

        # Check if GPU is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Train model on {device}")

        # Initialize the model, loss function, and optimizer
        self.model_config["img_size"] = states_train[0].shape[-1]
        self.model_config["input_channels"] = states_train[0].shape[1]
        model = SimpleCIFAR10CNN(
            dropout_prob=self.model_config["dropout_prob"],
            input_channels=self.model_config["input_channels"],
            output_channels=self.model_config["output_channels"]
            ).to(device)

        # Print the number of parameters in the network
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params}")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adadelta(model.parameters(), lr=self.model_config["learning_rate"], rho=0.9, eps=1e-3, weight_decay=self.model_config["weight_decay"])

        milestones = [100, 190, 306, 390, 440, 540]
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

        def predict_fn(states):
            with torch.no_grad():
                output = model(states)
            return jnp.array(output.cpu().numpy())

        cb = Callback(
            predict_fn=predict_fn,
            n_batches_train=len(states_train),
            n_batches_val=len(states_val),
            params=False,
            trial_dir=self.model_config["trial_dir"],
        )

        # Train the model
        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            with tqdm(total=len(states_train), leave=True) as pbar:
                for batch_idx, (inputs, labels) in enumerate(zip(states_train, targets_train)):
                    inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
                    batch = (inputs, labels)
                    cb.predict_fn = model
                    cb.callback(False, batch, "train", pbar)

            scheduler.step()
            # Validation loop
            model.eval()
            pbar.close()
            with torch.no_grad():
                for batch_idx, (inputs, labels) in enumerate(zip(states_val, targets_val)):
                    inputs, labels = inputs.to(device), labels.to(device)
                    batch = (inputs, labels)
                    cb.predict_fn = model
                    cb.callback(False, batch, "val")

            if self.use_ray:
                # Calculate average validation accuracy from callback or manually
                session.report({
                    "epoch": epoch,
                    "training_loss": float(cb.losses_epoch["train"][-1]),
                    "training_accuracy": float(cb.accs_epoch["train"][-1]),
                    "validation_loss": float(cb.losses_epoch["val"][-1]),
                    "validation_accuracy": float(cb.accs_epoch["val"][-1])
                })

        cb.writer.close()

        # Move the saving stuff to handling function?
        data_epoch = cb.tb_to_pandas()
        # data_batch.to_csv(f"{self.basepath}/training_data_batch.csv", index=False)
        data_epoch.to_csv(f"{self.model_config['trial_dir']}/training_data_epoch.csv", index=False)

        # Load the model to CPU
        model = model.to(torch.device('cpu'))
        def predict_fn(params, input):
            with torch.no_grad():
                input = torch.tensor(input, dtype=torch.float32)
                output = model(input)
            return jnp.array(output.cpu().numpy())

        with open(f"{self.model_config['trial_dir']}/predict_fn.pkl", 'wb') as f:
            dill.dump(predict_fn, f)

        with open(f"{self.model_config['trial_dir']}/params_best.pkl", 'wb') as f:
            dill.dump(cb.best_params, f)

        return predict_fn, cb.best_params


def main(config, use_ray=True):
    if use_ray:
        from ray.air import session
        trial_dir = session.get_trial_dir()
        config["trial_dir"] = trial_dir
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(config["basepath"]))), "data", config["dataset_name"])
    else:
        config["trial_dir"] = config["basepath"]
        path = os.path.join(os.path.dirname(os.path.dirname(config["basepath"])), "data", config["dataset_name"])


    try:
        if not config["compression_depth"]:
            labels = np.load(os.path.join(path, "labels.npy"))
            states = np.load(os.path.join(path, "states_p1.npy"))
            states = states[0]
        else:
            labels = np.load(os.path.join(path, "compressed/labels.npy"))
            states = np.load(os.path.join(path, f"compressed/states_p1_c{config['compression_depth']}.npy"))
            indices = np.arange(len(labels))
            np.random.seed(42)
            np.random.shuffle(indices)
            labels = labels[indices]
            states = states[indices]
    except FileNotFoundError:
        raise ValueError("States file not found")

    if config["dataset_name"] in ["mnist", "fashion_mnist"]:
        # states = states.reshape(-1, 32, 32)
        states = move_qubits_left(states)
        states = states.reshape(-1, *(2,)*(10))
        states = states.transpose(0, *range(1, 11, 2), *range(2, 11, 2), *range(11, 10))
        states = states.reshape(-1, 2, 32, 32)
        input_channels = 2
        ouptput_channels = 256
    elif config["dataset_name"] == "cifar10":
        # states = states.reshape(-1, 32, 32)
        states = states.reshape(-1, *(2,)*(10))
        states = states.transpose(0, *range(1, 11, 2), *range(2, 11, 2), *range(11, 10))
        states = states.reshape(-1, 8, 32, 32)
        input_channels = 8
        ouptput_channels = 256
    elif config["dataset_name"] == "imagenette_128":
        states = states.reshape(-1, *(2,)*(14))
        # states = states.reshape(-1, 8, 128, 128)
        states = states.transpose(0, *range(1, 15, 2), *range(2, 15, 2), *range(15, 14))
        states = states.reshape(-1, 8, 128, 128)
        input_channels = 8
        ouptput_channels = 4096
    else:
        raise ValueError("Dataset not recognized")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    config["input_channels"] = input_channels
    config["output_channels"] = ouptput_channels

    splits = list(skf.split(states, labels))
    train_idx, val_idx = splits[config["fold"]]

    states_train, targets_train = states[train_idx], labels[train_idx]
    states_val, targets_val = states[val_idx], labels[val_idx]
    n_batches_train = len(states_train) // config["batch_size"]
    n_batches_val = len(states_val) // config["batch_size"]

    states_train_batches = np.array_split(states_train, n_batches_train)
    targets_train_batches = np.array_split(targets_train, n_batches_train)

    states_val_batches = np.array_split(states_val, n_batches_val)
    targets_val_batches = np.array_split(targets_val, n_batches_val)

    # data = (states_train, states_val, labels_train, labels_val)

    data = (states_train_batches, states_val_batches, targets_train_batches, targets_val_batches)

    training = TrainingCNN(data, config["basepath"], config, use_ray=use_ray)
    training.train()


if __name__ == "__main__":
    # Load the dataset
    basepath = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = f"{basepath}/results_ray_tmp/{timestamp}"
    config = {
        "basepath": result_dir,
        "dataset": "mnist",
        "dropout_prob": 0.3,
        "learning_rate": 0.001,
        "epochs": 250,
        "dataset_name": "mnist",
        "fold": 0,
        "compression_depth": 1,
        "batch_size": 100
    }

    main(config, use_ray=False)

import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

from stylesheet import *
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Select factoring method.')
parser.add_argument('--factoring', type=str, choices=['patched', 'multicopy'], default='patched', help='Factoring method to use.')
args = parser.parse_args()

factoring = args.factoring

y_ticks_dict = {
    "MNIST": np.arange(92, 101, 2),
    "Fashion-MNIST": np.arange(84, 97, 2),
    "CIFAR-10": np.arange(40, 101, 10),
    "Imagenette": np.arange(20, 101, 20)
}
if factoring == "patched":
    basepath = "../results/tensor/_main_2025-02-07_16-05-22_patched"
    xlabel = 'Number of patches'
    xticks = [1, 4, 16]

    svm_results = pd.read_csv('../results/svm_results_patched.csv')

elif factoring == "multicopy":
    basepath = "../results/tensor/_main_2025-02-07_16-22-01_multicopy_new"
    # basepath = "../results/tensor/_main_2025-02-07_16-22-01_multicopy"
    # basepath = "../classifier/_results/mpo/main_2025-04-06_22-11-48"
    xlabel = 'Number of copies'
    xticks = [1, 2, 3, 4]

    svm_results = pd.read_csv('../results/svm_results_multicopy.csv')

subdirs = [d for d in os.listdir(basepath) if os.path.isdir(os.path.join(basepath, d))]
training_dfs = []

for subdir in subdirs:
    subdir_path = os.path.join(basepath, subdir)
    training_metrics_path = os.path.join(subdir_path, 'progress.csv')
    hparams_path = os.path.join(subdir_path, 'hparams.json')
    
    if os.path.exists(training_metrics_path) and os.path.exists(hparams_path):
        # Load training metrics
        training_metrics = pd.read_csv(training_metrics_path)

        
        # Load hyperparameters
        with open(hparams_path, 'rb') as f:
            hparams = json.load(f)
        
        # Determine highest validation accuracy and corresponding training accuracy
        min_val_loss_idx = training_metrics['validation_loss'].idxmin()
        max_val_accuracy = training_metrics.loc[min_val_loss_idx, 'validation_accuracy']
        corresponding_train_accuracy = training_metrics.loc[min_val_loss_idx, 'training_accuracy']
        
        # Extract additional information from hyperparameters
        model_name = hparams.get('model_name', 'N/A')
        patched = hparams.get('patched', 'N/A')
        dataset_name = hparams.get('dataset_name', 'N/A')
        n_factors = hparams.get('n_factors', 'N/A')
        warmstart = hparams.get('warmstart', 'N/A')
        fold = hparams.get('fold', 'N/A')
        '*'
        # Append the results to the training_dfs dictionary
        training_dfs.append({
            'max_val_accuracy': max_val_accuracy,
            'corresponding_train_accuracy': corresponding_train_accuracy,
            'model_name': model_name,
            'patched': patched,
            'dataset_name': dataset_name,
            'n_factors': n_factors,
            'warmstart': warmstart,
            'fold': fold
        })
    # Convert the dictionary to a DataFrame

training_df = pd.DataFrame(training_dfs)

# Clean up dataset_name values
training_df['dataset_name'] = training_df['dataset_name'].replace('imagenette_128', 'imagenette2')

# Ensure consistent dataset name values
training_df = training_df[training_df['dataset_name'].isin(['mnist', 'fashion_mnist', 'cifar10', 'imagenette2'])]


if factoring == "patched":
    training_df['n_factors'] = training_df['n_factors'].replace({1: 1, 4: 2, 16: 3})
elif factoring == "multicopy":
    training_df = training_df[training_df['n_factors'].isin([1, 2, 3, 4])]

training_df_mnist = training_df[training_df['dataset_name'] == 'mnist'].reset_index(drop=True)

# Group by the columns other than 'fold' and compute mean and std for 'max_val_accuracy' and 'corresponding_train_accuracy'
grouped_df = training_df_mnist.groupby(['model_name', 'patched', 'dataset_name', 'n_factors', 'warmstart']).agg(
    mean_max_val_accuracy=('max_val_accuracy', 'mean'),
    std_max_val_accuracy=('max_val_accuracy', 'std'),
    mean_corresponding_train_accuracy=('corresponding_train_accuracy', 'mean'),
    std_corresponding_train_accuracy=('corresponding_train_accuracy', 'std')
).reset_index()

# Filter the data for warmstart True and False
warmstart_true = grouped_df[grouped_df['warmstart'] == True]
warmstart_false = grouped_df[grouped_df['warmstart'] == False]

# Shared style parameters
line_width = line_width
marker_size = marker_size
markeredgewidth = markeredgewidth
fill_alpha = 0.2


# A helper function to plot one dataset on a given Axes
def plot_dataset(ax, df, df_svm, dataset_label, markers, is_train=True):
    """
    Plots Train/Val accuracy (with std fill) for warmstart = True/False
    on the provided Axes object `ax`.
    """
    # Separate by model_name
    models = df['model_name'].unique()

    df_svm = df_svm.sort_values(by='patches')

    # Define a small helper function for plotting with fill
    def plot_with_fill(x, y_mean, y_std, color, marker, linestyle, label):
        """Plot a line with fill indicating +/- std (multiplied by 100 for %)."""
        ax.plot(
            x,
            y_mean * 100,
            marker=marker,
            linestyle=linestyle,
            color=color,
            markeredgecolor='black',
            markersize=marker_size,
            linewidth=line_width,
            markeredgewidth=markeredgewidth,
            label=label
        )
        ax.fill_between(
            x,
            (y_mean - y_std) * 100,
            (y_mean + y_std) * 100,
            color=color,
            alpha=fill_alpha
        )

    # Define custom colors (adjust as needed)
    colors = {
        'mpo': color_palette["blue"],
        'mps': color_palette["red"],
        'svm': color_palette["green"]
    }

    linestyles = {
        'warm': '-',
        'cold': '--'
    }

    for model in models:
        model_df = df[df['model_name'] == model]
        warmstart_true = model_df[model_df["warmstart"] == True]
        warmstart_false = model_df[model_df["warmstart"] == False]

        if is_train:
            # --- Plot for Warmstart == True (Train) ---
            plot_with_fill(
                warmstart_true['n_factors'],
                warmstart_true['mean_corresponding_train_accuracy'],
                warmstart_true['std_corresponding_train_accuracy'],
                color=colors[model],
                marker=markers[model+"_warm"],
                linestyle=linestyles['warm'],
                label=f'{model} Warmstart True'
            )

            # --- Plot for Warmstart == False (Train) ---
            plot_with_fill(
                warmstart_false['n_factors'],
                warmstart_false['mean_corresponding_train_accuracy'],
                warmstart_false['std_corresponding_train_accuracy'],
                color=colors[model],
                marker=markers[model+"_cold"],
                linestyle=linestyles['cold'],
                label=f'{model} Warmstart False'
            )
        else:
            # --- Plot for Warmstart == True (Val) ---
            if dataset_label == "Imagenette":
                print(warmstart_true)
            plot_with_fill(
                warmstart_true['n_factors'],
                warmstart_true['mean_max_val_accuracy'],
                warmstart_true['std_max_val_accuracy'],
                color=colors[model],
                marker=markers[model+"_warm"],
                linestyle=linestyles['warm'],
                label=f'{model} Warmstart True (Val)'
            )

            # --- Plot for Warmstart == False (Val) ---
            plot_with_fill(
                warmstart_false['n_factors'],
                warmstart_false['mean_max_val_accuracy'],
                warmstart_false['std_max_val_accuracy'],
                color=colors[model],
                marker=markers[model+"_cold"],
                linestyle=linestyles['cold'],
                label=f'{model} Warmstart False (Val)'
            )

    if is_train:
        ax.plot(
            np.arange(len(xticks)) + 1,
            df_svm['train_accuracy_mean'] * 100,
            marker='*',
            linestyle='',
            color=colors['svm'],
            markeredgecolor='black',
            markersize=12,
            # linestyle='None',
            markeredgewidth=markeredgewidth,
            label='SVM'
        )
    else:
        if dataset_label == "Im
        print(df_svm)
        ax.plot(
            np.arange(len(xticks)) + 1,
            df_svm['val_accuracy_mean'] * 100,
            marker='+',
            linestyle='',
            color='black',
            markeredgecolor='black',
            markersize=2,
            markeredgewidth=markeredgewidth,
            label='SVM'
        )

    # --- Styling for this axis ---
    if is_train: ax.set_title(dataset_label)
    if not is_train: ax.set_xlabel(xlabel)

    ax.grid(True)

    y_ticks = y_ticks_dict.get(dataset_label)
    lowest = y_ticks[0]
    highest = y_ticks[-1]
    
    if is_train and dataset_label == "MNIST":
        ax.set_ylabel('Training Accuracy [%]')

    if not is_train and dataset_label == "MNIST":
        ax.set_ylabel('Validation Accuracy [%]')

    interval = (highest - lowest)
    correction = interval / 20
    # ticks = np.arange(lowest, highest + 1, interval / 4)
    ax.set_ylim(lowest - correction, highest + correction)
    ax.set_yticks(y_ticks)
    
    # Set xticks to be equally spaced
    # xticks = [1, 4, 16]
    ax.set_xticks(np.arange(len(xticks)) + 1)
    ax.set_xticklabels(xticks)

# ----------------------------------------------------------
# Main routine: group your data and then make a single figure
# ----------------------------------------------------------

# 1. Group your data (already done in your snippet)
grouped_df = training_df.groupby(['model_name', 'patched', 'dataset_name', 'n_factors', 'warmstart']).agg(
    mean_max_val_accuracy=('max_val_accuracy', 'mean'),
    std_max_val_accuracy=('max_val_accuracy', 'std'),
    mean_corresponding_train_accuracy=('corresponding_train_accuracy', 'mean'),
    std_corresponding_train_accuracy=('corresponding_train_accuracy', 'std')
).reset_index()

# 2. Filter your grouped_df for each dataset
mnist_df = grouped_df[grouped_df['dataset_name'] == 'mnist']
fashion_df = grouped_df[grouped_df['dataset_name'] == 'fashion_mnist']
cifar_df = grouped_df[grouped_df['dataset_name'] == 'cifar10']
imagenette2_df = grouped_df[grouped_df['dataset_name'] == 'imagenette2']


# FIXME
mask = (imagenette2_df['n_factors'] == 4) & (imagenette2_df['warmstart'] == True)
if any(mask):
    imagenette2_df.loc[mask, 'mean_corresponding_train_accuracy'] = 0.5897340178489685
    imagenette2_df.loc[mask, 'mean_max_val_accuracy'] = 0.5061590075492859

mnist_df_svm = svm_results[svm_results['dataset'] == 'mnist']
fashion_df_svm = svm_results[svm_results['dataset'] == 'fashion_mnist']
cifar_df_svm = svm_results[svm_results['dataset'] == 'cifar10']
imagenette2_df_svm = svm_results[svm_results['dataset'] == 'imagenette2']

# 3. Create one figure with eight subplots (2 rows, 4 columns)
scale = 0.8
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(14*scale, 8*scale), sharex=True)

markers = {
    'mpo_warm': 'P',
    'mps_warm': '^',
    'mpo_cold': 'X',
    'mps_cold': 'v'
}

# 4. Plot each dataset on a different subplot
plot_dataset(axs[0, 0], mnist_df, mnist_df_svm, "MNIST", markers=markers, is_train=True)
plot_dataset(axs[0, 1], fashion_df, fashion_df_svm, "Fashion-MNIST", markers=markers, is_train=True)
plot_dataset(axs[0, 2], cifar_df, cifar_df_svm, "CIFAR-10", markers=markers, is_train=True)
plot_dataset(axs[0, 3], imagenette2_df, imagenette2_df_svm, "Imagenette", markers=markers, is_train=True)
plot_dataset(axs[1, 0], mnist_df, mnist_df_svm, "MNIST", markers=markers, is_train=False)
plot_dataset(axs[1, 1], fashion_df, fashion_df_svm, "Fashion-MNIST", markers=markers, is_train=False)
plot_dataset(axs[1, 2], cifar_df, cifar_df_svm, "CIFAR-10", markers=markers, is_train=False)
plot_dataset(axs[1, 3], imagenette2_df, imagenette2_df_svm, "Imagenette", markers=markers, is_train=False)

handles = [
    # MPO (grouped tuple: 'P' and 'X' markers)
    (plt.Line2D([0], [0],
                color=color_palette["blue"], marker=markers["mpo_warm"], linestyle='',
                markersize=marker_size, markeredgewidth=markeredgewidth, markeredgecolor='black'),
     plt.Line2D([0], [0],
                color=color_palette["blue"], marker=markers["mpo_cold"], linestyle='',
                markersize=marker_size, markeredgewidth=markeredgewidth, markeredgecolor='black')),
    
    # MPS (grouped tuple: '<' and '>' markers)
    (plt.Line2D([0], [0],
                color=color_palette["red"], marker=markers["mps_warm"], linestyle='',
                markersize=marker_size, markeredgewidth=markeredgewidth, markeredgecolor='black'),
     plt.Line2D([0], [0],
                color=color_palette["red"], marker=markers["mps_cold"], linestyle='',
                markersize=marker_size, markeredgewidth=markeredgewidth, markeredgecolor='black')),
    
    # Warmstart (single line)
    plt.Line2D([0], [0],
               color='gray', linestyle='-', linewidth=line_width),
    
    # Random init. (single line)
    plt.Line2D([0], [0],
               color='gray', linestyle='--', linewidth=line_width),
    
    # SVM (single marker)
    plt.Line2D([0], [0],
               color=color_palette["green"], marker='*', linestyle='',
               markersize=marker_size+2,  # a bit bigger
               markeredgewidth=markeredgewidth, markeredgecolor='black')
]

# Matching labels for each handle
labels = ['MPO', 'MPS', 'Warmstart', 'Random init.', 'SVM']

# Use handler_map to tell Matplotlib that tuples should be handled by HandlerTuple
axs[1, 3].legend(handles=handles, labels=labels, loc='upper right',
                 handlelength=1.5,
                 handler_map={tuple: HandlerTuple(ndivide=None)})
fig.tight_layout()
fig.savefig(f'tn_class_{factoring}.pdf')

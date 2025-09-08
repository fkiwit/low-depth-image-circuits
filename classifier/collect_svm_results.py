import numpy as np
import os
import pandas as pd

mode = "multicopy"
# Directory containing the CSV files
# csv_dir = f"/raid/home/q615005/dev/data_compression/results/svm_{mode}"

csv_basedir = f"/raid/home/q615005/dev/data_compression/classifier/_results/svm"
extensions = [
    "mnist_20250407_104725",
    "fashion_mnist_20250407_104725",
    "cifar10_20250407_104725",
    "imagenette_128_20250407_104725"
]

# Load all CSV files into a list of DataFrames
csv_files = []
for ext in extensions:
    csv_dir = os.path.join(csv_basedir, ext)
    if os.path.exists(csv_dir):
        csv_files.extend([os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')])
dataframes = [pd.read_csv(file) for file in csv_files]

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)
print(combined_df)

# Compute mean and standard deviation of train and validation accuracy for each dataset and patch
stats = combined_df.groupby(["dataset", "factors"]).agg(
    train_accuracy_mean=("train_accuracy", "mean"),
    std_train_accuracy=("train_accuracy", "std"),
    val_accuracy_mean=("val_accuracy", "mean"),
    std_val_accuracy=("val_accuracy", "std")
).reset_index()

print(combined_df)
# Rename the column 'factors' to 'patches'
stats.rename(columns={"factors": "patches"}, inplace=True)
print(stats)
# Save the statistics to a CSV file
output_csv = os.path.join(f"/raid/home/q615005/dev/data_compression/results/svm_results_{mode}.csv")
stats.to_csv(output_csv)

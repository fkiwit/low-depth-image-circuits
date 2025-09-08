import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=str, default="../results/_TEST")
args = parser.parse_args()

data = np.load(os.path.join(args.basedir, 'compressed.npy'), allow_pickle=True)

df = pd.DataFrame(data)

losses_bfgs, parameterss_bfgs = [], []
for i, (filepath, filename) in enumerate(zip(df['filepath'], df['filename'])):
    loss_bfgs = np.load(f"{os.path.join(filepath, filename)}_bfgs_loss.npy")
    parameters_bfgs = np.load(f"{os.path.join(filepath, filename)}_bfgs_params.npy")
    losses_bfgs.append(loss_bfgs)
    parameterss_bfgs.append(parameters_bfgs)

df["loss_bfgs"] = losses_bfgs
df["parameters_bfgs"] = parameterss_bfgs

structured_array = df.to_records(index=False)
np.save(os.path.join(args.basedir, 'compressed_bfgs.npy'), structured_array)

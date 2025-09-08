import numpy as np
import os
import pandas as pd

path = "/global/homes/f/fkiwit/dev/data_compression/data/randoms"

npy_files = [f for f in os.listdir(path) if f.endswith('.npy')]

data = []
for file in npy_files:
    overlaps = np.load(os.path.join(path, file))
    qubits = int(file.split('_')[2])
    i = int(file.split('_')[3].split('.')[0])
    infidelity = 1 - np.abs(overlaps[1, -1]) ** 2
    data.append({'qubits': qubits, 'infidelity': infidelity, 'i': i})
print(overlaps.shape)
df = pd.DataFrame(data)
print(len(df))
print(df.head())
# quit()
means = df.groupby('qubits').mean()
stds = df.groupby('qubits').std()
percentiles = df.groupby('qubits').quantile([0.25, 0.75])

summary_df = pd.DataFrame({
    'qubits': means.index,
    'mean_infidelity': means['infidelity'],
    'std_infidelity': stds['infidelity'],
    '25th_percentile': percentiles.loc[(slice(None), 0.25), 'infidelity'].values,
    '75th_percentile': percentiles.loc[(slice(None), 0.75), 'infidelity'].values,
})
print(summary_df)
summary_df.to_csv(os.path.join(path, '_summary.csv'), index=False)

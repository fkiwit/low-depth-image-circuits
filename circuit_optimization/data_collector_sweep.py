import os
import pandas as pd
import numpy as np
from utils.image_encodings import FRQI_decoding, move_qubits_left, FRQI_RGBa_decoding
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=str, default="../data/cifar10/orthogonal_3_20250223-124430")

args = parser.parse_args()

all_dictionaries = []
all_files = os.walk(args.basedir)

with open(os.path.join(args.basedir, 'config.yml'), 'r') as f:
    config = yaml.safe_load(f)

counter = 0
for root, dirs, files in all_files:
    for file in files:
        if file.endswith('.npz'):
            file_path = os.path.join(root, file)
            data = np.load(file_path)
            data_dict = {key: data[key] for key in data}
            data_dict['loss_sweep_conv'] = data_dict['loss_sweep'][-1]
            counter += 1
            all_dictionaries.append(data_dict)
            if counter % 100 == 0:
                print(counter)

combined_dataframe = pd.DataFrame(all_dictionaries)

def convert_to_compressed(input_compressed):
    decoded = []
    for image in input_compressed:
        if config['color'] == 'rgb':
            decoded.append(FRQI_RGBa_decoding(image, shape=config['shape'], indexing="hierarchical"))
        else:
            image = move_qubits_left(image)
            decoded.append(FRQI_decoding(image, shape=config['shape'], indexing="hierarchical"))

    return decoded

index = config['layers'] - 1

combined_dataframe["image_sweep"] = convert_to_compressed(combined_dataframe[f"state_{index}"])
combined_dataframe["image_original"] = convert_to_compressed(combined_dataframe["frqi_target"])

structured_array = combined_dataframe.to_records(index=False)
print(structured_array.dtype.names)

print(np.unique(np.stack(structured_array["class_int"]), return_counts=True))

np.save(os.path.join(args.basedir, 'compressed.npy'), structured_array)

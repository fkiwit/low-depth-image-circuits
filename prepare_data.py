import os
import numpy as np
import tensorflow_datasets as tfds
from skimage.util import view_as_blocks
from PIL import Image
import pickle
import argparse

from compression.utils.image_encodings import FRQI_RGBa_encoding, FRQI_encoding, move_qubits_right
from compression.utils.image_compression import calc_MPS

def process_dataset(dataset, shape, color_mode):
    images, labels = [], []
    for image, label in dataset:
        image = image.numpy().squeeze()
        pil_img = Image.fromarray(image)

        if color_mode == "rgb" and pil_img.mode == "L":
            pil_img = pil_img.convert("RGB")
        elif color_mode == "gray":
            pil_img = pil_img.convert("L")

        width, height = pil_img.size
        new_size = min(width, height)
        pil_img = pil_img.crop(((width - new_size) // 2, (height - new_size) // 2, (width + new_size) // 2, (height + new_size) // 2))
        pil_img = pil_img.resize(shape, resample=Image.BILINEAR)
        images.append(np.array(pil_img) / 255.0)
        labels.append(label.numpy())
    return np.array(images), np.array(labels)

def divide_into_patches(images, block_shape):
    patches = [view_as_blocks(img, block_shape=block_shape).reshape(-1, *block_shape) for img in images]
    patches = np.array(patches).transpose(1, 0, *range(2, patches[0].ndim + 1))
    return patches

def encode_patches(images_patches, block_shape, n_patches, n_samples):
    if len(block_shape) == 3:
        assert block_shape[-1] == 3
        images_patches = images_patches.reshape((-1, *block_shape))
        images_patches = FRQI_RGBa_encoding(images_patches, indexing="hierarchical")
    elif len(block_shape) == 2:
        images_patches = images_patches.reshape((-1, *block_shape))
        images_patches = FRQI_encoding(images_patches, indexing="hierarchical")
        images_patches = move_qubits_right(images_patches)
    else:
        raise ValueError("Unsupported number of color channels")
    return images_patches.reshape((n_patches, n_samples, -1))

def main(dataset_name, n_patches=1, data_dir="./data"):
    dataset = tfds.load(dataset_name, as_supervised=True, data_dir=f"{data_dir}/_downloads")
    if 'validation' in dataset:
        train_data = dataset['train']
        val_data = dataset['validation']
        full_dataset = train_data.concatenate(val_data)
    elif 'test' in dataset:
        train_data = dataset['train']
        test_data = dataset['test']
        full_dataset = train_data.concatenate(test_data)
    else:
        full_dataset = dataset['train']

    config = {
        "imagenette/320px": ((128, 128), "rgb"),
        "cifar10": ((32, 32), "rgb"),
        "mnist": ((32, 32), "gray"),
        "fashion_mnist": ((32, 32), "gray")
    }
    shape, color_mode = config[dataset_name]
    images, labels = process_dataset(full_dataset, shape, color_mode)
    print("images processed")
    if color_mode == "gray":
        block_shape = (shape[0] // int(np.sqrt(n_patches)), shape[1] // int(np.sqrt(n_patches)))
    elif color_mode == "rgb":
        block_shape = (shape[0] // int(np.sqrt(n_patches)), shape[1] // int(np.sqrt(n_patches)), 3)
    images_patches = divide_into_patches(images, block_shape)
    output_dir = f"{data_dir}/{dataset_name.split('/')[0]}"
    print(f"Saving dataset to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "labels.npy"), labels)
    np.save(os.path.join(output_dir, f"images_p{n_patches}.npy"), images_patches)
    print("Encoding images as quantum states")
    states_patches = encode_patches(images_patches, block_shape, n_patches, len(images))
    np.save(os.path.join(output_dir, f"states_p{n_patches}.npy"), states_patches)

    mps_state = []
    for patch in states_patches:
        mps_state += calc_MPS(patch)[0]

    with open(os.path.join(output_dir, f"mps_p{n_patches}.pkl"), "wb") as f:
        pickle.dump(mps_state, f)
    print(f"Saved dataset to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process dataset with quantum encoding")
    parser.add_argument("--dataset_name", type=str, choices=["imagenette/320px", "cifar10", "mnist", "fashion_mnist"], help="Dataset name")
    parser.add_argument("--n_patches", type=int, default=1, help="Number of patches per image")
    args = parser.parse_args()
    main(args.dataset_name, args.n_patches)

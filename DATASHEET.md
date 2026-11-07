# Datasheet: Low-Depth Image Circuits

This datasheet follows the template of Gebru et al., *Datasheets for Datasets*, Communications of the ACM 64(12), 86--92, 2021, doi:10.1145/3458723. Questions that do not apply to this dataset are answered briefly with the reason.

The dataset accompanies the paper F. J. Kiwit, B. Jobst, A. Luckow, F. Pollmann, C. A. Riofrío, *Typical Machine Learning Datasets as Low-Depth Quantum Circuits*, Quantum Science and Technology (2025), arXiv:2505.03399. It is distributed via PennyLane at https://pennylane.ai/datasets/collection/low-depth-image-circuits, and the code that produced it is available at https://github.com/fkiwit/low-depth-image-circuits.

## Motivation

**For what purpose was the dataset created?** Loading classical data into a quantum state is a bottleneck for quantum machine learning: exact preparation of an amplitude-encoded state requires a number of gates that grows exponentially with the number of qubits, and published quantum machine learning studies therefore mostly work on heavily downscaled or dimensionality-reduced inputs. This dataset provides, for every image of four standard image-classification datasets, a low-depth quantum circuit that approximately prepares the image's quantum-encoded state. The goal is a standardized benchmark on which quantum machine learning models can be trained and compared on complete datasets at full image resolution, without each user having to repeat the costly circuit optimization.

**Who created the dataset and on behalf of which entity?** Florian J. Kiwit (Ludwig Maximilian University Munich; BMW Group), Bernhard Jobst (Technical University of Munich; Munich Center for Quantum Science and Technology), Andre Luckow (Ludwig Maximilian University Munich; BMW Group), Frank Pollmann (Technical University of Munich; Munich Center for Quantum Science and Technology), and Carlos A. Riofrío (BMW Group).

**Who funded the creation of the dataset?** As acknowledged in the accompanying paper:

- Compute resources of the National Energy Research Scientific Computing Center (NERSC) under allocations ERCAP0032477 and ERCAP0029512, "Characterization and Middleware for Hybrid Quantum-HPC Applications". The circuit optimization was run on the NERSC Perlmutter system.
- The Bavarian State Ministry of Economic Affairs, BenchQC project, grant DIK0425/03.
- F. Pollmann acknowledges support of the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany's Excellence Strategy EXC-2111-390814868, the European Research Council (ERC) under the European Union's Horizon 2020 research and innovation program (grant agreement No. 771537), and the Munich Quantum Valley, which the Bavarian state government supports with funds from the Hightech Agenda Bayern Plus.

## Composition

**What do the instances represent?** Each instance corresponds to one image of a source dataset. It consists of the exact quantum state that encodes the image and the parameters of quantum circuits that approximately prepare this state, at two circuit depths.

**How many instances are there?**

| Dataset | Instances | Resolution | Encoding | Qubits |
| --- | --- | --- | --- | --- |
| MNIST | 70,000 | 32x32, grayscale | FRQI | 11 |
| Fashion-MNIST | 70,000 | 32x32, grayscale | FRQI | 11 |
| CIFAR-10 | 60,000 | 32x32, color | MCRQI | 13 |
| Imagenette | 13,394 | 128x128, color | MCRQI | 17 |

**Does the dataset contain all possible instances or is it a sample?** All instances. Every image of the official training and test (for Imagenette: training and validation) splits of each source dataset is included; nothing was filtered or subsampled.

**What data does each instance consist of?** Each dataset file contains the following attributes, one entry per image:

| Attribute | Content |
| --- | --- |
| `exact_state` | the exactly encoded state vector (FRQI for grayscale, MCRQI for color images) that the circuits approximate |
| `labels` | the class label of the source image (integer, ten classes) |
| `params_d4`, `params_d8` | the RY rotation angles of the circuit with 4 and with 8 layers |
| `circuit_layout_d4`, `circuit_layout_d8` | the gate sequence of the 4- and 8-layer circuit as (gate name, wires), consisting of RY and CNOT gates; the layout is shared by all instances of a dataset |
| `fidelities_d4`, `fidelities_d8` | the fidelity between the state prepared by the 4- or 8-layer circuit and `exact_state` |

The circuits are built from two-qubit gates in the special orthogonal group SO(4), arranged in sequential (staircase) layers, and compiled into RY rotations and CNOT gates. Because all gates are real, the prepared states are real-valued, as are the FRQI and MCRQI states they approximate.

The fidelity attributes are contained in the published files but are not yet listed among the attributes on the PennyLane dataset pages.

**Is there a label associated with each instance?** Yes, the class label of the source dataset (ten classes for each dataset).

**Is any information missing from individual instances?** No.

**Are relationships between individual instances made explicit?** No relationships exist beyond the shared class labels. The instances are stored in the order of the source dataset as provided by TensorFlow Datasets, with the official training split first, followed by the official test split (for Imagenette: the validation split).

**Are there recommended data splits?** The official splits of the source datasets are recoverable from the instance order:

| Dataset | Training (first instances) | Test (remaining instances) |
| --- | --- | --- |
| MNIST | 60,000 | 10,000 |
| Fashion-MNIST | 60,000 | 10,000 |
| CIFAR-10 | 50,000 | 10,000 |
| Imagenette | 9,469 | 3,925 (official validation split) |

Using these splits makes results comparable to classical results reported on the source datasets. The accompanying paper instead merges both splits into one pool and evaluates classifiers with stratified five-fold cross-validation, with a fixed random seed of 42; results obtained with that protocol are comparable to the benchmark numbers of the paper.

**Are there any errors, sources of noise, or redundancies?** The circuits prepare the encoded states only approximately. The approximation error of every instance is given by `fidelities_d4` and `fidelities_d8`; it decreases with the number of layers and depends on the image content. The distribution of these fidelities per dataset is reported in the accompanying paper. Users should treat the circuit states as a compressed version of the images, not as exact copies. Any label errors of the source datasets are inherited unchanged.

**Is the dataset self-contained?** Yes for its intended use: the exact state of every image is included, so the original (preprocessed) pixel values can be recovered from `exact_state` without access to the source datasets. The data is derived from external datasets, whose licences apply (see Distribution).

**Does the dataset contain confidential data?** No.

**Does the dataset contain data that might be offensive?** Not intentionally. The source datasets are widely used benchmarks; Imagenette is a subset of ImageNet, whose photographs were collected from the web, and the dataset inherits whatever content those photographs contain.

**Does the dataset relate to people?** Not in the sense of the template. It contains no personal attributes. MNIST consists of digits handwritten by individuals, and ImageNet photographs may show people, but neither the source datasets' labels nor this dataset identify individuals. The remaining questions about subpopulations, identification, and sensitive data are therefore not applicable.

## Collection Process

**How was the data associated with each instance acquired?** It was derived from the source datasets MNIST, Fashion-MNIST, CIFAR-10, and Imagenette (variant `imagenette/320px`), all obtained through TensorFlow Datasets. No new images were collected.

**What mechanisms were used?** Each image was encoded into its exact quantum state (FRQI or MCRQI with hierarchical pixel ordering). The exact state was then converted to a matrix product state, and a circuit of sequential layers of two-qubit gates was optimized classically to approximate it, by iteratively adding layers and refining all gates with a sweeping optimization. The mechanism was validated by computing the fidelity of every optimized circuit state with the exact state, which is stored with each instance. The method is described in detail in the accompanying paper.

**If the dataset is a sample, what was the sampling strategy?** Not applicable; the complete datasets were processed.

**Who was involved in the data collection process?** The authors listed above. No crowdworkers or annotators were involved, and no labels were created.

**Over what timeframe was the data collected?** The circuit optimizations were run between February and April 2025. The source images were created considerably earlier: MNIST and CIFAR-10 date from before 2010, Fashion-MNIST from 2017, and the ImageNet photographs underlying Imagenette from before 2010.

**Were any ethical review processes conducted?** No. The dataset is derived from public benchmark datasets and involves no human subjects.

## Preprocessing

**Was any preprocessing done?** Yes. Each image was converted to grayscale (MNIST, Fashion-MNIST) or RGB (CIFAR-10, Imagenette), cropped to its largest centered square, resized with bilinear interpolation to 32x32 pixels (MNIST, Fashion-MNIST, CIFAR-10; the original MNIST and Fashion-MNIST images have 28x28 pixels) or 128x128 pixels (Imagenette), and scaled to the range [0, 1]. The training and test splits were concatenated in this order, so that the official splits remain recoverable (see Composition). The pixel values were then mapped to quantum states as described under Collection Process.

**Was the raw data saved?** The raw images are available from the source datasets, see Distribution. The preprocessed pixel values are recoverable from `exact_state`.

**Is the preprocessing software available?** Yes: `prepare_data.py` (preprocessing and encoding) and `circuit_optimization/` (circuit optimization and export) in https://github.com/fkiwit/low-depth-image-circuits.

## Uses

**Has the dataset been used for any tasks already?** In the accompanying paper, the circuits are used to benchmark quantum classifiers (variational quantum circuits, quantum kernel support vector machines, and tensor-network classifiers) against a classical convolutional neural network, on both the exact and the circuit-prepared states.

**Is there a repository that links to papers that use the dataset?** No list is maintained. Works using the dataset can be found through the citations of the accompanying paper.

**What other tasks could the dataset be used for?** Benchmarking other quantum machine learning models on realistic image data; benchmarking quantum hardware by executing the state-preparation circuits and comparing against the stored exact states; studying state preparation, circuit compilation, and the entanglement structure of image data.

**Is there anything about the composition that might impact future uses?** The circuit states are approximations. Results obtained on them reflect both the model under study and the approximation error, which users should report, e.g. by stating the circuit depth used and the fidelity distribution. Accuracy differences between the 4- and 8-layer circuits can serve to quantify this effect.

**Are there tasks for which the dataset should not be used?** It should not be used where exact pixel values matter, since the circuit states are lossy. For the Imagenette part, the usage restrictions of ImageNet apply (see Distribution).

## Distribution

**Will the dataset be distributed to third parties?** Yes, it is publicly available.

**How will the dataset be distributed?** Via PennyLane Datasets at https://pennylane.ai/datasets/collection/low-depth-image-circuits as four datasets, `low-depth-mnist`, `low-depth-fashion-mnist`, `low-depth-cifar-10`, and `low-depth-imagenette`, each loadable with `qml.data.load`. The metadata of the four entries is maintained in the repository https://github.com/XanaduAI/pennylane-datasets under `content/other/low-depth-image-circuits/`.

The dataset has no DOI of its own. It is identified and cited through the DOI of the accompanying paper, doi:10.1088/2058-9565/ae0123, which also names the PennyLane location of the data. The code that produced the dataset is archived on Zenodo, doi:10.5281/zenodo.24039339 (all versions).

**When will the dataset be distributed?** Low-Depth MNIST was released on 19 March 2025 and the other three datasets on 23 April 2025, all as version 0.1 (initial public release). The metadata was last modified on 23 April 2026.

**Under which licence is the dataset distributed?** The four datasets are distributed under the Creative Commons Attribution-ShareAlike 4.0 International licence (CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/). The citation requested on the dataset pages is the accompanying paper, doi:10.1088/2058-9565/ae0123.

The code in https://github.com/fkiwit/low-depth-image-circuits is licensed under the Apache License 2.0.

**Have any third parties imposed restrictions on the data?** The dataset is derived from the following datasets, whose terms apply to the derived data:

| Source | Terms |
| --- | --- |
| MNIST | no licence stated on the original distribution; third-party mirrors label it differently (e.g. MIT on Hugging Face) |
| Fashion-MNIST | MIT licence |
| CIFAR-10 | no licence stated on the original distribution; the creators ask users to cite Krizhevsky (2009) |
| Imagenette | Apache 2.0 for the subset definition; the images are taken from ImageNet, whose terms of access restrict use to non-commercial research and educational purposes and leave the copyright with the original image owners |

The CC BY-SA 4.0 licence of this dataset permits commercial use, but it cannot lift restrictions of the source data. For Low-Depth Imagenette in particular, users must additionally observe the ImageNet terms of access.

**Do any export controls or other regulatory restrictions apply?** Not to the knowledge of the authors.

## Maintenance

**Who will be supporting, hosting, and maintaining the dataset?** The data files are hosted by Xanadu as part of PennyLane Datasets. The dataset is maintained by Florian J. Kiwit.

**How can the maintainer be contacted?** Through the issue tracker of the code repository, https://github.com/fkiwit/low-depth-image-circuits/issues.

**Is there an erratum?** The data itself has no known errors. The metadata on the PennyLane dataset pages (version 0.1) contains the following errors:

- The description of `params_d4` refers to "a circuit layout with a depth of 8"; the parameters belong to the 4-layer layout `circuit_layout_d4`.
- On the MNIST, Fashion-MNIST, and CIFAR-10 pages, the usage notes state that `exact_state` encodes "Imagenette images"; it encodes the images of the respective dataset.
- The attributes `fidelities_d4` and `fidelities_d8` are contained in the files but not listed.

**Will the dataset be updated?** No new instances, datasets, or circuit depths are planned. Errors in the data or the metadata, including those listed above, are corrected in new versions on PennyLane Datasets (0.x), which are recorded in the changelog of the dataset pages and announced in the code repository.

**Will older versions continue to be supported?** No. A correction replaces the previous version, and the changelog documents what changed.

**If others want to extend the dataset, is there a mechanism for them to do so?** Contributions to the published dataset are not accepted. The code in the GitHub repository can be used to optimize circuits for further datasets, resolutions, or circuit depths and to publish them as separate derived datasets; derived datasets must be shared under the terms of CC BY-SA 4.0 and of the respective source datasets.

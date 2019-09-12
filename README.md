# Distilling Intractable Generative Models

This project includes all the matlab code and data files that are needed to reproduce the experiments in the paper:

> G. Papamakarios and I. Murray, _Distilling Intractable Generative Models_, NeurIPS workshop on Probabilistic Integration, 2015.
[[pdf]](https://gpapamak.github.io/papers/distilling_generative_models.pdf) [[bibtex]](https://gpapamak.github.io/bibtex/distilling_generative_models.bib)

## How to get started

* In the main folder, run ```install.m``` to add all necessary paths to the matlab path.
* Run the scripts in the folder ```experiments``` to run the experiments and visualize the results.

## What this folder contains

#### `experiments`
The scripts that run experiments and show results. In particular:

- `nade_fit_to_rbm.m`
Performs distillation, by training a NADE to mimic an RBM.

- `nade_fit_to_rbm_results.m`
Having trained a NADE with the previous script, run this one to visualize how well the distillation worked.

- `nade_estimate_rbm_logZ.m`
Uses the NADE trained above to estimate the partition function of the RBM.

- `nade_print_features.m`
Visualizes the features learnt by the RBM and the mimicking NADE.

- `nade_print_mnist_samples.m`
Shows some samples from the RBM and the NADE.

#### `nade`
The implementation of NADE. Includes code for training it and drawing samples from it.

#### `rbm`
The implementation of the RBM. Includes code that samples from it with Gibbs sampling.

#### `optimization`
It contains optimization routines, including _AdaDelta_ that is used in training NADE.

#### `util`
Various utility functions.

#### `outdir`
Folder where to save results, e.g. the trained NADEs. It already contains the binarized MNIST dataset and an RBM trained on it, taken from [here](http://www.utstat.toronto.edu/~rsalakhu/rbm_ais.html).

# episdet-transformer

This repository includes an implementation of a Transformer Neural Network for any-order epistasis detection.
This implementation has been devised to make use of network interpretation strategies (e.g., attention scores) to identify epistatic interactions and target various AI accelerators (e.g., GPUs, TPUs).

## Importance of Epistasis Detection

Genome-Wide Association Studies (GWAS) analyze the influence of individual genetic markers on well-known diseases. However, this approach ignores gene-gene interactions (epistasis), which are of utmost importance to understand complex diseases. Therefore, finding new SNP associations has a high impact on our understanding of these diseases, as well as precision medicine, contributing to improve personalized healthcare.

## Setup

### Requirements

* TensorFlow (version 2.6 or more recent)

## Usage example

Running the Transformer on a single GPU with a synthetic example dataset (fourth order interaction) with 1000 SNPs, 1600 samples (800 controls, 800 cases), splitting the dataset in 6 partitions, combining them in pairs, and selecting the top 5% SNPs:

```bash
$ python3 ED_Transformer.py -path Add4.txt -partitions 6 -comb 2 -top 0.05 -device GPU -n 1  
```

The Transformer implementation requires the following arguments:

```bash
$ -path PATH             a path to epistasis datasets (can be a .txt file, a folder with files, or a zip with files)
$ -partitions PARTITIONS the number of partitions
$ -comb COMB             the combination order to merge partitions
$ -top TOP               best SNP percentage to report after training (between 0 and 1)
$ -device DEVICE         the device to use (e.g., CPU, GPU, TPU)
$ -n N                   number of devices
$ -sparsity SPARSITY     an optional float (between 0 and 1) for the sparsity percentage on the transformer attention modules. Defaults to 0.9.
$ -epochs EPOCHS         an optional int for the epochs to train the transformer. Defaults to 15.  
```



## CG-AMP

Code and Datasets for "CG-AMP: Antimicrobial peptide prediction based on contrastive learning and gated convolutional neural network"

### Datasets

- dataset/AMPlify/AMPlify.fasta is a FASTA-formatted sequence file containing all peptide sequences from the AMPlify dataset. The dataset is divided into training, validation, and test sets in a ratio of 7:1:2.
- dataset/AMPScanner/AMPScanner.fasta is a FASTA-formatted sequence file containing all peptide sequences from the DAMP dataset. The dataset is divided into training, validation, and test sets in a ratio of 2:1:2.

### Model

model.pth is a trained CG-AMP model obtained using the AMPlify dataset. It can be used for inference or further evaluation on independent test sets.

### Usage

#### Environment Requirement

The code has been tested running under Python 3.7.16. The required packages are as follows:

- numpy == 1.21.6

- pandas == 1.3.5

- torch == 1.13.1+cu116

- pandas == 1.3.5

- tqdm == 4.66.5

- scipy == 1.7.3

- scilit-learn == 1.0.2

	(typo: the package name on pip is **scikit-learn**)

Additional packages needed to run this repo as-is:

- biopython (for FASTA parsing)
- einops (used in attention blocks)

If you do not already have the precomputed ESM2 embedding files (e.g. `dataset/AMPlify/amplify_esm2.npy`),
you must generate them before training/testing.

Quick install (recommended):

```bash
pip install -r requirements.txt
```

  

#### Training

```
git clone https://github.com/ghli16/CG-AMP
python main.py
```

Users can use their **own data** to train prediction models. 

#### Testing

```
git clone https://github.com/ghli16/CG-AMP
python test.py
```

#### Generating the missing ESM2 embeddings

This repository references ESM2 embeddings stored in `.npy` files (for example `dataset/AMPlify/amplify_esm2.npy`).
They are not included in the repo, so you need to generate them locally:

```bash
python prepare_esm2_embeddings.py --fasta dataset/AMPlify/AMPlify.fasta --out dataset/AMPlify/amplify_esm2.npy
```

Notes:

- The default model is `facebook/esm2_t33_650M_UR50D` (embedding dim 1280), which matches this repo's `model.pth`.
- Downloading this model is large and can take a while; using a GPU is strongly recommended.

Users can use their **own data** to train prediction models. 


#### Contact

Please feel free to contact us if you need any help.

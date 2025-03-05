# Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning  
This repository contains the implementation of our work, titled ["Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning"](https://openreview.net/forum?id=6wOmHdwCC4&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2025%2FConference%2FAuthors%23your-submissions)). Our work has been accepted for publication in ICLR 2024.

We proposed a novel and simple Divergence-enhanced Knowledge-guided Prompt Tuning (DeKg) method to address this issue. The key insight is that the bias toward pre-training can be alleviated by encouraging the independence between the learnable and the crafted prompt. Specifically, DeKg employs the Hilbert-Schmidt Independence Criterion (HSIC) to regularize the learnable prompts, thereby reducing their dependence on prior general knowledge, and enabling divergence induced by target knowledge.
![main figure](framework.pdf)
# How to Run

## GPU memory needed

All the experiments is able to run on a single graphic card. However, **if you want to get results on ImageNet, the memory on any single graphic card should be larger than 24 GB.** Around 12 GB is enough for other datasets. 


## How to Install
This code is built on top of the toolbox [Dassl.ProGrad.pytorch](https://github.com/BeierZhu/Prompt-align/tree/main/Dassl.ProGrad.pytorch). You can prepare the environment as follows:

```
# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Create a conda environment
conda create -n dassl python=3.7

# Activate the environment
conda activate dassl

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
# Please make sure you have installed the gpu version due to the speed.
# For example:
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

After that, run `pip install -r requirements.txt` under `DeKg/` to install a few more packages required by [CLIP](https://github.com/openai/CLIP) (this should be done when `dassl` is activated). Then, you are ready to go.

Follow [DATASETS.md](DATASETS.md) to install the datasets.


## Generalization From Base to New Classes

You will need `base2new_train_main.sh`, `base2new_test_main.sh`, and `run.sh`. The scripts with the prefix `base2new_train` train a model on base classes while the ones with the prefix `base2new_test` evaluate the trained model on new classes.  The valid names are the files' names in `DeKg/configs/datasets/`.

Below we provide an example on how to evaluate the model on ImageNet.

```bash
bash base2new_train.sh stanford_cars 8.0 6.0
bash base2new_test.sh stanford_cars 
```

When the evaluation is done, you can use `parse_test_res.py` to automatically calculate the average results.

Download checkpoints [here](https://drive.google.com/file/d/17xboHHQoDUCCqItX8ghJu7lDJijDkv6Q/view?usp=drive_link).

Then, to get the average performance on the base classes, run

```bash
python parse_test_res.py output/base2new/train_base/stanford_cars/shots_16/DeKg/vit_b16_ep100_ctxv1
```

To get the average performance on the new classes, run

```bash
python parse_test_res.py output/base2new/test_new/stanford_cars/shots_16/DeKg/vit_b16_ep100_ctxv1 --test-log
```

## Citation
```bibtex
@inproceedings{
li2025divergenceenhanced,
title={Divergence-enhanced Knowledge-guided Context Optimization for Visual-Language Prompt Tuning},
author={Yilun Li and Miaomiao Cheng and Xu Han and Wei Song},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=6wOmHdwCC4}
}
```


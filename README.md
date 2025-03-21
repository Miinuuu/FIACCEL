# FIACCEL in PyTorch

## Introduction
This is a PyTorch implementation of the FIACCEL

<p align='center'>
    <img src="fig/val_psnr.png" width="500" center>
</p>

## Running the code
To facilitate training we use `pytorch_lightning` and `hydra` to manage model configurations.

To create an environment with all the requirements for running this code, we suggest first verifying that the PyTorch version in the `fiaccel.yml` file is appropriate for your OS and server and selecting an appropriate one if it is not.
Then create a new `conda` environment by running:
```
conda env create -f fiaccel.yml
conda activate fiaccel
```
You can train a model with `model_train.py` but remember to modify the the training configuration (`config/train_config_*.yaml`) and include paths to an appropriate training dataset.


### Training data

fiaccel is trained on a proprietary dataset with one million internet video clips, each comprising 3 frames.

- We use the publcicly available Vimeo-90k dataset ([Xue et al., 2019](https://arxiv.org/abs/1711.09078)), which is commonly used dataset for video frame interpolation

## FIACCEL: Implementation details

## TOP
<p align='center'>
    <img src="fig/FIACCEL_HW.png" width="1000" center>
</p>
## TOP_SUB
<p align='center'>
    <img src="fig/TOP_SUB_FI.png" width="1000" center>
</p>
## Ecore
<p align='center'>
    <img src="fig/Ecore.png" width="1000" >
</p>

## Dcore
<p align='center'>
    <img src="fig/Dcore.png" width="1000" >
</p>

## PE
<p align='center'>
    <img src="fig/PE.png" width="1000" >
</p>

## Experimental_settings
<p align="center">
  <img src="fig/experimental_settings.png" width="1000" >>
</p>


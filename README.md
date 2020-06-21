# Pytorch VQ-VAE Implementation

Unofficial Pytorch implementation of [Neural Discrete Representation Learning (2017)](https://arxiv.org/abs/1711.00937)

> Based on [zalandoresearch/pytorch-vq-vae](https://github.com/zalandoresearch/pytorch-vq-vae) repository.

## Install 

``` shell
$ pip install -r requirements.txt
```

## Usage 

You can experiment with CIFAR-10 data by running the following command:

``` shell
python train.py
```

In the case of hyperparameters, you can experiment by tuning in `config.json`.

## Experiments (TBD)

## To Do

- [ ] VectorQuantize EMA ver.
- [ ] Warm Start (Save model & Load pre-trained model)
- [ ] Inference (unseen random size image)
- [ ] Visualization for Interpretability (Interactive Mode)

## Contact

If you have any question about the code, feel free to email me at `subinium@gmail.com`.
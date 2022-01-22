# Deep-Learning-Based Autofocusing of Bright-Field Microscopic Imagery  

## Introduction

This is the repo of for Deep-Learning-Based Autofocusing of Bright-Field Microscopic Imagery. The framework of the codes is modified from **[ PyTorch-GAN](https://github.com/eriklindernoren/PyTorch-GAN)** by [eriklindernoren](https://github.com/eriklindernoren/PyTorch-GAN/commits?author=eriklindernoren) (Pix2Pix part). The codes are developed by Yucheng Zhang(Email:yucheng.zhang009@gmail.com). This work is the co-work of Mr. Xingchen Dong(E-Mail:[xingchen.dong@tum.de](mailto:xingchen.dong@tum.de)).

![GAN](C:\Users\HONOR\Desktop\GAN.png)

**Input:** Images of [3,2048,1536] size. In the training, they would be cut into [3,256,256] tensors.

**Output:** Images of [3,2048,1536] size.

**Model:** Generator based on Pix2Pix U-Net model and modified. Discriminator as common CNN network.

**Loss function:** Mix of MSE and MS-SSIM

The generator (U-Net) and discriminator model is defined in `model.py`.

## How to train the network

First install the environment in `requirements.txt`. Note that this network needs CUDA to accelerate the training process, and you may need to change the pixel settings according to your images in function `cutimage` in `utils.py`.

Then set all other parameters like learning rate, epochs in `cfg.yaml` file. 

Then run the `train.py`.

## Using pretrained model to generate new image

Similarly, set the parameters in `cfg.yaml` file. The paths to be set are arguments with "cat_" beginning.

Then run the `predict.py`.


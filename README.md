# Skin40 Few-shot Classification

## Introduction

We investigate the effect of **the number of fully-connected layers** on the model generalization performance, and find that using more fully-connected layers can obtain better result when the finetuning dataset differs too much from the original pre-trained dataset images. In the experiments, our method obtains 76.8 Top-1 accuracy using CNN for backbone and 84.79 Top-1 accuracy using Swin Transformer for backbone on the Skin40 dataset.

## Detail

We try various models based on **CNN** and **Vit** backbones, including **ResNet**, **VGG**, **DenseNet**, and **Swin Transformer**. As a result, applying on several model structures is able to demonstrate the generalization ability of our method.

In addition, we employ **Normalization**, **Morphological Processing**,  **Data Augmentation** such as **RandomFlip**, **BatchMixup**, **RandomResizedCrop**, etc.

## Dataset

Skin40 is a dataset that contains 40 skin disease classes. The challenge is that only 60 images are available for each class. Due to the limited data, 5-fold cross-validation should be used to evaluate the model.

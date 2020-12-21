# Embedded Systems and Applications Final Project 2020
self-supervised contrastive learning with improved InfoNCE
# Requirements
* python == 3.6
* pytorch == 1.1.0
* torchvision == 0.3.0
* tensorboard == 2.4.0
* numpy == 1.19.2
* pillow == 8.0.1
* tqdm == 4.52.0
* yaml == 0.2.5
* yacs == 0.1.8
# Experiment environment 
* 2 Titan X GPUs
* CUDA 10.1
# Self-Supervised Pre-training
* MoCo with EqCo (K=512, alpha=16348) and DCL, dataset : CIFAR10, encoder : resnet18
```sh
python3 main.py --method moco --data cifar --arch resnet18 --use_eqco true --eqco_k 512 --use_dcl true
                --world-size 1 --rank 0 --dist-url tcp://localhost:10001
                --experiment_name moco_dcl_eqco_512_cifar_r18
```


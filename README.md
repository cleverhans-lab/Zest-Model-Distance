# Zest-Model-Distance

This repository is an implementation of the paper [A Zest of LIME: Towards Architecture-Independent Model Distances](https://openreview.net/forum?id=OUz_9TiTv9j). In this paper, we propose an architecture-independent distance metric that measures the similarity between ML models by comparing their global behaviors, approximated using LIME. For more details, please read the paper.


### Dependency
Our code is implemented and tested on PyTorch. Following packages are used:
```
numpy
pytorch==1.6.0
torchvision==0.7.0
lime
```
Note that `lime` can be installed by `pip install lime`. For more details, please see [this](https://github.com/marcotcr/lime).

### Train
To train 2 models so later we can compare them:
```
python main.py --id 1
python main.py --id 2
```
The default setting is ResNet20 trained on CIFAR10 following the implementation of [this](https://github.com/akamaster/pytorch_resnet_cifar10). If you want to use other dataset, architecture, and/or hyperparameters, please modify the arguments of `main.py` and (in case the architecture is not defined in `model.py` and is not from torchvision) add the code for the custom architecture to `model.py`.


### Compare
To compute the distance between the two models we just trained:
```
python compare.py --path1 models/ckpt_CIFAR10_1/model_epoch_200 --path2 models/ckpt_CIFAR10_2/model_epoch_200
```

In case of using other dataset and/or architecture, please modify the arguments of `compare.py` accordingly, and then set `path1` and `path2` to where the two models are stored. Note that the datasets and architectures used to train the two models do NOT need to be the same.

There is a `dist` argument, you could use one or more of `1`, `2`, `inf`, `cos` (if more than one, separate them by space). The first 3 are corresponding l_p norms, while `cos` is cosine distance. The default is to use all of them.

The distances computed by `compare.py` are unnormalized. If you want to normalize them, please train a few models with the same setting (in terms of dataset, architecture, and hyperparameters) and different random seeds, and use the average distances to do normalization. For example, if you want to know if model A (`CIFAR10, ResNet20`) is extracted from model B (`CIFAR10, ResNet50`), then you need to obtain a few (recommend to be >=5) pairs of models that one is trained on `CIFAR10+ResNet20` and the other is trained on `CIFAR10+ResNet50`, and compute the distances averaged over all the pairs to be the `reference distance`. If `distance(model A, model B) / reference distance < 1`, then we can claim model A and B are related (e.g., due to model extraction).


### Questions or suggestions
If you have any questions or suggestions, feel free to raise an issue or send me an email at nickhengrui.jia@mail.utoronto.ca


### Citing this work
If you use this repository for academic research, you are highly encouraged (though not required) to cite our paper:
```
@inproceedings{
jia2022a,
title={A Zest of {LIME}: Towards Architecture-Independent Model Distances},
author={Hengrui Jia and Hongyu Chen and Jonas Guan and Ali Shahin Shamsabadi and Nicolas Papernot},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=OUz_9TiTv9j}
}
```

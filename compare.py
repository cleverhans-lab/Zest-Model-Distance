import torchvision
import argparse
import numpy as np
import utils
import train
import model


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--model1', type=str, default="resnet20")
parser.add_argument('--path1', type=str, default="models/ckpt_CIFAR10_1/model_epoch_200")
parser.add_argument('--model2', type=str, default="resnet20")
parser.add_argument('--path2', type=str, default="models/ckpt_CIFAR10_2/model_epoch_200")
parser.add_argument('--dist', type=str, nargs='+', default=['1', '2', 'inf', 'cos'],
                    help='metric for computing distance, cos, 1, 2, or inf')
arg = parser.parse_args()

try:
    architecture_1 = eval(f"model.{arg.model1}")
except:
    architecture_1 = eval(f"torchvision.models.{arg.model1}")
try:
    architecture_2 = eval(f"model.{arg.model2}")
except:
    architecture_2 = eval(f"torchvision.models.{arg.model2}")

train_fn1 = train.TrainFn(batch_size=128, dataset=arg.dataset, architecture=architecture_1)
train_fn2 = train.TrainFn(batch_size=128, dataset=arg.dataset, architecture=architecture_2)

train_fn1.load(arg.path1)
train_fn2.load(arg.path2)
train_fn1.lime()
train_fn2.lime()

if not isinstance(arg.dist, list):
    dist = [arg.dist]
else:
    dist = arg.dist

distance = np.array(utils.parameter_distance(train_fn1.lime_mask, train_fn2.lime_mask, order=dist, lime=True))
print(distance)

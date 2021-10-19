import torchvision
import argparse
import model
import train


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--model', type=str, default="resnet20")
parser.add_argument('--save-freq', type=int, default=10, help='frequence of saving checkpoints')
parser.add_argument('--id', type=str, default='')
arg = parser.parse_args()

try:
    architecture = eval(f"model.{arg.model}")
except:
    architecture = eval(f"torchvision.models.{arg.model}")


train_fn = train.TrainFn(arg.lr, arg.batch_size, arg.dataset, architecture, exp_id=arg.id, save_freq=arg.save_freq)

for epoch in range(arg.epochs):
    train_fn.train(epoch)

train.validate(arg.dataset, train_fn.net)

import numpy as np
import os
import torch
import torch.optim as optim
import utils
import model
import lime_pytorch


seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


class TrainFn:
    def __init__(self, lr=0.01, batch_size=128, dataset='SVHN', architecture=model.resnet20, exp_id=None,
                 model_dir=None, save_freq=None, dec_lr=None,
                 trainset=None, lime_data_name=None, save_name=None, num_class=10,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        self.save_freq = save_freq
        self.num_class = num_class
        self.dataset = dataset
        self.batch_size = batch_size
        if lime_data_name is None:
            self.lime_data_name = f"{dataset}_lime"
        else:
            self.lime_data_name = f"{lime_data_name}_lime"
        self.device = device
        if save_name is None:
            save_name = self.dataset
        if save_freq is not None and save_freq > 0:
            if not os.path.exists("models"):
                os.mkdir("models")
            self.save_dir = os.path.join("models", f"ckpt_{save_name}_{exp_id}")
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)
        else:
            self.save_dir = None

        if not os.path.exists("data"):
            os.mkdir("data")
        if trainset is None:
            self.trainset = utils.load_dataset(self.dataset, True, download=True)
        else:
            self.trainset = trainset
        testset = utils.load_dataset(self.dataset, False, download=True)
        self.train_loader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=0, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                      shuffle=False, num_workers=0, pin_memory=True)

        self.net = architecture()
        if num_class != 10:
            self.net.linear = torch.nn.Linear(64, num_class)
        num_batch = self.trainset.__len__() / self.batch_size
        self.net.to(self.device)
        if self.dataset == 'MNIST':
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr)
            self.scheduler = None
        elif self.dataset == 'CIFAR10':
            if dec_lr is None:
                dec_lr = [100, 150]
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.1,
                                                            milestones=[round(i * num_batch) for i in dec_lr])
        elif self.dataset == 'CIFAR100':
            if dec_lr is None:
                dec_lr = [60, 120, 160]
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, gamma=0.2,
                                                            milestones=[round(i * num_batch) for i in dec_lr])
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr, eps=1e-8)
            self.scheduler = None
        if isinstance(self.trainset.__getitem__(0)[1], int) or 'extract' not in exp_id:
            self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        else:
            print("using MSE loss")
            self.criterion = torch.nn.MSELoss().to(self.device)

        if model_dir is not None:
            state = torch.load(model_dir)
            self.net.load_state_dict(state['net'])
            self.optimizer.load_state_dict(state['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(state['scheduler'])

        self.lime_data = None
        self.lime_mask = None
        self.lime_segment = None
        self.lime_dataset = None
        self.ref_dataset = None

    def save(self, epoch=None, save_path=None):
        assert epoch is not None or save_path is not None
        if save_path is None:
            save_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}")
        if not os.path.exists(save_path):
            if self.scheduler is None:
                state = {'net': self.net.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'lime': self.lime_mask
                         }
            else:
                state = {'net': self.net.state_dict(),
                         'optimizer': self.optimizer.state_dict(),
                         'scheduler': self.scheduler.state_dict(),
                         'lime': self.lime_mask
                         }
            torch.save(state, save_path)

    def load(self, path):
        states = torch.load(path)
        self.net.load_state_dict(states['net'])
        self.optimizer.load_state_dict(states['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(states['scheduler'])
        if 'lime' in states:
            self.lime_mask = states['lime']

    def train(self, epoch):
        self.net.train()
        if self.save_dir is not None:
            epoch_path = os.path.join(self.save_dir, f"model_epoch_{epoch + 1}")
            if os.path.exists(epoch_path):
                print(f"loading checkpoints for epoch {epoch + 1}")
                self.load(epoch_path)
                return True
            if self.save_freq > 1 and (epoch + 1) % self.save_freq != 0:
                next_save_epoch = ((epoch + 1) // self.save_freq + 1) * self.save_freq
                next_path = os.path.join(self.save_dir, f"model_epoch_{next_save_epoch}")
                if os.path.exists(next_path):
                    return True
        for _, data in enumerate(self.train_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        if self.save_freq is not None and (epoch + 1) % self.save_freq == 0 and self.save_freq > 0:
            self.save(epoch)
        return False

    def lime(self, save_name=None, cat=True):
        if save_name is None:
            save_name = self.lime_data_name
        self.net.eval()
        if self.lime_data is None:
            self.lime_data = lime_pytorch.prepare_lime_ref_data(save_name, self.trainset, self.batch_size)
        if self.lime_segment is None:
            self.lime_segment = lime_pytorch.prepare_lime_segment(save_name, self.lime_data, self.trainset)
        if self.ref_dataset is None or self.lime_dataset is None:
            self.ref_dataset, self.lime_dataset = lime_pytorch.prepare_lime_dataset(save_name, self.lime_data,
                                                                                    self.lime_segment)
        self.lime_mask = lime_pytorch.compute_lime_signature(self.net, self.ref_dataset, self.lime_dataset, cat=cat)
        self.net.train()


def validate(dataset, val_model, batch_size=128):
    device = torch.device('cuda:0' if next(val_model.parameters()).is_cuda else 'cpu')
    val_model.eval()
    testset = utils.load_dataset(dataset, False, download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0, pin_memory=True)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = val_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy: {100 * correct / total} %')
    return correct / total

import copy
import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms


def get_parameters(net, numpy=False):
    # get weights from a torch model as a list of numpy arrays
    parameter = torch.cat([i.data.reshape([-1]) for i in list(net.parameters())])
    if numpy:
        return parameter.cpu().numpy()
    else:
        return parameter


def get_model(model, architecture, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    state = torch.load(model)
    net = architecture()
    net.load_state_dict(state['net'])
    net.to(device)
    return net


def consistent_type(model, architecture=None,
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), half=False,
                    linear=False, lime=False):
    if isinstance(model, str):
        state = torch.load(model)
        if linear:
            weights = state['linear'].reshape(-1)
        elif lime:
            weights = state['lime']
            if isinstance(weights, np.ndarray):
                weights = torch.from_numpy(weights).float()
        else:
            assert architecture is not None
            net = architecture()
            net.load_state_dict(state['net'])
            weights = get_parameters(net)
    elif isinstance(model, np.ndarray):
        if lime:
            weights = torch.from_numpy(model).float()
        else:
            weights = torch.from_numpy(model).reshape(-1).float()
    elif not isinstance(model, torch.Tensor):
        weights = get_parameters(model)
    else:
        if not lime:
            weights = model.reshape(-1)
        else:
            weights = model
    if half:
        if half == 2:
            weights = weights.type(torch.IntTensor).type(torch.FloatTensor)
        else:
            weights = weights.half()
    return weights.to(device)


def compute_distance(a, b, order):
    if order == 'inf':
        order = np.inf
    if order == 'cos' or order == 'cosine':
        return (1 - torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).cpu().numpy()
    else:
        if order != np.inf:
            try:
                order = int(order)
            except:
                raise TypeError("input metric for distance is not understandable")
        return torch.norm(a - b, p=order).cpu().numpy()


def parameter_distance(model1, model2, order=2, architecture=None, half=False, linear=False, lime=False):
    # compute the difference between 2 checkpoints
    weights1 = consistent_type(model1, architecture, half=half, linear=linear, lime=lime)
    weights2 = consistent_type(model2, architecture, half=half, linear=linear, lime=lime)
    if not isinstance(order, list):
        orders = [order]
    else:
        orders = order
    res_list = []
    if lime:
        temp_w1 = copy.copy(weights1)
        temp_w2 = copy.copy(weights2)
    for o in orders:
        if lime:
            weights1, weights2 = lime_align(temp_w1, temp_w2, o)
        res = compute_distance(weights1, weights2, o)
        if isinstance(res, np.ndarray):
            res = float(res)
        res_list.append(res)
    return res_list


def load_dataset(dataset, train, download=False):
    try:
        dataset_class = eval(f"torchvision.datasets.{dataset}")
    except:
        raise NotImplementedError(f"Dataset {dataset} is not implemented by pytorch.")

    if "MNIST" in dataset:
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5,), (0.5,))])
    elif dataset == "CIFAR100":
        normalize = transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
                                         (0.2673342858792401, 0.2564384629170883, 0.27615047132568404))
        if train:
            transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(), transforms.RandomRotation(15),
                                            transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if train:
            transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4),
                                            transforms.ToTensor(), normalize])
        else:
            transform = transforms.Compose([transforms.ToTensor(), normalize])

    try:
        data = dataset_class(root='./data', train=train, download=download, transform=transform)
    except:
        if train:
            data = dataset_class(root='./data', split="train", download=download, transform=transform)
        else:
            data = dataset_class(root='./data', split="test", download=download, transform=transform)

    return data


def lime_align(w1, w2, order):
    shorter = int(w1.shape[1] <= w2.shape[1])
    w = [w1, w2] if shorter else [w2, w1]
    num_class = w1.shape[1] if shorter else w2.shape[1]
    new_w = [None] * num_class
    dist = np.zeros([w[0].shape[1], w[1].shape[1]])
    for j in range(w[0].shape[1]):
        for k in range(w[1].shape[1]):
            dist[j, k] = compute_distance(w[0][:, j], w[1][:, k], order)
    upper_bound = np.max(dist) + 1e10
    for i in range(w[0].shape[1]):
        ind1, ind2 = np.argmin(dist) // dist.shape[1], np.argmin(dist) % dist.shape[1]
        new_w[ind1] = w[1][:, ind2]
        dist[ind1, :] = upper_bound
        dist[:, ind2] = upper_bound
    new_w = torch.stack(new_w, 1)
    res = [w1, new_w] if shorter else [w2, new_w]
    return res[0].reshape([-1]), res[1].reshape([-1])


def mean_std_to_array(mean, std, rgb_last=True):
    mean = np.array(mean)
    std = np.array(std)
    if rgb_last:
        mean = mean.reshape([1, 1, 1, -1])
        std = std.reshape([1, 1, 1, -1])
    else:
        mean = mean.reshape([1, -1, 1, 1])
        std = std.reshape([1, -1, 1, 1])
    return mean, std

import os
import numpy as np
import torch
import copy
import torchvision
from lime.wrappers.scikit_image import SegmentationAlgorithm
from utils import mean_std_to_array


def get_reference_dataset(lime_data, ref_data, segment, fudged_image):
    sub_dataset = []
    for sample in lime_data:
        temp = copy.deepcopy(ref_data)
        zeros = np.where(sample == 0)[0]
        mask = np.zeros(segment.shape).astype(bool)
        for z in zeros:
            mask[segment == z] = True
        temp[mask] = fudged_image[mask]
        sub_dataset.append(temp)
    return np.stack(sub_dataset)


def prepare_lime_dataset(save_name, ref_data=None, segment=None, mean=np.array([0, 0, 0]), num_samples=1000):
    if os.path.exists(f"data/{save_name}/ref_dataset.npy") and os.path.exists(f"data/{save_name}/lime_dataset.npy"):
        ref_dataset = np.load(f"data/{save_name}/ref_dataset.npy", allow_pickle=True)
        lime_dataset = np.load(f"data/{save_name}/lime_dataset.npy", allow_pickle=True)[1:]
    else:
        if not os.path.exists(f"data/{save_name}"):
            os.mkdir(f"data/{save_name}")
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)
        fudged_image = np.zeros(ref_data.shape[1:])
        fudged_image += mean.reshape([1, 1, -1])
        ref_dataset = []
        lime_dataset = []

        for i in range(ref_data.shape[0]):
            n_features = np.unique(segment[i]).shape[0]
            lime_data = np.random.randint(0, 2, [num_samples, n_features])
            lime_data[0, :] = 1
            ref_dataset.append(get_reference_dataset(lime_data, ref_data[i], segment[i], fudged_image))
            lime_dataset.append(lime_data)
        np.save(f"data/{save_name}/ref_dataset.npy", ref_dataset, allow_pickle=True)
        np.save(f"data/{save_name}/lime_dataset.npy", [np.zeros([1])] + lime_dataset, allow_pickle=True)
    return ref_dataset, lime_dataset


def label_lime_dataset(lime_dataset, ref_dataset, model):
    device = torch.device('cuda:0' if next(model.parameters()).is_cuda else 'cpu')
    datasets = []
    with torch.no_grad():
        for i in range(len(lime_dataset)):
            lime_data = lime_dataset[i]
            data = ref_dataset[i]
            inputs = torch.from_numpy(data).to(device).permute(0, 3, 1, 2).float()
            outputs = model(inputs).detach().cpu().numpy()
            datasets.append([lime_data, outputs])
    return datasets


def train_lime_model(datasets, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), cat=True):
    weights = []
    with torch.no_grad():
        for data, label in datasets:
            data, label = torch.from_numpy(data).float().to(device), torch.from_numpy(label).float().to(device)
            w = torch.chain_matmul(torch.pinverse(torch.matmul(data.T, data)), data.T, label)
            weights.append(w)
    if cat:
        return torch.cat(weights)
    else:
        return weights


def prepare_lime_ref_data(save_name, dataset=None, data_size=128, ref_data=None, mean=None, std=None):
    if os.path.exists(f"data/{save_name}/ref_data.npy"):
        return np.load(f"data/{save_name}/ref_data.npy")
    else:
        if not os.path.exists(f"data/{save_name}"):
            os.mkdir(f"data/{save_name}")
        if ref_data is None:
            assert dataset is not None
            ref_data = dataset.data[:data_size]
            if hasattr(dataset, 'transform'):
                for transform in dataset.transform.transforms:
                    if isinstance(transform, torchvision.transforms.transforms.Normalize):
                        mean, std = mean_std_to_array(transform.mean, transform.std, ref_data.shape[-1] == 3)
                        ref_data = (ref_data / 255 - mean) / std
        else:
            if mean is not None and std is not None:
                mean, std = mean_std_to_array(mean, std, ref_data.shape[-1] == 3)
                ref_data = (ref_data / 255 - mean) / std

        np.save(f"data/{save_name}/ref_data.npy", ref_data)
        return ref_data


def prepare_lime_segment(save_name, ref_data=None, dataset=None, mean=None, std=None):
    if os.path.exists(f"data/{save_name}/segment.npy"):
        return np.load(f"data/{save_name}/segment.npy")
    else:
        if not os.path.exists(f"data/{save_name}"):
            os.mkdir(f"data/{save_name}")
        if dataset is not None:
            if hasattr(dataset, 'transform'):
                for transform in dataset.transform.transforms:
                    if isinstance(transform, torchvision.transforms.transforms.Normalize):
                        mean, std = mean_std_to_array(transform.mean, transform.std, ref_data.shape[-1] == 3)
                        ref_data = (ref_data * std + mean) * 255
        elif mean is not None and std is not None:
            mean, std = mean_std_to_array(mean, std, ref_data.shape[-1] == 3)
            ref_data = (ref_data * std + mean) * 255
        temp = []
        if ref_data.shape[1] == 3:
            ref_data = np.moveaxis(ref_data, 1, -1)
        segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4, ratio=0.2, max_dist=200)
        for image in ref_data:
            temp.append(segmentation_fn(image))
        lime_segment = np.stack(temp)
        np.save(f"data/{save_name}/segment.npy", lime_segment)
        return lime_segment


def compute_lime_signature(model, ref_dataset, lime_dataset, cat=True):
    labelled_dataset = label_lime_dataset(lime_dataset, ref_dataset, model)
    return train_lime_model(labelled_dataset, cat=cat)

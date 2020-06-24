import numpy as np
import torch
import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import Dataset


def get_imagenet_indices(labels, min_val, max_val):
    return filter_by_class(labels, min_val, max_val)


def get_indices(ix_dir, min_class, max_class, training, dataset_name):
    train_labels = np.load(os.path.join(ix_dir, '{}_indices/{}_train_labels.npy'.format(dataset_name, dataset_name)))
    val_labels = np.load(os.path.join(ix_dir, '{}_indices/{}_val_labels.npy'.format(dataset_name, dataset_name)))
    if training:
        curr_idx = get_imagenet_indices(train_labels, min_val=min_class, max_val=max_class)
        curr_labels = train_labels[np.array(curr_idx)]
    else:
        curr_idx = get_imagenet_indices(val_labels, min_val=min_class, max_val=max_class)
        curr_labels = val_labels[np.array(curr_idx)]
    return curr_idx, curr_labels


def filter_by_class(labels, min_class, max_class):
    """
    Return the indices for the desired classes in [min_class, max_class)
    :param labels: class indices from numpy files
    :param min_class: minimum class included
    :param max_class: maximum class excluded
    :return: list of indices
    """
    ixs = list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])
    return ixs


def get_imagenet_data_loader(dirname, label_dir, split, batch_size=128, shuffle=False, min_class=0, max_class=None,
                             sampler=None, batch_sampler=None, dataset_name='imagenet', return_item_ix=False,
                             num_workers=8):
    # filter out only the indices for the desired class
    if max_class is not None:
        _labels = np.load(
            os.path.join(label_dir, '{}_indices/{}_{}_labels.npy'.format(dataset_name, dataset_name, split)))
        idxs = filter_by_class(_labels, min_class=min_class, max_class=max_class)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataset = datasets.ImageFolder(dirname, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    if batch_sampler is None and sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    dataset = ImagenetDataset(dataset, idxs, return_item_ix)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler)

    if split == 'train':
        print('\nLoading the ' + split + ' data ... ({} samples)'.format(len(idxs)))
    return loader


class ImagenetDataset(Dataset):
    def __init__(self, data, indices, return_item_ix):
        self.data = data
        self.indices = indices
        self.return_item_ix = return_item_ix

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x, y = self.data[index]
        if not self.return_item_ix:
            return x, y
        else:
            return x, y, index


class IndexSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

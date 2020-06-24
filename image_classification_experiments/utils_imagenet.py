import numpy as np
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
from torch.utils.data import Dataset
import image_classification_experiments.utils as utils
import h5py
import os


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
    """Inputs:
        labels: class indices from numpy files
        min_val: xxx
        max_val: xxx
    """
    ixs = list(np.where(np.logical_and(labels >= min_class, labels < max_class))[0])
    return ixs


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


def get_imagenet_data_loader(dirname, label_dir, split, batch_size=128, shuffle=False, min_class=0, max_class=None,
                             sampler=None, batch_sampler=None, dataset_name='imagenet', return_item_ix=False,
                             num_workers=8, augment=False, augmentation_techniques=['crop', 'flip']):
    if max_class is not None:
        _labels = np.load(
            os.path.join(label_dir, '{}_indices/{}_{}_labels.npy'.format(dataset_name, dataset_name, split)))
        idxs = filter_by_class(_labels, min_class=min_class, max_class=max_class)

    print('\nLoading the ' + split + ' data...')
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if split == 'train' and augment:
        if augment:
            print('\nUsing standard data augmentation...')
        augmentation_transforms = []
        if 'crop' in augmentation_techniques:
            print("Using crops!")
            augmentation_transforms.append(transforms.RandomResizedCrop(224))
        if 'flip' in augmentation_techniques:
            print("Using flips!")
            augmentation_transforms.append(transforms.RandomHorizontalFlip())

        augmentation_transforms += [transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        dataset = datasets.ImageFolder(
            dirname,
            transforms.Compose(augmentation_transforms))
    else:
        dataset = datasets.ImageFolder(dirname, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    print("Loading indices")

    # if idxs is None:
    #     idxs = range(len(dataset))

    if batch_sampler is None and sampler is None:

        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    dataset = ImagenetDataset(dataset, idxs, return_item_ix)

    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler)

    print("Filtered Dataset size {}".format(len(idxs)))
    return loader


class IndexSampler(torch.utils.data.Sampler):
    """Samples elements sequentially, always in the same order.
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_imagenet_loader(images_path, min_class, max_class, training, batch_size=256):
    train_labels = np.load('./imagenet_indices/imagenet_train_labels.npy')
    val_labels = np.load('./imagenet_indices/imagenet_val_labels.npy')
    if training:
        curr_idx = get_imagenet_indices(train_labels, min_val=min_class, max_val=max_class)
        images_path += '/train'
    else:
        curr_idx = get_imagenet_indices(val_labels, min_val=min_class, max_val=max_class)
        images_path += '/val'

    loader = get_imagenet_data_loader(images_path, training, curr_idx, batch_size=batch_size, shuffle=False)
    return loader


class FeaturesDataset(Dataset):
    def __init__(self, h5_file_path, split, min_class=0, max_class=None, return_item_ix=False, dataset_name='imagenet',
                 transform=None):
        super(FeaturesDataset, self).__init__()
        self.h5_file_path = h5_file_path
        h5 = h5py.File(h5_file_path, 'r')
        keys = list(h5.keys())
        if 'reconstructions' in keys:
            self.features_key = 'reconstructions'
        elif 'image_features' in keys:
            self.features_key = 'image_features'
        else:
            self.features_key = 'features'

        features = h5[self.features_key]
        self.split = split
        self.min_class = min_class
        self.max_class = max_class
        self.dataset_len = len(features)
        if max_class is not None:
            _labels = np.load('./{}_indices/{}_{}_labels.npy'.format(dataset_name, dataset_name, self.split))
            indices = filter_by_class(_labels, min_class=min_class, max_class=max_class)
            self.dataset_len = len(indices)
        self.return_item_ix = return_item_ix

        self.transform = transform

    def __getitem__(self, index):
        if not hasattr(self, 'features'):
            self.h5 = h5py.File(self.h5_file_path, 'r')
            self.features = self.h5[self.features_key]
            self.labels = self.h5['labels']

        feat = self.features[index]
        if self.transform is not None:
            feat = self.transform(feat)

        if self.return_item_ix:
            return feat, self.labels[index], index
        else:
            return feat, self.labels[index]

    def __len__(self):
        return self.dataset_len


def get_features_dataloader(h5_file_path, split, batch_size=128, shuffle=False, min_class=0, max_class=None,
                            sampler=None, batch_sampler=None, dataset_name='imagenet', return_item_ix=False,
                            num_workers=8, random_resized_crops=False):
    if max_class is not None:
        _labels = np.load('./{}_indices/{}_{}_labels.npy'.format(dataset_name, dataset_name, split))
        idxs = filter_by_class(_labels, min_class=min_class, max_class=max_class)
    if batch_sampler is None and sampler is None:
        if shuffle:
            sampler = torch.utils.data.sampler.SubsetRandomSampler(idxs)
        else:
            sampler = IndexSampler(idxs)
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size=batch_size, drop_last=False)

    if random_resized_crops:
        transform = utils.RandomResizeCrop(7, scale=(2 / 7, 1.0))
    else:
        transform = None

    dataset = FeaturesDataset(h5_file_path, split, min_class=min_class, max_class=max_class,
                              return_item_ix=return_item_ix, dataset_name=dataset_name, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler)
    return loader

"""
Written by Kushal, modified by Robik
"""
import json
import random
import sys
from collections import Counter

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# from cnn_lava.streaming_using_features import RehearsalBatchSampler
from vqa_experiments.data_utils import RehearsalBatchSampler, FixedBufferRehearsalBatchSampler
from torch.utils.data.sampler import SubsetRandomSampler


def dictoflists2listofdicts(dictoflists):
    listofdicts = []
    for i in range(len(dictoflists['qid'])):
        ent = {}
        for k in dictoflists:
            ent[k] = dictoflists[k][i]
        listofdicts.append(ent)
    return listofdicts


def format_data(h5file, config,
                num_classes,
                arrangement='random',
                data_subset=1.0):
    """Choose number of classes and arrange data:
        Inputs:
            h5file: Preprocessed h5file
            num_classes: Number of classes to use
            arrangement: 'ans_class', ans_type', 'ques_type', or 'random'
        Returns:
    """
    mem_feat = dict()
    for dset in h5file.keys():
        if config.use_lstm and dset == 'qfeat':
            mem_feat[dset] = [0] * len(h5file['qid'])
        else:
            mem_feat[dset] = h5file[dset][:]

    mem_feat = dictoflists2listofdicts(mem_feat)

    mem_feat = mem_feat[:int(len(mem_feat) * data_subset)]

    data = []
    for d in mem_feat:
        if d['aidx'] < num_classes:
            data.append(d)

    if arrangement != 'random':
        data = sorted(data, key=lambda k: k[arrangement])
    elif arrangement == 'random':
        random.Random(666).shuffle(data)
    return data


def qid2fname(qid, split):
    rid = str(qid)[:-1].rjust(12, '0')
    fname = '{}2014/COCO_{}2014_{}.jpg'.format(split, split, rid)
    return fname


def build_target(ten_aidx, config):
    scores = torch.zeros((config.num_classes))
    ans_freq = Counter(ten_aidx)
    for c in ans_freq:
        if c < config.num_classes and c != -1:
            scores[c] = min(ans_freq[c] * 0.3, 1)
    return scores


class VQADataset(Dataset):
    def __init__(self, data, config, split, mem_feat, **kwargs):
        if split == 'train' and config.arrangement[split] != 'random':
            arr = config.arrangement[split]
            all_keys = list(set([d[arr] for d in data]))
            random.Random(666).shuffle(all_keys)
            keymap = {idx: key for idx, key in enumerate(all_keys)}
            data = sorted(data, key=lambda k: keymap[k[arr]])

            for idx, _ in enumerate(data):
                if keymap[data[idx][config.arrangement[split]]] < config.only_first_k[split]:
                    continue
                else:
                    break
        elif config.arrangement[split] != 'random':
            for idx, _ in enumerate(data):
                if data[idx][config.arrangement[split]] < config.only_first_k[split]:
                    continue
                else:
                    break
        else:
            idx = len(data) - 1

        data = data[:idx + 1]

        if 'err' in kwargs:
            data2 = []
            for dp in data:
                if torch.sum(kwargs['err'][str(dp['iid'])]).item() < 0.75 * 36:
                    data2.append(dp)
            data = data2
        self.data = data
        self.split = split
        self.map = json.load(open(config.map_path))
        self.config = config
        self.d = config.d
        if config.load_in_memory:
            self.feat = mem_feat

    def __len__(self):
        if self.config.fetch_all:
            return len(set([dp[self.config.arrangement[self.split]] for dp in self.data]))
        else:
            return len(self.data)

    def __getitem__(self, index):
        if not hasattr(self, 'feat'):
            # Loading the h5 file has to be done here and not in __init__ because when the DataLoader
            # forks for multiple works, every child would use the same file object and fail
            # Having multiple readers using different file objects is fine though, so we just init in here.
            self.feat = h5py.File(self.config.feat_path, 'r')

        if self.config.fetch_all:
            all_valid = [i for i, dp in enumerate(self.data) if dp[self.config.arrangement[self.split]] == index]
            data_batch = (self.get_datapoint(i) for i in all_valid)
            return data_batch
        else:
            return self.get_datapoint(index)

    def get_datapoint(self, index):

        dp = self.data[index]
        iid = str(dp['iid'])
        feat_index = self.map['image_id_to_ix'][str(iid)]

        if self.config.use_pooled:
            imfeat = self.feat['image_features'][feat_index]
            imfeat = np.mean(imfeat, axis=0)
        else:
            imfeat = self.feat['image_features'][feat_index]

        qfeat = dp['qfeat']
        if self.config.qnorm and not self.config.use_lstm:  # I THINK THIS IS CORRECT BUT WE NEED TO CHECK
            dp['qfeat'] = qfeat.astype('float32') / (np.linalg.norm(qfeat) + 1e-8)

        if self.config.imnorm:
            if self.config.use_pooled:
                imfeat = imfeat.astype('float32') / (np.linalg.norm(imfeat) + 1e-8)
            else:
                imfeat = imfeat.astype('float32') / (np.linalg.norm(imfeat, axis=1, keepdims=True) + 1e-8)

        # pos_feat = self.feat['spatial_features'][feat_index]
        #        imfeat = np.concatenate((imfeat,pos_feat),axis=1)
        if self.config.mkii:
            codebook_index = self.feat['codebook_indices'][feat_index]

        if self.config.dataset == 'clevr':
            l = 45
        else:
            l = 30
        qseq = torch.ones(l).long() * self.d.ntoken
        qtokens = self.d.tokenize(dp['q'], False)
        qlen = len(qtokens)
        qseq[:qlen] = torch.from_numpy(np.array(qtokens[:l - 1])).long()

        if self.config.soft_targets:
            aidx = build_target(dp['ten_aidx'], self.config)
        else:
            aidx = dp['aidx']
        if self.config.mkii:
            return qfeat, qseq, imfeat, codebook_index, dp['qid'], aidx, dp['ten_aidx'], qlen
        else:
            return qfeat, qseq, imfeat, dp['qid'], dp['iid'], aidx, dp['ten_aidx'], qlen


def collate_batch(data_batch):
    data_batch.sort(key=lambda x: x[-1], reverse=True)
    return torch.utils.data.dataloader.default_collate(data_batch)


# %%
def build_dataloaders(config, preloaded_feat, **kwargs):
    # Make val dataset, change arrangment and num_classes in config
    print('Loading Train Data')
    train_h5file = h5py.File(config.train_file, 'r')
    print('Filtering Train Data')

    if config.train_on == 'valid':
        nc = config.num_classes
    else:
        nc = sys.maxsize

    train_data = format_data(train_h5file, config, num_classes=nc, arrangement=config.arrangement['train'],
                             data_subset=config.data_subset)

    # TODO: Compute LUT for each run
    if 'err' in kwargs:
        train_dataset = VQADataset(train_data, config, 'train', preloaded_feat, err=kwargs['err'])
    else:
        train_dataset = VQADataset(train_data, config, 'train', preloaded_feat)

    print('Loading Test Data')
    val_h5file = h5py.File(config.val_file, 'r')

    print('Filtering Test Data')
    if config.test_on == 'valid':
        nc = config.num_classes
    else:
        nc = sys.maxsize
    val_data = format_data(val_h5file, config, num_classes=nc, arrangement=config.arrangement['val'], data_subset=1.0)
    val_dataset = VQADataset(val_data, config, 'val', preloaded_feat)

    if config.fetch_all:
        train_dataloader = train_dataset
        val_dataloader = val_dataset
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=config.train_batch_size,
                                      shuffle=config.shuffle, num_workers=12, collate_fn=collate_batch)

        val_dataloader = DataLoader(val_dataset, batch_size=config.val_batch_size,
                                    shuffle=False, num_workers=8, collate_fn=collate_batch)  # Never shuffle val data

    return train_dataloader, val_dataloader


def build_rehearsal_dataloader(dataset, rehearsal_ixs, num_rehearsal_samples):
    rehearsal_batch_sampler = RehearsalBatchSampler(rehearsal_ixs, num_rehearsal_samples)
    loader = DataLoader(dataset, batch_sampler=rehearsal_batch_sampler)
    return loader


def build_rehearsal_dataloader_with_limited_buffer(dataset, rehearsal_ixs, num_rehearsal_samples,
                                                   max_buffer_size, buffer_replacement_strategy):
    rehearsal_batch_sampler = FixedBufferRehearsalBatchSampler(max_buffer_size, num_rehearsal_samples,
                                                               buffer_replacement_strategy)
    loader = DataLoader(dataset, batch_sampler=rehearsal_batch_sampler)
    return loader


def build_base_init_dataloader(dataset, data_indices, batch_size):
    """
    Only goes through the data items whose indices are present in data_indices
    """
    index_sampler = SubsetRandomSampler(data_indices)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=index_sampler, num_workers=8,
                        collate_fn=collate_batch)
    return loader


def main():
    pass


if __name__ == '__main___':
    main()

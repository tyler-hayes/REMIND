import numpy as np
import torch
import os
import torch.nn as nn
import json
import random
import math
import torch.utils.data
from image_classification_experiments.resnet_models import *


class Counter:
    def __init__(self):
        self.count = 0

    def update(self):
        self.count += 1


class CMA(object):
    """
    A continual moving average for tracking loss updates.
    """

    def __init__(self):
        self.N = 0
        self.avg = 0.0

    def update(self, X):
        self.avg = (X + self.N * self.avg) / (self.N + 1)
        self.N = self.N + 1


def accuracy(output, target, topk=(1,), output_has_class_ids=False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if not output_has_class_ids:
        output = torch.Tensor(output)
    else:
        output = torch.LongTensor(output)
    target = torch.LongTensor(target)
    with torch.no_grad():
        maxk = max(topk)
        batch_size = output.shape[0]
        if not output_has_class_ids:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
        else:
            pred = output[:, :maxk].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


def save_predictions(y_pred, min_class_trained, max_class_trained, save_path, suffix='', order=None):
    if order is not None:
        name = 'core50_' + order + '_preds_min_trained_' + str(min_class_trained) + '_max_trained_' + str(
            max_class_trained)
    else:
        name = 'preds_min_trained_' + str(min_class_trained) + '_max_trained_' + str(max_class_trained) + suffix
    torch.save(y_pred, save_path + '/' + name + '.pt')


def save_accuracies(accuracies, min_class_trained, max_class_trained, save_path,suffix='', order=None):
    if order is not None:
        name = 'core50_' + order + '_accuracies_min_trained_' + str(min_class_trained) + '_max_trained_' + str(
            max_class_trained) + '.json'
    else:
        name = 'accuracies_min_trained_' + str(min_class_trained) + '_max_trained_' + str(max_class_trained) + suffix+'.json'
    json.dump(accuracies, open(os.path.join(save_path, name), 'w'))


def safe_load_dict(model, new_model_state, should_resume_all_params=False):
    old_model_state = model.state_dict()
    c = 0
    if should_resume_all_params:
        for old_name, old_param in old_model_state.items():
            assert old_name in list(new_model_state.keys()), "{} parameter is not present in resumed checkpoint".format(
                old_name)
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'module':
            name = '.'.join(end)
        if name not in old_model_state:
            # print('%s not found in old model.' % name)
            continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')


def save_model(model, optimizer, ckpt_path):
    if '/' in ckpt_path:
        ckpt_dir = '/'.join(ckpt_path.split('/')[:-1])
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

    state = {
        'state_dict': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }
    torch.save(state, ckpt_path)


def randint(max_val, num_samples):
    rand_vals = {}
    _num_samples = min(max_val, num_samples)
    while True:
        _rand_vals = np.random.randint(0, max_val, num_samples)
        for r in _rand_vals:
            rand_vals[r] = r
            if len(rand_vals) >= _num_samples:
                break

        if len(rand_vals) >= _num_samples:
            break
    return rand_vals.keys()


def build_classifier(classifier, classifier_ckpt, num_classes):
    classifier = eval(classifier)(num_classes=num_classes)

    if classifier_ckpt is None:
        print("Will not resume any checkpoints!")
    else:
        resumed = torch.load(classifier_ckpt)
        if 'state_dict' in resumed:
            state_dict_key = 'state_dict'
        else:
            state_dict_key = 'model_state'
        print("Resuming with {}".format(classifier_ckpt))
        safe_load_dict(classifier, resumed[state_dict_key], should_resume_all_params=True)
    return classifier


class RehearsalBatchSampler(torch.utils.data.Sampler):
    """
    A sampler that returns a generator obj30216ect which randomly samples from a list, that holds the indices that are
    eligible for rehearsal.
    The samples that are eligible for rehearsal grows over time, so we want it to be a 'generator' object and not an
    iterator object.
    """

    # See: https://github.com/pytorch/pytorch/issues/683
    def __init__(self, rehearsal_ixs, num_rehearsal_samples):
        # This makes sure that different workers have different randomness and don't end up returning the same data
        # item!
        self.rehearsal_ixs = rehearsal_ixs  # These are the samples which can be replayed. This list can grow over time.

        np.random.seed(os.getpid())
        self.num_rehearsal_samples = num_rehearsal_samples

    def __iter__(self):
        # We are returning a generator instead of an iterator, because the data points we want to sample from, differs
        # every time we loop through the data.
        # e.g., if we are seeing 100th sample, we may want to do a replay by sampling from 0-99 samples. But then,
        # when we see 101th sample, we want to replay from 0-100 samples instead of 0-99.
        while True:
            ix = randint(len(self.rehearsal_ixs), self.num_rehearsal_samples)
            yield np.array([self.rehearsal_ixs[_curr_ix] for _curr_ix in ix])

    def __len__(self):
        return 2 ** 64  # Returning a very large number because we do not want it to stop replaying.
        # The stop criteria must be defined in some other manner.


class RandomResizeCrop(object):
    """Randomly crops tensor then resizes uniformly between given bounds
    Args:
        size (sequence): Bounds of desired output sizes.
        scale (sequence): Range of size of the origin size cropped
        ratio (sequence): Range of aspect ratio of the origin aspect ratio cropped
        interpolation (int, optional): Desired interpolation. Default is 'bilinear'
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation='bilinear'):
        #        assert (isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.
        Args:
            img (3-d tensor (C,H,W)): Tensor to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        area = img.size(1) * img.size(2)

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size(1) and h <= img.size(2):
                i = random.randint(0, img.size(2) - h)
                j = random.randint(0, img.size(1) - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size(1) / img.size(2)
        if (in_ratio < min(ratio)):
            w = img.size(1)
            h = int(w / min(ratio))
        elif (in_ratio > max(ratio)):
            h = img.size(2)
            w = int(h * max(ratio))
        else:  # whole image
            w = img.size(1)
            h = img.size(2)
        i = int((img.size(2) - h) // 2)
        j = int((img.size(1) - w) // 2)
        return i, j, h, w

    def __call__(self, img):
        """
        Args:
            img (3-D tensor (C,H,W)): Tensor to be cropped and resized.
        Returns:
            Tensor: Randomly cropped and resized Tensor.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = img[:, i:i + h, j:j + w]  ##crop
        return torch.nn.functional.interpolate(img.unsqueeze(0), self.size, mode=self.interpolation,
                                               align_corners=False).squeeze(0)

    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, scale={1}, ratio={2}, interpolation={3})'.format(self.size,
                                                                                                      self.scale,
                                                                                                      self.ratio,
                                                                                                      interpolate_str)
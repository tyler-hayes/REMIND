"""
Written by Robik
"""
import torch, os
import numpy as np


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

    def update_buffer(self, item_ix, class_id=None):
        self.rehearsal_ixs.append(item_ix)

    def get_state(self):
        return {'rehearsal_ixs': self.rehearsal_ixs,
                'num_rehearsal_samples': self.num_rehearsal_samples}

    def load_state(self, state):
        rehearsal_ixs = state['rehearsal_ixs']
        while len(self.rehearsal_ixs) > 0:
            self.rehearsal_ixs.pop()
        self.rehearsal_ixs.extend(rehearsal_ixs)
        self.num_rehearsal_samples = state['num_rehearsal_samples']

    def get_rehearsal_ixs(self):
        return self.rehearsal_ixs


class FixedBufferRehearsalBatchSampler(torch.utils.data.Sampler):
    """
    Maintains a buffer for each class. The buffer is initialized with base init samples.
    Once the buffer is full, oldest sample from the class having largest # of samples is replaced.
    This implementation has been optimized for speed and not sexiness!"""

    def __init__(self, buffer_size, num_rehearsal_samples, buffer_replacement_strategy):
        self.buffer_size = buffer_size
        # A dictionary from class id to
        self.per_class_rehearsal_ixs = {}
        self.num_rehearsal_samples = num_rehearsal_samples
        self.buffer_replacement_strategy = buffer_replacement_strategy
        self.class_lens = {}
        self.total_len = 0
        self.device = 'cuda'
        np.random.seed(os.getpid())

    def find_class_having_max_samples(self):
        max_class = None
        max_num = 0
        for c in self.class_lens:
            class_len = self.class_lens[c]
            if class_len > max_num:
                max_num = class_len
                max_class = c
        return max_class, max_num

    def delete_sample_from_largest_class(self, args=None, train_data=None):
        max_class, max_num = self.find_class_having_max_samples()
        if self.buffer_replacement_strategy == 'random':
            del_ix = int(list(randint(max_num, 1))[0])
            del self.per_class_rehearsal_ixs[max_class][del_ix]
        self.class_lens[max_class] -= 1
        self.total_len -= 1

    def update_buffer(self, new_ix, class_id, args=None, train_data=None):
        new_ix = int(new_ix)
        class_id = int(class_id)
        if self.total_len >= self.buffer_size:
            self.delete_sample_from_largest_class(args, train_data)
        if class_id not in self.per_class_rehearsal_ixs:
            self.per_class_rehearsal_ixs[class_id] = []
            self.class_lens[class_id] = 0
        self.class_lens[class_id] += 1
        self.per_class_rehearsal_ixs[class_id].append(new_ix)
        self.total_len += 1

    def get_rehearsal_item_ix(self, ix):
        """
        Given a random integer 'ix', this function figures out the class and the index
        within the class this refers to, and returns that element.
        :param ix:
        :return:
        """
        cum_sum = 0
        for class_id, class_len in zip(list(self.class_lens.keys()), list(self.class_lens.values())):
            cum_sum += class_len
            if ix < cum_sum:
                class_item_ix = class_len - (cum_sum - ix)
                # print(
                #     f"class_item_ix {class_item_ix} class_len {class_len} len {len(self.per_class_rehearsal_ixs[class_id])}")
                return self.per_class_rehearsal_ixs[class_id][class_item_ix]

    def __iter__(self):
        while True:
            ixs = randint(self.total_len, self.num_rehearsal_samples)
            yield np.array([self.get_rehearsal_item_ix(ix) for ix in ixs])

    def __len__(self):
        return 2 ** 64  # Returning a very large number because we do not want it to stop replaying.
        # The stop criteria must be defined in some other manner.

    def get_state(self):
        return {
            'buffer_size': self.buffer_size,
            'per_class_rehearsal_ixs': self.per_class_rehearsal_ixs,
            'num_rehearsal_samples': self.num_rehearsal_samples,
            'class_lens': self.class_lens,
            'total_len': self.total_len
        }

    def load_state(self, state):
        self.buffer_size = state['buffer_size']
        self.num_rehearsal_samples = state['num_rehearsal_samples']
        self.total_len = state['total_len']
        for c in state['class_lens']:
            self.class_lens[c] = state['class_lens'][c]
            print(f"class len {c}: {self.class_lens[c]}")
        for c in state['per_class_rehearsal_ixs']:
            if c in self.per_class_rehearsal_ixs:
                while len(self.per_class_rehearsal_ixs[c]) > 0:
                    self.per_class_rehearsal_ixs[c].pop()
            else:
                self.per_class_rehearsal_ixs[c] = []
            self.per_class_rehearsal_ixs[c].extend(state['per_class_rehearsal_ixs'][c])

    def get_rehearsal_ixs(self):
        rehearsal_ixs = []
        for c in self.per_class_rehearsal_ixs:
            rehearsal_ixs += self.per_class_rehearsal_ixs[c]
        return rehearsal_ixs

    def get_len_of_rehearsal_ixs(self):
        return len(self.get_rehearsal_ixs())

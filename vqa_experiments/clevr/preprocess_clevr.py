"""
Written by Kushal, modified by Robik
"""

import json
import h5py
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm

PATH = '/hdd/robik/CLEVR'
CATEGORIES = {'count': ['count'],
              'compare_attribute':
                  ['equal_color',
                   'equal_material',
                   'equal_shape',
                   'equal_size'],

              'exist': ['exist'],

              'compare_integer': [
                  'greater_than',
                  'less_than',
                  'equal_integer'],

              'query_attribute': [
                  'query_color',
                  'query_material',
                  'query_shape',
                  'query_size']
              }

categories_fine = dict()

for k, v in CATEGORIES.items():
    for vi in v:
        categories_fine[vi] = k

annotations = dict()
for split in ['train', 'val']:
    annotations[split] = json.load(
        open(f'{PATH}/questions/CLEVR_{split}_questions.json'))['questions']

meta = defaultdict(list)

for ann in annotations['train']:
    ans = ann['answer']
    meta['a'].append(ans)
    meta['atype'].append('answer_type')
    meta['qtype'].append(categories_fine[ann['program'][-1]['function']])

lut = dict()

for m in ['a', 'atype', 'qtype']:
    most_common = Counter(meta[m]).most_common()
    lut[f'{m}2idx'] = {a[0]: idx for idx, a in enumerate(most_common)}

json.dump(lut, open('LUT_clevr.json', 'w'))
# %%
dt = h5py.special_dtype(vlen=str)
for split in ['train', 'val']:
    qfeat_file = h5py.File(f'{PATH}/questions_{split}_clevr.h5', 'r')

    mem_feat = dict()
    for dset in qfeat_file.keys():
        mem_feat[dset] = qfeat_file[dset][:]
    qids = mem_feat['qids'][:]
    qid2idx = {qid: idx for idx, qid in enumerate(qids)}
    num_instances = len(annotations[split])
    h5file = h5py.File(f'{PATH}/{split}_clevr.h5', 'w')
    h5file.create_dataset('qfeat', (num_instances, 2048), dtype=np.float32)
    h5file.create_dataset('qid', (num_instances,), dtype=np.int64)
    h5file.create_dataset('iid', (num_instances,), dtype=np.int64)
    h5file.create_dataset('q', (num_instances,), dtype=dt)
    h5file.create_dataset('a', (num_instances,), dtype=dt)
    h5file.create_dataset('ten_ans', (num_instances, 10), dtype=dt)
    h5file.create_dataset('aidx', (num_instances,), dtype=np.int32)
    h5file.create_dataset('ten_aidx', (num_instances, 10), dtype=np.int32)
    h5file.create_dataset('atypeidx', (num_instances,), dtype=np.int32)
    h5file.create_dataset('qtypeidx', (num_instances,), dtype=np.int32)

    for idx, ann in enumerate(tqdm(annotations[split])):
        qid = ann['question_index']
        if split == 'train':
            iid = int('1' + str(ann['image_index']))
        elif split == 'val':
            iid = int('2' + str(ann['image_index']))
        else:
            raise (Exception)

        feat_idx = qid2idx[qid]
        ten_ans = [ann['answer']] * 10
        ans = ten_ans[0]
        aidx = lut['a2idx'].get(ans, -1)
        ten_aidx = np.array([lut['a2idx'].get(a, -1) for a in ten_ans])
        atypeidx = lut['atype2idx'].get('answer_type', -1)
        qtype_clevr = categories_fine[ann['program'][-1]['function']]
        qtypeidx = lut['qtype2idx'].get(qtype_clevr, -1)
        h5file['qfeat'][idx] = mem_feat['feats'][feat_idx]
        h5file['qid'][idx] = qid
        h5file['iid'][idx] = iid
        h5file['q'][idx] = mem_feat['questions'][feat_idx]
        h5file['a'][idx] = ans
        h5file['ten_ans'][idx] = ten_ans
        h5file['aidx'][idx] = aidx
        h5file['atypeidx'][idx] = atypeidx
        h5file['qtypeidx'][idx] = qtypeidx
        h5file['ten_aidx'][idx] = ten_aidx
    h5file.close()

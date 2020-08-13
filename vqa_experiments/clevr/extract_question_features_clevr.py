"""
Written by Kushal, modified by Robik
"""

import sys
import json
import h5py
import numpy as np

DATA_PATH = '/hdd/robik/CLEVR'
GENSEN_PATH = '/hdd/robik/projects/gensen'
sys.path.append(f'{GENSEN_PATH}')

from gensen import GenSen, GenSenSingle

gensen_1 = GenSenSingle(
    model_folder=f'{GENSEN_PATH}/data/models',
    filename_prefix='nli_large_bothskip',
    cuda=True,
    pretrained_emb=f'{GENSEN_PATH}/data/embedding/glove.840B.300d.h5'
)

for split in ['train', 'val']:
    feat_h5 = h5py.File(f'{DATA_PATH}/questions_{split}_clevr.h5', 'w')
    ques = json.load(open(f'{DATA_PATH}/questions/CLEVR_{split}_questions.json'))
    ques = ques['questions']
    questions = [q['question'] for q in ques]
    qids = [q['question_index'] for q in ques]
    qids = np.int64(qids)
    dt = h5py.special_dtype(vlen=str)
    feat_h5.create_dataset('feats', (len(qids), 2048), dtype=np.float32)
    feat_h5.create_dataset('qids', (len(qids),), dtype=np.int64)
    feat_h5.create_dataset('questions', (len(qids),), dtype=dt)
    feat_h5['qids'][:] = qids
    feat_h5['questions'][:] = questions

    chunksize = 5000
    question_chunks = [questions[x:x + chunksize] for x in range(0, len(questions), chunksize)]

    done = 0
    for qchunk in question_chunks:
        print(done)
        _, reps_h_t = gensen_1.get_representation(
            qchunk, pool='last', return_numpy=True, tokenize=True
        )
        feat_h5['feats'][done:done + len(qchunk)] = reps_h_t
        done += len(qchunk)

    feat_h5.close()

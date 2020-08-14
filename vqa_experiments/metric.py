import h5py
from collections import defaultdict


def compute_clevr_accuracy(PATH, preds):
    gt_answers = h5py.File(f'{PATH}/val_clevr.h5', 'r')['aidx'][:]
    gt_qids = h5py.File(f'{PATH}/val_clevr.h5', 'r')['qid'][:]
    gt_qtypes = h5py.File(f'{PATH}/val_clevr.h5', 'r')['qtypeidx'][:]

    qid2qtype = {qid: gt for qid, gt in zip(gt_qids, gt_qtypes)}
    qid2gt = {qid: gt for qid, gt in zip(gt_qids, gt_answers)}

    acc = defaultdict(list)

    for qid in qid2gt:
        gt = qid2gt[qid]
        qtype = qid2qtype[qid]
        if gt == preds[str(qid)]:
            acc['overall'].append(1)
            acc[qtype].append(1)
        else:
            acc['overall'].append(0)
            acc[qtype].append(0)

    mpt = 0
    overall = 0
    for k in acc:
        if k == 'overall':
            overall = sum(acc[k]) / len(acc[k])
        else:
            mpt += sum(acc[k]) / len(acc[k])
    mpt = mpt / 5

    return mpt, overall


def compute_clevr_per_type_accuracies(path, preds):
    gt_answers = h5py.File(f'{path}/val_clevr.h5', 'r')['aidx'][:]
    gt_qids = h5py.File(f'{path}/val_clevr.h5', 'r')['qid'][:]
    gt_qtypes = h5py.File(f'{path}/val_clevr.h5', 'r')['qtypeidx'][:]

    qid2qtype = {qid: gt for qid, gt in zip(gt_qids, gt_qtypes)}
    qid2gt = {qid: gt for qid, gt in zip(gt_qids, gt_answers)}

    acc = {}
    some_qids = {}
    for qid in qid2gt:
        gt = qid2gt[qid]
        qtype = qid2qtype[qid]
        if qtype not in acc:
            acc[qtype] = {
                'total': 0,
                'correct': 0
            }
            some_qids[qtype] = qid
        if gt == preds[str(qid)]:
            acc[qtype]['correct'] += 1
        acc[qtype]['total'] += 1

    for qtype in [0, 1, 2, 3, 4]:
        # print(f"{acc[qtype]['correct'] / acc[qtype]['total']}")
        # print("%.2f" % (acc[qtype]['total']))
        print("%.2f" % (acc[qtype]['correct'] / acc[qtype]['total'] * 100))

    print("\n\nQuestion ids")
    print(some_qids)


def compute_tdiuc_accuracy(PATH, preds):
    gt_answers = h5py.File(f'{PATH}/val_tdiuc.h5')['aidx'][:]
    gt_qids = h5py.File(f'{PATH}/val_tdiuc.h5')['qid'][:]
    gt_qtypes = h5py.File(f'{PATH}/val_tdiuc.h5')['qtypeidx'][:]

    qid2qtype = {qid: gt for qid, gt in zip(gt_qids, gt_qtypes)}
    qid2gt = {qid: gt for qid, gt in zip(gt_qids, gt_answers)}

    acc = defaultdict(list)

    for qid in qid2gt:
        gt = qid2gt[qid]
        qtype = qid2qtype[qid]
        if gt == preds[str(qid)]:
            acc['overall'].append(1)
            acc[qtype].append(1)
        else:
            acc['overall'].append(0)
            acc[qtype].append(0)

    mpt = 0
    overall = 0
    for k in acc:
        if k == 'overall':
            overall = sum(acc[k]) / len(acc[k])
        else:
            mpt += sum(acc[k]) / len(acc[k])
    mpt = mpt / 12

    return mpt, overall


def compute_accuracy(path, dataset, preds):
    if dataset == 'clevr':
        mpt, overall = compute_clevr_accuracy(path, preds)
    elif dataset == 'tdiuc':
        mpt, overall = compute_tdiuc_accuracy(path, preds)
    print(f"Mean Per Type: {mpt}, Overall: {overall}")

"""
Written by Kushal, modified by Robik
"""
import faiss
import numpy as np
import h5py
import json

PATH = '/hdd/robik/CLEVR'
feat_name = f'{PATH}/all_clevr_resnet_largestage3'
train_filename = f'{PATH}/train_clevr.h5'
lut_name = f'{PATH}/map_clevr_resnet_largestage3.json'
streaming_type = 'iid'

feat_dim = 1024
num_feat_maps = 196

train_data = h5py.File(train_filename, 'r')
lut = json.load(open(lut_name))
feat_h5 = h5py.File(f'{feat_name}.h5', 'r')
print(f"# images {len(feat_h5['image_features'])}")
if streaming_type == 'iid':
    ids = train_data['iid'][:]
    feat_idxs = list(set([lut['image_id_to_ix'][str(id)] for id in ids]))
    feat_idxs_base_init = feat_idxs[:int(0.1 * len(feat_idxs))]
else:
    feat_idxs_base_init = list(set([lut['image_id_to_ix'][str(iid)] for qtypeidx, iid in
                                    zip(train_data['qtypeidx'], train_data['iid']) if qtypeidx == 0]))
print(f'len feat_idxs_base_init {len(feat_idxs_base_init)}')
train_data_base_init = np.array([feat_h5['image_features'][bidx] for bidx in feat_idxs_base_init], dtype=np.float32)

# train set
train_data_base_init = np.reshape(train_data_base_init, (-1, feat_dim))

print('Training Product Quantizer')

d = feat_dim  # data dimension
cs = 32  # code size (bytes)
pq = faiss.ProductQuantizer(d, cs, 8)
pq.train(train_data_base_init)

print('Encoding, Decoding and saving Reconstructed Features')
del train_data_base_init
feats = feat_h5['image_features']
start = 0
batch = 10000
reconstructed_h5 = h5py.File(f'{feat_name}pq_{streaming_type}.h5', 'w')
reconstructed_h5.create_dataset('image_features', shape=feats.shape, dtype=np.float32)

while start < len(feats):
    print(start, ' feats done out of ', len(feats))
    data_batch = feats[start:start + batch]
    num_feats = len(data_batch)
    data_batch = np.reshape(data_batch, (-1, feat_dim))
    codes = pq.compute_codes(data_batch)
    data_batch_reconstructed = pq.decode(codes)
    data_batch_reconstructed = np.reshape(data_batch_reconstructed, (-1, num_feat_maps, feat_dim))
    reconstructed_h5['image_features'][start:start + num_feats] = data_batch_reconstructed
    start = start + batch

reconstructed_h5.close()

# boundary points: [251749, 377403, 542809, 605903, 699989]

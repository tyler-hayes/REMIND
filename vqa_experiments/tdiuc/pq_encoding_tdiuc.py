"""
Written by Kushal, modified by Robik
"""
import faiss
import numpy as np
import h5py
import json

print("Starting the encoding process...")
# Change these based on data set
PATH = '/hdd/robik/TDIUC'  # Change this
streaming_type = 'qtype'  # Change this

# Probably don't need to be changed
feat_name = f'{PATH}/all_tdiuc_resnet'
train_filename = f'{PATH}/train_tdiuc.h5'
lut_name = f'{PATH}/map_tdiuc_resnet.json'

feat_dim = 2048
num_feat_maps = 49

train_data = h5py.File(train_filename, 'r')
lut = json.load(open(lut_name))
feat_h5 = h5py.File(f'{feat_name}.h5', 'r')

if streaming_type == 'iid':
    ids = train_data['iid'][:]
    feat_idxs = list(set([lut['image_id_to_ix'][str(id)] for id in ids]))
    feat_idxs_base_init = feat_idxs[:int(0.1 * len(feat_idxs))]
else:
    feat_idxs_base_init = list(set([lut['image_id_to_ix'][str(iid)] for qtypeidx, iid in
                                    zip(train_data['qtypeidx'], train_data['iid']) if qtypeidx == 0]))

print(f"# samples for base init {len(feat_idxs_base_init)}")
train_data_base_init = np.array([feat_h5['image_features'][bidx] for bidx in feat_idxs_base_init], dtype=np.float32)

# train set

train_data_base_init = np.reshape(train_data_base_init, (-1, feat_dim))

print('Training Product Quantizer')

d = feat_dim  # data dimension
cs = 32  # code size (bytes)
pq = faiss.ProductQuantizer(d, cs, 8)
pq.train(train_data_base_init)

print('Encoding, Decoding and saving Reconstructed Features')

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

# Boundary points: [21602, 463412, 575269, 821512, 840988, 903850, 905311, 905661, 950335, 976377, 982225, 1115299]

"""
Written by Kushal, modified by Robik
"""
import vqa_experiments.vqa_models as vqa_models
import torch
from vqa_experiments.dictionary import Dictionary
import sys
from vqa_experiments.vqa_models import WordEmbedding
from vqa_experiments.s_mac import s_mac

# Model and runtime configuration choices. A copy of this will be saved along with the model
# weights so that it is easy to reproduce later.

# Data
data_path = '/hdd/robik/CLEVR'
dataset = 'clevr'
img_feat = 'resnet'  # updn, resnet, updnmkii, resnetmkii
mkii = False  # If you want to also load codebook indices
data_subset = 1.0
d = Dictionary.load_from_file('vqa_experiments/data/dictionary_{}.pkl'.format(dataset))

map_path = f'{data_path}/map_clevr_resnet_largestage3.json'

train_file = f'{data_path}/train_{dataset}.h5'
val_file = f'{data_path}/val_{dataset}.h5'

train_batch_size = 64
val_batch_size = 64
num_classes = 28  # Number of classifier units 1480 for TDIUC, 31xx for VQA,28 for CLEVR

train_on = 'full'
test_on = 'full'  # 'full' or 'valid'

arrangement = dict()
arrangement['train'] = 'random'  # 'random', 'aidx', 'atypeidx', 'qtypeidx'
arrangement['val'] = 'random'  # 'random', 'aidx', 'atypeidx', 'qtypeidx'

# How many to train/test on
# How many of "indices" to train on, E.g., if arrangement is ans_class, it refers
# to how many answer classes to use, if arrangement is ques_type, it refers to how
# many question_types to use -1 means all possible

only_first_k = dict()
only_first_k['train'] = sys.maxsize  # Use sys.maxsize to load all
only_first_k['val'] = sys.maxsize  # Use sys.maxsize to load all

qnorm = True  # Normalize ques feat?
imnorm = False  # Normalize img feat?

shuffle = False

fetch_all = False

if fetch_all:  # For ques_type, ans_class or ans_type arrangement, get all qualifying data
    assert (not shuffle)
    train_batch_size = 1
    val_batch_size = 1  # Dataset[i] will return all qualifying data of idx 1

load_in_memory = False
use_all = False
use_pooled = False
use_lstm = True

# Training
overwrite_expt_dir = True  # Set to True during dev phase
max_epochs = 20
test_interval = 8

# Model
attn_type = 'old'  # new or old
num_attn_hops = 2
soft_targets = False
bidirectional = True
lstm_out = 512
emb_dim = 300
cnn_feat_size = 2048  # 2048 for resnet/updn/clevr_layer4 ; 1024 for clevr layer_3
if dataset == 'clevr':
    cnn_feat_size = 1024

classfier_dropout = True
embedding_dropout = False
attention_dropout = True
num_hidden = 1024
use_model = s_mac.sMacNetwork  # BLAH
optimizer = torch.optim.Adamax
lr = 1e-4
save_models = False
if not soft_targets:
    train_on = 'valid'
num_rehearsal_samples = 50

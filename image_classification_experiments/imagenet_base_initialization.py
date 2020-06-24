import numpy as np
import image_classification_experiments.utils_imagenet as utils_imagenet
import image_classification_experiments.utils as utils
from image_classification_experiments.retrieve_any_layer import ModelWrapper
from image_classification_experiments.utils import build_classifier
import time
from collections import defaultdict
import faiss


def extract_features(model, data_loader, data_len, num_channels=512, num_feats=7):
    start = time.time()
    model.eval()
    model.cuda()

    # allocate space for features and labels
    features_data = np.empty((data_len, num_channels, num_feats, num_feats), dtype=np.float32)
    labels_data = np.empty((data_len, 1), dtype=np.int)
    item_ixs_data = np.empty((data_len, 1), dtype=np.int)

    # put features and labels into arrays
    start_ix = 0
    for batch_ix, (batch_x, batch_y, batch_item_ixs) in enumerate(data_loader):
        batch_feats = model(batch_x.cuda())
        end_ix = start_ix + len(batch_feats)
        features_data[start_ix:end_ix] = batch_feats.cpu().numpy()
        labels_data[start_ix:end_ix] = np.atleast_2d(batch_y.numpy().astype(np.int)).transpose()
        item_ixs_data[start_ix:end_ix] = np.atleast_2d(batch_item_ixs.numpy().astype(np.int)).transpose()
        start_ix = end_ix
    print("Completed in {} secs".format(time.time() - start))
    return features_data, labels_data, item_ixs_data


def extract_base_init_features(imagenet_path, label_dir, extract_features_from, classifier_ckpt, arch,
                               max_class, num_channels, num_feats, batch_size=128):
    core_model = build_classifier(arch, classifier_ckpt, num_classes=None)

    model = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True)

    base_train_loader = utils_imagenet.get_imagenet_data_loader(imagenet_path + '/train', label_dir, split='train',
                                                                batch_size=batch_size, shuffle=False,
                                                                return_item_ix=True, min_class=0,
                                                                max_class=max_class, augment=False)

    base_train_features, base_train_labels, base_item_ixs = extract_features(model, base_train_loader,
                                                                             len(base_train_loader.dataset),
                                                                             num_channels=num_channels,
                                                                             num_feats=num_feats)
    return base_train_features, base_train_labels, base_item_ixs


def fit_pq(feats_base_init, labels_base_init, item_ix_base_init, num_channels, num_feats, num_codebooks, codebook_size,
           batch_size=128, counter=utils.Counter()):
    train_data_base_init = np.transpose(feats_base_init, (0, 2, 3, 1))
    train_data_base_init = np.reshape(train_data_base_init, (-1, num_channels))
    num_samples = len(train_data_base_init)

    print('Training Product Quantizer')
    start = time.time()
    nbits = int(np.log2(codebook_size))
    pq = faiss.ProductQuantizer(num_channels, num_codebooks, nbits)
    pq.train(train_data_base_init)
    print("Completed in {} secs".format(time.time() - start))
    del train_data_base_init

    print('Encoding, Decoding, and Storing Base Init Codes')
    start = time.time()
    latent_dict = {}
    class_id_to_item_ix_dict = defaultdict(list)
    rehearsal_ixs = []
    mb = min(batch_size, num_samples)
    for i in range(0, num_samples, mb):
        start = min(i, num_samples - mb)
        end = start + mb
        data_batch = feats_base_init[start:end]
        batch_labels = labels_base_init[start:end]
        batch_item_ixs = item_ix_base_init[start:end]

        data_batch = np.transpose(data_batch, (0, 2, 3, 1))
        data_batch = np.reshape(data_batch, (-1, num_channels))
        codes = pq.compute_codes(data_batch)
        codes = np.reshape(codes, (-1, num_feats, num_feats, num_codebooks))

        # put codes and labels into buffer (dictionary)
        for j in range(len(batch_labels)):
            ix = int(batch_item_ixs[j])
            latent_dict[ix] = [codes[j], batch_labels[j]]
            rehearsal_ixs.append(ix)
            class_id_to_item_ix_dict[int(batch_labels[j])].append(ix)
            counter.update()

    print("Completed in {} secs".format(time.time() - start))
    return pq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict


if __name__ == '__main__':
    max_class = 10
    imagenet_path = '/media/tyler/nvme_drive/data/ImageNet2012'
    label_dir = '/home/tyler/Desktop/codes/lifelong-vqa-iccv/lifelong-vqa/cnn_lava/'
    ckpt_dir = '/home/tyler/Desktop/codes/lifelong-vqa-iccv/lifelong-vqa/REMIND/resnet_imagenet_ckpts'
    ckpt_file = 'best_ResNet18ClassifyAfterLayer4_1_100.pth'
    arch = 'ResNet18ClassifyAfterLayer4_1'
    extract_features_from = 'model.layer4.0'
    num_channels = 512
    num_feats = 7
    num_codebooks = 32
    codebook_size = 256

    feat_data, label_data, item_ix_data = extract_base_init_features(imagenet_path, label_dir, extract_features_from,
                                                                     ckpt_dir + '/' + ckpt_file,
                                                                     arch, max_class, num_channels,
                                                                     num_feats)
    pq, latent_dict, rehearsal_ixs, class_id_to_item_ix_dict = fit_pq(feat_data, label_data, item_ix_data, num_channels,
                                                                      num_feats, num_codebooks,
                                                                      codebook_size)

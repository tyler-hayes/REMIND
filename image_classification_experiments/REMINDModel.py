import time
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys
import random
import os
import faiss
import pickle
import image_classification_experiments.utils as utils
from image_classification_experiments.retrieve_any_layer import ModelWrapper

sys.setrecursionlimit(10000)


def randint(max_val, num_samples):
    """
    return num_samples random integers in the range(max_val)
    """
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


class REMINDModel(object):
    """
    The REMIND class for streaming training with replay from our paper (https://arxiv.org/abs/1910.02509).
    """

    def __init__(self, num_classes, classifier_G='ResNet18ClassifyAfterLayer4_1',
                 extract_features_from='model.layer4.0',
                 classifier_F='ResNet18_StartAt_Layer4_1', classifier_ckpt=None, weight_decay=1e-5, lr_mode=None,
                 lr_step_size=100, start_lr=0.1, end_lr=0.001, lr_gamma=0.5, num_samples=50, use_mixup=False,
                 mixup_alpha=0.2, grad_clip=None, num_channels=512, num_feats=7, num_codebooks=32, codebook_size=256,
                 use_random_resize_crops=True, max_buffer_size=None):

        # make the classifier
        self.classifier_F = utils.build_classifier(classifier_F, classifier_ckpt, num_classes=num_classes)
        core_model = utils.build_classifier(classifier_G, classifier_ckpt, num_classes=None)
        self.classifier_G = ModelWrapper(core_model, output_layer_names=[extract_features_from], return_single=True)

        # make the optimizer
        trainable_params = self.get_trainable_params(self.classifier_F, start_lr)
        self.optimizer = optim.SGD(trainable_params, momentum=0.9, weight_decay=weight_decay)

        # setup lr decay
        if lr_mode in ['step_lr_per_class']:
            self.lr_scheduler_per_class = {}
            for class_ix in range(0, num_classes):
                self.lr_scheduler_per_class[class_ix] = optim.lr_scheduler.StepLR(self.optimizer,
                                                                                  step_size=lr_step_size,
                                                                                  gamma=lr_gamma)
        else:
            self.lr_scheduler_per_class = None

        # setup parameters
        self.num_classes = num_classes
        self.lr_mode = lr_mode
        self.lr_step_size = lr_step_size
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.lr_gamma = lr_gamma
        self.num_samples = num_samples
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.grad_clip = grad_clip
        self.num_channels = num_channels
        self.num_feats = num_feats
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.use_random_resize_crops = use_random_resize_crops
        self.random_resize_crop = utils.RandomResizeCrop(7, scale=(2 / 7, 1.0))
        self.max_buffer_size = max_buffer_size

    def get_trainable_params(self, classifier, start_lr):
        trainable_params = []
        for k, v in classifier.named_parameters():
            trainable_params.append({'params': v, 'lr': start_lr})
        return trainable_params

    def fit_incremental_batch(self, curr_loader, latent_dict, pq, rehearsal_ixs=None, class_id_to_item_ix_dict=None,
                              verbose=True, counter=utils.Counter()):
        """
        Fit REMIND on samples from a data loader one at a time.
        :param curr_loader: the data loader of new samples to be fit (returns (images, labels, item_ixs)
        :param latent_dict: dictionary containing latent codes for replay samples
        :param pq: trained PQ object for decoding latent codes
        :param rehearsal_ixs: list of item_ixs eligible for replay
        :param class_id_to_item_ix_dict: dictionary of visited classes with associated item_ixs visited
        :param verbose: true for printing loss to console
        :param counter: object to track how many samples are in buffer
        :return: None
        """

        ongoing_class = None

        # put classifiers on GPU and set plastic portion of network to train
        classifier_F = self.classifier_F.cuda()
        classifier_F.train()
        classifier_G = self.classifier_G.cuda()
        classifier_G.eval()

        criterion = nn.CrossEntropyLoss(reduction='none')

        msg = '\rSample %d -- train_loss=%1.6f -- elapsed_time=%d secs'

        start_time = time.time()
        total_loss = utils.CMA()
        c = 0
        for batch_images, batch_labels, batch_item_ixs in curr_loader:

            # get features from G and latent codes from PQ
            data_batch = classifier_G(batch_images.cuda()).cpu().numpy()
            data_batch = np.transpose(data_batch, (0, 2, 3, 1))
            data_batch = np.reshape(data_batch, (-1, self.num_channels))
            codes = pq.compute_codes(data_batch)
            codes = np.reshape(codes, (-1, self.num_feats, self.num_feats, self.num_codebooks))

            # train REMIND on one new sample at a time
            for x, y, item_ix in zip(codes, batch_labels, batch_item_ixs):
                if self.lr_mode == 'step_lr_per_class' and (ongoing_class is None or ongoing_class != y):
                    ongoing_class = y

                if self.use_mixup:
                    # gather two batches of previous data for mixup and replay
                    data_codes = np.empty(
                        (2 * self.num_samples + 1, self.num_feats, self.num_feats, self.num_codebooks),
                        dtype=np.uint8)
                    data_labels = torch.empty((2 * self.num_samples + 1), dtype=torch.int).cuda()
                    data_codes[0] = x
                    data_labels[0] = y
                    ixs = randint(len(rehearsal_ixs), 2 * self.num_samples)
                    ixs = [rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
                    for ii, v in enumerate(ixs):
                        data_codes[ii + 1] = latent_dict[v][0]
                        data_labels[ii + 1] = torch.from_numpy(latent_dict[v][1])

                    # reconstruct/decode samples with PQ
                    data_codes = np.reshape(data_codes, (
                        (2 * self.num_samples + 1) * self.num_feats * self.num_feats, self.num_codebooks))
                    data_batch_reconstructed = pq.decode(data_codes)
                    data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                          (-1, self.num_feats, self.num_feats,
                                                           self.num_channels))
                    data_batch_reconstructed = torch.from_numpy(
                        np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).cuda()

                    # perform random resize crop augmentation on each tensor
                    if self.use_random_resize_crops:
                        transform_data_batch = torch.empty_like(data_batch_reconstructed)
                        for tens_ix, tens in enumerate(data_batch_reconstructed):
                            transform_data_batch[tens_ix] = self.random_resize_crop(tens)
                        data_batch_reconstructed = transform_data_batch

                    # MIXUP: Do mixup between two batches of previous data
                    x_prev_mixed, prev_labels_a, prev_labels_b, lam = self.mixup_data(
                        data_batch_reconstructed[1:1 + self.num_samples],
                        data_labels[1:1 + self.num_samples],
                        data_batch_reconstructed[1 + self.num_samples:],
                        data_labels[1 + self.num_samples:],
                        alpha=self.mixup_alpha)

                    data = torch.empty((self.num_samples + 1, self.num_channels, self.num_feats, self.num_feats))
                    data[0] = data_batch_reconstructed[0]
                    data[1:] = x_prev_mixed.clone()
                    labels_a = torch.zeros(self.num_samples + 1).long()
                    labels_b = torch.zeros(self.num_samples + 1).long()
                    labels_a[0] = y.squeeze()
                    labels_b[0] = y.squeeze()
                    labels_a[1:] = prev_labels_a
                    labels_b[1:] = prev_labels_b

                    # fit on replay mini-batch plus new sample
                    output = classifier_F(data.cuda())
                    loss = self.mixup_criterion(criterion, output, labels_a.cuda(), labels_b.cuda(), lam)
                else:
                    # gather previous data for replay
                    data_codes = np.empty(
                        (self.num_samples + 1, self.num_feats, self.num_feats, self.num_codebooks),
                        dtype=np.uint8)
                    data_labels = torch.empty((self.num_samples + 1), dtype=torch.long).cuda()
                    data_codes[0] = x
                    data_labels[0] = y
                    ixs = randint(len(rehearsal_ixs), self.num_samples)
                    ixs = [rehearsal_ixs[_curr_ix] for _curr_ix in ixs]
                    for ii, v in enumerate(ixs):
                        data_codes[ii + 1] = latent_dict[v][0]
                        data_labels[ii + 1] = torch.from_numpy(latent_dict[v][1])

                    # reconstruct/decode samples with PQ
                    data_codes = np.reshape(data_codes, (
                        (self.num_samples + 1) * self.num_feats * self.num_feats, self.num_codebooks))
                    data_batch_reconstructed = pq.decode(data_codes)
                    data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                          (-1, self.num_feats, self.num_feats,
                                                           self.num_channels))
                    data_batch_reconstructed = torch.from_numpy(
                        np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).cuda()

                    # perform random resize crop augmentation on each tensor
                    if self.use_random_resize_crops:
                        transform_data_batch = torch.empty_like(data_batch_reconstructed)
                        for tens_ix, tens in enumerate(data_batch_reconstructed):
                            transform_data_batch[tens_ix] = self.random_resize_crop(tens)
                        data_batch_reconstructed = transform_data_batch

                    # fit on replay mini-batch plus new sample
                    output = classifier_F(data_batch_reconstructed)
                    loss = criterion(output, data_labels)

                loss = loss.mean()
                self.optimizer.zero_grad()  # zero out grads before backward pass because they are accumulated
                loss.backward()

                # if gradient clipping is desired
                if self.grad_clip is not None:
                    nn.utils.clip_grad_norm_(classifier_F.parameters(), self.grad_clip)

                self.optimizer.step()

                total_loss.update(loss.item())
                if verbose:
                    print(msg % (c, total_loss.avg, time.time() - start_time), end="")
                c += 1

                # since we have visited item_ix, it is now eligible for replay
                rehearsal_ixs.append(int(item_ix.numpy()))
                latent_dict[int(item_ix.numpy())] = [x, y.numpy()]
                class_id_to_item_ix_dict[int(y.numpy())].append(int(item_ix.numpy()))

                # if buffer is full, randomly replace previous example from class with most samples
                if self.max_buffer_size is not None and counter.count >= self.max_buffer_size:
                    # class with most samples and random item_ix from it
                    max_key = max(class_id_to_item_ix_dict, key=lambda x: len(class_id_to_item_ix_dict[x]))
                    max_class_list = class_id_to_item_ix_dict[max_key]
                    rand_item_ix = random.choice(max_class_list)

                    # remove the random_item_ix from all buffer references
                    max_class_list.remove(rand_item_ix)
                    latent_dict.pop(rand_item_ix)
                    rehearsal_ixs.remove(rand_item_ix)
                else:
                    counter.update()

                # update lr scheduler
                if self.lr_scheduler_per_class is not None:
                    self.lr_scheduler_per_class[int(y)].step()

    def mixup_data(self, x1, y1, x2, y2, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        mixed_x = lam * x1 + (1 - lam) * x2
        y_a, y_b = y1, y2
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * criterion(pred, y_a.squeeze()) + (1 - lam) * criterion(pred, y_b.squeeze())

    def predict(self, data_loader, pq):
        """
        Perform inference with REMIND.
        :param data_loader: data loader of test images (images, labels)
        :param pq: trained PQ model
        :return: (label predictions, probabilities, ground truth labels)
        """
        with torch.no_grad():
            self.classifier_F.eval()
            self.classifier_F.cuda()
            self.classifier_G.eval()
            self.classifier_G.cuda()

            probas = torch.zeros((len(data_loader.dataset), self.num_classes))
            all_lbls = torch.zeros((len(data_loader.dataset)))
            start_ix = 0
            for batch_ix, batch in enumerate(data_loader):
                batch_x, batch_lbls = batch[0], batch[1]
                batch_x = batch_x.cuda()

                # get G features
                data_batch = self.classifier_G(batch_x).cpu().numpy()

                # quantize test data so features are in the same space as training data
                data_batch = np.transpose(data_batch, (0, 2, 3, 1))
                data_batch = np.reshape(data_batch, (-1, self.num_channels))
                codes = pq.compute_codes(data_batch)
                data_batch_reconstructed = pq.decode(codes)
                data_batch_reconstructed = np.reshape(data_batch_reconstructed,
                                                      (-1, self.num_feats, self.num_feats, self.num_channels))
                data_batch_reconstructed = torch.from_numpy(np.transpose(data_batch_reconstructed, (0, 3, 1, 2))).cuda()

                batch_lbls = batch_lbls.cuda()
                logits = self.classifier_F(data_batch_reconstructed)
                end_ix = start_ix + len(batch_x)
                probas[start_ix:end_ix] = F.softmax(logits.data, dim=1)
                all_lbls[start_ix:end_ix] = batch_lbls.squeeze()
                start_ix = end_ix

            preds = probas.data.max(1)[1]

        return preds.numpy(), probas.numpy(), all_lbls.int().numpy()

    def save(self, inc, save_full_path, rehearsal_ixs, latent_dict, class_id_to_item_ix_dict, pq):
        """
        Save REMIND parameters to files.
        :param inc: which increment number is being saved
        :param save_full_path: path where weights should be saved
        :param rehearsal_ixs: list of item_ixs eligible for replay
        :param latent_dict: dictionary containing latent codes for replay samples
        :param class_id_to_item_ix_dict: dictionary of visited classes with associated item_ixs visited
        :param pq: trained PQ object for decoding latent codes
        :return: None
        """

        if not os.path.exists(save_full_path):
            os.makedirs(save_full_path)

        state = {
            'model_state_dict': self.classifier_F.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        print(f'\nSaving REMIND model to {save_full_path}')
        torch.save(state, os.path.join(save_full_path, 'remind_classifier_F_%d.pth' % inc))

        # get PQ centroids/codebooks
        centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, pq.ksub, pq.dsub)

        d = {'latent_dict': latent_dict, 'rehearsal_ixs': rehearsal_ixs,
             'class_id_to_item_ix_dict': class_id_to_item_ix_dict, 'pq_centroids': centroids}

        with open(os.path.join(save_full_path, 'remind_buffer_%d.pkl' % inc), 'wb') as f:
            pickle.dump(d, f)

    def resume(self, inc, resume_full_path):
        """
        Load previous REMIND model to continue training.
        :param inc: which increment number was saved
        :param resume_full_path: path where weights are saved
        :return: (classifier state dict, latent dict, rehearsal ixs list, class id to item ix dict)
        """

        print(f'\nResuming REMIND model from {resume_full_path}')
        state = torch.load(os.path.join(resume_full_path, 'remind_classifier_F_%d.pth' % inc))
        self.classifier_F.load_state_dict(state['model_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])

        # load parameters
        with open(os.path.join(resume_full_path, 'remind_buffer_%d.pkl' % inc), 'rb') as f:
            d = pickle.load(f)
        nbits = int(np.log2(self.codebook_size))
        pq = faiss.ProductQuantizer(self.num_channels, self.num_codebooks, nbits)
        faiss.copy_array_to_vector(d['pq_centroids'].ravel(), pq.centroids)
        return state, d['latent_dict'], d['rehearsal_ixs'], d['class_id_to_item_ix_dict'], pq

import argparse
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', metavar='DIR', help='path to dataset')  # path to ImageNet images
    parser.add_argument('--labels_dir', type=str, default='./imagenet_files/imagenet_indices')
    parser.add_argument('--class_order_text_file', type=str, default='./imagenet_files/imagenet_class_order.txt')
    args = parser.parse_args()

    print('\nloading the data...')
    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    # determine default load order of labels and class names from pytorch (alphabetical)
    default_train_labels = train_dataset.targets
    default_val_labels = val_dataset.targets
    default_class_order_per_sample = [train_dataset.samples[i][0].split('/')[-2] for i in
                                      range(len(train_dataset.samples))]
    default_class_order = []
    for v in default_class_order_per_sample:
        if v not in default_class_order:
            default_class_order.append(v)

    # load in user desired order from text file
    with open(args.class_order_text_file) as f:
        lines = [line.rstrip() for line in f]  # grab each class name from line in text file

    # compute mapping from default pytorch order to user order
    map = []
    for v in default_class_order:
        ix = lines.index(v)
        map.append(ix)

    # relabel all samples and save to numpy files
    new_train_labels = np.empty_like(default_train_labels)
    new_val_labels = np.empty_like(default_val_labels)

    for i in range(len(new_train_labels)):
        new_train_labels[i] = map[default_train_labels[i]]

    for i in range(len(new_val_labels)):
        new_val_labels[i] = map[default_val_labels[i]]

    print('Saving numpy file to {}'.format(
        os.path.join(args.labels_dir, 'imagenet_train_labels'.format(args.labels_dir))))
    np.save(os.path.join(args.labels_dir, 'imagenet_train_labels'.format(args.labels_dir)), np.array(new_train_labels))
    print(
        'Saving numpy file to {}'.format(os.path.join(args.labels_dir, 'imagenet_val_labels'.format(args.labels_dir))))
    np.save(os.path.join(args.labels_dir, 'imagenet_val_labels'.format(args.labels_dir)), np.array(new_val_labels))


if __name__ == '__main__':
    main()

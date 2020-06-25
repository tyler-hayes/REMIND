REMIND
=====================================
This is a PyTorch implementation of the REMIND algorithm from our paper. An [arXiv pre-print](https://arxiv.org/abs/1910.02509) of our paper is available.

## Dependences 
- Tested with Python 3.7.6 and PyTorch 1.3.1, torchvision 0.4.2, NumPy, FAISS, NVIDIA GPU

## Setup ImageNet-2012
The ImageNet Large Scale Visual Recognition Challenge (ILSVRC) dataset has 1000 categories and 1.2 million images. The images do not need to be preprocessed or packaged in any database, but the validation images need to be moved into appropriate subfolders. [See link.](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset)

1. Download the images from http://image-net.org/download-images

2. Extract the training data:
  ```bash
  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..
  ```

3. Extract the validation data and move images to subfolders:
  ```bash
  mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
  wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
  ```
  
## Installing FAISS
[See link.](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md)
# CPU version only
conda install faiss-cpu -c pytorch

# GPU version
conda install faiss-gpu cudatoolkit=8.0 -c pytorch # For CUDA8
conda install faiss-gpu cudatoolkit=9.0 -c pytorch # For CUDA9
conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA10

## Citation
If using this code, please cite our paper.
```
@article{hayes2019remind,
  title={REMIND Your Neural Network to Prevent Catastrophic Forgetting},
  author={Hayes, Tyler L and Kafle, Kushal and Shrestha, Robik and Acharya, Manoj and Kanan, Christopher},
  journal={arXiv preprint arXiv:1910.02509},
  year={2019}
}


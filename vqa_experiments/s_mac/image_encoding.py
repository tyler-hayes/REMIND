#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2018 Kim Seonghyeon
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ------------------------------------------------------------------------------
#
# Copyright (C) IBM Corporation 2018
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
image_encoding.py: Implementation of the image processing done by the input unit in the MAC network. See \
https://arxiv.org/pdf/1803.03067.pdf for the reference paper.

"""
__author__ = "Vincent Marois"

import torch
from torch.nn import Module


class ImageProcessing(Module):
    """
    Image encoding using a 2-layers CNN assuming the images have been already \
    preprocessed by `ResNet101`.
    """

    def __init__(self, dim):
        """
        Constructor for the 2-layers CNN.

        :param dim: global 'd' hidden dimension
        :type dim: int

        """

        # call base constructor
        super(ImageProcessing, self).__init__()

        self.conv = torch.nn.Sequential(torch.nn.Conv2d(1024, dim, 3, padding=1),
                                        torch.nn.ELU(),
                                        torch.nn.Conv2d(dim, dim, 3, padding=1),
                                        torch.nn.ELU())
        # specify weights initialization
        torch.nn.init.kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        torch.nn.init.kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

    def forward(self, feature_maps):
        """
        Apply the constructed CNN model on the feature maps (coming from \
        `ResNet101`).

        :param feature_maps: [batch_size x nb_kernels x feat_H x feat_W] coming from `ResNet101`. \
               Should have [nb_kernels x feat_H x feat_W] = [1024 x 14 x 14].
        :type feature_maps: torch.tensor

        :return feature_maps: feature map, shape [batch_size, dim, new_height, new_width]

        """

        new_feature_maps = self.conv(feature_maps)

        return new_feature_maps

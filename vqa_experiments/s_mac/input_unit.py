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
input_unit.py: Implementation of the input unit for the MAC network. Cf https://arxiv.org/abs/1803.03067 for \
the reference paper.
"""
__author__ = "Vincent Marois"
import torch
from torch.nn import Module

from vqa_experiments.s_mac.utils_mac import linear
from vqa_experiments.s_mac.image_encoding import ImageProcessing

class InputUnit(Module):
    """
    Implementation of the ``InputUnit`` of the MAC network.
    """

    def __init__(self, dim, embedded_dim):
        """
        Constructor for the ``InputUnit``.

        :param dim: global 'd' hidden dimension
        :type dim: int

        :param embedded_dim: dimension of the word embeddings.
        :type embedded_dim: int

        """

        # call base constructor
        super(InputUnit, self).__init__()

        self.dim = dim



        # instantiate image processing (2-layers CNN)
        self.conv = ImageProcessing(dim)

        # define linear layer for the projection of the knowledge base
        self.kb_proj_layer = linear(dim, dim, bias=True)

        # create bidirectional LSTM layer
        self.lstm = torch.nn.LSTM(input_size=embedded_dim, hidden_size=self.dim,
                            num_layers=1, batch_first=True, bidirectional=True)

        # linear layer for projecting the word encodings from 2*dim to dim
        # TODO: linear(2*self.dim, self.dim, bias=True) ?
        self.lstm_proj = torch.nn.Linear(2 * self.dim, self.dim)

    def forward(self, questions, questions_len, feature_maps):
        """
        Forward pass of the ``InputUnit``.

        :param questions: tensor of the questions words, shape [batch_size x maxQuestionLength x embedded_dim].
        :type questions: torch.tensor

        :param questions_len: Unpadded questions length.
        :type questions_len: list

        :param feature_maps: [batch_size x nb_kernels x feat_H x feat_W] coming from `ResNet101`.
        :type feature_maps: torch.tensor

        :return:

            - question encodings: [batch_size x 2*dim] (torch.tensor),
            - word encodings: [batch_size x maxQuestionLength x dim] (torch.tensor),
            - images_encodings: [batch_size x nb_kernels x (H*W)] (torch.tensor).


        """
        batch_size = feature_maps.shape[0]

        # images processing
        feature_maps = self.conv(feature_maps)

        # reshape feature maps as channels first
        feature_maps = feature_maps.view(batch_size, self.dim, -1)

        # pass feature maps through linear layer
        kb_proj = self.kb_proj_layer(
            feature_maps.permute(0, 2, 1)).permute(0, 2, 1)

        # avoid useless computations on padding elements: pack sequences
        embed = torch.nn.utils.rnn.pack_padded_sequence(
            questions, questions_len, batch_first=True)

        # LSTM layer: words & questions encodings
        lstm_out, (h, _) = self.lstm(embed)

        # reshape packed sequences to a padded tensor
        lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True)

        # get final words encodings using linear layer
        lstm_out = self.lstm_proj(lstm_out)

        # reshape last hidden states for questions encodings -> [batch_size x
        # (2*dim)]
        h = h.permute(1, 0, 2).contiguous().view(batch_size, -1)

        # return everything
        return feature_maps, kb_proj, lstm_out, h

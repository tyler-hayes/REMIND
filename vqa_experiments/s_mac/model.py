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
model.py:

    - Implementation of the ``MAC`` network, reusing the different units implemented in separated files.
    - Cf https://arxiv.org/abs/1803.03067 for the reference paper.


"""
__author__ = "Vincent Marois , Vincent Albouy"

import os
import nltk
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms

from vqa_experiments.s_mac.input_unit import InputUnit
from vqa_experiments.s_mac.mac_unit import MACUnit
from vqa_experiments.s_mac.output_unit import OutputUnit

from vqa_experiments.vqa_models import WordEmbedding

class MACNetwork(nn.Module):
    """
    Implementation of the entire ``MAC`` network.
    """

    def __init__(self, params):
        """
        Constructor for the ``MAC`` network.

        :param params: dict of parameters (read from configuration ``.yaml`` file).
        :type params: utils.ParamInterface

        :param problem_default_values_: default values coming from the ``Problem`` class.
        :type problem_default_values_: dict
        """

        # call base constructor
        super(MACNetwork, self).__init__()

        # parse params dict
        self.dim = 512
        self.embed_hidden = params.emb_dim  # embedding dimension
        self.max_step = 12
        self.self_attention = False
        self.memory_gate = False

        self.max_step = 12
        self.dropout = 0.15


        try:
            self.nb_classes = params.num_classes
        except KeyError:
            self.logger.warning("Couldn't retrieve one or more value(s) from problem_default_values_.")

        self.name = 'MAC'
        self.embedding = WordEmbedding(params.d.ntoken, params.emb_dim, params.embedding_dropout)

        # instantiate units
        self.input_unit = InputUnit(
            dim=self.dim, embedded_dim=self.embed_hidden)

        self.mac_unit = MACUnit(
            dim=self.dim,
            max_step=self.max_step,
            self_attention=self.self_attention,
            memory_gate=self.memory_gate,
            dropout=self.dropout)

        self.output_unit = OutputUnit(dim=self.dim, nb_classes=self.nb_classes)

#        self.data_definitions = {'images': {'size': [-1, 1024, 14, 14], 'type': [np.ndarray]},
#                                 'questions': {'size': [-1, -1, -1], 'type': [torch.Tensor]},
#                                 'questions_length': {'size': [-1], 'type': [list, int]},
#                                 'targets': {'size': [-1, self.nb_classes], 'type': [torch.Tensor]}
#                                 }

#        # transform for the image plotting
#        self.transform = transforms.Compose(
#            [transforms.Resize([224, 224]), transforms.ToTensor()])

    def forward(self, q, v, ql, dropout=0.15):
        """
        Forward pass of the ``MAC`` network. Calls first the ``InputUnit``, then the recurrent \
        MAC cells and finally the ```OutputUnit``.

        :param data_dict: input data batch.
        :type data_dict: utils.DataDict

        :param dropout: dropout rate.
        :type dropout: float

        :return: Predictions of the model.
        """

        # reset cell state history for visualization

        # unpack data_dict
        v = torch.transpose(v,1,2)
        images = v.view(-1,1024,14,14)
        questions = self.embedding(q)
        questions_length = ql

        # input unit
        img, kb_proj, lstm_out, h = self.input_unit(
            questions, questions_length, images)

        # recurrent MAC cells
        memory = self.mac_unit(lstm_out, h, img, kb_proj)

        # output unit
        logits = self.output_unit(memory, h)

        return logits

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
s_mac_unit.py:

    - Implementation of the :py:class:`MACUnit` for the ``S-MAC`` network (simplified MAC).
    - Cf https://arxiv.org/abs/1803.03067 for the reference MAC paper (Hudson and Manning, ICLR 2018).

"""
__author__ = "Vincent Marois & T.S. Jayram"

import torch
from torch.nn import Module

from vqa_experiments.s_mac.s_control_unit import ControlUnit
from vqa_experiments.s_mac.s_read_unit import ReadUnit
from vqa_experiments.s_mac.s_write_unit import WriteUnit



class MACUnit(Module):
    """
    Implementation of the :py:class:`MACUnit` (iteration over the MAC cell) of the ``S-MAC`` network.

    .. note::

        This implementation is part of a simplified version of the MAC network, where modifications regarding \
        the different units have been done to reduce the number of linear layers (and thus number of parameters).

        The implementation being simplified, we are not using the optional `self-attention` & `memory-gating` in \
        the :py:class:`WriteUnit`.

        This is part of a submission to the ViGIL workshop for NIPS 2018. Feel free to use this model and refer to it \
        with the following BibTex:

        ::

            @article{marois2018transfer,
                    title={On transfer learning using a MAC model variant},
                    author={Marois, Vincent and Jayram, TS and Albouy, Vincent and Kornuta, Tomasz and Bouhadjar, Younes and Ozcan, Ahmet S},
                    journal={arXiv preprint arXiv:1811.06529},
                    year={2018}
            }

    """

    def __init__(self, dim, max_step=12, dropout=0.15):
        """
        Constructor for the :py:class:`MACUnit`, which represents the recurrence over the \
        MACCell for the ``S-MAC`` network.

        :param dim: global 'd' hidden dimension.
        :type dim: int

        :param max_step: maximal number of MAC cells. Default: 12.
        :type max_step: int

        :param dropout: dropout probability for the variational dropout mask. Default: 0.15.
        :type dropout: float

        """

        # call base constructor
        super(MACUnit, self).__init__()

        # instantiate the units
        self.control = ControlUnit(dim=dim, max_step=max_step)
        self.read = ReadUnit(dim=dim)
        self.write = WriteUnit(dim=dim)

        # initialize hidden states
        self.mem_0 = torch.nn.Parameter(torch.zeros(1, dim))
        self.control_0 = torch.nn.Parameter(torch.zeros(1, dim))

        self.dim = dim
        self.max_step = max_step
        self.dropout = dropout

        # for the visualization
        self.cell_state_history = []

    @staticmethod
    def get_dropout_mask(x, dropout):
        """
        Create a dropout mask to be applied on x.

        :param x: tensor of arbitrary shape to apply the mask on.
        :type x: :py:class:`torch.Tensor`

        :param dropout: dropout rate.
        :type dropout: float

        :return: mask (:py:class:`torch.Tensor`)

        """
        # create a binary mask, where the probability of 1's is (1-dropout)
        mask = torch.empty_like(x).bernoulli_(1 - dropout)

        # normalize the mask so that the average value is 1 and not (1-dropout)
        mask /= (1 - dropout)

        return mask

    def forward(self, context, question, kb_proj):
        """
        Forward pass of the :py:class:`MACUnit`, which represents the recurrence over the \
        MACCell for the ``S-MAC`` network.

        :param context: contextual words, shape `[batch_size x maxQuestionLength x dim]`
        :type context: :py:class:`torch.Tensor`

        :param question: questions encodings, shape `[batch_size x 2*dim]`
        :type question: :py:class:`torch.Tensor`

        :param kb_proj: Linear projection of the knowledge_base (feature maps extracted by a CNN), shape \
        `[batch_size x dim x (feat_H * feat_W)]`.
        :type kb_proj: :py:class:`torch.Tensor`

        :return: Last memory state (:py:class:`torch.Tensor`)

        """
        # get batch size
        batch_size = question.size(0)

        # expand the hidden states to whole batch
        control = self.control_0.expand(batch_size, self.dim)
        memory = self.mem_0.expand(batch_size, self.dim)

        # apply variational dropout during training
        if self.training:  # TODO: check
            control_mask = self.get_dropout_mask(control, self.dropout)
            memory_mask = self.get_dropout_mask(memory, self.dropout)
            control = control * control_mask
            memory = memory * memory_mask

        # main loop of recurrence over the MACCell
        for i in range(self.max_step):

            # control unit
            control = self.control(step=i, contextual_words=context, question_encoding=question,
                                   ctrl_state=control)

            # apply variational dropout
            if self.training:
                control = control * control_mask

            # read unit
            read = self.read(memory_state=memory, ctrl_state=control, kb_proj=kb_proj)

            # write unit
            memory = self.write(read_vector=read)

            # apply variational dropout
            if self.training:
                memory = memory * memory_mask

            # # store attention weights for visualization
            # if app_state.visualize:
            #     self.cell_state_history.append(
            #         (self.read.rvi.cpu().detach(), self.control.cvi.cpu().detach()))

        return memory

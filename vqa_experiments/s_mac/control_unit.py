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
control_unit.py: Implementation of the Control Unit for the MAC network. Cf https://arxiv.org/abs/1803.03067 for the \
reference paper.
"""
__author__ = "Vincent Marois"

import torch
from torch.nn import Module
from vqa_experiments.s_mac.utils_mac import linear


class ControlUnit(Module):
    """
    Implementation of the ``ControlUnit`` of the MAC network.
    """

    def __init__(self, dim, max_step):
        """
        Constructor for the control unit.

        :param dim: global 'd' hidden dimension
        :type dim: int

        :param max_step: maximum number of steps -> number of MAC cells in the network.
        :type max_step: int

        """

        # call base constructor
        super(ControlUnit, self).__init__()

        # define the linear layers (one per step) used to make the questions
        # encoding
        self.pos_aware_layers = torch.nn.ModuleList()
        for _ in range(max_step):
            self.pos_aware_layers.append(linear(2 * dim, dim, bias=True))

        # define the linear layer used to create the cqi values
        self.ctrl_question = linear(2 * dim, dim, bias=True)

        # define the linear layer used to create the attention weights. Should
        # be one scalar weight per contextual word
        self.attn = linear(dim, 1, bias=True)
        self.step = 0

    def forward(self, step, contextual_words, question_encoding, ctrl_state):
        """
        Forward pass of the ``ControlUnit``.

        :param step: index of the current MAC cell.
        :type step: int

        :param contextual_words: tensor of shape [batch_size x maxQuestionLength x dim] containing the words \
        encodings ('representation of each word in the context of the question').
        :type contextual_words: torch.tensor


        :param question_encoding: question representation, of shape [batch_size x 2*dim].
        :type question_encoding: torch.tensor

        :param ctrl_state: previous control state, of shape [batch_size x dim]
        :type ctrl_state: torch.tensor

        :return: new control state, [batch_size x dim]

        """
        self.step = step
        # select current 'position aware' linear layer & pass questions through
        # it
        pos_aware_question_encoding = self.pos_aware_layers[step](
            question_encoding)

        cqi = torch.cat([ctrl_state, pos_aware_question_encoding], dim=-1)
        cqi = self.ctrl_question(cqi)  # [batch_size x dim]

        # compute element-wise product between cqi & contextual words
        # [batch_size x maxQuestionLength x dim]
        context_ctrl = cqi.unsqueeze(1) * contextual_words

        # compute attention weights
        cai = self.attn(context_ctrl)  # [batch_size x maxQuestionLength x 1]

        self.cvi = torch.nn.functional.softmax(cai, dim=1)

        # compute next control state
        # [batch_size x dim]
        next_ctrl_state = (self.cvi * contextual_words).sum(1)

        return next_ctrl_state

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
s_write_unit.py:

    - Implementation of the :py:class:`WriteUnit` for the ``S-MAC`` network (simplified MAC).
    - Cf https://arxiv.org/abs/1803.03067 for the reference MAC paper (Hudson and Manning, ICLR 2018).

"""
__author__ = "Vincent Marois & T.S. Jayram"

from torch.nn import Module
from vqa_experiments.s_mac.utils_mac import linear


class WriteUnit(Module):
    """
    Implementation of the :py:class:`WriteUnit` for the ``S-MAC`` model.

    .. note::

        This implementation is part of a simplified version of the MAC network, where modifications regarding \
        the different units have been done to reduce the number of linear layers (and thus number of parameters).

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

    def __init__(self, dim):
        """
        Constructor for the :py:class:`WriteUnit` of the ``S-MAC`` model.

        :param dim: global 'd' hidden dimension.
        :type dim: int

        """

        # call base constructor
        super(WriteUnit, self).__init__()

        # linear layer to create the new memory state from the current read vector (coming from the read unit)
        self.concat_layer = linear(dim, dim, bias=True)

    def forward(self, read_vector):
        """
        Forward pass of the :py:class:`WriteUnit` for the ``S-MAC`` model.

        :param read_vector: current read vector (output of the :py:class:`ReadUnit`), shape `[batch_size x dim]`.
        :type read_vector: :py:class:`torch.Tensor`

        :return: current memory state, shape [batch_size x mem_dim] (:py:class:`torch.Tensor`).
        """
        return self.concat_layer(read_vector)

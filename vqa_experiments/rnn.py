import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class RNN(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers=1, bidirect=True, dropout=0, rnn_type='GRU'):
        super(RNN, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def init_hidden_with(self, qn):
        qn = qn.unsqueeze(0).repeat(self.nlayers * self.ndirections, 1, 1)
        return qn

    def forward(self, x, init_hid_with=None, q_one_hot=None):
        batch_size = x.size(0)
        if init_hid_with is None:
            hidden = self.init_hidden(batch_size)
        else:
            hidden = self.init_hidden_with(init_hid_with)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden) # output: B x num_objs x (gru_dim * 2), hidden: 2 x B x gru_dim

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        ret = torch.cat((forward_, backward), dim=1)
        return ret

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output

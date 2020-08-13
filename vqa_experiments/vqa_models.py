"""
Written by Kushal, modified by Robik
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from vqa_experiments.rnn import RNN


# TODO: Test with and without weight norm

class WordEmbedding(nn.Module):
    """Word Embedding
    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim)
        self.dropout = nn.Dropout()
        self.ntoken = ntoken
        self.emb_dim = emb_dim
        self.use_dropout = dropout

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        if self.use_dropout:
            emb = self.dropout(emb)
        return emb


class Classifier(nn.Module):
    def __init__(self, num_input, num_hidden, num_classes, use_dropout=True):
        super(Classifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.lin1 = nn.Linear(num_input, num_hidden)
        self.classifier = nn.Linear(num_hidden, num_classes)
        self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, feat):
        # Tile ques_vector, concatenate
        projection = self.relu(self.lin1(feat))
        if self.use_dropout:
            projection = self.dropout(projection)
        preds = self.classifier(projection)
        return preds


class MultiModalCore(nn.Module):
    """
    Concatenates visual and linguistic features and passes them through an MLP.
    """

    def __init__(self, config):
        super(MultiModalCore, self).__init__()
        self.config = config
        self.v_dim_orig = self.config.cnn_feat_size
        self.v_dim = 512

        if config.bidirectional:
            self.q_emb_dim = config.lstm_out * 2
        else:
            self.q_emb_dim = config.lstm_out

        self.mmc_sizes = [1024, 1024, 1024, 1024]
        self.mmc_layers = []
        nonlin = nn.ReLU()

        self.conv = torch.nn.Sequential(torch.nn.Conv2d(self.v_dim_orig, self.v_dim, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(self.v_dim, self.v_dim, 3, padding=1),
                                        torch.nn.ReLU())
        # specify weights initialization
        torch.nn.init.kaiming_uniform_(self.conv[0].weight)
        self.conv[0].bias.data.zero_()
        torch.nn.init.kaiming_uniform_(self.conv[2].weight)
        self.conv[2].bias.data.zero_()

        # Create MLP with early fusion in the first layer followed by batch norm
        for mmc_ix in range(len(self.mmc_sizes)):
            if mmc_ix == 0:
                in_s = self.v_dim + self.q_emb_dim
                self.batch_norm_fusion = nn.BatchNorm1d(in_s)
            else:
                in_s = self.mmc_sizes[mmc_ix - 1]
            out_s = self.mmc_sizes[mmc_ix]
            lin = nn.Linear(in_s, out_s)
            self.mmc_layers.append(lin)
            self.mmc_layers.append(nonlin)

        self.mmc_layers = nn.ModuleList(self.mmc_layers)
        self.batch_norm_mmc = nn.BatchNorm1d(self.mmc_sizes[-1])

        # Aggregation
        out_s += self.q_emb_dim
        self.batch_norm_before_aggregation = nn.BatchNorm1d(out_s)
        self.aggregator = RNN(out_s, 1024, nlayers=1, bidirect=True)

    #        self.aggregator = nn.GRU(out_s, 1024, bidirectional=True,batch_first=True)

    def forward(self, v, q, labels=None):
        """
        :param v: B x num_objs x emb_size
        :param b: B x num_objs x 6 (Boxes)
        :param q: B x emb_size
        :param labels
        :return:
         """
        q = q.unsqueeze(1).repeat(1, v.shape[1], 1)

        nfeat = int(np.sqrt(v.size(1)))
        v = torch.transpose(v, 1, 2)
        v = v.view(-1, self.v_dim_orig, nfeat, nfeat)
        v = self.conv(v)
        v = v.view(-1, self.v_dim, nfeat * nfeat)
        v = torch.transpose(v, 1, 2)
        x = torch.cat([v, q], dim=2)  # B x num_objs x (2 * emb_size)
        num_objs = x.shape[1]
        emb_size = x.shape[2]
        x = x.view(-1, emb_size)
        x = self.batch_norm_fusion(x)
        x = x.view(-1, num_objs, emb_size)

        curr_lin_layer = -1

        # Pass through MMC
        for mmc_layer in self.mmc_layers:
            if isinstance(mmc_layer, nn.Linear):
                curr_lin_layer += 1
                mmc_out = mmc_layer(x)
                x_new = mmc_out
                if curr_lin_layer > 0:
                    x_new = x + mmc_out
                if x_new is None:
                    x_new = mmc_out
                x = x_new
            else:
                x = mmc_layer(x)

        x = x.view(-1, self.mmc_sizes[-1])
        x = self.batch_norm_mmc(x)
        x = x.view(-1, num_objs, self.mmc_sizes[-1])

        x = torch.cat((x, q), dim=2)
        curr_size = x.size()
        x = x.view(-1, curr_size[2])
        x = self.batch_norm_before_aggregation(x)
        x = x.view(curr_size)
        x_aggregated = self.aggregator(x)
        return x, x_aggregated


class QuestionEncoder(nn.Module):
    def __init__(self, config):
        super(QuestionEncoder, self).__init__()
        self.embedding = WordEmbedding(config.d.ntoken, config.emb_dim, config.embedding_dropout)
        self.lstm = nn.LSTM(input_size=config.emb_dim,
                            hidden_size=config.lstm_out,
                            num_layers=1, bidirectional=config.bidirectional)
        self.config = config

    #        self.lstm2 = nn.LSTM(input_size=config.emb_dim,
    #                            hidden_size=config.lstm_out,
    #                            num_layers=1, batch_first=False)

    def forward(self, q, q_len):
        q_embed = self.embedding(q)
        packed = pack_padded_sequence(q_embed, q_len, batch_first=True)
        o, (h, c) = self.lstm(packed)
        #        o1, (h1, c1) = self.lstm(q_embed)
        #        print(o1.shape)
        #        print(h1.shape)
        #        print(h.shape)
        #        o,_ = pad_packed_sequence(o)
        #        print(o.shape)
        #
        #        print("false")
        #
        #        o, (h, c) = self.lstm2(packed)
        #        o1, (h1, c1) = self.lstm2(q_embed)
        #        print(o1.shape)
        #        print(h1.shape)
        #        print(h.shape)
        #        o,_ = pad_packed_sequence(o)
        h = torch.transpose(h, 0, 1)
        return torch.flatten(h, start_dim=1)


class Attention(nn.Module):
    def __init__(self, imfeat_size, qfeat_size, use_dropout=True):
        super(Attention, self).__init__()
        self.nlin = nn.Linear(qfeat_size + imfeat_size, 1024)
        self.attnmap = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, qfeat, imfeat):
        num_objs = imfeat.size(1)
        qtile = qfeat.unsqueeze(1).repeat(1, num_objs, 1)
        #        print(qtile.shape)
        #        print(imfeat.shape)
        qi = torch.cat((imfeat, qtile), 2)
        qi = self.relu(self.nlin(qi))
        if self.use_dropout:
            qi = self.dropout(qi)
        attn_map = self.attnmap(qi)
        attn_map = nn.functional.softmax(attn_map, 1)
        return attn_map


class NewAttention(nn.Module):
    def __init__(self, imfeat_size, qfeat_size, use_dropout=True):
        super(NewAttention, self).__init__()
        self.q_proj = nn.Linear(qfeat_size, 1024)
        self.v_proj = nn.Linear(imfeat_size, 1024)

        self.attnmap = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.use_dropout = use_dropout

    def forward(self, qfeat, imfeat):
        num_objs = imfeat.size(1)
        qfeat_proj = self.relu(self.q_proj(qfeat))
        imfeat_proj = self.relu(self.v_proj(imfeat))
        qtile = qfeat_proj.unsqueeze(1).repeat(1, num_objs, 1)
        #        print(qtile.shape)
        #        print(imfeat.shape)
        qi = imfeat_proj * qtile
        #        print(qi.shape)
        if self.use_dropout:
            qi = self.dropout(qi)
        attn_map = self.attnmap(qi)
        attn_map = nn.functional.softmax(attn_map, 1)
        return attn_map


class QI(nn.Module):
    def __init__(self, config):
        super(QI, self).__init__()
        assert config.use_pooled
        self.config = config
        qfeat_dim = 2048
        if config.use_lstm:
            self.ques_encoder = QuestionEncoder(config)
            if config.bidirectional:
                qfeat_dim = config.lstm_out * 2
            else:
                qfeat_dim = config.lstm_out

        self.classifier = Classifier(qfeat_dim + self.config.cnn_feat_size,
                                     config.num_hidden,
                                     config.num_classes,
                                     config.classfier_dropout
                                     )

    def forward(self, q, imfeat, ql):
        if self.config.use_lstm:
            qfeat = self.ques_encoder(q, ql)
        else:
            qfeat = q
        concat_feat = torch.cat([qfeat, imfeat], dim=1)
        preds = self.classifier(concat_feat)
        return preds


class Q_only(nn.Module):
    def __init__(self, config):
        super(Q_only, self).__init__()
        assert config.use_pooled
        self.config = config
        qfeat_dim = 2048
        if config.use_lstm:
            self.ques_encoder = QuestionEncoder(config)
            if config.bidirectional:
                qfeat_dim = config.lstm_out * 2
            else:
                qfeat_dim = config.lstm_out

        self.classifier = Classifier(qfeat_dim,
                                     config.num_hidden,
                                     config.num_classes,
                                     config.classfier_dropout
                                     )

    def forward(self, q, imfeat, ql):
        if self.config.use_lstm:
            qfeat = self.ques_encoder(q, ql)
        else:
            qfeat = q
        preds = self.classifier(qfeat)
        return preds


class UpDown(nn.Module):
    def __init__(self, config):
        super(UpDown, self).__init__()
        assert not config.use_pooled
        self.config = config
        qfeat_dim = 2048
        if config.use_lstm:
            self.ques_encoder = QuestionEncoder(config)
            if config.bidirectional:
                qfeat_dim = config.lstm_out * 2
            else:
                qfeat_dim = config.lstm_out
        attention = []
        if config.attn_type == 'old':
            for i in range(config.num_attn_hops):
                attention.append(Attention(self.config.cnn_feat_size, qfeat_dim, config.attention_dropout))
        else:
            for i in range(config.num_attn_hops):
                attention.append(NewAttention(self.config.cnn_feat_size, qfeat_dim, config.attention_dropout))

        self.attention = nn.ModuleList(attention)
        self.classifier = Classifier(qfeat_dim + self.config.cnn_feat_size * config.num_attn_hops,
                                     config.num_hidden * 2,
                                     config.num_classes,
                                     config.classfier_dropout
                                     )

    def forward(self, q, imfeat, ql):
        #        print(imfeat.shape)
        if self.config.use_lstm:
            qfeat = self.ques_encoder(q, ql)
        else:
            qfeat = q
        for i in range(self.config.num_attn_hops):
            attn_map = self.attention[i](qfeat, imfeat)
            scaled_imfeat = (attn_map * imfeat).sum(1)
            if i == 0:
                concat_feat = torch.cat([qfeat, scaled_imfeat], dim=1)
            else:
                concat_feat = torch.cat([concat_feat, scaled_imfeat], dim=1)
        preds = self.classifier(concat_feat)
        return preds


class Ramen(nn.Module):
    def __init__(self, config):
        super(Ramen, self).__init__()
        self.config = config
        self.mmc_net = MultiModalCore(config)
        if config.use_lstm:
            self.ques_encoder = QuestionEncoder(config)
            if config.bidirectional:
                qfeat_dim = config.lstm_out * 2
            else:
                qfeat_dim = config.lstm_out

        self.classifier = Classifier(qfeat_dim + 1024*2,
                                     config.num_hidden * 2,
                                     config.num_classes,
                                     config.classfier_dropout
                                     )

    def forward(self, q, v, ql):
        """Forward

        v: [batch, num_objs, v_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits
        """
        batch_size, num_objs, v_emb_dim = v.size()
        q_emb = self.ques_encoder(q, ql)
        mmc, mmc_aggregated = self.mmc_net(v, q_emb)  # B x num_objs x num_hid and B x num_hid
        # print(mmc_aggregated.shape)
        concat_feat = torch.cat([q_emb, mmc_aggregated], dim=1)
        logits = self.classifier(concat_feat)
        return logits


def main():
    pass


if __name__ == '__main___':
    main()

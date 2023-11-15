import torch
import torch.nn as nn

import consts as consts


class KPRN(nn.Module):
    def __init__(self, type_embedding_dim, relation_embedding_dim, entity_embedding_dim, lstm_hidden_dim, type_size,
                 relation_size, entity_size, gamma):
        super(KPRN, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim
        self.gamma = gamma

        # because latter use "pack_padded_sequence", so it doesn't need embedding padding index
        self.type_embedding = nn.Embedding(type_size, type_embedding_dim,)
        self.relation_embedding = nn.Embedding(relation_size, relation_embedding_dim)
        self.entity_embedding = nn.Embedding(entity_size, entity_embedding_dim)

        self.lstm = nn.LSTM(input_size=type_embedding_dim + relation_embedding_dim + entity_embedding_dim,
                            hidden_size=lstm_hidden_dim, batch_first=True)

        self.fc1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)
        # output is the plausibility of this path, it is a binary classify so output dimension is 2
        self.fc2 = nn.Linear(lstm_hidden_dim, 2)
        self.relu = nn.ReLU()

    def forward(self, paths, paths_length):
        # shape: (N, 6, 32)
        type_embed = self.type_embedding(paths[:, :, 1])
        relation_embed = self.relation_embedding(paths[:, :, 2])
        # shape: (N, 6, 64)
        entity_embed = self.entity_embedding(paths[:, :, 0])

        # shape: (N, 6, 128)
        input_embed = torch.cat([entity_embed, type_embed, relation_embed], dim=-1)

        # 参考链接:https://zhuanlan.zhihu.com/p/342685890
        # pack_padded_sequence的输入(第一个参数)是经过pad_sequence处理之后的数据(即进行了padding处理的数据)
        # lengths参数(第二个参数)对应一个batch中序列的实际长度

        # 返回对象共有四个，其中data和batch_sizes是最重要的
        # batch_sizes告诉lstm每一个time step要吃入多少个序列
        pack_embed = nn.utils.rnn.pack_padded_sequence(input_embed, paths_length, batch_first=True)

        # hn is what we need, it is the last time step output
        # shape: (1, N, 256)
        packed_out, (hn, cn) = self.lstm(pack_embed)
        # lstm_out, lstm_out_lengths = nn.utils.rnn.pad_packed_sequence(packed_out)

        # shape: (N, 256)
        hn = hn[0, :, :]

        # shape: (N, 2)
        interactions_path_plausibility = self.fc2(self.relu(self.fc1(hn)))

        return interactions_path_plausibility

    def weighted_pooling(self, interaction_path_plausibility):

        weighted_out = torch.div(interaction_path_plausibility, self.gamma)
        exp_out = torch.exp(weighted_out)
        sum_out = torch.sum(exp_out, dim=0)
        g = torch.log(sum_out)

        return g

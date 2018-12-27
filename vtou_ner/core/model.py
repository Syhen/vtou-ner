# -*- coding: utf-8 -*-
"""
create on 2018-12-25 下午7:52

author @heyao
"""

import torch
import torch.nn as nn

from vtou_ner.layers import CRF

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


class BiLSTMCRF(nn.Module):
    def __init__(self, seq_len, vocab_size, num_tags, embedding_dim=64, hidden_size=200, embedding_matrix=None):
        super(BiLSTMCRF, self).__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding_layer.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
            self.embedding_layer.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_size // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.lstm_encoder = nn.Linear(hidden_size, num_tags)
        self.crf = CRF(num_tags)

    def _get_features(self, seq):
        embeddings = self.embedding_layer(seq)
        features, _ = self.lstm(embeddings)
        return self.lstm_encoder(features)

    def forward(self, seq):
        features = self._get_features(seq)
        tags = self.crf(features)
        return tags

    def loss(self, seq, tags):
        features = self._get_features(seq)
        return self.crf.loss(features, tags)

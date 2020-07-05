import torch
import torch.nn as nn
import torch.nn.functional as F


class BoW(nn.Module):
    def __init__(self, word_embedding, num_classes, dropout_prob):
        super(BoW, self).__init__()
        self.word_embedding = nn.Embedding(word_embedding.shape[0], word_embedding.shape[1], padding_idx=0)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(word_embedding))
        self.word_embedding.weight.requires_grad=False
        
        self.dense = nn.Linear(word_embedding.shape[1], word_embedding.shape[1])
        self.dropout = nn.Dropout(p=dropout_prob)
        self.output_layer = nn.Linear(word_embedding.shape[1], num_classes)

    def forward(self, word_ids):
        word_sum_vector = self.word_embedding(word_ids).sum(1)
        feature_vector = self.dropout(word_sum_vector)
        feature_vector = self.dense(feature_vector)
        feature_vector = torch.tanh(feature_vector)
        feature_vector = self.dropout(feature_vector)
        return self.output_layer(feature_vector)
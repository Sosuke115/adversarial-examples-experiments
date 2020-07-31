import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


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
    
    
class CNN_Text(nn.Module):
    
    def __init__(self, word_embedding, num_classes, dropout_prob):
        super(CNN_Text, self).__init__()
#         self.args = args
        
        V = word_embedding.shape[0]
        D = word_embedding.shape[1]
        C = num_classes
        Ci = 1
        Co = 100
        Ks = [3,4,5]
        self.word_embedding = nn.Embedding(V, D)
        self.word_embedding.weight = nn.Parameter(torch.FloatTensor(word_embedding))
#         self.word_embedding.weight.requires_grad=False
#         self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(len(Ks)*Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, word_ids):
        x = self.word_embedding(word_ids)  # (N, W, D)
        
#         if self.args.static:
#         x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
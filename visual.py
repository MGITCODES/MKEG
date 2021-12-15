import torch
import torch.nn as nn
import torchvision
from torch.nn.modules.module import Module
import math
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import cv2
from PIL import Image
import os
from torchvision import transforms



class VisualFeatureExtractor(nn.Module):
    def __init__(self, in_features, output_features):
        super(VisualFeatureExtractor, self).__init__()
        self.model = self.__get_model()
        self.activation = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=1024,out_channels=20,kernel_size=(1,1),stride=(1,1),padding=0 ,bias=False)
        # self.globalmaxpooling = F.max_pool2d(in_features, output_features)


    def __get_model(self):
        densenet = torchvision.models.densenet121(pretrained=True)
        densenet = densenet.features
        model_state = torch.load('./CheXpert_Baseline_PyTorch-master/checkpoints/m_0808_032627.pth')
        old_state = densenet.state_dict()
        new_state = model_state['state_dict']
        densenet.load_state_dict({k[16:]: v for k, v in new_state.items() if k[16:] in old_state})
        return densenet

    def forward(self, images):
        visual_features = self.model(images)
        visual_features_raw = self.conv(visual_features)
        # visual_features_raw = self.globalmaxpooling(visual_features)

        return visual_features_raw


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        #for 3_D batch, need a loop!!!


        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN_En(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class GCN_En2(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN_En2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nembed)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        return x

class GCN_VisualFeature(nn.Module):
    def __init__(self, nembed, nhid, output_features, dropout):
        super(GCN_VisualFeature, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.globalmaxpooling = F.max_pool2d(nhid, output_features)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        graph_based_disease_feature = self.globalmaxpooling(x)

        return graph_based_disease_feature


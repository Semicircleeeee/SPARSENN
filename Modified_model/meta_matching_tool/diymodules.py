import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init

############### Sparse-nn function #################
def truncated_normal_(tensor,mean=0,std=0.09):
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size+(4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor
    
# class myLinear(nn.Module):
#     __constants__ = ['bias']
 
#     def __init__(self, , bias=True):
#         super(myLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
 
#     def reset_parameters(self):
#         self.weight = truncated_normal_(self.weight, mean = 0, std = 0.1)
#         if self.bias is not None:
#             fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             init.uniform_(self.bias, -bound, bound)
 
    
#     def forward(self, input):
#         return F.linear(input, self.weight, self.bias)
    
class SparseLinear(nn.Module):
    """
    Define our linear connection layer which enabled sparse connection
    """
    def __init__(self, m, residual_connection, bias=True):
        super(SparseLinear, self).__init__()
        self.in_features = m.shape[0]
        self.out_features = m.shape[1]
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.residual_connection = residual_connection
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        indices_mask = [np.where(m==1)[1].tolist(),np.where(m==1)[0].tolist()]
 
        # def backward_hook(grad):
        #     # Clone due to not being allowed to modify in-place gradients
        #     out = grad.clone()
        #     out[self.mask] = 0
        #     return out

        self.mask = torch.zeros([self.out_features, self.in_features]).bool()
        self.mask[indices_mask] = 1 # create mask

        # self.linear.weight.data[self.mask == 0] = 0 # zero out bad weights
        # self.linear.weight.register_hook(backward_hook) # hook to zero out bad gradients

    def reset_parameters(self):
        self.weight = truncated_normal_(self.weight, mean = 0, std = 0.1)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # print(self.mask.shape, self.weight.shape, input.shape)
        out = input @ (self.mask * self.weight).T + self.bias
        out = F.relu(out, inplace=False)
        # Residual connection
        out = input[:, self.residual_connection] + out
        return out
    
# Below is a naive idea of residual connection... Not sure whether it will work. TODO: check this
class lastResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, m):
        indices_mask = [np.where(m==1)[1].tolist(),np.where(m==1)[0].tolist()]
        
        super(lastResidualBlock, self).__init__()
        
    def forward(self, input):
        out = self.linear(input)
        out = out + input
        return
"""
Author   : Bao-lin Yin
Data     : 10.23 2023
Version  : V1.0
Function : Defining the different models used to train different dataset
"""
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, dim_input, dim_output):
        super(LinearRegression, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.network = nn.Sequential(nn.Linear(self.dim_input, 360),
                                     nn.Linear(360, self.dim_output))

    def forward(self, x):
        output = self.network(x)
        return output

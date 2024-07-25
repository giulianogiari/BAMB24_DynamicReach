"""
paper
https://github.com/BeNeuroLab/2022-preserved-dynamics/blob/main/rnn/simulation/networks.py
https://github.com/BeNeuroLab/2022-preserved-dynamics/blob/main/rnn/simulation/runner.py
"""

import torch
from torch import nn
    
class RNN(nn.Module):
    # https://github.com/bambschool/BAMB2023/blob/main/4-recurrent_neural_networks/ops.py#L71
    def __init__(self, input_size, hidden_size, output_size, nonlinearity='relu'):
        super(RNN, self).__init__()
        # get input
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        
        # build a recurrent neural network with a single recurrent layer and rectified linear units
        self.recurrent = nn.RNN(self.input_size, 
                              self.hidden_size, 
                              nonlinearity=nonlinearity)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # Apply the weight initialization
        self.apply(init_weights)


    def forward(self, x):
        # get the output of the network for a given input
        out, _ = self.recurrent(x)
        x = self.linear(out)
        return x, out
    

def init_weights(m):
    # Define the weight initialization function
    # set them to low values to avoid exploding gradients
    # https://machinelearningmastery.com/exploding-gradients-in-neural-networks/
    if isinstance(m, nn.RNN):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data) * 0.001
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data) * 0.001
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
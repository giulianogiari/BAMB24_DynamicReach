"""
Train the neural network
"""
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from rnn_model import RNN
from rnn_task import MyEnv, Dataset
from torch import nn

# check if GPU is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Set up config:
# network
net_kwargs = {'hidden_size': 64,
            'output_size': 2, 
            'input_size': 3,
            'nonlinearity': 'tanh', 
            'learning_rate': 0.01,
            'weight_decay': 0.01,
            'batch_size': 32,
            'n_epochs': 1_000,
            'n_hidden': 64, 
            } 
# enviroment
FS = 130
T_MAX = 2
N_TRLS_SEQ = 3
env_kwargs = {'dt': int(np.round(1/FS * 1_000, 1)),
              't_max': 2,
              'n_trls_seq': 3,
              'fs': FS,
              'seq_len': FS * T_MAX * N_TRLS_SEQ}

""" DEFINE THE ENVIRONMENT """
env = MyEnv(dt=env_kwargs['dt'])
dataset = Dataset(env, 
                  batch_size=net_kwargs['batch_size'], 
                  seq_len=env_kwargs['seq_len'])

""" SETUP THE NETWORK """
rnn = RNN(hidden_size=net_kwargs['hidden_size'], 
          output_size=net_kwargs['output_size'], 
          input_size=net_kwargs['input_size'])
# Move network to the device (CPU or GPU)
rnn = rnn.to(device)
# Define loss: we use mse which corresponds to euclidean distance between the predicted and the true action
criterion = nn.MSELoss()
# Define optimizer
# weight decay here is a regularization term that penalizes large weights
optimizer = torch.optim.Adam(rnn.parameters(), 
                             lr=net_kwargs['learning_rate'], 
                             weight_decay=net_kwargs['weight_decay'])

# Save config
kwargs = {}
kwargs['env_kwargs'] = env_kwargs
kwargs['net_kwargs'] = net_kwargs
with open('rnn_config.json', 'w') as f:
    json.dump(kwargs, f)

# We'll keep track of the loss as we train.
# It is initialized to zero and then monitored over training interations
running_loss = 0.0
when = np.round(np.linspace(0, net_kwargs['n_epochs'], 10), 0)
for i in range(net_kwargs['n_epochs']):

    # get inputs and labels and pass them to the GPU
    inputs, labels = dataset()
    inputs = torch.from_numpy(inputs).type(torch.float).to(device)
    labels = torch.from_numpy(labels.flatten()).type(torch.long).to(device)
    # print shapes of inputs and labels
    if i == 0:
        print('inputs shape: ', inputs.shape)
        print('labels shape: ', labels.shape)
        print('Max labels: ', labels.max())
    # we need zero the parameter gradients to re-initialize and avoid they accumulate across epochs
    optimizer.zero_grad()

    # FORWARD PASS: get the output of the network for a given input
    outputs, _ = rnn(inputs)

    #reshape outputs so they have the same shape as labels
    outputs = outputs.view(-1, env.action_space.shape[0])

    # compute loss with respect to the labels
    labels = labels.view(-1, env.action_space.shape[0])
    loss = criterion(outputs, labels.float())

    # compute gradients
    loss.backward()
    
    # update weights
    optimizer.step()

    # print average loss over last 200 training iterations and save the current network
    running_loss += loss.item()
    if any(when == i):
        print('{:d} loss: {:0.5f}'.format(i + 1, running_loss / 200))
        running_loss = 0.0
print('Finished Training')
torch.save(rnn.state_dict(), 'rnn_net.pth')


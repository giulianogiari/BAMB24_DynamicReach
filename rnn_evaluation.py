"""

"""
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from rnn_task import MyEnv
from rnn_model import RNN

# Load the configuration file
with open('rnn_config.json', 'r') as f:
    config = json.load(f)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Since we will not train the network anymore, we can turn off the gradient computation. The most commun way to do this is to use the context manager torch.no_grad() as follows:
with torch.no_grad():
    rnn = RNN(input_size=config['net_kwargs']['input_size'],
              hidden_size=config['net_kwargs']['hidden_size'],
              output_size=config['net_kwargs']['output_size'])

    net = rnn.to(device) # pass to GPU for running forwards steps

    # load the trained network's weights from the saved file
    rnn.load_state_dict(torch.load('rnn_net.pth', 
                                   map_location=torch.device(device)))

# define the environment
env = MyEnv(dt=config['env_kwargs']['dt'], jump_percent=1)
# create new trial
info = env.new_trial()
# read out the inputs in that trial
inputs = torch.from_numpy(env.ob[:, None, :]).type(torch.float)
labels = torch.from_numpy(env.gt[:, None, :]).type(torch.long)
# as before you can print the shapes of the variables to understand what they are and how to use them
# do this for the rest of the variables as you build the code
# 
predictions, _ = rnn(inputs)
predictions = predictions.detach().numpy()

fig, ax = plt.subplots(2,1)
ax[0].plot(labels[..., 0].flatten(), label='true x')
ax[0].plot(predictions[..., 0], '.-', label='x_hat')
ax[0].legend()
ax[1].plot(labels[..., 1].flatten(), label='true y')
ax[1].plot(predictions[..., 1],  '.-', label='y_hat')
ax[1].legend()
#plt.setp(ax, xlim=(200, 300))

# in time domain, the network oscillates at the time of the transition
# this possibly reflects the Gibbs phenomenon
# https://en.wikipedia.org/wiki/Gibbs_phenomenon

markersize = 5

fig, ax = plt.subplots(1,1, figsize=(7,6))
# plot initial position
ax.plot(0,0,'o', color='k',markersize=markersize)
# add possible targets
ax.plot(env.possible_target_locations[:, 0], 
        env.possible_target_locations[:, 1], 'o', 
        color='k', markersize=markersize)
# add target location
ax.plot(info['ground_truth'][0], 
        info['ground_truth'][1], 'o', 
        color='r', markersize=markersize)
if info['is_jump']:
        ax.plot(info['first_target'][0], 
                info['first_target'][1], 'o', 
                color='g', markersize=markersize)

# plot the trajectory
x = predictions.squeeze()[:, 0]
y = predictions.squeeze()[:, 1]
c = np.arange(len(x))
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Normalize the colors based on the variable c
lc = LineCollection(segments, 
                    linewidths=2,
                    cmap='viridis')
lc.set_array(c)
ax.add_collection(lc)

# Add a colorbar
cb = plt.colorbar(lc, ax=ax)
cb.set_label('Time (Samples)')

plt.setp(ax, xlim=(-100,100), ylim=(-100,100), 
         xticks=[], yticks=[])
  
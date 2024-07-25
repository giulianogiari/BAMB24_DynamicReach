"""
Evaluate the trained network on a new trial and visualize the results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection
from rnn_task import MyEnv, plot_simulation, Dataset
from rnn_model import RNN
import pandas as pd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
jump_percent_train = 0
jump_percent_test = 1

# in time domain, the network oscillates at the time of the transition
# this possibly reflects the Gibbs phenomenon
# https://en.wikipedia.org/wiki/Gibbs_phenomenon


""" Compare mse across training procedures """
jump_percent_test = 1
# define the environment
env = MyEnv(dt=7, jump_percent=jump_percent_test)
inputs_list, labels_list, info_list = [], [], []
for _ in range(1000):
        info = env.new_trial()
        info_list.append([info['jump_time'], info['jump_angle'], 
                          info['ground_truth'], info['first_target']])
        inputs_list.append(torch.from_numpy(env.ob[:, None, :]).type(torch.float))
        labels_list.append(torch.from_numpy(env.gt[:, None, :]).type(torch.long))

mse, jump_percent, jump_time, jump_angle = [], [], [], []
for jump_percent_train  in [0, .3, .5, .8]:
        # Load the configuration file
        with open(f'./trained_models/rnn_config_jump{jump_percent_train}.json', 'r') as f:
                config = json.load(f)
        # intialize the network
        rnn = RNN(input_size=config['net_kwargs']['input_size'],
                hidden_size=config['net_kwargs']['hidden_size'],
                output_size=config['net_kwargs']['output_size'])
        net = rnn.to(device) # pass to GPU for running forwards steps
        # load the trained network's weights from the saved file
        rnn.load_state_dict(torch.load(f'./trained_models/rnn_net_jump{jump_percent_train}.pth', 
                                        map_location=torch.device(device)))
        # Switch model to evaluation mode
        rnn.eval()
        # Disable gradient computation for inference
        torch.set_grad_enabled(False)

        trajectories = []
        for inputs, labels, info in zip(inputs_list, labels_list, info_list):
                predictions, _ = rnn(inputs)
                # store the full trajectory
                trajectories.append(predictions.detach().numpy()[:, 0, :])
                # compute mse
                # restrict values to re-preparation time period, i.e., after the jump
                jump_start = info[0]
                mse.append(np.mean((labels[jump_start:, ...] - predictions[jump_start:, ...]).numpy()**2))
                jump_percent.append(jump_percent_train)
                jump_time.append(info[0])
                jump_angle.append(info[1])
        
        """ plot predictions in time domain """
        fig, ax = plt.subplots(2,1)
        ax[0].plot(labels[..., 0].flatten(), label='true')
        ax[0].plot(predictions[..., 0], '.-', label='predicted')
        ax[0].legend()
        ax[0].set_title('X position')
        ax[0].set_ylabel('position')
        ax[1].plot(labels[..., 1].flatten(), label='true')
        ax[1].plot(predictions[..., 1],  '.-', label='predicted')
        ax[1].legend()
        ax[1].set_xlabel('Time (Samples)')
        ax[1].set_title('Y position')
        ax[1].set_ylabel('position')
        fig.tight_layout()
        fig.savefig(f'./figures/predictions_jump{jump_percent_train}.png', dpi=200)

        """ plot the trajectory in 2d """
        markersize = 30
        fig, ax = plt.subplots(1,1, figsize=(7,6))
        # plot initial position
        ax.scatter(0,0, color='white',s=markersize, edgecolor='k')
        # add possible targets
        ax.scatter(env.possible_target_locations[:, 0], 
                env.possible_target_locations[:, 1],
                color='k', s=markersize)
        # add target location
        ax.scatter(info[2][0], 
                info[2][1], label='target',
                color='r', s=markersize+10, edgecolor='k')
        if all(info[2]!=info[3]):
                ax.scatter(info[3][0], 
                        info[3][1], label='target before jump',
                        color='g', s=markersize+10, edgecolor='k')
        # plot the trajectory
        x = predictions.squeeze()[:, 0]
        y = predictions.squeeze()[:, 1]
        c = np.arange(len(x))
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # Normalize the colors based on the variable c
        lc = LineCollection(segments, 
                        linewidths=3, alpha=.9, 
                        cmap='viridis')
        lc.set_array(c)
        ax.add_collection(lc)
        # Add a colorbar
        cb = plt.colorbar(lc, ax=ax)
        cb.set_label('Time (Samples)')
        ax.set_title('RNN Trajectory')
        ax.legend()
        fig.tight_layout()
        plt.setp(ax, xlim=(-100,100), ylim=(-100,100), 
                xticks=[], yticks=[]);
        fig.savefig(f'./figures/trajectory2d_jump{jump_percent_train}.png', dpi=200)

        """ plot the trajectories in 2d """
        fig, ax = plt.subplots(1,1, figsize=(7,6))
        # plot initial position
        ax.scatter(0,0, color='white',s=markersize, edgecolor='k')
        # add possible targets
        ax.scatter(env.possible_target_locations[:, 0], 
                env.possible_target_locations[:, 1],
                color='k', s=markersize)
        # plot the trajectory
        for t in trajectories:
                ax.plot(t[:, 0], t[:, 1], alpha=0.5, color='g')
        ax.set_title('RNN Trajectory')
        ax.legend()
        fig.tight_layout()
        plt.setp(ax, xlim=(-100,100), ylim=(-100,100), 
                xticks=[], yticks=[]);
        fig.savefig(f'./figures/trajectories_jump{jump_percent_train}.png', dpi=200)

# create the dataframe
df = pd.DataFrame({'jump_percent': jump_percent, 'mse': mse, 'jump_time': jump_time, 'jump_angle': jump_angle}) 

""" plot the mse as a function of jump percent during training """
fig, ax = plt.subplots(1,1)
m = df.groupby('jump_percent').mean().reset_index()
sd = df.groupby('jump_percent').std().reset_index()
ax.plot(m['jump_percent'], m['mse'], '.-')
ax.fill_between(m['jump_percent'], m['mse']-sd['mse'], m['mse']+sd['mse'], alpha=0.5)
ax.set_xlabel('Jump Percent during Training')
ax.set_ylabel('MSE')
ax.set_xticks([0, .3, .5, .8])
fig.tight_layout()
fig.savefig(f'./figures/mse_jump_percent.png', dpi=200)

""" plot the mse as a function of jump percent and jump time """
fig, ax = plt.subplots(1,1, figsize=(10,5))
m = df.groupby(['jump_percent', 'jump_time']).mean().reset_index()
sd = df.groupby(['jump_percent', 'jump_time']).std().reset_index()
for jump_percent in [0, .3, .5, .8]:
        ax.plot(m[m['jump_percent']==jump_percent]['jump_time'].values-env.start_ind['reach'], 
                m[m['jump_percent']==jump_percent]['mse'], '.-', label=f'{jump_percent}')
        ax.fill_between(m[m['jump_percent']==jump_percent]['jump_time'].values-env.start_ind['reach'], 
                        m[m['jump_percent']==jump_percent]['mse']-sd[m['jump_percent']==jump_percent]['mse'], 
                        m[m['jump_percent']==jump_percent]['mse']+sd[m['jump_percent']==jump_percent]['mse'], alpha=0.3)
ax.set_xlabel('Jump time before Reach')
ax.set_ylabel('MSE')
ax.legend(title='Jump Percent during Training')
fig.tight_layout()
fig.savefig(f'./figures/mse_jump_time.png', dpi=200)

""" plot the mse as a function of jump angle """
fig, ax = plt.subplots(1,1, figsize=(10,5))
m = df.groupby(['jump_percent', 'jump_angle']).mean().reset_index()
sd = df.groupby(['jump_percent', 'jump_angle']).std().reset_index()
for jump_percent in [0, .3, .5, .8]:
        ax.plot(m[m['jump_percent']==jump_percent]['jump_angle'], 
                m[m['jump_percent']==jump_percent]['mse'], '.-', label=f'{jump_percent}')
        ax.fill_between(m[m['jump_percent']==jump_percent]['jump_angle'], 
                        m[m['jump_percent']==jump_percent]['mse']-sd[m['jump_percent']==jump_percent]['mse'], 
                        m[m['jump_percent']==jump_percent]['mse']+sd[m['jump_percent']==jump_percent]['mse'], alpha=0.3)
ax.set_xlabel('Jump angle')
ax.set_ylabel('MSE')
ax.legend()
ax.legend(title='Jump Percent during Training')
fig.tight_layout()
fig.savefig(f'./figures/mse_jump_angle.png', dpi=200)

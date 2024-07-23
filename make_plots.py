## BASIC PLOT OF TASK ENVIRONMENT ##
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

def simulate_data(env, agent, N, target=None, postjump_target=None):
  traj_list = []
  belief_list = []
  target_list = []
  for _ in range(N):
    agent.reset_belief()
    state, terminated = env.reset(set_target=target, set_postjump_target=postjump_target)
    trajectory = {'x': [], 'y': []}
    belief = []
    tg = []
    while not terminated:
      action = agent.act(state)
      next_state, reward, terminated = env.step(action)
      belief.append(agent.p)
      tg.append(next_state['target'])
      if next_state['time'] > 0:
        x, y = next_state['position']
        trajectory['x'].append(x)
        trajectory['y'].append(y)
      state = next_state
    traj_list.append(trajectory)
    belief_list.append(belief)
    target_list.append(tg)

  return traj_list, belief_list, target_list

def plot_traj(env, agent, N, target=None, postjump_target=None, smooth=True):
    traj_list, _, _ = simulate_data(env, agent, N, target, postjump_target)
    # set up figure
    plt.figure(figsize=(3,3), dpi=200)
    plt.plot(0,0,'o', color='black',markersize=2)
    plt.xlim(-100,100)
    plt.ylim(-100,100)
    plt.xticks([])
    plt.yticks([])
  
    # add possible targets
    for i in range(8):
      if i==target:
        color='red'
      elif i==postjump_target:
        color='blue'
      else:
        color='gray'
      x = 80*np.sin([0*np.pi/180, 45*np.pi/180, 90*np.pi/180, 135*np.pi/180, 180*np.pi/180, 225*np.pi/180, 270*np.pi/180, 315*np.pi/180][i])
      y = 80*np.cos([0*np.pi/180, 45*np.pi/180, 90*np.pi/180, 135*np.pi/180, 180*np.pi/180, 225*np.pi/180, 270*np.pi/180, 315*np.pi/180][i])
      plt.plot(x, y, 'o', label=str(i), color=color)
  
    # add trajectory
    for trajectory in traj_list:
      plt.plot(trajectory['x'], trajectory['y'], linewidth=0.5)
    

## TASK: DYNAMIC REACH ENVIRONMENT ##

import numpy as np
from gym import utils
from gym import Env
from gym.spaces import Dict,Box,Discrete

class DynamicReach(Env):
  def __init__(self, allow_jump=False):
    # observation space: infinite box, 8 possible targets, 390 timesteps
    self.observation_space = Dict({'position': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32), 'target': Discrete(8), 'time': Discrete(390)})

    # action space: reach angle [-pi, pi], 2 possible velocities (stand still or constant velocity of 1 mm/ms)
    self.action_space = Dict({'angle': Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32), 'velocity': Discrete(2)})

    # whether to allow target jumps
    self.allow_jump = allow_jump

    # add some info about possible target locations
    self.locationX = [80*np.sin(0*np.pi/180), 80*np.sin(45*np.pi/180), 80*np.sin(90*np.pi/180), 80*np.sin(135*np.pi/180),
                      80*np.sin(180*np.pi/180), 80*np.sin(225*np.pi/180), 80*np.sin(270*np.pi/180), 80*np.sin(315*np.pi/180)]
    self.locationY = [80*np.cos(0*np.pi/180), 80*np.cos(45*np.pi/180), 80*np.cos(90*np.pi/180), 80*np.cos(135*np.pi/180),
                      80*np.cos(180*np.pi/180), 80*np.cos(225*np.pi/180), 80*np.cos(270*np.pi/180), 80*np.cos(315*np.pi/180)]

  def reset(self, set_target=None):
    # determine target
    if set_target:
      self.target = set_target
    else:
      self.target = self.observation_space['target'].sample()

    # determine location
    self.target_location = (self.locationX[self.target], self.locationY[self.target])

    # check if target jumps, if so update location
    if self.allow_jump:
      if np.random.rand() < 0.30:
        self.do_jump = True
        self.jump_time = 260 - int(np.random.uniform(150,550)/1000*130)
        tg_indx = [0,1,2,3,4,5,6,7].index(self.target)
        self.postjump_target = [0,1,2,3,4,5,6,7][tg_indx + np.random.choice([-3,-1,1,3])] # -130, -45, 45, or 130 degrees jump
        self.postjump_target_location = (self.locationX[self.postjump_target], self.locationY[self.postjump_target])
      else:
        self.do_jump = False
        self.jump_time = 390
        self.postjump_target = self.target
        self.postjump_target_location = self.target_location
    else:
      self.do_jump = False
      self.jump_time = 390
      self.postjump_target = self.target
      self.postjump_target_location = self.target_location

    # assemble state: start from coordinates x=0, y=0, measured in mm
    self.state = {'position': (0,0), 'target': self.target, 'time': 0}
    terminated = False

    return self.state, terminated

  def step(self, action):
    terminated = False

    # perform movement
    x,y = self.state['position']
    if action['velocity']==1:
      x = x + (1000/130)*np.cos(action['angle'])
      y = y + (1000/130)*np.sin(action['angle'])

    # check if target is reached
    tloc_x, tloc_y = [self.target_location, self.postjump_target_location][int(self.allow_jump)*int(self.do_jump)]
    if np.sqrt((x-tloc_x)**2 + (y-tloc_y)**2) < 12.5:
      reward = 1
      terminated = True
    else:
      reward = 0

    # let time pass
    if self.state['time']+1 >= 390:
      time = self.state['time']
      terminated = True
    else:
      time = self.state['time']+1

    # assemble next state
    target = [self.target, self.postjump_target][int(self.allow_jump)*int(self.do_jump)*int(time>=self.jump_time)]
    next_state = {'position': (x,y), 'target': target, 'time': time}
    self.state = next_state

    return next_state, reward, terminated

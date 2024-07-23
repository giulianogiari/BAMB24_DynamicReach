## TASK: DYNAMIC REACH ENVIRONMENT ##
import numpy as np
from gym import utils
from gym import Env
from gym.spaces import Dict,Box,Discrete

class DynamicReach(Env):
  def __init__(self, allow_jump=False):
    # observation space: infinite box, 8 possible targets, 390 timesteps
    self.observation_space = Dict({'position': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32), 'target': Discrete(8), 'time': Box(low=-1500, high=1000, shape=(1,), dtype=np.float32)})

    # action space: reach angle [-pi, pi], 4 possible velocities (stand still or velocity of 1 mm/ms, 2 mm/ms, 3 mm/ms)
    self.action_space = Dict({'angle': Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32), 'velocity': Discrete(4)})

    # whether to allow target jumps
    self.allow_jump = allow_jump

    # add some info about possible target locations
    self.locationX = [80*np.sin(0*np.pi/180), 80*np.sin(45*np.pi/180), 80*np.sin(90*np.pi/180), 80*np.sin(135*np.pi/180),
                      80*np.sin(180*np.pi/180), 80*np.sin(225*np.pi/180), 80*np.sin(270*np.pi/180), 80*np.sin(315*np.pi/180)]
    self.locationY = [80*np.cos(0*np.pi/180), 80*np.cos(45*np.pi/180), 80*np.cos(90*np.pi/180), 80*np.cos(135*np.pi/180),
                      80*np.cos(180*np.pi/180), 80*np.cos(225*np.pi/180), 80*np.cos(270*np.pi/180), 80*np.cos(315*np.pi/180)]

    # add parameters for distance & moving cost penalties
    self.dist_cost = 0.5
    self.move_cost = 0.001

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
        self.jump_time = 0 - int(np.random.uniform(150,550))
        tg_indx = [0,1,2,3,4,5,6,7].index(self.target)
        self.postjump_target = [0,1,2,3,4,5,6,7][tg_indx + np.random.choice([-3,-1,1,3])] # -130, -45, 45, or 130 degrees jump
        self.postjump_target_location = (self.locationX[self.postjump_target], self.locationY[self.postjump_target])
      else:
        self.do_jump = False
        self.jump_time = 1000
        self.postjump_target = self.target
        self.postjump_target_location = self.target_location
    else:
      self.do_jump = False
      self.jump_time = 1000
      self.postjump_target = self.target
      self.postjump_target_location = self.target_location

    # assemble state: start from coordinates x=0, y=0, measured in mm
    self.state = {'position': (0,0), 'target': self.target, 'time': -2000}
    terminated = False

    return self.state, terminated

  def step(self, action, belief=None):
    terminated = False

    # perform movement at chosen velocity
    x,y = self.state['position']
    x = x + action['velocity'] * np.cos(action['angle'])
    y = y + action['velocity'] * np.sin(action['angle'])

    # calculate cost of movement
    if belief:
      Jx1 = belief * np.sqrt((x-self.target_location[0])**2 + (y-self.target_location[1])**2)
      Jx2 = (1 - belief) * np.sqrt((x-self.postjump_target_location[0])**2 + (y-self.postjump_target_location[1])**2)
      Jx = self.dist_cost * Jx1 + self.dist_cost * Jx2
    else:
      Jx = self.dist_cost * np.sqrt((x-self.target_location[0])**2 + (y-self.target_location[1])**2)
    Ju = (self.move_cost * (self.state['time']/1000)) * action['velocity'] # move cost increases with time & velocity
    cost = Jx + Ju

    # check if target is reached
    tloc_x, tloc_y = [self.target_location, self.postjump_target_location][int(self.allow_jump)*int(self.do_jump)]
    if np.sqrt((x-tloc_x)**2 + (y-tloc_y)**2) < 12.5:
      terminated = True

    # let time pass
    if self.state['time']+1 >= 1000:
      time = self.state['time']
      terminated = True
    else:
      time = self.state['time']+1

    # assemble next state
    target = [self.target, self.postjump_target][int(self.allow_jump)*int(self.do_jump)*int(time>=self.jump_time)]
    next_state = {'position': (x,y), 'target': target, 'time': time}
    self.state = next_state

    return next_state, cost, terminated
    
  def simulate_step(self, action, belief=None):
    terminated = False

    # perform movement at chosen velocity
    x,y = self.state['position']
    x = x + action['velocity'] * np.cos(action['angle'])
    y = y + action['velocity'] * np.sin(action['angle'])

    # calculate cost of movement
    if belief:
      Jx1 = belief * np.sqrt((x-self.target_location[0])**2 + (y-self.target_location[1])**2)
      Jx2 = (1 - belief) * np.sqrt((x-self.postjump_target_location[0])**2 + (y-self.postjump_target_location[1])**2)
      Jx = self.dist_cost * Jx1 + self.dist_cost * Jx2
    else:
      Jx = self.dist_cost * np.sqrt((x-self.target_location[0])**2 + (y-self.target_location[1])**2)
    Ju = (self.move_cost * (self.state['time']/1000)) * action['velocity'] # move cost increases with time & velocity
    cost = Jx + Ju

    # check if target is reached
    tloc_x, tloc_y = [self.target_location, self.postjump_target_location][int(self.allow_jump)*int(self.do_jump)]
    if np.sqrt((x-tloc_x)**2 + (y-tloc_y)**2) < 12.5:
      terminated = True

    # let time pass
    if self.state['time']+1 >= 4000:
      time = self.state['time']
      terminated = True
    else:
      time = self.state['time']+1

    # assemble next state
    target = [self.target, self.postjump_target][int(self.allow_jump)*int(self.do_jump)*int(time>=self.jump_time)]
    next_state = {'position': (x,y), 'target': target, 'time': time}

    return next_state, cost, terminated


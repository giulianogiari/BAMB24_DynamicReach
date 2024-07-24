## TASK: DYNAMIC REACH ENVIRONMENT ##
import numpy as np
from gym import utils
from gymnasium import Env
from gymnasium.spaces import Box

class DynamicReachPPO(Env):
  def __init__(self, do_jump=False, dist_cost=0.01, move_cost=0.01):
    super(DynamicReachPPO, self).__init__()
    # observation space: infinite box, 8 possible targets, 390 timesteps
    self.observation_space = Box(low=np.array([-200, -200, 0, -10]), high=np.array([200, 200, 8, 1000]), dtype=np.float32)
    #self.observation_space = Dict({'position': Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32), 'target': Discrete(8), 'time': Box(low=-1500, high=1000, shape=(1,), dtype=np.float32)})

    # action space: reach angle [-1, 1] (normalized), velocity [0, 1]
    self.action_space = Box(low=np.array([-1.0, 0]), high=np.array([1,1]), dtype=np.float32)

    # whether to do target jump
    self.do_jump = do_jump
    self.jumped = 0

    # add some info about possible target locations
    self.locationX = [80*np.sin(0*np.pi/180), 80*np.sin(45*np.pi/180), 80*np.sin(90*np.pi/180), 80*np.sin(135*np.pi/180),
                      80*np.sin(180*np.pi/180), 80*np.sin(225*np.pi/180), 80*np.sin(270*np.pi/180), 80*np.sin(315*np.pi/180)]
    self.locationY = [80*np.cos(0*np.pi/180), 80*np.cos(45*np.pi/180), 80*np.cos(90*np.pi/180), 80*np.cos(135*np.pi/180),
                      80*np.cos(180*np.pi/180), 80*np.cos(225*np.pi/180), 80*np.cos(270*np.pi/180), 80*np.cos(315*np.pi/180)]

    # add parameters for distance & moving cost penalties
    self.dist_cost = dist_cost
    self.move_cost = move_cost

  def reset(self, seed=0, set_target=None, set_postjump_target=None):
    # determine target
    '''if set_target:
      self.target = set_target
    else:
      self.target = np.random.choice([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0])'''
    self.target=1
    # determine location
    self.target_location = (self.locationX[int(self.target)], self.locationY[int(self.target)])

    # determine if target jumps & update location
    if not self.do_jump:
      self.postjump_target = self.target
      self.postjump_target_location = self.target_location
      self.jump_time = 1000
    else:
      if set_postjump_target:
        self.postjump_target = set_postjump_target
      else:
        self.postjump_target = [0,1,2,3,4,5,6,7][int(self.target) + np.random.choice([-3,-1,1,3])] # -130, -45, 45, or 130 degrees jump
      
      # determine time & location
      self.jump_time = 0 - int(np.random.uniform(150,550))
      self.postjump_target_location = (self.locationX[int(self.postjump_target)], self.locationY[int(self.postjump_target)])

    # assemble state: start from coordinates x=0, y=0, measured in mm
    self.state = np.array([0.0, 0.0, self.target, -10.0])
    #self.state = {'position': np.array([0,0]), 'target': self.target, 'time': -2000}
    info = {}

    return self.state, info

  def step(self, action, belief=None):
    terminated = False
    angle = action[0]
    velocity = action[1]

    # perform movement at chosen velocity (only if time>=0)
    x,y = self.state[:2]
    
    if self.state[3]>=0:
      x = x + velocity * np.cos(np.pi * angle)
      y = y + velocity * np.sin(np.pi * angle)

    # calculate cost of movement
    if self.state[3]>0:
      if belief:
        Jx1 = belief * np.sqrt((x-self.target_location[0])**2 + (y-self.target_location[1])**2)
        Jx2 = (1 - belief) * np.sqrt((x-self.postjump_target_location[0])**2 + (y-self.postjump_target_location[1])**2)
        Jx = self.dist_cost * Jx1 + self.dist_cost * Jx2
      else:
        Jx = self.dist_cost * np.sqrt((x-self.target_location[0])**2 + (y-self.target_location[1])**2)
      Ju = (self.move_cost * (self.state[3]/1000)) * velocity # move cost increases with time & velocity
      cost = Jx + Ju
      reward = 1-cost
    else:
      reward = 0

    # check if target is reached & determine reward
    tloc_x, tloc_y = [self.target_location, self.postjump_target_location][self.jumped]
    if np.sqrt((x-tloc_x)**2 + (y-tloc_y)**2) < 2:
      terminated = True
      reward += 10

    # let time pass
    if self.state[3]+1 >= 1000:
      time = self.state[3]
      terminated = True
    else:
      time = self.state[3]+1
      if time >= self.jump_time:
        self.jumped = 1

    # assemble next state
    target = [self.target, self.postjump_target][self.jumped]
    next_state = np.array([x, y, target, time])
    # next_state = {'position': np.array([x,y]), 'target': target, 'time': time}
    self.state = next_state

    return next_state, reward, terminated, False, {}

  def close(self):
    pass
    

## AGENT: BASIC STRUCTURE ##
import numpy as np

# RANDOM #
class RandomAgent:
  def __init__(self, action_space, observation_space):
    self.action_space = action_space
    self.observation_space = observation_space

  def act(self, state):
    # determine velocity based on time
    if state['time']<260:
      velocity = 0
    else:
      velocity = 1

    # pick random angle
    angle = self.action_space['angle'].sample()[0]

    # assemble and return action
    action = {'angle': angle, 'velocity': velocity}
    return action

# OPTIMAL WITH MOTOR NOISE #
class OptimalNoisyAgent:
  def __init__(self, env, motor_noise=1):
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.locationX = env.locationX
    self.locationY = env.locationY
    # implement parameters
    self.motor_noise = motor_noise

  def act(self, state):
    # determine velocity based on time
    if state['time']<260:
      velocity = 0
    else:
      velocity = 1

    # angle that minimizes distance to target + small amount of gaussian noise
    tloc_x, tloc_y = (self.locationX[state['target']], self.locationY[state['target']])
    x,y = state['position']
    possible_actions = np.linspace(-np.pi, np.pi, 360)
    dist = [np.sqrt(((x+1000/130*np.cos(ori))-tloc_x)**2 + ((y+1000/130*np.sin(ori))-tloc_y)**2) for ori in possible_actions]
    angle = possible_actions[dist.index(min(dist))] + np.random.normal(0, self.motor_noise)

    # assemble and return action
    action = {'angle': angle, 'velocity': velocity}
    return action

# OPTIMAL WITH MOTOR NOISE, SMOOTHER TRAJECTORY #
class OptimalSmoothNoisyAgent:
  def __init__(self, env, motor_noise=1):
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.locationX = env.locationX
    self.locationY = env.locationY
    # implement parameters
    self.motor_noise = motor_noise
    # initialize angle
    self.angle = 0

  def act(self, state):
    # determine velocity based on time
    if state['time']<260:
      velocity = 0
    else:
      velocity = 1

    # angle that minimizes distance to target + small amount of gaussian noise
    tloc_x, tloc_y = (self.locationX[state['target']], self.locationY[state['target']])
    x,y = state['position']
    possible_actions = np.linspace(self.angle-np.pi/4, self.angle+np.pi/4, 360)
    dist = [np.sqrt(((x+1000/130*np.cos(ori))-tloc_x)**2 + ((y+1000/130*np.sin(ori))-tloc_y)**2) for ori in possible_actions]
    angle = possible_actions[dist.index(min(dist))] + np.random.normal(0, self.motor_noise)
    self.angle = angle

    # assemble and return action
    action = {'angle': angle, 'velocity': velocity}
    return action



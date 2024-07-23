## AGENT: BASIC STRUCTURE ##
import numpy as np

# RANDOM #
class RandomAgent:
  def __init__(self, action_space, observation_space):
    self.action_space = action_space
    self.observation_space = observation_space

  def act(self, state):
    # determine velocity based on time
    if state['time']<0:
      velocity = 0
    else:
      velocity = self.action_space['velocity'].sample()[0]

    # pick random angle
    angle = self.action_space['angle'].sample()[0]

    # assemble and return action
    action = {'angle': angle, 'velocity': velocity}
    return action

# OPTIMAL WITH MOTOR NOISE #
class OptimalAgent:
  def __init__(self, env, motor_noise=1):
    self.env = env
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.locationX = env.locationX
    self.locationY = env.locationY
    # implement parameters
    self.motor_noise = motor_noise

  def act(self, state):
    # determine velocity based on time
    if state['time']<0:
      angle = 0
      velocity = 0
      action = {'angle': angle, 'velocity': velocity}
      return action
    
    else:
      # choose angle and velocity to minimize cost
      possible_angles = np.linspace(-np.pi, np.pi, 360)
      possible_velocities = [0,1,2,3]
      min_cost, best_action = 999999, None
      for angle in possible_angles:
        for velocity in possible_velocities:
          action = {'angle': angle, 'velocity': velocity}
          _, cost, _ = self.env.simulate_step(state, action)
          if cost < min_cost:
            min_cost = cost
            best_action = action
      
      # apply some motor noise and return
      best_action['angle'] = best_action['angle'] + np.random.normal(0, self.motor_noise)

      return best_action

# OPTIMAL WITH BELIEF UPDATING # NOT FINISHED
class OptimalBeliefAgent:
  def __init__(self, env, lr=0.02, belief_noise=0.01, motor_noise=1):
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.locationX = env.locationX
    self.locationY = env.locationY
    # implement parameters
    self.lr = lr
    self.belief_noise = belief_noise
    self.motor_noise = motor_noise
    # initialize belief about target
    self.p = 0.5

  def act(self, state):
    self.target = env.target
    self.jump_time = env.jump_time
    self.postjump_target = env.postjump_target
    # start with belief updating step
    if self.target!=self.postjump_target and state['time']>self.jump_time:
        self.p -= self.lr+np.random.normal(0, self.belief_noise)
    else:
        self.p += self.lr+np.random.normal(0, self.belief_noise)
        
    # determine velocity based on time
    if state['time']<0:
      velocity = 0
    else:
      velocity = 1
    
    # use belief to figure out the goal target location
    if np.random.rand() < self.p:
        tloc_x, tloc_y = (self.locationX[self.target], self.locationY[self.target])
    else:
        tloc_x, tloc_y = (self.locationX[self.postjump_target], self.locationY[self.postjump_target])
    
    # angle that minimizes distance to believed target + small amount of gaussian noise
    x,y = state['position']
    possible_actions = np.linspace(-np.pi, np.pi, 360)
    dist = [np.sqrt(((x+1000/130*np.cos(ori))-tloc_x)**2 + ((y+1000/130*np.sin(ori))-tloc_y)**2) for ori in possible_actions]
    angle = possible_actions[dist.index(min(dist))] + np.random.normal(0, self.motor_noise)

    # assemble and return action
    action = {'angle': angle, 'velocity': velocity}
    return action

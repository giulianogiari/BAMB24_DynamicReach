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
      velocity = self.action_space['velocity'].sample()

    # pick random angle
    angle = self.action_space['angle'].sample()[0]

    # assemble and return action
    action = {'angle': angle, 'velocity': velocity}
    return action


# OPTIMAL WITH MOTOR NOISE #
class OptimalAgent:
  def __init__(self, env, motor_noise=0.5):
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
  def __init__(self, env, lr=0.001, belief_noise=0.01, motor_noise=0.5):
    self.env = env
    self.action_space = env.action_space
    self.observation_space = env.observation_space
    self.locationX = env.locationX
    self.locationY = env.locationY
    # initialize target and jump target
    self.target = None
    self.postjump_target = None
    # implement parameters
    self.lr = lr
    self.belief_noise = belief_noise
    self.motor_noise = motor_noise
    # initialize belief about target
    self.p = 0.5
  
  def reset_belief(self):
    self.p = 0.5
    self.target = None
    self.postjump_target = None

  def update_belief(self, state):
    # if first time, update target
    if not self.target:
      self.target = state['target']
    # if target jumped, update postjump target
    if state['target']!= self.target:
      self.postjump_target = state['target']

    # do belief updating
    if state['target']==self.target:
      self.p = min(0.99999, self.p+self.lr+np.random.normal(0, self.belief_noise)) 
    elif state['target']==self.postjump_target:
      self.p = max(0.00001, self.p-self.lr+np.random.normal(0, self.belief_noise))
    
    return self.p

  def act(self, state):
    # first update belief
    p = self.update_belief(state)
        
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
          _, cost, _ = self.env.simulate_step(state, action, p)
          if cost < min_cost:
            min_cost = cost
            best_action = action
      
      # apply some motor noise and return
      best_action['angle'] = best_action['angle'] + np.random.normal(0, self.motor_noise)

      return best_action

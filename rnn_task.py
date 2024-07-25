"""
Environment for a reaching task
based on neurogym
https://neurogym.github.io/understanding_neurogym_task.html
"""

import matplotlib.pyplot as plt
import numpy as np
import random
# import gym and neurogym to create tasks
from neurogym import spaces
import neurogym as ngym
from neurogym.utils.data import Dataset


class MyEnv(ngym.TrialEnv):
    def __init__(self, dt=7, timing=None, jump_percent=.3):
        """
        dt: int, time step in ms, default 7ms corresponding to 130 Hz
        """
        # Python boilerplate to initialize base class
        super().__init__(dt=dt) 

        # define timing for the task
        self.jump_percent = jump_percent

        self.timing = {
            'fixation1': 750,
            'jump': 600,
            'fixation2': 150,
            'reach': 500}
        if timing:
            self.timing.update(timing)

        # possible target locations
        # this is now defined as a 2D array ranging from -80 to 80
        self.possible_target_locations = np.c_[
            80 * np.sin(np.arange(0, 360, 45) * np.pi / 180),
            80 * np.cos(np.arange(0, 360, 45) * np.pi / 180)]

        # A two-dimensional box with minimum and maximum value set by low and high
        # define the label of the observation space dimensions
        name = {'target_position_x': 0, 'target_position_y': 1, 'movement': 2} 
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), 
            name=name, dtype=np.float32)

        # A two-dimensional box with minimum and maximum value set by low and high
        # define the label of the action space dimensions
        name = {'position_x': 0, 'position_y': 1}
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,),
            name=name, dtype=np.float32)


    def _new_trial(self, **kwargs):
        """
        new_trial() is called when a trial ends to generate the next trial.
        The following variables are created:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and
            decision periods)
            ground truth: correct response for the trial
            obs: observation
        """
        # Periods
        self.add_period(list(self.timing.keys()))

        target_ind = self.rng.choice(np.arange(len(self.possible_target_locations)))
        target_pos = self.possible_target_locations[target_ind, :]
        if self.jump_percent > self.rng.rand():
            is_jump = True
            new_target = _choose_coordinate(self.possible_target_locations, 
                                            target_pos)
            # random time for the jump
            # between 150 and 550 ms before the reach period following the paper
            jump_time = self.rng.randint(self.start_ind['jump'],
                                        self.end_ind['jump'])
        else:
            is_jump = False
            new_target = target_pos
            jump_time = None

        # Trial info
        trial = {
            'ground_truth': new_target,
            'is_jump': is_jump,
            'first_target': target_pos,
            'second_target': new_target,
        }
        trial.update(kwargs)

        # Observations
        # period relates to timing in self.timing
        # where here relates to name in self.observation space

        # add observation for movement
        self.add_ob(0, period=['fixation1', 'jump', 'fixation2'], where='movement')
        self.add_ob(1, period='reach', where='movement')

        # add observation for target position
        self.add_ob(target_pos[0], 
                    period=['fixation1', 'jump', 'fixation2', 'reach'], 
                    where='target_position_x')
        self.add_ob(target_pos[1], 
                    period=['fixation1', 'jump', 'fixation2', 'reach'], 
                    where='target_position_y')
        if is_jump:
            # here it creates a "view" of the observation,
            # thus we can change the observation variable and it will be reflected in the environment
            stim = self.view_ob()
            stim[jump_time:, :2] = new_target
            #self.add_ob(target_pos[0], 
            #        period=['jump', 'fixation2', 'reach'], where='target_position_x')
            #self.add_ob(target_pos[1], 
            #        period=['jump', 'fixation2', 'reach'], where='target_position_y')
            
        # Ground truth
        # this is kind of a constanst velocity 
        self._set_groundtruth([0,0], 
                              period=['fixation1', 'jump', 'fixation2'])
        self._set_groundtruth(trial['ground_truth'], 
                             period='reach')
        return trial
    

    def _set_groundtruth(self, value, period=None, where=None):
        """
        Set groundtruth value.
        overwrites the default function 
        """
        if not self._gt_built:
            self._init_gt()

        if where is not None:
            value = self.observation_space.name[where]
        if isinstance(period, str):
            self.gt[self.start_ind[period]: self.end_ind[period], :] = \
                np.repeat(np.array(value)[None, :], self.end_ind[period] - self.start_ind[period], axis=0)
        elif period is None:
            self.gt[:] = value
        else:
            for p in period:
                self._set_groundtruth(value, p)
    

    def _step(self, action):
        """
        step is not needed for supervised learning tasks, but only for reinforcement learning tasks
        see here for more details: 
        https://neurogym.github.io/envs/Reaching1D-v0.html
        https://github.com/neurogym/ngym_usage/blob/master/training/auto_notebooks/rl/Reaching1D-v0.ipynb
        if not included in the environment it will raise an error
        """
        return None, None, None, {'trial': self.trial}
    

def _opposite_coordinate(coord, center=[0,0]):
    """ calculate the opposite coordinate of a given coordinate"""
    x_center, y_center = center
    x, y = coord
    x_opposite = 2 * x_center - x
    y_opposite = 2 * y_center - y
    return (x_opposite, y_opposite)

def _choose_coordinate(coords, current_coord):
    """ choose a coordinate from a list of coordinates that is not the opposite of the current coordinate"""
    opposite_coord = _opposite_coordinate(current_coord)
    # Filter out the opposite coordinate
    valid_coords = [coord for coord in coords if all(coord != opposite_coord)]
    # Randomly select a valid coordinate
    chosen_coord = random.choice(valid_coords)
    return chosen_coord


def plot_simulation(inputs, labels):
    """ Plot the simulated data """
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(inputs[:, 0, 0], '.-', label='x', color='r')
    ax[0].plot(inputs[:, 0, 1], '.-', label='y', color='b')
    # add different yaxis for ax[0] using twinx
    ax[0].legend()
    ax[0].set_ylabel('position')
    ax2 = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.plot(inputs[:, 0, 2], '.-', label='fixation', color='k')
    ax2.set_ylabel('movement')
    ax[0].set_title('Inputs')
    ax[1].plot(labels[:, 0, 0], '.-',label='x')
    ax[1].plot(labels[:, 0, 1], '.-',label='y')
    ax[1].set_ylabel('position')
    ax[1].legend()
    ax[1].set_title('Labels')
    ax[1].set_xlabel('Time (Samples)')
    plt.show()
    return fig, ax
    

if __name__ == '__main__':

    BATCH_SIZE = 16
    SEQ_LEN = 260
    DT = 7

    # create an instance of the environment
    env = MyEnv(dt=DT, jump_percent=.3)

    # from neurogym.utils.data import Dataset
    # this line of code in neurogym does not work, thus i had to copy the class here
    # https://github.com/neurogym/neurogym/blob/master/neurogym/utils/data.py#L30C26-L30C44

    dataset = Dataset(env, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    inputs, labels = dataset()
    print(inputs)
    print(labels)
    assert inputs.shape == (SEQ_LEN, BATCH_SIZE, env.observation_space.shape[0])
    assert labels.shape == (SEQ_LEN, BATCH_SIZE, env.action_space.shape[0])

    plot_simulation(inputs, labels)

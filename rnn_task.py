"""
Environment for a reaching task
based on neurogym
https://neurogym.github.io/understanding_neurogym_task.html
TODO:
- add jump
- add proper labels
"""

import numpy as np
# import gym and neurogym to create tasks
from neurogym import spaces
import neurogym as ngym


class Dataset(object):
    """Make an environment into an iterable dataset for supervised learning.

    Create an iterator that at each call returns
        inputs: numpy array (sequence_length, batch_size, input_units)
        target: numpy array (sequence_length, batch_size, output_units)

    Args:
        env: str for env id or gym.Env objects
        env_kwargs: dict, additional kwargs for environment, if env is str
        batch_size: int, batch size
        seq_len: int, sequence length
        max_batch: int, maximum number of batch for iterator, default infinite
        batch_first: bool, if True, return (batch, seq_len, n_units), default False
        cache_len: int, default length of caching
    """

    def __init__(self, env, env_kwargs=None, batch_size=1, seq_len=None, max_batch=np.inf,
                 batch_first=False, cache_len=None):
        # self.envs = [copy.deepcopy(env) for _ in range(batch_size)]
        if env_kwargs is None:
            env_kwargs = {}
        env.reset()
        self.env = env
        self.seed()
        self.batch_size = batch_size
        self.batch_first = batch_first

        if seq_len is None:
            seq_len = 1000

        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if len(action_shape) == 0:
            self._expand_action = True
        else:
            self._expand_action = False
        if cache_len is None:
            # Infer cache len
            cache_len = 1e5  # Probably too low
            cache_len /= (np.prod(obs_shape) + np.prod(action_shape))
            cache_len /= batch_size
        cache_len = int((1 + (cache_len // seq_len)) * seq_len)

        self.seq_len = seq_len
        self._cache_len = cache_len

        if batch_first:
            shape1, shape2 = [batch_size, seq_len], [batch_size, cache_len]
        else:
            shape1, shape2 = [seq_len, batch_size], [cache_len, batch_size]

        self.inputs_shape = shape1 + list(obs_shape)
        self.target_shape = shape1 + list(action_shape)
        self._cache_inputs_shape = shape2 + list(obs_shape)
        self._cache_target_shape = shape2 + list(action_shape)

        self._inputs = np.zeros(self._cache_inputs_shape, dtype=env.observation_space.dtype)
        self._target = np.zeros(self._cache_target_shape, dtype=env.action_space.dtype)

        self._cache()

        self._i_batch = 0
        self.max_batch = max_batch

    def _cache(self, **kwargs):
        for i in range(self.batch_size):
            env = self.env
            seq_start = 0
            seq_end = 0
            while seq_end < self._cache_len:
                env.new_trial(**kwargs)
                ob, gt = env.ob, env.gt
                seq_len = ob.shape[0]
                seq_end = seq_start + seq_len
                if seq_end > self._cache_len:
                    seq_end = self._cache_len
                    seq_len = seq_end - seq_start
                if self.batch_first:
                    self._inputs[i, seq_start:seq_end, ...] = ob[:seq_len]
                    self._target[i, seq_start:seq_end, ...] = gt[:seq_len]
                else:
                    self._inputs[seq_start:seq_end, i, ...] = ob[:seq_len]
                    self._target[seq_start:seq_end, i, ...] = gt[:seq_len]
                seq_start = seq_end

        self._seq_start = 0
        self._seq_end = self._seq_start + self.seq_len

    def __iter__(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.__next__(**kwargs)

    def __next__(self, **kwargs):
        self._i_batch += 1
        if self._i_batch > self.max_batch:
            self._i_batch = 0
            raise StopIteration

        self._seq_end = self._seq_start + self.seq_len

        if self._seq_end >= self._cache_len:
            self._cache(**kwargs)

        if self.batch_first:
            inputs = self._inputs[:, self._seq_start:self._seq_end, ...]
            target = self._target[:, self._seq_start:self._seq_end, ...]
        else:
            inputs = self._inputs[self._seq_start:self._seq_end]
            target = self._target[self._seq_start:self._seq_end]

        self._seq_start = self._seq_end
        return inputs, target
        # return inputs, np.expand_dims(target, axis=2)

    def seed(self, seed=None):
        self.env.seed(seed)


class MyEnv(ngym.TrialEnv):
    def __init__(self, dt=7, timing=None):
        """
        dt: int, time step in ms, default 7ms corresponding to 130 Hz
        """
        # Python boilerplate to initialize base class
        super().__init__(dt=dt) 

        # define timing for the task
        # TODO: add jump
        self.timing = {
            'fixation': 1500,
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
        # Trial info
        target_ind = self.rng.choice(np.arange(len(self.possible_target_locations)))
        trial = {
            'ground_truth': self.possible_target_locations[target_ind, :],
            'target_ind': target_ind
        }
        trial.update(kwargs)

        # Periods
        self.add_period(list(self.timing.keys()))

        # Observations
        # period relates to timing in self.timing
        # where here relates to name in self.observation space

        # add observation for movement
        self.add_ob(0, period='fixation', where='movement')
        self.add_ob(1, period='reach', where='movement')

        # add observation for target position
        self.add_ob(self.possible_target_locations[target_ind, 0], 
                    period=['fixation', 'reach'], where='target_position_x')
        self.add_ob(self.possible_target_locations[target_ind, 1], 
                    period=['fixation', 'reach'], where='target_position_y')
        # TODO: add jump

        # Ground truth
        # this is kind of a constanst velocity 
        self._set_groundtruth([0,0], 
                              period='fixation')
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
    

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    BATCH_SIZE = 16
    SEQ_LEN = 260
    DT = 7

    # create an instance of the environment
    env = MyEnv(dt=DT)

    # from neurogym.utils.data import Dataset
    # this line of code in neurogym does not work, thus i had to copy the class here
    # https://github.com/neurogym/neurogym/blob/master/neurogym/utils/data.py#L30C26-L30C44

    dataset = Dataset(env, batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    inputs, labels = dataset()
    print(inputs)
    print(labels)
    assert inputs.shape == (SEQ_LEN, BATCH_SIZE, env.observation_space.shape[0])
    assert labels.shape == (SEQ_LEN, BATCH_SIZE, env.action_space.shape[0])

    # plot the simulated data
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    ax[0].plot(inputs[:, 0, 0], '.-', label='x', color='r')
    ax[0].plot(inputs[:, 0, 1], '.-', label='y', color='b')
    # add different yaxis for ax[0] using twinx
    ax[0].legend()
    ax2 = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis
    ax2.plot(inputs[:, 0, 2], '.-', label='fixation', color='k')
    ax[1].plot(labels[:, 0, 0], '.-',label='x')
    ax[1].plot(labels[:, 0, 1], '.-',label='y')
    ax[1].legend()
    #plt.setp(ax, xlim=(200, 250))

    # output angle
'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
import copy
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
set_random_seed(0, using_cuda=True)

class Executor():
	def __init__(self, id, mode, I=None, Beta=None, Circumstance=None, basic=False):
		super().__init__()
		self.id = id
		self.I = I
		self.Circumstance = Circumstance
		self.Beta = Beta
		self.basic = basic
		self.mode = mode
		self.policy = None

	def path_to_json(self):
		return {self.id:self.policy}

class Executor_RL(Executor):
    def __init__(self, id, alg, policy, I, Beta, Circumstance=None, basic=False, nulified_action_indexes=[], wrapper=None, horizon=None):
        super().__init__(id, "RL", I, Beta, Circumstance, basic)
        self.alg = alg
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.wrapper = wrapper
        self.horizon = horizon

    def execute(self, env, obs, goal, symgoal, render=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        horizon = self.horizon if self.horizon is not None else 500
        dummy_env = self.wrapper(env, nulified_action_indexes=self.nulified_action_indexes, horizon=horizon) if self.wrapper is not None else env
        print("\tTask goal: ", symgoal)
        print("\tLoading policy {}".format(self.policy))
        print("\tNumber of nulified indexes: ", len(self.nulified_action_indexes))
        print("\tAction space: ", dummy_env.action_space)
        if self.model is None:
            self.model = self.alg.load(self.policy, 
                                       env=dummy_env,
                                       custom_objects={'observation_space': dummy_env.observation_space, 
                                                       'action_space': dummy_env.action_space,
                                                       #'replay_buffer_class': None,
                                                       })
        step_executor = 0
        done = False
        success = False
        while not done:
            if goal is not None:
                #print("\tLow level goal: ", goal)
                #goal_copy = copy.deepcopy(goal)
                goal_copy = np.copy(goal)
                obs = np.concatenate((obs, goal_copy))
                #print("\tObservation shape: ", obs.shape)
                #print("\tObservation: ", obs)
            action, _states = self.model.predict(obs)
            # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
            if self.nulified_action_indexes:
                for index in self.nulified_action_indexes:
                    action = np.insert(action, index, 0)
            try: 
                obs, reward, terminated, truncated, info = env.step(action)
                #print(obs.shape)
                done = terminated or truncated
            except:
                obs, reward, done, info = env.step(action)
            step_executor += 1
            success = self.Beta(env, symgoal)
            if success:
                print("\tSuccess: Task completed in {} steps\n".format(step_executor))
                break
            done = success
            if step_executor > 500:
                done = True
            if render:
                env.render()
        return obs, success
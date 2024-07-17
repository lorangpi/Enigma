'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
import copy
import torch
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from omegaconf import DictConfig
from imitation_learning.environments import D4RLEnv
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
            #print("Input action: ", action)
            # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
            if self.nulified_action_indexes:
                for index in self.nulified_action_indexes:
                    action = np.insert(action, index, 0)
            #print("Transformed action: ", action)        
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
    

class Executor_GAIL(Executor):
    def __init__(self, id, alg, policy, I, Beta, Circumstance=None, basic=False, nulified_action_indexes=[], wrapper=None, horizon=None):
        super().__init__(id, "RL", I, Beta, Circumstance, basic)
        self.alg = alg
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.wrapper = wrapper
        self.horizon = horizon
        # Define the configuration dictionary
        config_dict = {
            'hidden_size': 256,
            'depth': 2,
            'activation': 'relu'
        }

        # Create a DictConfig instance
        self.model_cfg = DictConfig(config_dict)

    def execute(self, env, obs, goal, symgoal, render=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        horizon = self.horizon if self.horizon is not None else 500
        dummy_env = self.wrapper(env, nulified_action_indexes=self.nulified_action_indexes, horizon=horizon) if self.wrapper is not None else env
        D4RLEnv(dummy_env, True, max_episode_steps=700)
        print("\tTask goal: ", symgoal)
        print("\tLoading policy {}".format(self.policy))
        print("\tNumber of nulified indexes: ", len(self.nulified_action_indexes))
        print("\tAction space: ", dummy_env.action_space)
        if self.model is None:
            self.model = self.alg(dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0], self.model_cfg)
            # Load the entire dictionary
            checkpoint = torch.load(self.policy)
            self.model.load_state_dict(checkpoint['actor'])
        step_executor = 0
        done = False
        success = False 
        reached_success = False
        extra_steps = 0
        while not done:
            if goal is not None:
                #print("\tLow level goal: ", goal)
                #goal_copy = copy.deepcopy(goal)
                goal_copy = np.copy(goal)
                obs = np.concatenate((obs, goal_copy))
                #print("\tObservation shape: ", obs.shape)
                #print("\tObservation: ", obs)
            # Convert the numpy array to a tensor
            obs = torch.from_numpy(obs)
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(dim=0)  # Add batch dimension to state
            obs = torch.cat([obs, torch.zeros(obs.size(0), 1)], dim=1) # absorbing state (D4rl env)
            action = self.model.get_greedy_action(obs)  # Take greedy action
            #print("Input action: ", action)
            # Detach the tensor and convert it to a numpy array
            action = action.detach().numpy()
            # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
            if self.nulified_action_indexes:
                for index in self.nulified_action_indexes:
                    action = np.insert(action, index, 0)
            #print("Transformed action: ", action)   
            # convert the action back to a torch tensor
            #action = torch.tensor(action, dtype=torch.float32).unsqueeze(dim=0)
            #d4rl_env = D4RLEnv(env, True, max_episode_steps=700)     
            try: 
                #obs, reward, terminated = d4rl_env.step(action)
                obs, reward, terminated, truncated, info = env.step(action)
                #print(obs.shape)
                done = terminated or truncated
                # convert the obs to numpy and remove the absorbing state
                #obs = obs.numpy()
                #obs = obs[0][:-1]
            except:
                obs, reward, done, info = env.step(action)
            if render:
                env.render()
            step_executor += 1
            success = self.Beta(env, symgoal)
            if success:
                # pause the simulation for 2 seconds
                #time.sleep(2)
                # Run 2 extra steps to make sure the goal is reached
                reached_success = True
            if reached_success:
                extra_steps += 1
            if extra_steps > 5:
                print("\tSuccess: Task completed in {} steps\n".format(step_executor))
                done = True
            if step_executor > 300:
                done = True 
        return obs, reached_success

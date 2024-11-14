'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
import dill
import torch
import hydra
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from omegaconf import DictConfig
from imitation_learning.environments import D4RLEnv
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import TrainDiffusionTransformerLowdimWorkspace
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
            if self.nulified_action_indexes != []:
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

    def load_policy(self, env):
        horizon = self.horizon if self.horizon is not None else 500
        dummy_env = self.wrapper(env, nulified_action_indexes=self.nulified_action_indexes, horizon=horizon) if self.wrapper is not None else env
        if dummy_env.observation_space.shape[0] < 48:
            dummy_env = D4RLEnv(dummy_env, True, max_episode_steps=700)
        print("\tLoading policy {}".format(self.policy))
        print("\tNumber of nulified indexes: ", len(self.nulified_action_indexes))
        print("\tAction space: ", dummy_env.action_space)
        horizon = self.horizon if self.horizon is not None else 500
        print("\tCreating the model")
        print("\tObservation space: ", dummy_env.observation_space.shape[0])
        self.model = self.alg(dummy_env.observation_space.shape[0], dummy_env.action_space.shape[0], self.model_cfg)
        # Load the entire dictionary
        checkpoint = torch.load(self.policy)
        self.model.load_state_dict(checkpoint['actor'])

    def execute(self, env, obs, goal, symgoal, render=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        goal_check = np.zeros(3)
        horizon = self.horizon if self.horizon is not None else 500


        print("\tTask goal: ", symgoal)

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
            #obs_to_check = obs[0][-3:].numpy() if obs.shape[1] > 1 else obs[-3:].numpy()
            #if any(goal_check != obs_to_check):
            #    print("\t\t\n\nPrevious goal: {}, New goal: {}\n\n".format(goal_check, obs_to_check))
            #goal_check = obs_to_check
            action = self.model.get_greedy_action(obs)  # Take greedy action
            #print("Input action: ", action)
            # Detach the tensor and convert it to a numpy array
            action = action.detach().numpy()
            # If action shape is matrix, convert it to a vector
            if len(action.shape) > 1:
                action = action[0]
            # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
            if self.nulified_action_indexes != []:
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
            if step_executor > horizon:
                done = True 
        return obs, reached_success

class Executor_Diffusion(Executor):
    def __init__(self, id, policy, I, Beta, Circumstance=None, basic=False, nulified_action_indexes=[], oracle=False, wrapper=None, horizon=None, device="cpu"):
        super().__init__(id, "RL", I, Beta, Circumstance, basic)
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.wrapper = wrapper
        self.horizon = horizon
        self.device = device
        self.oracle = oracle

    def load_policy(self):
        path = self.policy
        # load checkpoint
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        #cls = hydra.utils.get_class(cfg._target_)
        #target = "diffusion_policy.diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace.TrainDiffusionTransformerLowdimWorkspace"
        #cls = hydra.utils.get_class(target)
        cls = TrainDiffusionTransformerLowdimWorkspace
        #workspace = cls(cfg, output_dir="../data/")
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        #device = torch.device(self.device)
        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.model = policy

    def obs_mapping(self, obs, action_step="PickPlace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3)}
        trace_obs_list = ["gripper_pos", "aperture", "place_to_drop_pos"]
        reach_pick_obs_list = ["gripper_pos"]
        pick_obs_list = ["gripper_z", "aperture"]
        reach_drop_obs_list = ["gripper_pos"]
        drop_obs_list = ["gripper_z", "aperture"]

        oracle = np.array([])
        if action_step == "PickPlace":
            for key in trace_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "ReachPick":
            for key in reach_pick_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "Grasp":
            for key in pick_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "ReachDrop":
            for key in reach_drop_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "Drop":
            for key in drop_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        return oracle

    def relative_obs_mapping(self, obs, action_step="PickPlace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (9,10), "place_to_drop_z": (6,7)}
        # trace_obs_list = obj_to_pick_pos - gripper_pos, aperture, place_to_drop_pos - gripper_pos
        # reach_pick_obs_list = obj_to_pick_pos - gripper_pos
        # pick_obs_list = obj_to_pick_z - gripper_z, aperture
        # reach_drop_obs_list = place_to_drop_pos - gripper_pos
        # drop_obs_list = place_to_drop_z - gripper_z, aperture

        oracle = np.array([])
        if action_step == "PickPlace":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "ReachPick":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "Grasp":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        elif action_step == "ReachDrop":
            oracle = np.concatenate([obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "Drop":
            oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        else:
            oracle = obs
        return oracle

    def keypoint_mapping(self, obs, action_step="PickPlace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (9,10), "place_to_drop_z": (6,7)}
        trace_key = "obj_to_pick_pos"
        reach_pick_key = "obj_to_pick_pos"
        pick_key = "obj_to_pick_z"
        reach_drop_key = "place_to_drop_pos"
        drop_key = "place_to_drop_z"

        if action_step == "PickPlace":
            keypoint = obs[index_obs[trace_key][0]:index_obs[trace_key][1]]
        elif action_step == "ReachPick":
            keypoint = obs[index_obs[reach_pick_key][0]:index_obs[reach_pick_key][1]]
        elif action_step == "Grasp":
            keypoint = obs[index_obs[pick_key][0]:index_obs[pick_key][1]]
        elif action_step == "ReachDrop":
            keypoint = obs[index_obs[reach_drop_key][0]:index_obs[reach_drop_key][1]]
        elif action_step == "Drop":
            keypoint = obs[index_obs[drop_key][0]:index_obs[drop_key][1]]
        return keypoint

    def prepare_obs(self, obs, action_step="PickPlace"):
        obs_dim = {"PickPlace": 10, "ReachPick": 6, "Grasp": 3, "ReachDrop": 6, "Drop": 3}
        if action_step not in obs_dim.keys():
            return obs
        returned_obs = np.zeros((obs.shape[0], len(obs[0]), obs_dim[action_step]))
        for j, env_n_obs in enumerate(obs):
            for i in range(len(env_n_obs)):
                obs_step = env_n_obs[i]
                # Prepare the observation for the policy
                obs_policy = self.relative_obs_mapping(obs_step, action_step=action_step)
                keypoint_policy = self.keypoint_mapping(obs_step, action_step=action_step)
                concatenated_obs = np.concatenate([keypoint_policy, obs_policy], axis=-1)
                
                # Resize env_n_obs[i] to match the new shape
                returned_obs[j][i] = concatenated_obs
        #print("Returned obs shape: ", returned_obs.shape)
        #print("Original obs shape: ", obs.shape)
        return returned_obs

    def execute(self, env, obs, goal, symgoal, render=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        horizon = self.horizon if self.horizon is not None else 500
        print("\tTask goal: ", symgoal)

        step_executor = 0
        done = False
        success = False 
        while not done:
            # Prepare the observation for the policy
            if self.oracle:
                obs = self.prepare_obs(obs, action_step=self.id)
            # create obs dict
            np_obs_dict = {
                'obs': obs.astype(np.float32)
            }
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=self.device))
            # run policy
            with torch.no_grad():
                action_dict = self.model.predict_action(obs_dict)
            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action']
            # If the actions in action (array) do not have 4 elements, then concatenate [0] to the action array
            if len(action[0][0]) < 4:
                # Create a column of zeros
                zeros_column = np.zeros((action.shape[0], action.shape[1], 1))
                # Concatenate the zeros column to the original array
                #action = np.concatenate((action, zeros_column), axis=2)
                # Concatenate zeros_colum to action at self.nulified_action_indexes
                for index in self.nulified_action_indexes:
                    action = np.insert(action, index, 0, axis=2)
            #print("Action: ", action)
            # step env
            try: 
                obs, reward, terminated, truncated, info = env.step(action)
                #done = terminated or truncated
            except:
                obs, reward, done, info = env.step(action)
            #done = np.all(done)
            if done:
                print("Environment terminated")
            step_executor += 1
            state = info[0]['state'][-1]
            success = self.Beta(state, symgoal)
            success = success or info[0]['is_success'][-1]
            if success:
                done = True
            if step_executor > horizon:
                print("Reached executor horizon")
                done = True 
        return obs, success

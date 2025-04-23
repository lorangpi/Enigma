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


class Executor_Diffusion_Diarc(Executor):
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
        cls = TrainDiffusionTransformerLowdimWorkspace
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
    
    def get_obs(self):
        # TODO: Get the observation from Diarc
        # The obs is supposed to be a list of each environment 4 latest steps observations (parallele runing), 
        # but we only consider one environment here. Thus, the obs should be obs = [[obs1, obs2, obs3, obs4]]
        # obs = Diarc.get_obs()
        obs = None
        return obs

    def obs_mapping(self, obs, action_step="PickPlace"):
        # TODO: add the mapping of the input observation to the oracle (i.e., the input obs of the policy)
        # PickPlace: 7 pick_pos, aperture, place_to_drop_pos (relative to gripper_pos)
        # ReachPick: 3 pick_pos (relative to gripper_pos)
        # Grasp: 2 obj_to_pick_z (relative to gripper_z), aperture
        # ReachDrop: 3 place_to_drop_pos (relative to gripper_pos)
        # Drop: 2 place_to_drop_z (relative to gripper_z), aperture
        oracle = np.array([])
        if action_step == "PickPlace":
            oracle = None
        elif action_step == "ReachPick":
            oracle = None
        elif action_step == "Grasp":
            oracle = None
        elif action_step == "ReachDrop":
            oracle = None
        elif action_step == "Drop":
            oracle = None
        else:
            oracle = obs
        return oracle
    
    def step(self, action):
        # TODO: Send the action to Diarc and get the observation
        obs, reward, done, info = None, None, None, None
        return obs, reward, done, info

    def get_success(self, action_step="PickPlace"):
        # TODO: Get the success of the action step (i.e., if the action step is successful and terminated or not)
        success = False
        return success

    def prepare_obs(self, obs, action_step="PickPlace"):
        obs_dim = {"PickPlace": 7, "ReachPick": 3, "Grasp": 2, "ReachDrop": 3, "Drop": 2}
        if action_step not in obs_dim.keys():
            return obs
        returned_obs = np.zeros((obs.shape[0], len(obs[0]), obs_dim[action_step]))
        for j, env_n_obs in enumerate(obs):
            for i in range(len(env_n_obs)):
                obs_step = env_n_obs[i]
                # Prepare the observation for the policy
                obs_policy = self.obs_mapping(obs_step, action_step=action_step)
                returned_obs[j][i] = obs_policy
        #print("Returned obs shape: ", returned_obs.shape)
        #print("Original obs shape: ", obs.shape)
        return returned_obs

    def prepare_act(self, act, action_step="PickPlace"):
        return act
    
    def control_void_act(self, action, obs):
        if len(action[0][0]) < 4:
            # Concatenate zeros_colum to action at self.nulified_action_indexes
            for index in self.nulified_action_indexes:
                error = obs[0][-1][index] - self.control_static[index]
                action = np.insert(action, index, -error, axis=2)
        return action

    def valid_state_f(self, state):
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
        if len(state) != 3:
            return False
        # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
        pegs = []
        for relation, value in state.items():
            _, peg = relation.split('(')[1].split(',')
            pegs.append(peg)
        if len(pegs) != len(set(pegs)):
            #print("Two or more cubes are on the same peg")
            return False
        return True

    def execute(self, env, symgoal, info = {}):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        obs = self.get_obs()
        self.control_static = {0: obs[0][-1][0], 1: obs[0][-1][1], 2: obs[0][-1][2], 3: obs[0][-1][3]}
        horizon = self.horizon if self.horizon is not None else 500
        print("\tTask goal: ", symgoal)
        step_executor = 0
        done = False
        success = False 
        while not done:
            # Prepare the observation for the policy
            obs_copy = np.copy(obs)
            if self.oracle:
                obs = self.prepare_obs(obs, action_step=self.id)
            # Diffusion Library Formatting of obs
            np_obs_dict = {
                'obs': obs.astype(np.float32)
            }
            # Device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=self.device))
            # Run policy
            with torch.no_grad():
                action_dict = self.model.predict_action(obs_dict)
            # Device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action']
            # If the actions in action (array) do not have 4 elements, then concatenate [0] to the action array
            action = self.control_void_act(action, obs_copy)
            # Step env
            for i in range(len(action[0])):
                action = self.prepare_act(action, action_step=self.id)
            obs, reward, done, info = env.step(action)
            if done:
                print("Environment terminated")
            step_executor += 1
            #state = info[0]['state'][-1]
            #success = self.Beta(state, symgoal)
            success = success or info[0]['is_success'][-1]
            if success:
                done = True
            if step_executor > horizon:
                print("Reached executor horizon")
                done = True 
        return success


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
        cls = TrainDiffusionTransformerLowdimWorkspace
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

    def relative_obs_mapping(self, obs, action_step="PickPlace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (9,10), "place_to_drop_z": (6,7)}
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

    def prepare_obs(self, obs, action_step="PickPlace"):
        obs_dim = {"PickPlace": 7, "ReachPick": 3, "Grasp": 2, "ReachDrop": 3, "Drop": 2}
        if action_step not in obs_dim.keys():
            return obs
        returned_obs = np.zeros((obs.shape[0], len(obs[0]), obs_dim[action_step]))
        for j, env_n_obs in enumerate(obs):
            for i in range(len(env_n_obs)):
                obs_step = env_n_obs[i]
                # Prepare the observation for the policy
                obs_policy = self.relative_obs_mapping(obs_step, action_step=action_step)
                returned_obs[j][i] = obs_policy
        #print("Returned obs shape: ", returned_obs.shape)
        #print("Original obs shape: ", obs.shape)
        return returned_obs

    def prepare_act(self, act, action_step="PickPlace"):
        return act
    
    def control_void_act(self, action, obs):
        if len(action[0][0]) < 4:
            # Concatenate zeros_colum to action at self.nulified_action_indexes
            for index in self.nulified_action_indexes:
                error = obs[0][-1][index] - self.control_static[index]
                action = np.insert(action, index, -error, axis=2)
        return action

    def obs_base_from_info(self, info):
        obs_base = []
        for i in range(len(info)):
            obs_base.append(info[i]["obs_base"])
        return np.array(obs_base)

    def valid_state_f(self, state):
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
        if len(state) != 3:
            return False
        # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
        pegs = []
        for relation, value in state.items():
            _, peg = relation.split('(')[1].split(',')
            pegs.append(peg)
        if len(pegs) != len(set(pegs)):
            #print("Two or more cubes are on the same peg")
            return False
        return True

    def execute(self, env, obs, goal, symgoal, render=False, info = {}, setting="3x3", obj_centric=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        self.control_static = {0: obs[0][-1][0], 1: obs[0][-1][1], 2: obs[0][-1][2], 3: obs[0][-1][3]}
        horizon = self.horizon if self.horizon is not None else 500
        print("\tTask goal: ", symgoal)

        obs_base = False

        if isinstance(obs, np.ndarray):
            obs_base = np.any(obs == None)
        else:
            obs_base = obs == None

        step_executor = 0
        done = False
        success = False 
        while not done:
            # Prepare the observation for the policy
            obs_copy = np.copy(obs)
            if self.oracle:
                obs = self.prepare_obs(obs, action_step=self.id)
            if obs_base:
                obs = self.obs_base_from_info(info)
            #else:
            #    print("Observation: ", obs)
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
            if obs_base:
                    print("Action: ", action)
                    obj_to_pick = action[0][0][0]
                    obj_to_drop = action[0][0][1]
                    success = round(obj_to_drop) != round(obj_to_pick)
                    if not success:
                        action = (round(obj_to_pick), round(obj_to_drop))
                        print("Invalid task: ", action)
                        return None, success
                    obj_to_pick = "cube" + str(round(obj_to_pick))
                    obj_to_drop = "peg" + str(round(obj_to_drop)-3) if round(obj_to_drop) >= 4 else "cube" + str(round(obj_to_drop))
                    print("New task: ", (obj_to_pick, obj_to_drop))
                    env.set_task((obj_to_pick, obj_to_drop))
                    return (obj_to_pick, obj_to_drop), success
            # If the actions in action (array) do not have 4 elements, then concatenate [0] to the action array
            action = self.control_void_act(action, obs_copy)
            # step env
            for i in range(len(action[0])):
                action = self.prepare_act(action, action_step=self.id)
            #print("Transformed action: ", action)
            try: 
                obs, reward, terminated, truncated, info = env.step(action)
            except:
                obs, reward, done, info = env.step(action)
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
        if setting == "3x3":
            valid_state = self.valid_state_f(state)
            success = success and valid_state
            if not valid_state:
                print("Invalid HANOI state")
        return obs, success

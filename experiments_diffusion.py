import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from statistics import mean 
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick_place import PickPlaceWrapper
from robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector
from PDDL.planner import *
from PDDL.executor import *

# diffusion policy import
from typing import Dict
import numpy as np
from tqdm.auto import tqdm

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

# env import
import gym
#import pymunk.pygame_util

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render the environment')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--hanoi', action='store_true', help='Use the Hanoi environment')
args = parser.parse_args()

def termination_indicator(operator):
    if operator == 'pick':
        def Beta(state, symgoal):
            condition = state[f"grasped({symgoal[0]})"]
            return condition
    elif operator == 'drop':
        def Beta(state, symgoal):
            condition = state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"]
            return condition
    elif operator == 'reach_pick':
        def Beta(state, symgoal):
            condition = state[f"over(gripper,{symgoal[0]})"]
            return condition
    elif operator == 'reach_drop':
        def Beta(state, symgoal):
            condition = state[f"over(gripper,{symgoal[1]})"]
            return condition
    return Beta

# Load executors
reach_pick = Executor_Diffusion(id='ReachPick', 
                         #policy="/home/lorangpi/Enigma/saved_policies/reach_pick/epoch=7900-train_loss=0.008.ckpt",
                         policy="/home/lorangpi/Enigma/saved_policies_27u/reach_pick/epoch=2550-train_loss=0.062.ckpt",
                         I={}, 
                         Beta=termination_indicator('reach_pick'),
                         nulified_action_indexes=[3],
                         oracle=True,
                         wrapper = ReachPickWrapper,
                         horizon=10)
grasp = Executor_Diffusion(id='Grasp', 
                   #policy="/home/lorangpi/Enigma/saved_policies/grasp/epoch=7700-train_loss=0.021.ckpt", 
                   policy="/home/lorangpi/Enigma/saved_policies_27u/grasp/epoch=3250-train_loss=0.027.ckpt",
                   I={}, 
                   Beta=termination_indicator('pick'),
                   nulified_action_indexes=[0, 1],
                   oracle=True,
                   wrapper = PickWrapper,
                   horizon=10)
reach_drop = Executor_Diffusion(id='ReachDrop', 
                         #policy="/home/lorangpi/Enigma/saved_policies/reach_place/epoch=6450-train_loss=0.011.ckpt", 
                         policy="/home/lorangpi/Enigma/saved_policies_27u/reach_drop/epoch=2050-train_loss=0.064.ckpt",
                         I={}, 
                         Beta=termination_indicator('reach_drop'),
                         nulified_action_indexes=[3],
                         oracle=True,
                         wrapper = ReachDropWrapper,
                         horizon=17)
drop = Executor_Diffusion(id='Drop', 
                   #policy="/home/lorangpi/Enigma/saved_policies/drop/epoch=7850-train_loss=0.021.ckpt", 
                   policy="/home/lorangpi/Enigma/saved_policies_27u/drop/epoch=3350-train_loss=0.051.ckpt",
                   I={}, 
                   Beta=termination_indicator('drop'),
                   nulified_action_indexes=[0, 1],
                   oracle=True,
                   wrapper = DropWrapper,
                   horizon=10)
pickplace = Executor_Diffusion(id='PickPlace', 
                   policy="/home/lorangpi/Enigma/saved_policies/pick_place/epoch=5900-train_loss=0.010.ckpt", 
                   #policy="/home/lorangpi/Enigma/saved_policies/epoch=3400-train_loss=0.020.ckpt", 
                   I={}, 
                   Beta=termination_indicator('drop'),
                   nulified_action_indexes=[],
                   oracle=True,
                   wrapper = DropWrapper,
                   horizon=100)
place = Executor_Diffusion(id='Place', 
                   policy="/home/lorangpi/Enigma/saved_policies/place/epoch=3400-train_loss=0.015.ckpt", 
                   I={}, 
                   Beta=termination_indicator('drop'),
                   nulified_action_indexes=[],
                   wrapper = DropWrapper,
                   horizon=60)
pick = Executor_Diffusion(id='Pick', 
                   policy="/home/lorangpi/Enigma/saved_policies/pick/epoch=4450-train_loss=0.011.ckpt", 
                   I={}, 
                   Beta=termination_indicator('pick'),
                   nulified_action_indexes=[],
                   wrapper = DropWrapper,
                   horizon=40)

Move_action = [reach_pick, grasp, reach_drop, drop]
#Move_action = [pick, reach_drop, drop]
#Move_action = [pickplace]

# Create an env wrapper which transforms the outputs of reset() and step() into gym formats (and not gymnasium formats)
class GymnasiumToGymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        # set up observation space
        self.obs_dim = 10

        high = np.inf * np.ones(self.obs_dim)
        low = -high
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

    def reset(self):
        obs, info = self.env.reset()
        #keypoint = obs[-3:]#info["keypoint"]#obs[-3:]
        #obs = np.concatenate([
        #    keypoint, 
        #    obs], axis=-1)
        #obs = np.concatenate([obs, info["keypoint"]])
        return obs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        #keypoint = obs[-3:]#info["keypoint"]#obs[-3:]
        #obs = np.concatenate([
        #    keypoint, 
        #    obs], axis=-1)
        return obs, reward, terminated or truncated, info

    def render(self, mode='human', *args, **kwargs):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def set_task(self, task):
        self.env.set_task(task)

# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

device = "cpu"
def env_fn():
    env = suite.make(
        "Hanoi",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        horizon=20000 if args.hanoi else 2000,
        use_camera_obs=False,
        render_camera="robot0_eye_in_hand",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=True,
    )

    # Wrap the environment
    env = GymWrapper(env)
    env = PickPlaceWrapper(env, render_init=args.render, horizon=20000 if args.hanoi else 2000, hanoi=args.hanoi)
    env = GymnasiumToGymWrapper(env)
    env = MultiStepWrapper(
        env=env,
        n_obs_steps=n_obs_steps,
        n_action_steps=n_action_steps,
        max_episode_steps=max_steps
    )
    return env

n_obs_steps = 4
n_action_steps = 8
max_steps = 20000 if args.hanoi else 2000
env_fns = [env_fn]
dummy_env = env_fn()

print(dummy_env.observation_space)

obs_dim = 10
high = np.inf * np.ones(obs_dim)
low = -high
observation_space = gym.spaces.Box(low, high, dtype=np.float64)
action_space = gym.spaces.Box(low=dummy_env.action_space.low, high=dummy_env.action_space.high, dtype=np.float64)

print(observation_space)


def gen_dummy_env():
    def dummy_env_fn():
        # Avoid importing or using env in the main process
        # to prevent OpenGL context issue with fork.
        # Create a fake env whose sole purpos is to provide 
        # obs/action spaces and metadata.
        env = gym.Env()
        # env.observation_space = gym.spaces.Box(
        #     -8, 8, shape=(15,), dtype=np.float32)
        # env.action_space = gym.spaces.Box(
        #     -8, 8, shape=(4,), dtype=np.float32)
        env.observation_space = observation_space
        env.action_space = action_space
        env.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': 12
        }
        env = MultiStepWrapper(
            env=env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
        return env
    return dummy_env_fn

env = AsyncVectorEnv(env_fns, dummy_env_fn=gen_dummy_env())

#Reset the environment
try:
    obs, info = env.reset()
except Exception as e:
    obs = env.reset()

obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
print("Info: ", info)
state = info[0]['state'][-1]
# Detect the state of the environment
# detector = Robosuite_Hanoi_Detector(env)
# state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
# print("Initial state: ", state)

# Create a lambda function that maps "on(cube1,peg1)" to "p1(o1,o3)"
def map_predicate(predicate):
    # Extract the objects from the predicate
    objects = predicate.split('(')[1].split(')')[0].split(',')
    # Map the objects to their corresponding ids
    obj_mapping = {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o6', 'peg1': 'o3', 'peg2': 'o4', 'peg3': 'o5'}
    # Map the predicate to the PDDL format
    return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[1]]})"
def change_predicate(predicate):
    # Extract the objects from the predicate
    objects = predicate.split('(')[1].split(')')[0].split(',')
    # Change clear(cube1) to p1(o1,o1)
    obj_mapping = {'cube1': 'o1', 'cube2': 'o2', 'cube3': 'o6', 'peg1': 'o3', 'peg2': 'o4', 'peg3': 'o5'}
    return f"p1({obj_mapping[objects[0]]},{obj_mapping[objects[0]]})"
# Filter and keep only the predicates that are "on" and are True and map them to the PDDL format
init_predicates = {map_predicate(predicate): True for predicate in state.keys() if 'on' in predicate and state[predicate]}
# Filter and keep only the predicates that are "clear" and are True and map them to the PDDL format
init_predicates.update({change_predicate(predicate): True for predicate in state.keys() if 'clear' in predicate and state[predicate]})
print("Initial predicates: ", init_predicates)

# Usage
add_predicates_to_pddl('problem_static.pddl', init_predicates)

# Generate a plan
plan, _ = call_planner("domain_asp", "problem_dummy")
print("Plan: ", plan)

# # Detected objects
# cube1_body = env.sim.model.body_name2id('cube1_main')
# cube2_body = env.sim.model.body_name2id('cube2_main')
# cube3_body = env.sim.model.body_name2id('cube3_main')
# peg1_body = env.sim.model.body_name2id('peg1_main')
# peg2_body = env.sim.model.body_name2id('peg2_main')
# peg3_body = env.sim.model.body_name2id('peg3_main')
# obj_body_mapping = {
#     'o1': cube1_body,
#     'o2': cube2_body,
#     'o6': cube3_body,
#     'o3': peg1_body,
#     'o4': peg2_body,
#     'o5': peg3_body
# }
# robosuite_obj_body_mapping = {
#     'cube1': cube1_body,
#     'cube2': cube2_body,
#     'cube3': cube3_body,
#     'peg1': peg1_body,
#     'peg2': peg2_body,
#     'peg3': peg3_body
# }
obj_mapping = {'o1': 'cube1', 'o2': 'cube2', 'o6': 'cube3', 'o3': 'peg1', 'o4': 'peg2', 'o5': 'peg3'}
# area_pos = {'peg1': env.pegs_xy_center[0], 'peg2': env.pegs_xy_center[1], 'peg3': env.pegs_xy_center[2]}

def valid_state_f(state):
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
    #print(state)
    return True

reset_gripper_pos = np.array([-0.14193391, -0.03391656,  0.95828137]) * 1000
successes = 0
pick_place_failure = 0
successful_operations = []
percentage_advancement = []

def reset_gripper(env):
    print("Resetting gripper")
    # First move up
    for _ in range(5):
        action = np.array([0, 0, 500, 0])
        obs, reward, done, info = env.step([[action, action, action, action]])
    # Second move to the initial position
    obs = obs[-1][-1]
    current_pos = obs[:3]
    delta = reset_gripper_pos - current_pos
    action = 5*np.array([delta[0], delta[1], delta[2], 0])
    while np.linalg.norm(delta) > 10:
        #print("Curent pos: ", current_pos)
        action = 5*np.array([delta[0], delta[1], delta[2], 0])
        action = action * 0.9
        obs, reward, done, info = env.step([[action, action, action, action]])
        obs = obs[-1][-1]
        current_pos = obs[:3]
        delta = reset_gripper_pos - current_pos
        #print(f"Delta: {delta}, Current pos: {current_pos}, Reset pos: {reset_gripper_pos}")

for i in range(100):
    print("Episode: ", i)
    success = False
    valid_state = False
    plan = False
    # Reset the environment until a valid state is reached
    while plan == False:
        while not valid_state:
            # Reset the environment
            try:
                obs, info = env.reset()
            except Exception as e:
                obs = env.reset()
            obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
            state = info[0]['state'][-1]
            valid_state = valid_state_f(state)
        # Generate the plan
        init_predicates = {map_predicate(predicate): True for predicate in state.keys() if 'on' in predicate and state[predicate]}
        init_predicates.update({change_predicate(predicate): True for predicate in state.keys() if 'clear' in predicate and state[predicate]})

        add_predicates_to_pddl('problem_static.pddl', init_predicates)
        plan, _ = call_planner("domain_asp", "problem_dummy")
    print("Plan: ", plan)

    num_successful_operations = 0
    # Execute the first operator in the plan
    for operator in plan:
        print("\nExecuting operator: ", operator)
        # Concatenate the observations with the operator effects
        if not args.hanoi:
            obj_to_pick = info[0]['task'][0][0]
            obj_to_drop = info[0]['task'][0][1]
        else:
            obj_to_pick = obj_mapping[operator.split(' ')[2].lower()]
            obj_to_drop = obj_mapping[operator.split(' ')[1].lower()]
            #env.set_attr('obj_to_pick', obj_to_pick)
            #env.set_attr('place_to_drop', obj_to_drop)
            env.set_task((obj_to_pick, obj_to_drop))
            print("Set attributes: ", obj_to_pick, obj_to_drop)
            obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
        print("Picking object: {}, Dropping object: {}".format(obj_to_pick, obj_to_drop))
        #pick_loc = env.sim.data.body_xpos[robosuite_obj_body_mapping[obj_to_pick]][:3]
        #drop_loc = env.sim.data.body_xpos[robosuite_obj_body_mapping[obj_to_drop]][:3]
        for action_step in Move_action:
            #if action_step.model == None:
            action_step.load_policy()
            print("\tExecuting action: ", action_step.id)
            symgoal = (obj_to_pick, obj_to_drop)
            goal = []
            obs, success = action_step.execute(env, obs, goal, symgoal, render=args.render)
            if not success:
                print("Execution failed.\n")
                pick_place_failure += 1
                #break
        reset_gripper(env)#, obs[-1][-1])
        if not success:
            # Print the number of operators that were successfully executed out of the total number of operators in the plan
            print("--- Object not picked and placed.")
            print(f"Successfull operations: {num_successful_operations}, Out of: {len(plan)}, Percentage advancement: {num_successful_operations/len(plan)}")
            break
        num_successful_operations += 1
        print("+++ Object successfully picked and placed.")
        print(f"Successfull operations: {num_successful_operations}, Out of: {len(plan)}, Percentage advancement: {num_successful_operations/len(plan)}")
        # Move up the gripper again
        #for _ in range(5):
        #    action = np.array([0, 0, 500, 0])
        #    obs, reward, done, info = env.step([[action, action, action, action]])
    successful_operations.append(num_successful_operations)
    percentage_advancement.append(num_successful_operations/len(plan))
    if success:
        successes += 1
        print("Execution succeeded.\n")
    print("Success rate: ", successes/(i+1))
    print("\n\n")

print("Successfull operations: ", successful_operations)
print("Percentage advancement: ", percentage_advancement)
print("Mean Successful operations: ", mean(successful_operations))
print("Mean Percentage advancement: ", mean(percentage_advancement))

print("Success rate: ", successes/(100))
print("Pick Place failure rate: ", pick_place_failure/(100))

# Write the results to a file results_seed_{args.seed}.txt
with open(f"results_seed_{args.seed}.txt", 'w') as file:
    file.write("Success rate: {}\n".format(successes/(100)))
    file.write("Pick Place failure rate: {}\n".format(pick_place_failure/(100)))
    file.write("Mean Successful operations: {}\n".format(mean(successful_operations)))
    file.write("Mean Percentage advancement: {}\n".format(mean(percentage_advancement)))


import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from detector import Robosuite_Hanoi_Detector
from stable_baselines3 import sac
from PDDL.planner import *
from PDDL.executor import *


# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--policy', type=str, help='Path to the policy')
args = parser.parse_args()

def termination_indicator(operator):
    if operator == 'pick':
        def Beta(env, symgoal):
            detector = Robosuite_Hanoi_Detector(env)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            condition = state[f"grasped({symgoal})"]
            return condition
    elif operator == 'drop':
        def Beta(env, symgoal):
            detector = Robosuite_Hanoi_Detector(env)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            condition = state[f"on({symgoal[0]},{symgoal[1]})"]
            return condition
    elif operator == 'reach_pick':
        def Beta(env, symgoal):
            detector = Robosuite_Hanoi_Detector(env)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            condition = state[f"over(gripper,{symgoal})"]
            return condition
    elif operator == 'reach_drop':
        def Beta(env, symgoal):
            detector = Robosuite_Hanoi_Detector(env)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            condition = state[f"over(gripper,{symgoal})"]
    return Beta


# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

env = suite.make(
    "Hanoi",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    horizon=5000,
    use_camera_obs=False,
    render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    random_reset=False,
)

# Wrap the environment
env = GymWrapper(env)

# Load executors
reach_pick = Executor_RL(id='ReachPick', 
                         alg=sac.SAC, 
                         policy="/home/lorangpi/Enigma/data/demo_seed_4/2024-04-26_11:01:51_reach_pick/policy/best_model.zip", 
                         I={}, 
                         Beta=termination_indicator('reach_pick'),
                         nulified_action_indexes=[3],
                         wrapper = ReachPickWrapper,
                         horizon=200)
pick = Executor_RL(id='Pick', 
                   alg=sac.SAC, 
                   policy="/home/lorangpi/Enigma/data/demo_seed_4/2024-04-26_11:02:20_pick/policy/best_model.zip", 
                   I={}, 
                   Beta=termination_indicator('pick'),
                   nulified_action_indexes=[0,1],
                   wrapper = PickWrapper,
                   horizon=70)
reach_drop = Executor_RL(id='ReachDrop', 
                         alg=sac.SAC, 
                         policy="/home/lorangpi/Enigma/data/demo_seed_4/2024-04-26_11:01:53_reach_drop/policy/best_model.zip", 
                         I={}, 
                         Beta=termination_indicator('reach_drop'),
                         nulified_action_indexes=[3],
                         wrapper = ReachDropWrapper,
                         horizon=200)
drop = Executor_RL(id='Drop', 
                   alg=sac.SAC, 
                   policy="/home/lorangpi/Enigma/data/demo_seed_4/2024-04-26_11:02:36_drop/policy/best_model.zip", 
                   I={}, 
                   Beta=termination_indicator('drop'),
                   nulified_action_indexes=[0,1],
                   wrapper = DropWrapper,
                   horizon=50)

Move_action = [reach_pick, pick, reach_drop, drop]

# Generate a plan
plan, _ = call_planner("domain_asp", "problem_asp")
print("Plan: ", plan)

# Detected objects
cube1_body = env.sim.model.body_name2id('cube1_main')
cube2_body = env.sim.model.body_name2id('cube2_main')
cube3_body = env.sim.model.body_name2id('cube3_main')
peg1_body = env.sim.model.body_name2id('peg1_main')
peg2_body = env.sim.model.body_name2id('peg2_main')
peg3_body = env.sim.model.body_name2id('peg3_main')
obj_body_mapping = {
    'o1': cube1_body,
    'o2': cube2_body,
    'o6': cube3_body,
    'o3': peg1_body,
    'o4': peg2_body,
    'o5': peg3_body
}
obj_mapping = {'o1': 'cube1', 'o2': 'cube2', 'o6': 'cube3', 'o3': 'peg1', 'o4': 'peg2', 'o5': 'peg3'}

# Evaluate the agent
done = False
try:
    obs, _ = env.reset()
except Exception as e:
    obs = env.reset()
for operator in plan:
    print("\nExecuting operator: ", operator)
    # Concatenate the observations with the operator effects
    obj_to_pick = operator.split(' ')[2].lower()
    obj_to_drop = operator.split(' ')[1].lower()
    print("Picking object: {}, Dropping object: {}".format(obj_to_pick, obj_to_drop))
    pick_loc = env.sim.data.body_xpos[obj_body_mapping[obj_to_pick]][:3]
    drop_loc = env.sim.data.body_xpos[obj_body_mapping[obj_to_drop]][:3]
    for action_step in Move_action:
        print("\tExecuting action: ", action_step.id)
        if 'Pick' in action_step.id:
            goal = pick_loc
            symgoal = obj_mapping[obj_to_pick]
        elif 'Drop' in action_step.id:
            goal = drop_loc
            if 'Reach' in action_step.id:
                symgoal = obj_mapping[obj_to_drop]
            else:
                symgoal = (obj_mapping[obj_to_pick],obj_mapping[obj_to_drop])
        obs, success = action_step.execute(env, obs, goal, symgoal, render=True)
        if not success:
            print("Execution failed.")
            pass
    if not success:
        break

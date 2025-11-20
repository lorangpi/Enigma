import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from statistics import mean 
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick_place import PickPlaceWrapper
from robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector
from PDDL.planner import *
from PDDL.executor import *
from imitation_learning.models import SoftActor
from imitation_learning.environments import D4RLEnv

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render the environment')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
args = parser.parse_args()

def termination_indicator(operator):
    if operator == 'place':
        def Beta(env, symgoal):
            detector = Robosuite_Hanoi_Detector(env)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            condition = state[f"on({symgoal[0]},{symgoal[1]})"]
            return condition
    elif operator == 'pick':
        def Beta(env, symgoal):
            detector = Robosuite_Hanoi_Detector(env)
            state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            condition = state[f"grasped({symgoal[0]})"]
            return condition
    return Beta


# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

env = suite.make(
    "Hanoi",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=args.render,
    has_offscreen_renderer=True,
    horizon=100000,
    use_camera_obs=False,
    render_camera="robot0_eye_in_hand",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    random_reset=True,
)

# Wrap the environment
env = GymWrapper(env)

# Load executors
pick = Executor_GAIL(id='Pick',
                            alg=SoftActor,
                            #policy="/home/lorangpi/Enigma/hpc/07-20_04-16-37/agent_1650000.pth",
                            policy="/home/lorangpi/Enigma/hpc/pick_policy.pth",
                            I={},
                            Beta=termination_indicator('pick'),
                            #wrapper = PickPlaceWrapper,
                            horizon=70)
place = Executor_GAIL(id='Place',
                            alg=SoftActor,
                            policy="/home/lorangpi/Enigma/imitation_learning/outputs/GAIL_reach_and_place/10-02_11-19-18/agent_1280000.pth",
                            #policy="/home/lorangpi/Enigma/hpc/place_policy.pth",
                            I={},
                            Beta=termination_indicator('place'),
                            #wrapper = PickPlaceWrapper,
                            horizon=170)

Move_action = [pick, place]

# Detect the state of the environment
detector = Robosuite_Hanoi_Detector(env)
state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

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
robosuite_obj_body_mapping = {
    'cube1': cube1_body,
    'cube2': cube2_body,
    'cube3': cube3_body,
    'peg1': peg1_body,
    'peg2': peg2_body,
    'peg3': peg3_body
}
obj_mapping = {'o1': 'cube1', 'o2': 'cube2', 'o6': 'cube3', 'o3': 'peg1', 'o4': 'peg2', 'o5': 'peg3'}
area_pos = {'peg1': env.pegs_xy_center[0], 'peg2': env.pegs_xy_center[1], 'peg3': env.pegs_xy_center[2]}

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


env = PickPlaceWrapper(env)
# Reset the environment
try:
    obs, _ = env.reset(seed=args.seed)
except Exception as e:
    obs = env.reset(seed=args.seed)


successes = 0
pick_place_failure = 0
successful_operations = []
percentage_advancement = []

for action_step in Move_action:
    action_step.load_policy(env)

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
                obs, _ = env.reset()
            except Exception as e:
                obs = env.reset()
            valid_state = valid_state_f(detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False))
        # Generate the plan
        state = detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
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
        obj_to_pick = env.obj_to_pick
        obj_to_drop = env.place_to_drop
        print("Picking object: {}, Dropping object: {}".format(obj_to_pick, obj_to_drop))
        pick_loc = env.sim.data.body_xpos[robosuite_obj_body_mapping[obj_to_pick]][:3]
        drop_loc = env.sim.data.body_xpos[robosuite_obj_body_mapping[obj_to_drop]][:3]
        for action_step in Move_action:
            print("\tExecuting action: ", action_step.id)
            symgoal = (obj_to_pick, obj_to_drop)
            goal = []
            obs, success = action_step.execute(env, obs, goal, symgoal, render=args.render)
            if not success:
                print("Execution failed.\n")
                pick_place_failure += 1
                break
        if not success:
            break
        num_successful_operations += 1
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


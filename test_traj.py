
import warnings
warnings.filterwarnings("ignore")

import os, argparse
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector
from stable_baselines3 import sac
from imitation.data import serialize
from imitation.data.types import AnyPath, TrajectoryWithRew

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', type=str, default='data/', help='Data Directory')
args = parser.parse_args()
# Set the random seed
np.random.seed(0)

ds = [d for d in os.listdir(args.data_dir + "/hf_traj/")] # there's a directory d for each action "pick", "drop", "reach_drop", "reach_pick"
demo_auto_trajectories = {}
for d in ds:
    demo_trajectories_for_act_dataset = serialize.load(args.data_dir + "/hf_traj/" + d)
    demo_auto_trajectories[d] = demo_trajectories_for_act_dataset

# If the class has a `__dict__` attribute, print it to see all attributes and their values
if hasattr(demo_trajectories_for_act_dataset, '__dict__'):
    print(demo_trajectories_for_act_dataset.__dict__)

# Find indexes of the action space in the trajectories that are never used in the expert demonstrations or are always the same value
def find_constant_indexes(action):
    # Transpose the action array to get actions at each index
    action_t = np.transpose(action)

    # Find indexes where all values are the same or never used
    constant_indexes = [i for i, a in enumerate(action_t) if np.all(a == a[0])]

    return constant_indexes


# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# Create the environment
env = suite.make(
    "Hanoi",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    horizon=500,
    use_camera_obs=False,
    render_camera="frontview", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    random_reset=True,
)


# Wrap the environment
env = GymWrapper(env)


# Test the trajectory
for episodes in range(10):
    print("Episode = ", episodes)
    step = 0
    done = False
    obs = env.reset()
    # While the environment is not reset in a state that matches the trajectory 0 in the dataset
    while not np.linalg.norm(demo_auto_trajectories['reach_pick'][0].obs[0][:63] - obs) < 0.08:
        obs = env.reset()
    # Select one trajectory from the dataset demo_auto_trajectories['pick'] where demo_auto_trajectories['pick'][i].obs[-1] == obs[-1]
    # traj_goal = None
    # while traj_goal != obs[-1]:
    #     traj = np.random.choice(expert_traj)
    #     traj_goal = traj.obs[0][-1]
    # traj_goal = [1000,1000,1000]
    # while np.linalg.norm(traj_goal - obs[-3:]) > 0.01:
    #     traj = np.random.choice(expert_traj)
    #     traj_goal = traj.obs[0][-3:]
    while not done:
        # Create a new TrajectoryWithRew instance where we
        # Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
        expert_traj = [TrajectoryWithRew(
            obs=demo.obs,
            acts=demo.acts,
            rews=demo.rews,
            infos=demo.infos,
            terminal=demo.terminal
        ) for demo in demo_auto_trajectories['reach_pick']]

        traj = expert_traj[0]

        print("Step = ", step)
        # Follow the trajectory
        action = traj.acts[step]
        print("Action = ", action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated
        env.render()
        step += 1
        if step >= len(traj.acts):
            done = True
    done = False
    step = 0
    while not done:
        # Create a new TrajectoryWithRew instance where we
        # Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
        expert_traj = [TrajectoryWithRew(
            obs=demo.obs,
            acts=demo.acts,
            rews=demo.rews,
            infos=demo.infos,
            terminal=demo.terminal
        ) for demo in demo_auto_trajectories['pick']]

        traj = expert_traj[0]

        print("Step = ", step)
        # Follow the trajectory
        action = traj.acts[step]
        print("Action = ", action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated
        env.render()
        step += 1
        if step >= len(traj.acts):
            done = True
    done = False
    step = 0
    while not done:
        # Create a new TrajectoryWithRew instance where we
        # Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
        expert_traj = [TrajectoryWithRew(
            obs=demo.obs,
            acts=demo.acts,
            rews=demo.rews,
            infos=demo.infos,
            terminal=demo.terminal
        ) for demo in demo_auto_trajectories['reach_drop']]

        traj = expert_traj[0]

        print("Step = ", step)
        # Follow the trajectory
        action = traj.acts[step]
        print("Action = ", action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated
        env.render()
        step += 1
        if step >= len(traj.acts):
            done = True
    done = False
    step = 0
    while not done:
        # Create a new TrajectoryWithRew instance where we
        # Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
        expert_traj = [TrajectoryWithRew(
            obs=demo.obs,
            acts=demo.acts,
            rews=demo.rews,
            infos=demo.infos,
            terminal=demo.terminal
        ) for demo in demo_auto_trajectories['drop']]

        traj = expert_traj[0]

        print("Step = ", step)
        # Follow the trajectory
        action = traj.acts[step]
        print("Action = ", action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated
        env.render()
        step += 1
        if step >= len(traj.acts):
            done = True
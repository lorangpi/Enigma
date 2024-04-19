
import warnings
warnings.filterwarnings("ignore")

import os, argparse
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
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

nulified_indexes = find_constant_indexes(demo_auto_trajectories['pick'][0].acts)
print("Nulified indexes = ", nulified_indexes)

# Create a new TrajectoryWithRew instance where we
# Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
expert_traj = [TrajectoryWithRew(
    obs=demo.obs,
    acts=np.delete(demo.acts, nulified_indexes, axis=1),
    rews=demo.rews,
    infos=demo.infos,
    terminal=demo.terminal
) for demo in demo_auto_trajectories['pick']]

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
    #render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    random_reset=True,
)


# Wrap the environment
env = GymWrapper(env)
env = PickWrapper(env, nulified_action_indexes=nulified_indexes)

# Test the trajectory
for episodes in range(10):
    step = 0
    done = False
    obs, _ = env.reset()
    # Select one trajectory from the dataset demo_auto_trajectories['pick'] where demo_auto_trajectories['pick'][i].obs[-1] == obs[-1]
    traj_goal = None
    while traj_goal != obs[-1]:
        traj = np.random.choice(expert_traj)
        traj_goal = traj.obs[0][-1]
    print("action = ", traj.acts[0])
    while not done:
        # Follow the trajectory
        action = traj.acts[step]
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated
        env.render()
        step += 1
        if step >= len(traj.acts):
            done = True
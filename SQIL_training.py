

#%%
import numpy as np
import os
import argparse
import robosuite as suite
import time
from robosuite.wrappers import GymWrapper
from imitation.algorithms import sqil
from imitation.util.util import make_vec_env
from stable_baselines3 import sac
from imitation.data import serialize
from record_demos_automation import RecordDemos, to_datestring

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='demo', choices=['demo'], 
                    help='Name of the experiment. Used to name the log and model directories. Augmented means that the observations are augmented with the detector observation.')
parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
parser.add_argument('--episodes', type=int, default=int(200), help='Number of episodes to train for')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
parser.add_argument('--render', action='store_true', help='Render the initial state')
parser.add_argument('--split_action', action='store_true', help='Split the MOVE action into reach_pick, pick, reach_drop, drop')
parser.add_argument('--data_dir', type=str, default='data/', help='Data Directory')
args = parser.parse_args()
# Set the random seed
np.random.seed(args.seed)
SEED=args.seed

#%%
# load data

#path = "data/demo_seed_0/2024-04-15_19:58:45" + "/hf_traj/"
ds = [d for d in os.listdir(args.data_dir + "/hf_traj/")]
demo_auto_trajectories = {}
for d in ds:
    demo_trajectories_for_act_dataset = serialize.load(args.data_dir + "/hf_traj/" + d)
    demo_auto_trajectories[d] = demo_trajectories_for_act_dataset

#%%
# 
# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')
# Create the environment
env = suite.make(
    "Hanoi",
    robots="Kinova3",
    controller_configs=controller_config,
    has_renderer=args.render,
    has_offscreen_renderer=True,
    horizon=100000000,
    use_camera_obs=False,
    #render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
    random_reset=True,
)
data_folder = args.data_folder
experiment_name = args.experiment + '_seed_' + str(args.seed)
experiment_id = f"{to_datestring(time.time())}"#self.hashid 
if args.name is not None:
    experiment_id = args.name
args.experiment_dir = os.path.join(data_folder, experiment_name, experiment_id)

print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

# Create the directories
args.graphs = args.experiment_dir + '/graphs_train/'
args.pddl = args.experiment_dir + '/pddl_train/'
args.traces = args.experiment_dir + '/traces_train/'
os.makedirs(args.experiment_dir, exist_ok=True)
os.makedirs(args.graphs, exist_ok=True)
os.makedirs(args.pddl, exist_ok=True)
os.makedirs(args.traces, exist_ok=True)
# Wrap the environment
env = GymWrapper(env)
env = RecordDemos(env, args)


# %%
# try training a 'pick' policy
print("Observation space:", env.observation_space)
print("Sample observation:", env.observation_space.sample())

sqil_trainer = sqil.SQIL(
    venv=env,
    demonstrations=demo_auto_trajectories['pick'],
    policy="MlpPolicy",
    rl_algo_class=sac.SAC,
    rl_kwargs=dict(seed=SEED),
)

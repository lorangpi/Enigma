
import os, argparse, time, functools
import numpy as np
import robosuite as suite
import gymnasium as gym
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from imitation.algorithms import sqil
from imitation.util.util import make_seeds
from stable_baselines3 import sac
from stable_baselines3.common import monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from imitation.data import serialize
from record_demos_automation import to_datestring
from typing import (Callable, List,)

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

#path = "data/demo_seed_0/2024-04-15_19:58:45" + "/hf_traj/"
#args.data_dir = "/home/lorangpi/Enigma/data/demo_seed_0/2024-04-16_08:14:04"
ds = [d for d in os.listdir(args.data_dir + "/hf_traj/")]
demo_auto_trajectories = {}
for d in ds:
    demo_trajectories_for_act_dataset = serialize.load(args.data_dir + "/hf_traj/" + d)
    demo_auto_trajectories[d] = demo_trajectories_for_act_dataset

print(len(demo_trajectories_for_act_dataset[0].obs))
print(len(demo_trajectories_for_act_dataset[0].acts))
print(len(demo_trajectories_for_act_dataset[0].rews))
print(demo_trajectories_for_act_dataset[0].terminal)

# If the class has a `__dict__` attribute, print it to see all attributes and their values
if hasattr(demo_trajectories_for_act_dataset, '__dict__'):
    print(demo_trajectories_for_act_dataset.__dict__)
# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

data_folder = args.data_folder
experiment_name = args.experiment + '_seed_' + str(args.seed)
experiment_id = f"{to_datestring(time.time())}"#self.hashid 
if args.name is not None:
    experiment_id = args.name
args.experiment_dir = os.path.join(data_folder, experiment_name, experiment_id)

print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

# Create the directories
args.logs = args.experiment_dir + '/logs/'
os.makedirs(args.experiment_dir, exist_ok=True)
os.makedirs(args.logs, exist_ok=True)

def make_env(i: int, this_seed: int):
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

    # Wrap the environment
    env = GymWrapper(env)
    env = PickWrapper(env)
    env.reset(seed=int(this_seed))
    env = monitor.Monitor(env, args.logs)
    return env


env = make_env(0, SEED)

#Create venv
n_envs = 1
rng=np.random.default_rng(seed=SEED)
env_seeds = make_seeds(rng, n_envs)
env_fns: List[Callable[[], gym.Env]] = [
    functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
]
venv = DummyVecEnv(env_fns)

sqil_trainer = sqil.SQIL(
    venv=venv,
    demonstrations=demo_auto_trajectories['pick'],
    policy="MlpPolicy",
    rl_algo_class=sac.SAC,
    rl_kwargs=dict(seed=SEED),
)

reward_before_training, _ = evaluate_policy(sqil_trainer.policy, venv, 100)
print(f"Reward before training: {reward_before_training}")


sqil_trainer.train(
    total_timesteps=1000,
)  # Note: set to 300_000 to obtain good results
reward_after_training, _ = evaluate_policy(sqil_trainer.policy, venv, 100)
print(f"Reward after training: {reward_after_training}")
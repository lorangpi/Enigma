
import warnings
warnings.filterwarnings("ignore")

import os, argparse, time, functools
import numpy as np
import robosuite as suite
import gymnasium as gym
from pathlib import Path
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.behavior_cloning.reset_with_action_policies import PoliciesResetWrapper
from imitation.algorithms import sqil
from imitation.util.util import make_seeds
from imitation.policies.serialize import save_stable_model
from imitation.data.types import AnyPath, TrajectoryWithRew
from stable_baselines3 import sac
from stable_baselines3.common import monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from eval_callback import CustomEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from imitation.data import serialize
from record_demos_automation import to_datestring
from typing import (Callable, List,)
from bc.off_sqil import Off_SQIL
from execution import *

env_map = {'pick': PickWrapper, 'drop': DropWrapper, 'reach_pick': ReachPickWrapper, 'reach_drop': ReachDropWrapper}
env_horizon = {'pick': 70, 'drop': 50, 'reach_pick': 200, 'reach_drop': 200}
prev_action_policies_executors = {'pick':[reach_pick], 'drop':[reach_pick, pick, reach_drop], 'reach_pick':[], 'reach_drop':[reach_pick, pick]}

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--experiment', type=str, default='demo', choices=['demo'], 
                    help='Name of the experiment. Used to name the log and model directories. Augmented means that the observations are augmented with the detector observation.')
parser.add_argument('--data_dir', type=str, default='data/', help='Data Directory')
parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
parser.add_argument('--episodes', type=int, default=int(200), help='Number of episodes to train for')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--render', action='store_true', help='Render the initial state')
parser.add_argument('-lr', '--learning_rate', type=float, default=3e-3, help='Learning rate')
parser.add_argument('-steps', '--total_timesteps', type=int, default=500_000, help='Total timesteps')
parser.add_argument('-save', '--save_interval', type=int, default=5_000, help='Save interval')
parser.add_argument('-action', type=str, default='trace', help='Possible action step to train reach_pick, pick, reach_drop, drop')
parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
parser.add_argument('-off_rl', '--off_rl', action='store_true', help='Use Off-Policy RL')
args = parser.parse_args()
# Set the random seed
np.random.seed(args.seed)
SEED=args.seed

ds = [d for d in os.listdir(args.data_dir + "/hf_traj/")] # there's a directory d for each action "pick", "drop", "reach_drop", "reach_pick"
demo_auto_trajectories = {}
for d in ds:
    demo_trajectories_for_act_dataset = serialize.load(args.data_dir + "/hf_traj/" + d)
    demo_auto_trajectories[d] = demo_trajectories_for_act_dataset

#print("Observations: ", demo_auto_trajectories['pick'][-1].obs)
#print("Actions: ", demo_auto_trajectories['pick'][-1].acts)
# If the class has a `__dict__` attribute, print it to see all attributes and their values
if hasattr(demo_trajectories_for_act_dataset, '__dict__'):
    print(demo_trajectories_for_act_dataset.__dict__)

#Find the average length of the demonstrations
demo_lengths = [len(demo.acts) for demo in demo_auto_trajectories['pick']]
print("Average length of demonstrations: ", np.mean(demo_lengths))

# Find indexes of the action space in the trajectories that are never used in the expert demonstrations or are always the same value
def find_constant_indexes(action):
    # Transpose the action array to get actions at each index
    action_t = np.transpose(action)

    # Find indexes where all values are the same or never used
    constant_indexes = [i for i, a in enumerate(action_t) if np.all(a == a[0])]

    return constant_indexes

nulified_indexes = find_constant_indexes(demo_auto_trajectories[args.action][0].acts)
print("Nulified indexes = ", nulified_indexes)

# Create a new TrajectoryWithRew instance where we
# Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
expert_traj = [TrajectoryWithRew(
    obs=demo.obs,
    acts=np.delete(demo.acts, nulified_indexes, axis=1),
    rews=demo.rews,
    infos=demo.infos,
    terminal=demo.terminal
) for demo in demo_auto_trajectories[args.action]]

# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

data_folder = args.data_folder
experiment_name = args.experiment + '_seed_' + str(args.seed)
experiment_id = f"{to_datestring(time.time())}"#self.hashid 
if args.name is not None:
    experiment_id = args.name
if args.action is not None:
    experiment_id = experiment_id + "_" + args.action
args.experiment_dir = os.path.join(data_folder, experiment_name, experiment_id)

print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

# Create the directories
args.logs = args.experiment_dir + '/logs/'
args.tensorboard = args.experiment_dir + '/tensorboard/'
os.makedirs(args.experiment_dir, exist_ok=True)
os.makedirs(args.logs, exist_ok=True)
os.makedirs(args.tensorboard, exist_ok=True)
# make a directory to save the trained policy
policy_dir = Path(os.path.join(args.experiment_dir, "policy"))
os.makedirs(policy_dir, exist_ok=True)
policy_name = "model.zip"
policy_path = policy_dir / policy_name


def make_env(i: int, this_seed: int):
    # Create the environment
    print("Creating environment with seed: ", this_seed)
    env = suite.make(
        "Hanoi",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        horizon=1000,
        use_camera_obs=False,
        #render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=True,
    )

    # Wrap the environment
    env = GymWrapper(env)
    if args.action in env_map:
        env = env_map[args.action](env, nulified_action_indexes=nulified_indexes, horizon=env_horizon[args.action])
        env = PoliciesResetWrapper(env=env, nulified_action_indexes=nulified_indexes, horizon=env_horizon[args.action], prev_action_policies=prev_action_policies_executors[args.action])
    env.reset(seed=int(this_seed))
    env = monitor.Monitor(env, args.logs)
    return env

#Create venv
n_envs = 1
rng=np.random.default_rng(seed=SEED)
env_seeds = make_seeds(rng, n_envs)
env_fns: List[Callable[[], gym.Env]] = [
    functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
]
venv = DummyVecEnv(env_fns)

# Linear schedule for the learning rate
def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value

    return func

# Create the SQIL trainer
if args.off_rl:
    print("Using Off-Policy RL.")
    sqil_alg = Off_SQIL
else:
    print("Using On-Policy RL.")
    sqil_alg = sqil.SQIL
sqil_trainer = sqil_alg(
    venv=venv,
    demonstrations=expert_traj,
    policy="MlpPolicy",
    rl_algo_class=sac.SAC,
    rl_kwargs=dict(seed=SEED, 
                   verbose=1,
                   learning_rate=linear_schedule(3e-3, 1e-4),#linear_schedule(4e-4, 7e-5),#linear_schedule(7e-4, 2e-4),#linear_schedule(2e-3, 5e-4), #3e-3,
                   tensorboard_log=args.tensorboard,
                   policy_kwargs=dict(net_arch=[128, 256, 64]),
                   #policy_kwargs=dict(net_arch=[128, 256, 32]),
                   )
)

#print("Evaluation before training.")
#reward_before_training, _ = evaluate_policy(sqil_trainer.policy, venv, 50)
#print(f"Reward before training: {reward_before_training}")

# Define the evaluation callback to evaluate the policy and save the best one each 100000 steps
eval_env = make_env(0, SEED)
eval_callback = CustomEvalCallback(
    eval_env,
    best_model_save_path=policy_dir,
    log_path=args.experiment_dir + '/evaluations.npz',
    eval_freq=args.save_interval,
    n_eval_episodes=30,
    deterministic=True,
    render=False,
    verbose=1
)
#callbacks = [eval_callback, CheckpointCallback(save_freq=args.save_interval, save_path=policy_dir)]
callbacks = [eval_callback]

# Train the policy
print("Launching the SQIL training.")
sqil_trainer.train(
    total_timesteps=args.total_timesteps,
    log_interval=10,
    tb_log_name="SQIL",
    callback=callbacks,
)  # Note: set to 300_000 to obtain good results

# Evaluate the trained policy
print("Evaluation after training.")
reward_after_training, _ = evaluate_policy(sqil_trainer.policy, venv, 50)
print(f"Reward after training: {reward_after_training}")

# Save the trained policy
save_stable_model(policy_dir, sqil_trainer.rl_algo)
print(f"Policy saved to {policy_path}")

# Load the policy
print(f"Loading the policy from {policy_path}")
learned_model = sac.SAC.load(policy_path)

# Now you can use the loaded policy
reward_after_loading, _ = evaluate_policy(learned_model.policy, venv, 1)
print(f"Reward after loading: {reward_after_loading}")

import warnings
warnings.filterwarnings("ignore")

import os, argparse, time, functools
import numpy as np
import robosuite as suite
import gymnasium as gym
from pathlib import Path
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_seeds
from imitation.policies.serialize import save_stable_model
from imitation.data.types import AnyPath, TrajectoryWithRew
from stable_baselines3 import sac
from stable_baselines3.common import monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv
from eval_callback import CustomEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
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

data_folder = args.data_folder
experiment_name = args.experiment + '_seed_' + str(args.seed)
experiment_id = f"{to_datestring(time.time())}"#self.hashid 
if args.name is not None:
    experiment_id = args.name
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
        horizon=150,
        use_camera_obs=False,
        #render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=True,
    )

    # Wrap the environment
    env = GymWrapper(env)
    env = PickWrapper(env, nulified_action_indexes=nulified_indexes)
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

learner = sac.SAC(env=venv, policy="MlpPolicy", verbose=1, tensorboard_log=args.tensorboard, seed=SEED)

reward_net = BasicRewardNet(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    normalize_input_layer=RunningNorm,
)

gail_trainer = GAIL(
    demonstrations=expert_traj,
    demo_batch_size=1024,
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=venv,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
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
    eval_freq=100_000,
    n_eval_episodes=10,
    deterministic=True,
    render=False,
    verbose=1
)

# Train the policy
print("Launching the GAIL training.")
for steps in range(10_000):
    gail_trainer.train(
        total_timesteps=10_000,
    )  # Note: set to 300_000 to obtain good results
    save_stable_model(policy_dir, learner)
    gail_trainer.gen_algo.save(str(policy_path) + "_" + str(steps))

# Evaluate the trained policy
print("Evaluation after training.")
reward_after_training, _ = evaluate_policy(learner, venv, 50)
print(f"Reward after training: {reward_after_training}")

# Save the trained policy
save_stable_model(policy_dir, learner)
print(f"Policy saved to {policy_path}")

# Load the policy
print(f"Loading the policy from {policy_path}")
learned_model = sac.SAC.load(policy_path)

# Now you can use the loaded policy
reward_after_loading, _ = evaluate_policy(learned_model.policy, venv, 1)
print(f"Reward after loading: {reward_after_loading}")
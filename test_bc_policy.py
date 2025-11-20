
import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from stable_baselines3 import sac

# Define the command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('policy', type=str, help='Path to the policy')
args = parser.parse_args()

# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

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
env = PickWrapper(env, nulified_action_indexes=[0,1])

# Load the policy
print(f"Loading the policy from {args.policy}")
learned_model = sac.SAC.load(args.policy)
print("Policy loaded successfully")

# Evaluate the policy
done = False
obs, _ = env.reset()
print("Evaluating the policy")
while not done:
    action, _ = learned_model.predict(obs)
    print(action)
    obs, rew, terminated, truncated, info = env.step(action)
    done = terminated
    env.render()
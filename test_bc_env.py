import numpy as np
import robosuite as suite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action


controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

# Create the environment
env = suite.make(
    "Hanoi",
    robots="Kinova3",
    #robots="Fetch",
    controller_configs=controller_config,
    has_renderer=True,
    has_offscreen_renderer=True,
    horizon=100000000,
    render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
)
env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
env = PickWrapper(env, render_init=True)

device = Keyboard()
env.viewer.add_keypress_callback(device.on_press)

#Initialize device control
device.start_control()

env.reset()
counter = 0

while True:
    # Set active robot
    active_robot = env.robots[0]

    # Get the newest action
    action, grasp = input2action(
        device=device, robot=active_robot, active_arm="right", env_configuration="single-arm-opposed"
    )

    # If action is none, then this a reset so we should break
    if action is None:
        break

    # If the current grasp is active (1) and last grasp is not (-1) (i.e.: grasping input just pressed),
    # toggle arm control and / or camera viewing angle if requested

    # Update last grasp
    last_grasp = grasp

    # Fill out the rest of the action space if necessary
    rem_action_dim = env.action_dim - action.size
    if rem_action_dim > 0:
        # Initialize remaining action space
        rem_action = np.zeros(rem_action_dim)
        # This is a multi-arm setting, choose which arm to control and fill the rest with zeros
        action = np.concatenate([action, rem_action])

    elif rem_action_dim < 0:
        # We're in an environment with no gripper action space, so trim the action space to be the action dim
        action = action[: env.action_dim]

    # Step through the simulation and render
    try:
        obs, reward, terminated, truncated, info = env.step(action)
    except:
        obs, reward, done, info = env.step(action)
    
    env.render()
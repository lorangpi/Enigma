import os, argparse, time, zipfile, pickle
import numpy as np
import robosuite as suite
from datetime import datetime
from robosuite.src.robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector
from robosuite.devices import Keyboard
from robosuite.utils.input_utils import input2action
from graph_learner import GraphLearner

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')
device = Keyboard()

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

# def save_buffer(data_buffer, zip_path):
#     # Save the replay buffer as a zip file
#     with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zip_file:
#         if len(data_buffer[0]) == 3:
#             for i, (obs, act, next_obs) in enumerate(data_buffer):
#                 print(obs.reshape(-1).shape, act.shape, next_obs.reshape(-1).shape)
#                 obs_bytes = obs.tobytes()
#                 act_bytes = act.tobytes()
#                 next_obs_bytes = next_obs.tobytes()
#                 terminated = terminated.tobytes()
#                 truncated = truncated.tobytes()
#                 reward = reward.tobytes()
#                 zip_file.writestr(f'transition_{i}/obs', obs_bytes)
#                 zip_file.writestr(f'transition_{i}/act', act_bytes)
#                 zip_file.writestr(f'transition_{i}/next_obs', next_obs_bytes)
#         elif len(data_buffer[0]) == 5:
#             for i, (obs, act, next_obs, done, reward) in enumerate(data_buffer):
#                 print(obs.reshape(-1).shape, act.shape, next_obs.reshape(-1).shape)
#                 obs_bytes = obs.tobytes()
#                 act_bytes = act.tobytes()
#                 next_obs_bytes = next_obs.tobytes()
#                 terminated = terminated.tobytes()
#                 truncated = truncated.tobytes()
#                 reward = reward.tobytes()
#                 zip_file.writestr(f'transition_{i}/obs', obs_bytes)
#                 zip_file.writestr(f'transition_{i}/act', act_bytes)
#                 zip_file.writestr(f'transition_{i}/next_obs', next_obs_bytes)
#                 zip_file.writestr(f'transition_{i}/terminated', done)
#                 zip_file.writestr(f'transition_{i}/reward', reward)
#         else:
#             for i, (obs, act, next_obs, terminated, truncated, reward) in enumerate(data_buffer):
#                 print(obs.reshape(-1).shape, act.shape, next_obs.reshape(-1).shape)
#                 obs_bytes = obs.tobytes()
#                 act_bytes = act.tobytes()
#                 next_obs_bytes = next_obs.tobytes()
#                 terminated = terminated.tobytes()
#                 truncated = truncated.tobytes()
#                 reward = reward.tobytes()
#                 zip_file.writestr(f'transition_{i}/obs', obs_bytes)
#                 zip_file.writestr(f'transition_{i}/act', act_bytes)
#                 zip_file.writestr(f'transition_{i}/next_obs', next_obs_bytes)
#                 zip_file.writestr(f'transition_{i}/terminated', terminated)
#                 zip_file.writestr(f'transition_{i}/truncated', truncated)
#                 zip_file.writestr(f'transition_{i}/reward', reward)

def save_buffer(data_buffer, file_path):
    # Convert the data buffer to bytes
    data_bytes = pickle.dumps(data_buffer)

    # Write the bytes to a zip file
    with zipfile.ZipFile(file_path, 'w') as zip_file:
        with zip_file.open('data.pkl', 'w') as file:
            file.write(data_bytes)

def load_buffer(file_path):
    # Read the bytes from the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        with zip_file.open('data.pkl', 'r') as file:
            data_bytes = file.read()

    # Convert the bytes back to a list of lists
    data_buffer = pickle.loads(data_bytes)

    return data_buffer


def demo(env, detector, Graph, args):
    # Reset the environment
    try:
        obs, _ = env.reset()
    except:
        obs = env.reset()

    state = detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
    symbolic_buffer = list()
    data_buffer = list()
    episode_buffer = list() # 1 episode here consists of a trajectory between 2 symbolic nodes
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
            next_obs, reward, terminated, truncated, info = env.step(action)
            episode_buffer.append((obs, action, next_obs, terminated, truncated, reward))
        except:
            next_obs, reward, done, info = env.step(action)
            episode_buffer.append((obs, action, next_obs, done, reward))

        new_state = detector.get_groundings(as_dict=True, binary_to_float=True, return_distance=False)
        if new_state != state:
            # Check if the change is linked a grounded predicate 'on(o1,o2)' or 'clear(o1)'
            diff = {k: new_state[k] for k in new_state if k not in state or new_state[k] != state[k]}
            # If any key in diff has 'on' in it and verify that the value associated with that key is 1.0
            if any(['on' in k and diff[k] == 1.0 for k in diff]):
                if not(any(['grasped' in k and diff[k] == 1.0 for k in diff])):
                    print("Change detected: {}".format(diff))
                    print("State: {}".format(new_state))
                    # Filter only the keys that have 'on' in them
                    state = {k: state[k] for k in state if 'on' in k}
                    new_state = {k: new_state[k] for k in new_state if 'on' in k}
                    Graph.learn(state, "MOVE", new_state)
                    symbolic_buffer.append((state, "MOVE", new_state))
                    data_buffer.append(episode_buffer)
                    save_buffer(data_buffer, args.traces + 'traj.zip')
                    save_buffer(symbolic_buffer, args.traces + 'sym.zip')
                    state = new_state
        env.render()
        obs = next_obs


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='demo', choices=['demo'], 
                        help='Name of the experiment. Used to name the log and model directories. Augmented means that the observations are augmented with the detector observation.')
    parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--timesteps', type=int, default=int(5e5), help='Number of timesteps to train for')
    parser.add_argument('--eval_freq', type=int, default=20000, help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=20, help='Number of evaluation episodes')
    parser.add_argument('--no_transfer', action='store_true', help='No transfer learning')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate') # 0.00005 0.00001
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')

    args = parser.parse_args()
    # Set the random seed
    np.random.seed(args.seed)

    # Define the directories
    data_folder = args.data_folder
    experiment_name = args.experiment + '_seed_' + str(args.seed)
    experiment_id = f"{to_datestring(time.time())}"#self.hashid 
    if args.name is not None:
        experiment_id = args.name
    args.experiment_dir = os.path.join(data_folder, experiment_name, experiment_id)

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))

    # Create the directories
    args.graphs = args.experiment_dir + '/graphs/'
    args.pddl = args.experiment_dir + '/pddl/'
    args.traces = args.experiment_dir + '/traces/'
    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(args.graphs, exist_ok=True)
    os.makedirs(args.pddl, exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)

    # Create the environment
    env = suite.make(
        "Hanoi",
        robots="Fetch",
        controller_configs=controller_config,
        has_renderer=True,
        has_offscreen_renderer=True,
        horizon=100000000,
        render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=False,
    )
    env.viewer.add_keypress_callback(device.on_press)

    # Initialize the detector
    detector = Robosuite_Hanoi_Detector(env)
    # Initialize the graph learner
    Graph = GraphLearner(file_path=args.graphs)
    #Initialize device control
    device.start_control()

    demo(env, detector, Graph, args)
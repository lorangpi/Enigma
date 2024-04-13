import os, argparse, time, zipfile, pickle, copy
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from datetime import datetime
import gymnasium as gym
import numpy as np
from detector import Robosuite_Hanoi_Detector
from graph_learner import GraphLearner

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

class RecordDemos(gym.Wrapper):
    def __init__(self, env, args):
        # Run super method
        super().__init__(env=env)
        self.env = env
        self.args = args
        # Initialize the detector
        self.detector = Robosuite_Hanoi_Detector(self)
        # Initialize the graph learner
        self.Graph = GraphLearner(file_path=args.graphs)
        # Define needed variables
        self.cube1_body = self.env.sim.model.body_name2id('cube1_main')
        self.cube2_body = self.env.sim.model.body_name2id('cube2_main')
        self.cube3_body = self.env.sim.model.body_name2id('cube3_main')
        self.peg1_body = self.env.sim.model.body_name2id('peg1_main')
        self.peg2_body = self.env.sim.model.body_name2id('peg2_main')
        self.peg3_body = self.env.sim.model.body_name2id('peg3_main')
        self.obj_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.obj_to_pick = 'cube1_main'
        self.place_to_drop = 'cube2_main'
        self.count_step = 0
        # Adjust the positions to its referential (the bin position)
        #self.peg1_xy = self.env.peg1_pos[:2] + self.env.bin1_pos[:2]
        #self.peg2_xy = self.env.peg2_pos[:2] + self.env.bin1_pos[:2]
        #self.peg3_xy = self.env.peg3_pos[:2] + self.env.bin1_pos[:2]
        # Environment parameters
        self.ray_bins = {'peg': 0.15}
        self.detector = Robosuite_Hanoi_Detector(self)
        self.area_pos = {'peg1': self.pegs_xy_center[0], 'peg2': self.pegs_xy_center[1], 'peg3': self.pegs_xy_center[2]}
        self.render_init = args.render
        self.max_distance = 10
        self.gripper_on = True
        self.picked = []
        self.placed = []
        self.action_steps = []

        # Set up the buffers
        self.symbolic_buffer = list()
        self.data_buffer = dict()

    def pick_reset(self, obs):
        """
        Resets the environment to a state where the gripper is holding the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state

        self.reset_step_count = 0
        #print("Moving up...")
        for _ in range(5):
            next_obs, _, _, _, _  = self.env.step([0,0,1,0])
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.env.render() if self.render_init else None
            self.state_memory = self.record_demos(obs, [0,0,1,0], next_obs, self.state_memory, next_state, action_step="reach_pick")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state

        #print("Moving gripper over object...")
        while not state['over(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 500:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            net_obs = self.env.step([0,0,0,-0.1])
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, [0,0,0,-0.1], next_obs, self.state_memory, next_state, action_step="pick")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="pick")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 400:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(self.obj_to_pick)]:
            next_obs, _, _, _, _  = self.env.step([0,0,0,0.1])
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, [0,0,0,0.1], next_obs, self.state_memory, next_state, action_step="pick")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Lifting object...")
        while not state['picked_up({})'.format(self.obj_to_pick)]:
            next_obs, _, _, _, _  = self.env.step([0,0,0.4,0])
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, [0,0,0.2,0], next_obs, self.state_memory, next_state, action_step="pick")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 300:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def drop_reset(self, obs):
        """
        Resets the environment to a state where the gripper is holding the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)

        self.reset_step_count = 0
        #print("Moving up...")
        #for _ in range(10):
        #    next_obs, _, _, _, _  = self.env.step([0,0,0.5,0])
        #    next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        #    self.state_memory = self.record_demos(obs, [0,0,0.5,0], next_obs, self.state_memory, next_state)
        #    obs, state = next_obs, next_state
        #    self.env.render() if self.render_init else None

        #print("Moving gripper over place to drop...")
        while not state['over(gripper,{})'.format(self.place_to_drop)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if 'peg' in self.place_to_drop:
                object_pos = self.area_pos[self.place_to_drop]
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            action = 5*np.concatenate([dist_xy_plan, [0, 0]])
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="reach_drop")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 500:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down picked object on place to drop...")
        while not state['on({},{})'.format(self.obj_to_pick, self.place_to_drop)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            place_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]])
            dist_z_axis = [- (object_pos[2] - place_pos[2])]
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]])
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="drop")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("dropping object...")
        while state['grasped({})'.format(self.obj_to_pick)]:
            next_obs, _, _, _, _  = self.env.step([0,0,0,-0.1])
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, [0,0,0,-0.1], next_obs, self.state_memory, next_state, action_step="drop")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving up...")
        for _ in range(5):
            next_obs, _, _, _, _  = self.env.step([0,0,1,0])
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.env.render() if self.render_init else None
            self.state_memory = self.record_demos(obs, [0,0,0.5,0], next_obs, self.state_memory, next_state, action_step="drop")
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state

        return True, obs

    def reset(self, seed=None):
        # Reset the environment
        #obs = self.env.viewer._get_observations() if self.env.viewer_get_obs else self.env._get_observations()
        # Flatten the observation
        #obs = self.env._flatten_obs(obs)
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.symbolic_buffer = list()
        obs = self.env.reset()
        self.sample_task()
        self.sim.forward()
        return obs

    def next_episode(self):
        # Data buffer saves a tuple of (trajectory[obs, action, next_obs, done, reward], symbolic trajectory[state, "MOVE", next_state], task)
        for step in self.action_steps:
            if step not in self.data_buffer.keys():
                self.data_buffer[step] = [self.episode_buffer[step], self.symbolic_buffer, "on({},{})".format(self.obj_to_pick, self.place_to_drop)]
            else:
                self.data_buffer[step].append([self.episode_buffer[step], self.symbolic_buffer, "on({},{})".format(self.obj_to_pick, self.place_to_drop)])
        self.save_buffer(self.data_buffer, self.args.traces)
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.symbolic_buffer = list()
        self.sample_task()

    def step_episode(self, obs):
        done_pick, obs = self.pick_reset(obs)
        if not(done_pick):
            return done_pick
        done_drop, obs = self.drop_reset(obs)
        return done_drop

    def record_demos(self, obs, action, next_obs, state_memory, new_state, sym_action="MOVE", action_step="default", reward=-1.0, done=False, info=None):
        # Step through the simulation and render
        if action_step not in self.action_steps:
            self.action_steps.append(action_step)
        if action_step not in self.episode_buffer.keys():
            self.episode_buffer[action_step] = [(obs, action, next_obs, reward, done)]
        else:
            self.episode_buffer[action_step].append((obs, action, next_obs, reward, done))

        #print("Memory: {}".format(self.state_memory['on(cube1,cube2)']))
        state = copy.deepcopy(state_memory)
        if new_state != state:
            # Check if the change is linked a grounded predicate 'on(o1,o2)' or 'clear(o1)'
            transition = {k: new_state[k] for k in new_state if k not in state or new_state[k] != state[k]}
            # If any key in transition has 'on' in it and verify that the value associated with that key is 1.0
            if self.switched_graph_state(transition):
                print("Change detected: {}".format(transition))
                #print("State: {}".format(new_state))
                # Filter only the keys that have 'on' in them
                state_memory = copy.deepcopy(new_state)
                state = {k: state[k] for k in state if 'on' in k}
                new_state = {k: new_state[k] for k in new_state if 'on' in k}
                # Filter only the values that are True
                state = {key: value for key, value in state.items() if value}
                new_state = {key: value for key, value in new_state.items() if value}
                # if state has not 3 keys, return None
                if len(state) != 3 or len(new_state) != 3:
                    return None
                # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
                for test_state in [state, new_state]:
                    pegs = []
                    for relation, value in test_state.items():
                        _, peg = relation.split('(')[1].split(',')
                        pegs.append(peg)
                    if len(pegs) != len(set(pegs)):
                        return None
                self.Graph.learn(state, sym_action, new_state)
                self.symbolic_buffer.append((state, sym_action, new_state))
                state = new_state
        return state_memory

    def switched_graph_state(self, transition, mode='simple'):
        if mode == 'simple':
            if any(['on' in k and transition[k] == 1.0 for k in transition]):
                if not(any(['grasped' in k and transition[k] == 1.0 for k in transition])):
                    return True
        return False

    def sample_task(self):
        # Sample a random task
        valid_task = False
        while not valid_task:
            # Sample a random task, bias towards the least sampled tasks (cf. self.picked and self.placed)
            pick_counts = np.bincount(self.picked, minlength=4)
            place_counts = np.bincount(self.placed, minlength=7)
            pick_weights = 1 / (pick_counts[1:] + 1)  # Add 1 to avoid division by zero
            place_weights = 1 / (place_counts[1:] + 1)
            cube_to_pick = np.random.choice(np.arange(1, 4), p=pick_weights / pick_weights.sum())
            place_to_drop = np.random.choice(np.arange(1, 7), p=place_weights / place_weights.sum())
            pick = cube_to_pick
            place = place_to_drop
            if cube_to_pick >= place_to_drop:
                continue
            if place_to_drop < 4:
                place_to_drop = 'cube{}'.format(place_to_drop)
            else:
                place_to_drop = 'peg{}'.format(place_to_drop - 3)
            cube_to_pick = 'cube{}'.format(cube_to_pick)
            state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            if state['on({},{})'.format(cube_to_pick, place_to_drop)]:
                continue
            if state['clear({})'.format(cube_to_pick)] and state['clear({})'.format(place_to_drop)]:
                valid_task = True
        # print("State: {}".format(state))
        self.picked.append(pick)
        self.placed.append(place)
        # Set the task
        self.obj_to_pick = cube_to_pick
        self.place_to_drop = place_to_drop
        print("Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))

    def save_buffer(self, data_buffer, dir_path):
        # Decompose the data buffer into action steps
        for step in self.action_steps:
            # Convert the data buffer to bytes
            data_bytes = pickle.dumps(data_buffer[step])
            file_path = dir_path + step + '.zip'
            # Write the bytes to a zip file
            with zipfile.ZipFile(file_path, 'w') as zip_file:
                with zip_file.open('data.pkl', 'w', force_zip64=True) as file:
                    file.write(data_bytes)


if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='demo', choices=['demo'], 
                        help='Name of the experiment. Used to name the log and model directories. Augmented means that the observations are augmented with the detector observation.')
    parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--episodes', type=int, default=int(200), help='Number of episodes to train for')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the initial state')

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
        render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=True,
    )

    # Reset the environment
    try:
        obs, _ = env.reset()
    except:
        obs = env.reset()

    # Wrap the environment
    env = GymWrapper(env)
    env = RecordDemos(env, args)
    env.reset()
    # Run the environment
    done = False
    for episode in range(args.episodes):
        print("Episode: {}".format(episode+1))
        keys = list(env.data_buffer.keys())
        if len(keys) > 0:
            print("Number of recorded episodes: {}".format(len(env.data_buffer[keys[0]])))
        done = env.step_episode(obs)
        if done:
            obs = env.next_episode()
        else:
            obs = env.reset()
        done = False
        if episode % 10 == 0 and episode > 0:
            obs = env.reset()
        if episode % 50 == 0:
            print("\n Graph mapping: ", env.Graph.state_mapping)
    print("\n Graph mapping: ", env.Graph.state_mapping)
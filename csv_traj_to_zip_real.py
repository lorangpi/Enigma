import os, argparse, time, zipfile, pickle, copy
import numpy as np
import csv

from datetime import datetime
import numpy as np

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

class RecordDemos():
    def __init__(self, args, randomize=True):
        # Run super method
        super().__init__()
        self.args = args

        self.action_steps = []

        # Set up the buffers
        self.symbolic_buffer = list()
        self.data_buffer = dict()
        self.previous_action = None

    def get_obs_from_row(self, row):
        """
        Returns the step data from the row
        """
        # Access the value of the specified column and convert to float
        # pallet_x = float(row['pallet_x'])
        # pallet_y = float(row['pallet_y'])
        # pallet_yaw = float(row['pallet_yaw'])
        # crayler_x = float(row['fork_x'])
        # crayler_y = float(row['fork_y'])
        # crayler_yaw = float(row['fork_yaw'])
        

        # relative_x = crayler_x - pallet_x
        # relative_y = crayler_y - pallet_y
        # relative_theta = crayler_yaw - pallet_yaw

        #relative_x = float(row['x_error'])
        #relative_y = float(row['y_error'])
        #relative_theta = float(row['theta_error'])

        #relative_x = float(row['fork_x']) - float(row['pallet_x'])
        #relative_y = float(row['fork_y']) - float(row['pallet_y'])
        # For relative theta, use the cosine of the angle between the pallet and the fork 
        # to avoid the discontinuity at 0 and 2pi, need to convert degrees to radians first
        #relative_theta = np.cos(np.radians(float(row['fork_yaw']) - float(row['pallet_yaw'])))
        if self.args.name != None and 'lift' in self.args.name or 'drop' in self.args.name:
            absolute_z = float(row['z_pallet'])
            obs = np.array([absolute_z], dtype=np.float64)
            return obs

        relative_x = float(row['x_rel'])
        relative_y = float(row['y_rel'])
        relative_theta = float(row['theta_rel'])

        drive_vel = float(row['c_drive_vel']) 
        steer_vel = float(row['steering_rate'])
        steer_pos = float(row['steering_pos'])
        forks_shift = float(row['shift_pos'])


        #obs = np.array([relative_x, relative_y, relative_theta, forks_shift, drive_vel, steer_vel, steer_pos], dtype=np.float64)
        obs = np.array([relative_x, relative_y, relative_theta, forks_shift], dtype=np.float64)
        #obs = np.array([pallet_x, pallet_y, pallet_yaw, crayler_x, crayler_y, crayler_yaw, forks_shift, drive_vel, steer_vel, steer_pos], dtype=np.float64)

        return obs

    def get_action_from_row(self, row):
        """
        Returns the action data from the row
        """
        # Access the value of the specified column and convert to float
        if self.args.name != None and 'lift' in self.args.name or 'drop' in self.args.name:
            c_lift_pos_ref = float(row['c_lift'])#float(row['c_lift_pos_ref'])
            action = np.array([c_lift_pos_ref], dtype=np.float64)
        else:
            c_shift_pos_ref = float(row['c_shift'])#float(row['c_shift_pos_ref'])
            c_drive_vel_ref = float(row['c_drive'])#float(row['c_drive_vel_ref'])
            c_steer_vel_ref = float(row['c_steer'])#float(row['c_steer_vel_ref'])

            action = np.array([c_drive_vel_ref, c_steer_vel_ref, c_shift_pos_ref], dtype=np.float64)

        if self.previous_action is not None:
            # If one slot of the action is equal to zero and the previous action is not equal to zero, then replace the zero with the previous action
            for i in range(len(action)):
                if action[i] == 0 and self.previous_action[i] != 0:
                    action[i] = self.previous_action[i]
        self.previous_action = action

        return action

    def read_csv_row(self, bag_name):
        """
        Resets the environment to a state where the gripper is holding the object
        """
        state = "at_pallet_area"
        self.state_memory = state

        # read the csv file (self.args.data_folder + bag_name + '.csv')
        csv_file = open(self.args.data_folder + bag_name + '.csv', 'r')
        csv_reader = csv.DictReader(csv_file)

        i = 0
        # Loop through the csv file except the first row
        for row in csv_reader:
            if i == 0:
                obs = self.get_obs_from_row(row)
                action = self.get_action_from_row(row)
                i += 1
                continue
            next_obs = self.get_obs_from_row(row)
            #if not(float(row['fork_x']) == 0 or float(row['fork_y']) == 0 or float(row['fork_yaw']) == 0):
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory)
            obs = next_obs
            action = self.get_action_from_row(row)
        self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory)
        # Add 10 steps of nul action to the end of the trajectory
        for i in range(10):
            if self.args.name != None and 'lift' in self.args.name or 'drop' in self.args.name:
                self.state_memory = self.record_demos(next_obs, np.array([0]), next_obs, self.state_memory)
            else:
                self.state_memory = self.record_demos(next_obs, np.array([0, 0, 0]), next_obs, self.state_memory)
        
        # Close the csv file
        csv_file.close()
        return True
            
    def reset(self):
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.symbolic_buffer = list()


    def next_trajectory(self):
        # Data buffer saves a tuple of (trajectory[obs, action, next_obs, done, reward], symbolic trajectory[state, "MOVE", next_state], task)
        for step in self.action_steps:
            if step in self.episode_buffer.keys():
                if step not in self.data_buffer.keys():
                    self.data_buffer[step] = [(self.episode_buffer[step], self.symbolic_buffer, "GoToPallet")]
                else:
                    self.data_buffer[step].append((self.episode_buffer[step], self.symbolic_buffer, "GoToPallet"))

        self.save_buffer(self.data_buffer, self.args.traces)
        self.reset()


    def record_demos(self, obs, action, next_obs, state_memory, action_step="trace", reward=-1.0, done=False, info=None):
        keypoint = next_obs
        transition = (obs, action, next_obs, keypoint, reward, done)
        if action_step not in self.action_steps:
            self.action_steps.append(action_step)
        if action_step not in self.episode_buffer.keys():
            self.episode_buffer[action_step] = [transition]
        else:
            self.episode_buffer[action_step].append(transition)

        return state_memory

    def sample_task(self):
        self.obj_to_pick = self.env.obj_to_pick
        self.place_to_drop = self.env.place_to_drop
    #     print("Task: Pick {} and drop it on {}".format(self.obj_to_pick, self.place_to_drop))

    def save_buffer(self, data_buffer, dir_path):
        # Decompose the data buffer into action steps
        for step in self.action_steps:
            if step not in data_buffer.keys():
                continue
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
    parser.add_argument('--data_folder', type=str, default='./forklift_data/csv/', help='Path to the data folder')
    parser.add_argument('--episodes', type=int, default=int(200), help='Number of episodes to train for')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')

    args = parser.parse_args()

    # Define the directories
    data_folder = args.data_folder
    experiment_name = args.experiment
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


    sample = RecordDemos(args)
    sample.reset()

    # Run the sampling
    done = False
    num_recorder_eps = 0
    episode = 1
    while num_recorder_eps < args.episodes and episode < args.episodes * 2:
        print("Episode: {}".format(episode+1))
        keys = list(sample.data_buffer.keys())
        try:
            done = sample.read_csv_row(bag_name="bag_{}data".format(episode))
        except Exception as e:
            print("Error reading csv file: {}".format(e))
            done = True
        if done:
            sample.next_trajectory()
        else:
            sample.reset()
        done = False
        episode += 1
        if len(keys) > 0:
            num_recorder_eps = len(sample.data_buffer[keys[0]])
            print("Number of recorded episodes: {}".format(num_recorder_eps))
import zipfile, pickle, copy, json, argparse, os
import datasets
import numpy as np
from datasets import Dataset, Features, Value, ClassLabel, Sequence
from imitation.data import types
from imitation.data import huggingface_utils, serialize

def load_data_from_zip(dir_path):
    # List all zip files in the directory
    zip_files = [f for f in os.listdir(dir_path) if f.endswith('.zip')]
    
    # Dictionary to hold the data extracted from each zip file
    data_buffers = {}

    # Iterate over each zip file
    for zip_file_name in zip_files:
        # Construct the full path to the zip file
        file_path = os.path.join(dir_path, zip_file_name)
        
        # Open the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            # Extract data.pkl
            with zip_file.open('data.pkl') as file:
                # Deserialize the data
                data = pickle.load(file)
                
                # Use the file name without extension as the key
                data_buffers[zip_file_name.split('.zip')[0]] = data

    return data_buffers

def load_buffer(file_path):
    # Read the bytes from the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        with zip_file.open('data.pkl', 'r') as file:
            data_bytes = file.read()

    # Convert the bytes back to a list of lists
    data_buffer = pickle.loads(data_bytes)

    return data_buffer

def convert_to_dict_format(data_buffer):
    dict_format_buffer = []
    for episode_buffer, symbolic_buffer, task in data_buffer:
        for transition in episode_buffer:
            obs, act, next_obs, reward, done = transition
            dict_format_buffer.append({
                "obs": obs,
                "acts": act,
                "next_obs": next_obs,
                "rews": reward,
                "dones": done,
                "infos": {"symbolic_trajectory": symbolic_buffer, "task": task}
            })
    return dict_format_buffer




features = Features({
    'observations': Sequence(Value('float32')),  # Assuming observations are vectors of floats
    'actions': Sequence(Value('float32')),  # Assuming actions are vectors of floats
    'next_observations': Sequence(Value('float32')),  # Same as observations
    'rewards': Sequence(Value('float32')),  # Rewards are floats
    'dones': Sequence(ClassLabel(names=["false", "true"]))  # Boolean done flags
})


def prepare_data_for_dataset(trajectories):
    trajectory_objects = []

    for trajectory in trajectories:
        if not trajectory:
            continue
        episode = trajectory[0]
        # Assuming each step has observations, actions, next_obs, rewards, and done flags
        obs = [step[0] for step in episode]
        obs.append(episode[-1][2]) # add the `next_obs` from the last trajectory as the final obs
        obs = np.array(obs) # turn it into np.array
        acts = np.array([step[1] for step in episode])
        rews = np.array([step[3] for step in episode])
        infos = np.array([{} for _ in episode])  # Assuming empty dicts for infos
        terminal = True #episode[-1][4]  # The 'done' flag of the last step
        #if terminal:
        #    print("an episode that reached the goal")
        traj_obj = types.TrajectoryWithRew(obs=obs, acts=acts, infos=infos, rews=rews, terminal=terminal)
        trajectory_objects.append(traj_obj)

    return trajectory_objects




def save_buffer_as_json(data_buffer, file_path):
    # Convert the data buffer to dictionary format
    dict_format_buffer = convert_to_dict_format(data_buffer)

    # Convert the dictionary format buffer to JSON
    json_data = json.dumps(dict_format_buffer)

    # Write the JSON data to a zip file
    with zipfile.ZipFile(file_path, 'w') as zip_file:
        with zip_file.open('data.json', 'w', force_zip64=True) as file:
            file.write(json_data.encode('utf-8'))

def load_json_buffer(file_path):
    # Open the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        # Open the JSON file
        with zip_file.open('data.json', 'r') as file:
            # Load the JSON data
            json_data = file.read().decode('utf-8')
            data_buffer = json.loads(json_data)

    return data_buffer

if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/', help='Data Directory')
    args = parser.parse_args()

    # Load the buffer from the zip file
    data_buffers = load_data_from_zip(args.data_dir + '/traces/')

    # Convert the buffer to a dict
    # dict_format_buffer = convert_to_dict_format(data_buffer)

    # Covert each buffer to a list of HuggingFace Trajectory
    demo_auto_trajectories = {}
    for act, buffer in data_buffers.items():
        demo_trajectories_for_act = prepare_data_for_dataset(buffer) # a buffer is a list of trajectories for the high level act e.g reach_pick
        # Convert the list of Trajectory to a HuggingFace Dataset
        demo_trajectories_for_act_dataset = huggingface_utils.trajectories_to_dataset(demo_trajectories_for_act)
        # Convert the dataset to a format usable by the imitation library.
        #demo_trajectories_for_act_dataset = huggingface_utils.TrajectoryDatasetSequence(demo_trajectories_for_act_dataset)
        demo_auto_trajectories[act] = demo_trajectories_for_act_dataset
    
        # save as a HuggingFace Dataset
        save_dir = args.data_dir + '/hf_traj/' + act + '/'
        os.makedirs(save_dir, exist_ok=True)
        serialize.save(save_dir, demo_trajectories_for_act)
    




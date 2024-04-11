import zipfile, pickle, copy, json, argparse, os

def load_buffer(self, file_path):
    # Read the bytes from the zip file
    with zipfile.ZipFile(file_path, 'r') as zip_file:
        with zip_file.open('data.pkl', 'r') as file:
            data_bytes = file.read()

    # Convert the bytes back to a list of lists
    data_buffer = pickle.loads(data_bytes)

    return data_buffer

def convert_to_dict_format(self, data_buffer):
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

def save_buffer_as_json(self, data_buffer, file_path):
    # Convert the data buffer to dictionary format
    dict_format_buffer = self.convert_to_dict_format(data_buffer)

    # Convert the dictionary format buffer to JSON
    json_data = json.dumps(dict_format_buffer)

    # Write the JSON data to a zip file
    with zipfile.ZipFile(file_path, 'w') as zip_file:
        with zip_file.open('data.json', 'w', force_zip64=True) as file:
            file.write(json_data.encode('utf-8'))

def load_json_buffer(self, file_path):
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
    data_buffer = load_buffer(args.data_dir + '/traces/' + 'traj.zip')

    # Convert the buffer to a dict
    dict_format_buffer = convert_to_dict_format(data_buffer)
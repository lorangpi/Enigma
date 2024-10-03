import os
import torch
from imitation.data import serialize
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

data_dir = "/home/lorangpi/Enigma/data/demo_seed_200/full/"
action = 'trace'

ds = [d for d in os.listdir(data_dir + "/hf_traj/")] # there's a directory d for each action "pick", "drop", "reach_drop", "reach_pick"
demo_auto_trajectories = {}
for d in ds:
    demo_trajectories_for_act_dataset = serialize.load(data_dir + "/hf_traj/" + d)
    demo_auto_trajectories[d] = demo_trajectories_for_act_dataset


# Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
dataset = dict()

# Get the maximum length of the trajectories
max_len_obs = max([len(demo.obs) for demo in demo_auto_trajectories[action]])
max_len_act = max([len(demo.acts) for demo in demo_auto_trajectories[action]])

# Pad each trajectory with the last observation value until it reaches max_len
padded_obs = []
padded_act = []
next_obs = []
terminals = []
for demo in demo_auto_trajectories[action]:
    obs = demo.obs
    act = demo.acts
    obs_len = len(obs)
    act_len = len(act)
    terminal = [False] * obs_len  # Not terminal during the actual trajectory
    if obs_len < max_len_obs:
        # Repeat the last observation until the length is max_len
        obs = np.concatenate((obs, np.repeat(obs[-1:], max_len_obs - obs_len, axis=0)))
        terminal += [True] * (max_len_obs - obs_len)  # Terminal during the padding
    if act_len < max_len_act:
        # Repeat the last action until the length is max_len
        act = np.concatenate((act, np.repeat(act[-1:], max_len_act - act_len, axis=0)))
    padded_obs.append(obs[:-1])  # Exclude the last observation
    next_obs.append(obs[1:])  # Exclude the first observation
    padded_act.append(act)
    terminals.append(terminal[1:])


# Flatten the lists to get a single list of observations, actions, next_observations, and terminals
padded_obs = np.concatenate(padded_obs)
next_obs = np.concatenate(next_obs)
padded_act = np.concatenate(padded_act)
terminals = np.concatenate(terminals)

# Convert to tensor
dataset['observations'] = torch.as_tensor(padded_obs, dtype=torch.float32)
dataset['next_observations'] = torch.as_tensor(next_obs, dtype=torch.float32)
dataset['terminals'] = torch.as_tensor(terminals, dtype=torch.float32)
dataset['actions'] = torch.as_tensor(padded_act, dtype=torch.float32)
dataset['timeouts'] = torch.zeros_like(dataset['terminals'], dtype=torch.float32)

# Get the number of trajectories
num_trajectories = len(demo_auto_trajectories[action])

# Initialize the lists of start and end indices
start_indices = [i * max_len_act for i in range(num_trajectories)]
end_indices = [(i + 1) * max_len_act for i in range(num_trajectories)]

# # Step 1: Convert the tensor to a numpy array and reshape it to 2D if necessary
# actions = dataset['actions'].numpy()
# actions = actions.reshape((actions.shape[0], -1))

# # Step 2: Apply PCA to reduce the dimensionality of your data
# pca = PCA(n_components=2)  # or any other number less than the number of features
# actions_pca = pca.fit_transform(actions)

# # Step 3: Apply K-means clustering to the transformed data
# kmeans = KMeans(n_clusters=4)  # or any other number depending on the number of steps you expect
# kmeans.fit(actions_pca)

# # Step 4: Label the original data points based on the cluster they belong to
# labels = kmeans.labels_

# Step 1: Convert the tensor to a numpy array and reshape it to 2D if necessary
actions = dataset['actions'].numpy()
actions = actions.reshape((actions.shape[0], -1))
spectral = SpectralClustering(n_clusters=4)
labels = spectral.fit_predict(actions)
plt.figure(figsize=(10, 7))
plt.plot(range(0, max_len_act), labels[start_indices[0]:end_indices[0]])
plt.title(f'Cluster labels over time for trajectory starting at index {start}')
plt.show()
#print(actions[start_indices[0]:end_indices[0]])
for val, index in enumerate(actions[start_indices[0]:end_indices[0]]):
    print(val, "\t", index)
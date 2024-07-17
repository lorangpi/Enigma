from collections import deque
import os
import time

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

from environments import D4RLEnv, ENVS, ROBOS_ENVS
from evaluation import evaluate_agent
from memory import ReplayMemory
from models import GAILDiscriminator, GMMILDiscriminator, PWILDiscriminator, REDDiscriminator, SoftActor, RewardRelabeller, TwinCritic, create_target_network, make_gail_input, mix_expert_agent_transitions
from training import adversarial_imitation_update, behavioural_cloning_update, sac_update, target_estimation_update
from utils import cycle, flatten_list_dicts, lineplot

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick_place import PickPlaceWrapper
from stable_baselines3.common import monitor
from imitation.data import serialize
from datetime import datetime

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

@hydra.main(version_base=None, config_path='conf', config_name='train_config')
def main(cfg: DictConfig):
  return train(cfg)

# Load the controller config
controller_config = suite.load_controller_config(default_controller='OSC_POSITION')
env_map = {'pick': PickWrapper, 'drop': DropWrapper, 'reach_pick': ReachPickWrapper, 'reach_drop': ReachDropWrapper, 'trace': PickPlaceWrapper}
env_horizon = {'pick': 70, 'drop': 50, 'reach_pick': 200, 'reach_drop': 200, 'trace': 400}

# Find indexes of the action space in the trajectories that are never used in the expert demonstrations or are always the same value
def find_constant_indexes(action):
    # Transpose the action array to get actions at each index
    action_t = np.transpose(action)

    # Find indexes where all values are the same or never used
    constant_indexes = [i for i, a in enumerate(action_t) if np.all(a == a[0])]

    return constant_indexes

def make_env(cfg: DictConfig):
    data_folder = cfg.data_dir
    experiment_name = 'demo' + '_seed_' + str(cfg.seed)
    experiment_id = f"{to_datestring(time.time())}"#self.hashid 
    if cfg.action is not None:
        experiment_id = experiment_id + "_" + cfg.action

    print("Starting experiment {}.".format(os.path.join(experiment_name, experiment_id)))
    # Create the directories
    logs_dir = os.path.join(data_folder, experiment_name, experiment_id) + '/logs/'


    ds = [d for d in os.listdir(cfg.data_dir + "/hf_traj/")] # there's a directory d for each action "pick", "drop", "reach_drop", "reach_pick"
    demo_auto_trajectories = {}
    for d in ds:
        demo_trajectories_for_act_dataset = serialize.load(cfg.data_dir + "/hf_traj/" + d)
        demo_auto_trajectories[d] = demo_trajectories_for_act_dataset


    #print(demo_auto_trajectories[cfg.action][0].acts)
    nulified_indexes = find_constant_indexes(demo_auto_trajectories[cfg.action][0].acts)
    print("Nulified indexes: ", nulified_indexes)

    # Remove the actions slots in the demonstrations.acts that correspond to the nulified indexes
    dataset = dict()
    # # Print the size of the dataset
    # print("Size of the dataset: ", len(demo_auto_trajectories[cfg.action]))
    # print("Size of the first trajectory: ", len(demo_auto_trajectories[cfg.action][0].acts))
    # print("Size of the observations: ", len(demo_auto_trajectories[cfg.action][0].obs))
    # # print all sizes of the observations in the dataset
    # print("Sizes of the observations in the dataset: ", [len(demo.obs) for demo in demo_auto_trajectories[cfg.action]])

    # Get the maximum length of the trajectories
    max_len_obs = max([len(demo.obs) for demo in demo_auto_trajectories[cfg.action]])
    max_len_act = max([len(demo.acts) for demo in demo_auto_trajectories[cfg.action]])

    env_horizon[cfg.action] = max_len_obs + 10

    # Pad each trajectory with the last observation value until it reaches max_len
    padded_obs = []
    padded_act = []
    next_obs = []
    terminals = []
    for demo in demo_auto_trajectories[cfg.action]:
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
        #print("Observation shape: ", obs.shape)
        padded_obs.append(obs[:-1])  # Exclude the last observation
        next_obs.append(obs[1:])  # Exclude the first observation
        # Remove the actions in the nulified indexes
        act = np.delete(act, nulified_indexes, axis=1)
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

    # Create the environment
    print("Creating environment with seed: ", cfg.seed)
    env = suite.make(
        "Hanoi",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        horizon=10000,
        use_camera_obs=False,
        #render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=True,
    )

    # Wrap the environment
    env = GymWrapper(env)
    if cfg.action in env_map:
        env = env_map[cfg.action](env, nulified_action_indexes=nulified_indexes, horizon=env_horizon[cfg.action])
    obs = env.reset(seed=cfg.seed)

    print("Observation space: ", env.observation_space)
    #env = monitor.Monitor(env, logs_dir)

    eval_env = suite.make(
        "Hanoi",
        robots="Kinova3",
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        horizon=10000,
        use_camera_obs=False,
        #render_camera="agentview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_reset=True,
    )

    # Wrap the environment
    eval_env = GymWrapper(eval_env)
    if cfg.action in env_map:
        eval_env = env_map[cfg.action](eval_env, nulified_action_indexes=nulified_indexes, horizon=env_horizon[cfg.action])
    eval_env.reset(seed=cfg.seed)
  
    env, eval_env = D4RLEnv(env, cfg.imitation.absorbing, dataset=dataset, max_episode_steps=env_horizon[cfg.action]), D4RLEnv(eval_env, cfg.imitation.absorbing, max_episode_steps=env_horizon[cfg.action])
    print("Observation space: ", env.observation_space)
    return env, eval_env


def train(cfg: DictConfig, file_prefix: str='') -> float:
  # Configuration check
  assert cfg.algorithm in ['AdRIL', 'BC', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED', 'SAC']
  assert cfg.data_dir != '' and os.path.exists(cfg.data_dir)
  assert cfg.env in ENVS or cfg.env in ROBOS_ENVS
  cfg.memory.size = min(cfg.steps, cfg.memory.size)  # Set max replay memory size to min of environment steps and replay memory size
  assert cfg.bc_pretraining.iterations >= 0
  assert cfg.imitation.trajectories >= 0
  assert cfg.imitation.subsample >= 1
  assert cfg.imitation.mix_expert_data in ['none', 'mixed_batch', 'prefill_memory']
  if cfg.algorithm == 'AdRIL': 
    assert cfg.imitation.mix_expert_data == 'mixed_batch'
    assert cfg.imitation.update_freq >= 0
  elif cfg.algorithm == 'DRIL': 
    assert 0 <= cfg.imitation.quantile_cutoff <= 1
  elif cfg.algorithm == 'GAIL':
    assert cfg.imitation.mix_expert_data != 'prefill_memory'  # Technically possible, but makes the control flow for training the discriminator more complicated
    assert cfg.imitation.discriminator.reward_function in ['AIRL', 'FAIRL', 'GAIL']
    assert cfg.imitation.grad_penalty >= 0
    assert cfg.imitation.entropy_bonus >= 0
    assert cfg.imitation.loss_function in ['BCE', 'Mixup', 'PUGAIL']
    if cfg.imitation.loss_function == 'Mixup': assert cfg.imitation.mixup_alpha > 0
    if cfg.imitation.loss_function == 'PUGAIL': assert 0 <= cfg.imitation.pos_class_prior <= 1 and cfg.imitation.nonnegative_margin >= 0
  assert cfg.logging.interval >= 0

  # General setup
  np.random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)

  # Set up environment
  env, eval_env = make_env(cfg)
  env.seed(cfg.seed)
  eval_env.seed(cfg.seed)
  #normalization_max, normalization_min = env.env.ref_max_score, env.env.ref_min_score
  normalization_max, normalization_min = 1, 0

  expert_memory = env.get_dataset(trajectories=cfg.imitation.trajectories, subsample=cfg.imitation.subsample)  # Load expert trajectories dataset
  state_size, action_size = env.observation_space.shape[0], env.action_space.shape[0]

  # Set up agent
  actor, critic, log_alpha = SoftActor(state_size, action_size, cfg.reinforcement.actor), TwinCritic(state_size, action_size, cfg.reinforcement.critic), torch.zeros(1, requires_grad=True)
  target_critic, entropy_target = create_target_network(critic), cfg.reinforcement.target_temperature * action_size  # Entropy target heuristic from SAC paper for continuous action domains
  actor_optimiser, critic_optimiser, temperature_optimiser = optim.AdamW(actor.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay), optim.AdamW(critic.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay), optim.Adam([log_alpha], lr=cfg.training.learning_rate)
  memory = ReplayMemory(cfg.memory.size, state_size, action_size, cfg.imitation.absorbing)

  # Set up imitation learning components
  if cfg.algorithm in ['AdRIL', 'DRIL', 'GAIL', 'GMMIL', 'PWIL', 'RED']:
    if cfg.algorithm == 'AdRIL':
      discriminator = RewardRelabeller(cfg.imitation.update_freq, cfg.imitation.balanced)  # Balanced sampling (switching between expert and policy data every update) is stateful
    if cfg.algorithm == 'DRIL':
      discriminator = SoftActor(state_size, action_size, cfg.imitation.discriminator)
    elif cfg.algorithm == 'GAIL':
      discriminator = GAILDiscriminator(state_size, action_size, cfg.imitation, cfg.reinforcement.discount)
    elif cfg.algorithm == 'GMMIL':
      discriminator = GMMILDiscriminator(state_size, action_size, cfg.imitation)
    elif cfg.algorithm == 'PWIL':
      discriminator = PWILDiscriminator(state_size, action_size, cfg.imitation, expert_memory, env.max_episode_steps)
    elif cfg.algorithm == 'RED':
      discriminator = REDDiscriminator(state_size, action_size, cfg.imitation)
    if cfg.algorithm in ['DRIL', 'GAIL', 'RED']:
      discriminator_optimiser = optim.AdamW(discriminator.parameters(), lr=cfg.imitation.learning_rate, weight_decay=cfg.imitation.weight_decay)

  # Metrics
  metrics = dict(train_steps=[], train_returns=[], test_steps=[], test_returns=[], test_returns_normalized=[], update_steps=[], predicted_rewards=[], alphas=[], entropies=[], Q_values=[])
  score = []  # Score used for hyperparameter optimization 

  last_step = 1
  # Test if a "load" path is given as a config arg # load actor, critic, discriminator and metric from load path
  if cfg.load:
      print("Loading model from path: ", cfg.load)
      # Path is load path + file_prefix + "_"last step number + ".pth"
      # Search for the last step number in the load path
      for file in os.listdir(cfg.load):
          if file.startswith(file_prefix) and file.endswith(".pth"):
              step = int(file.split("_")[-1].split(".")[0])
              if step > last_step:
                  last_step = step
      print("Latest step number as source policy: ", last_step)
      agent_path = cfg.load + "agent" + "_" + str(last_step) + ".pth"
      discriminator_path = cfg.load + "discriminator" + "_" + str(last_step) + ".pth"
      metrics_path = cfg.load + "metrics" + "_" + str(last_step) + ".pth"
      # Load actor, critic, discriminator and metrics from the load pat
      agent = torch.load(agent_path)
      actor.load_state_dict(agent['actor'])
      critic.load_state_dict(agent['critic'])
      log_alpha = agent['log_alpha']
      if cfg.algorithm in ['DRIL', 'GAIL', 'RED']:
          discriminator.load_state_dict(torch.load(discriminator_path))
      metrics = torch.load(metrics_path)

  if cfg.check_time_usage: start_time = time.time()  # Performance tracking

  # Behavioural cloning pretraining
  if cfg.bc_pretraining.iterations > 0:
    expert_dataloader = iter(cycle(DataLoader(expert_memory, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)))
    actor_pretrain_optimiser = optim.AdamW(actor.parameters(), lr=cfg.bc_pretraining.learning_rate, weight_decay=cfg.bc_pretraining.weight_decay)  # Create separate pretraining optimiser
    for _ in tqdm(range(cfg.bc_pretraining.iterations), leave=False):
      expert_transitions = next(expert_dataloader)
      behavioural_cloning_update(actor, expert_transitions, actor_pretrain_optimiser)

    if cfg.algorithm == 'BC':  # Return early if algorithm is BC
      if cfg.check_time_usage: metrics['pre_training_time'] = time.time() - start_time
      test_returns = evaluate_agent(actor, eval_env, cfg.evaluation.episodes)
      test_returns_normalized = (np.array(test_returns) - normalization_min) / (normalization_max - normalization_min)
      steps = [*range(0, cfg.steps, cfg.evaluation.interval)]
      metrics['test_steps'], metrics['test_returns'], metrics['test_returns_normalized'] = [0], [test_returns], [list(test_returns_normalized)]
      lineplot(steps, len(steps) * [test_returns], filename=f'{file_prefix}test_returns', title=f'{cfg.algorithm}: {cfg.env}')

      torch.save(dict(actor=actor.state_dict()), f'{file_prefix}agent.pth')
      torch.save(metrics, f'{file_prefix}metrics.pth')
      env.close()
      eval_env.close()
      return np.mean(test_returns_normalized)

  # Pretraining "discriminators"
  if cfg.algorithm in ['DRIL', 'RED']:
    expert_dataloader = iter(cycle(DataLoader(expert_memory, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True)))
    for _ in tqdm(range(cfg.imitation.pretraining.iterations), leave=False):
      expert_transition = next(expert_dataloader)
      if cfg.algorithm == 'DRIL':
        behavioural_cloning_update(discriminator, expert_transition, discriminator_optimiser)  # Perform behavioural cloning updates offline on policy ensemble (dropout version)
      elif cfg.algorithm == 'RED':
        target_estimation_update(discriminator, expert_transition, discriminator_optimiser)  # Train predictor network to match random target network

    with torch.inference_mode():
      if cfg.algorithm == 'DRIL':
        discriminator.set_uncertainty_threshold(expert_memory['states'], expert_memory['actions'], cfg.imitation.quantile_cutoff)
      elif cfg.algorithm == 'RED':
        discriminator.set_sigma(expert_memory['states'][:cfg.training.batch_size], expert_memory['actions'][:cfg.training.batch_size])  # Estimate on a minibatch for computational feasibility

    if cfg.check_time_usage:
      metrics['pre_training_time'] = time.time() - start_time
      start_time = time.time()

    if cfg.imitation.mix_expert_data == 'prefill_memory': memory.transfer_transitions(expert_memory)  # Once pretraining is over, transfer expert transitions to agent replay memory
  elif cfg.algorithm == 'PWIL':
    if cfg.imitation.mix_expert_data != 'none':
      with torch.inference_mode():
        for i, transition in tqdm(enumerate(expert_memory), leave=False):
          expert_memory.rewards[i] = discriminator.compute_reward(transition['states'].unsqueeze(dim=0), transition['actions'].unsqueeze(dim=0))  # Greedily calculate the reward for PWIL for expert data and rewrite memory
          if transition['terminals'] or transition['timeouts']: discriminator.reset()  # Reset the expert data for PWIL
    if cfg.imitation.mix_expert_data == 'prefill_memory': memory.transfer_transitions(expert_memory)  # Once rewards have been calculated, transfer expert transitions to agent replay memory
  elif cfg.algorithm == 'GMMIL':
    if cfg.imitation.mix_expert_data == 'prefill_memory': memory.transfer_transitions(expert_memory)

  # Training
  t, state, terminal, train_return = 0, env.reset(), False, 0
  if cfg.algorithm in ['GAIL', 'RED']: discriminator.eval()  # Set the "discriminator" to evaluation mode (except for DRIL, which explicitly uses dropout)
  pbar = tqdm(range(last_step, cfg.steps + 1), unit_scale=1, smoothing=0)
  for step in pbar:
    # Collect set of transitions by running policy π in the environment
    with torch.inference_mode():
      action = actor(state).sample()
      next_state, reward, terminal = env.step(action)
      t += 1
      train_return += reward
      if cfg.algorithm == 'PWIL': reward = discriminator.compute_reward(state, action)  # Greedily calculate the reward for PWIL
      memory.append(step, state, action, reward, next_state, terminal and t != env.max_episode_steps, t == env.max_episode_steps)  # True reward stored for SAC, should be overwritten by IL algorithms; if env terminated due to a time limit then do not count as terminal (store as timeout)
      state = next_state

    # Reset environment and track metrics on episode termination
    if terminal:  # If terminal (or timed out)
      if cfg.imitation.absorbing and t != env.max_episode_steps: memory.wrap_for_absorbing_states()  # Wrap for absorbing state if terminated without time limit
      if cfg.algorithm == 'PWIL': discriminator.reset()  # Reset the expert data for PWIL
      # Store metrics and reset environment
      metrics['train_steps'].append(step)
      metrics['train_returns'].append([train_return])
      pbar.set_description(f'Step: {step} | Return: {train_return}')
      t, state, train_return = 0, env.reset(), 0

    # Train agent and imitation learning component
    if step >= cfg.training.start and step % cfg.training.interval == 0:
      # Sample a batch of transitions
      transitions, expert_transitions = memory.sample(cfg.training.batch_size), expert_memory.sample(cfg.training.batch_size)

      if cfg.algorithm in ['AdRIL', 'DRIL', 'GAIL', 'GMMIL', 'RED']:  # Note that PWIL predicts and stores rewards online during environment interaction
        # Train discriminator
        if cfg.algorithm == 'GAIL':
          discriminator.train()
          adversarial_imitation_update(actor, discriminator, transitions, expert_transitions, discriminator_optimiser, cfg.imitation)
          discriminator.eval()

        # Optionally, mix expert data into agent data for training
        if cfg.imitation.mix_expert_data == 'mixed_batch' and cfg.algorithm != 'AdRIL': mix_expert_agent_transitions(transitions, expert_transitions)
        # Predict rewards
        states, actions, next_states, terminals, weights = transitions['states'], transitions['actions'], transitions['next_states'], transitions['terminals'], transitions['weights']
        expert_states, expert_actions, expert_next_states, expert_terminals, expert_weights = expert_transitions['states'], expert_transitions['actions'], expert_transitions['next_states'], expert_transitions['terminals'], expert_transitions['weights']  # Note that using the entire dataset is prohibitively slow in off-policy case (for relevant algorithms)

        with torch.inference_mode():
          if cfg.algorithm == 'AdRIL':
            discriminator.resample_and_relabel(transitions, expert_transitions, step, memory.num_trajectories, expert_memory.num_trajectories)  # Uses a mix of expert and policy data and overwrites transitions (including rewards) inplace
          elif cfg.algorithm == 'DRIL':
            transitions['rewards'] = discriminator.predict_reward(states, actions)
          elif cfg.algorithm == 'GAIL':
            transitions['rewards'] = discriminator.predict_reward(**make_gail_input(states, actions, next_states, terminals, actor, cfg.imitation.discriminator.reward_shaping, cfg.imitation.discriminator.subtract_log_policy))
          elif cfg.algorithm == 'GMMIL':
            transitions['rewards'] = discriminator.predict_reward(states, actions, expert_states, expert_actions, weights, expert_weights)
          elif cfg.algorithm == 'RED':
            transitions['rewards'] = discriminator.predict_reward(states, actions)

      # Perform a behavioural cloning update (optional)
      if cfg.imitation.bc_aux_loss: behavioural_cloning_update(actor, expert_transitions, actor_optimiser)
      # Perform a SAC update
      log_probs, Q_values = sac_update(actor, critic, log_alpha, target_critic, transitions, actor_optimiser, critic_optimiser, temperature_optimiser, cfg.reinforcement.discount, entropy_target, cfg.reinforcement.polyak_factor)
      # Save auxiliary metrics
      if cfg.logging.interval > 0 and step % cfg.logging.interval == 0:
        metrics['update_steps'].append(step)
        metrics['predicted_rewards'].append(transitions['rewards'].numpy())
        metrics['alphas'].append(log_alpha.exp().detach().numpy())
        metrics['entropies'].append((-log_probs).numpy())  # Actions are sampled from the policy distribution, so "p" is already included
        metrics['Q_values'].append(Q_values.numpy())

    # Evaluate agent and plot metrics
    if step % cfg.evaluation.interval == 0 and not cfg.check_time_usage:
      test_returns = evaluate_agent(actor, eval_env, cfg.evaluation.episodes)
      test_returns_normalized = (np.array(test_returns) - normalization_min) / (normalization_max - normalization_min)
      score.append(np.mean(test_returns_normalized))
      metrics['test_steps'].append(step)
      metrics['test_returns'].append(test_returns)
      metrics['test_returns_normalized'].append(list(test_returns_normalized))
      lineplot(metrics['test_steps'], metrics['test_returns'], filename=f"{file_prefix}test_returns", title=f'{cfg.algorithm}: {cfg.env} Test Returns')
      if len(metrics['train_returns']) > 0:  # Plot train returns if any
        lineplot(metrics['train_steps'], metrics['train_returns'], filename=f"{file_prefix}train_returns", title=f'Training {cfg.algorithm}: {cfg.env} Train Returns')
      if cfg.logging.interval > 0 and len(metrics['update_steps']) > 0:
        if cfg.algorithm != 'SAC': lineplot(metrics['update_steps'], metrics['predicted_rewards'], filename=f'{file_prefix}predicted_rewards', yaxis='Predicted Reward', title=f'{cfg.algorithm}: {cfg.env} Predicted Rewards')
        lineplot(metrics['update_steps'], metrics['alphas'], filename=f'{file_prefix}sac_alpha', yaxis='Alpha', title=f'{cfg.algorithm}: {cfg.env} Alpha')
        lineplot(metrics['update_steps'], metrics['entropies'], filename=f'{file_prefix}sac_entropy', yaxis='Entropy', title=f'{cfg.algorithm}: {cfg.env} Entropy')
        lineplot(metrics['update_steps'], metrics['Q_values'], filename=f'{file_prefix}Q_values', yaxis='Q-value', title=f'{cfg.algorithm}: {cfg.env} Q-values')

      # Save agent and metrics
      torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), log_alpha=log_alpha), f'{file_prefix}agent_{step}.pth')
      if cfg.algorithm in ['DRIL', 'GAIL', 'RED']: torch.save(discriminator.state_dict(), f'{file_prefix}discriminator_{step}.pth')
      torch.save(metrics, f'{file_prefix}metrics_{step}.pth')

  if cfg.check_time_usage:
    metrics['training_time'] = time.time() - start_time

  if cfg.save_trajectories:
    # Store trajectories from agent after training
    _, trajectories = evaluate_agent(actor, eval_env, cfg.evaluation.episodes, return_trajectories=True, render=cfg.render)
    torch.save(trajectories, f'{file_prefix}trajectories.pth')
  # Save agent and metrics
  torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), log_alpha=log_alpha), f'{file_prefix}agent.pth')
  if cfg.algorithm in ['DRIL', 'GAIL', 'RED']: torch.save(discriminator.state_dict(), f'{file_prefix}discriminator.pth')
  torch.save(metrics, f'{file_prefix}metrics.pth')

  env.close()
  eval_env.close()
  return np.mean(score)


if __name__ == '__main__':
  main()

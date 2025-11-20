import warnings
warnings.filterwarnings("ignore")

import argparse
import robosuite as suite
import numpy as np
from statistics import mean 
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick import PickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_drop import DropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_pick import ReachPickWrapper
from robosuite.wrappers.behavior_cloning.hanoi_reach_drop import ReachDropWrapper
from robosuite.wrappers.behavior_cloning.hanoi_pick_place import PickPlaceWrapper
from robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector
from PDDL.planner import *
from PDDL.executor import *

# diffusion policy import
from typing import Dict
import numpy as np
from tqdm.auto import tqdm

from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper

# env import
import gym
#import pymunk.pygame_util

if __name__ == "__main__":

    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the environment')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--hanoi', action='store_true', help='Use the Hanoi environment')
    parser.add_argument('--demos', type=int, default=0, help='Number of demonstrations used by the learned policies')
    args = parser.parse_args()

    def termination_indicator(operator):
        if operator == 'pick':
            def Beta(state, symgoal):
                condition = state[f"grasped({symgoal[0]})"]
                return condition
        elif operator == 'drop':
            def Beta(state, symgoal):
                condition = state[f"on({symgoal[0]},{symgoal[1]})"] and not state[f"grasped({symgoal[0]})"]
                return condition
        elif operator == 'reach_pick':
            def Beta(state, symgoal):
                condition = state[f"over(gripper,{symgoal[0]})"]
                return condition
        elif operator == 'reach_drop':
            def Beta(state, symgoal):
                condition = state[f"over(gripper,{symgoal[1]})"]
                return condition
        else:
            def Beta(state, symgoal):
                return False
        return Beta

    # Load executors
    reasoner = Executor_Diffusion(id='PickPlace', 
                    #policy="/home/lorangpi/Enigma/results_baselines/outputs/2025.01.20/07.35.41_train_diffusion_transformer_lowdim_500_oracle_lowdim/checkpoints/epoch=7950-train_loss=0.004.ckpt", 
                    policy=f"./policies/hil_dense_{args.demos}/reasoner.ckpt",
                    I={}, 
                    Beta=termination_indicator('other'),
                    nulified_action_indexes=[],
                    oracle=False,
                    wrapper = DropWrapper,
                    horizon=1)
    pickplace = Executor_Diffusion(id='PickPlace', 
                    #policy="/home/lorangpi/Enigma/results_baselines/outputs/2025.01.20/07.36.55_train_diffusion_transformer_lowdim_500_pick_place_rel_lowdim/checkpoints/epoch=1900-train_loss=0.027.ckpt", 
                    policy=f"./policies/hil_{args.demos}/pickplace.ckpt",
                    I={}, 
                    Beta=termination_indicator('drop'),
                    nulified_action_indexes=[],
                    oracle=True,
                    wrapper = DropWrapper,
                    horizon=50)

    Move_action = [pickplace]

    # Create an env wrapper which transforms the outputs of reset() and step() into gym formats (and not gymnasium formats)
    class GymnasiumToGymWrapper(gym.Env):
        def __init__(self, env):
            self.env = env
            self.action_space = env.action_space
            # set up observation space
            self.obs_dim = 10

            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float64)

        def reset(self):
            obs, info = self.env.reset()
            #keypoint = obs[-3:]#info["keypoint"]#obs[-3:]
            #obs = np.concatenate([
            #    keypoint, 
            #    obs], axis=-1)
            #obs = np.concatenate([obs, info["keypoint"]])
            return obs

        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            #keypoint = obs[-3:]#info["keypoint"]#obs[-3:]
            #obs = np.concatenate([
            #    keypoint, 
            #    obs], axis=-1)
            return obs, reward, terminated or truncated, info

        def render(self, mode='human', *args, **kwargs):
            self.env.render()

        def close(self):
            self.env.close()

        def seed(self, seed=None):
            self.env.seed(seed)

        def set_task(self, task):
            self.env.set_task(task)

    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSITION')

    device = "cpu"
    def env_fn():
        env = suite.make(
            "Hanoi",
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=True,
            has_offscreen_renderer=True,
            horizon=20000 if args.hanoi else 2000,
            use_camera_obs=False,
            render_camera="frontview",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
            random_reset=False,
        )

        # Wrap the environment
        env = GymWrapper(env)
        env = PickPlaceWrapper(env, render_init=args.render, horizon=20000 if args.hanoi else 2000, hanoi=args.hanoi, oracle=True)
        env = GymnasiumToGymWrapper(env)
        env = MultiStepWrapper(
            env=env,
            n_obs_steps=n_obs_steps,
            n_action_steps=n_action_steps,
            max_episode_steps=max_steps
        )
        return env

    n_obs_steps = 4
    n_action_steps = 8
    max_steps = 20000 if args.hanoi else 2000
    env_fns = [env_fn]
    dummy_env = env_fn()

    print(dummy_env.observation_space)

    obs_dim = 10
    high = np.inf * np.ones(obs_dim)
    low = -high
    observation_space = gym.spaces.Box(low, high, dtype=np.float64)
    action_space = gym.spaces.Box(low=dummy_env.action_space.low, high=dummy_env.action_space.high, dtype=np.float64)

    print(observation_space)


    def gen_dummy_env():
        def dummy_env_fn():
            # Avoid importing or using env in the main process
            # to prevent OpenGL context issue with fork.
            # Create a fake env whose sole purpos is to provide 
            # obs/action spaces and metadata.
            env = gym.Env()
            env.observation_space = observation_space
            env.action_space = action_space
            env.metadata = {
                'render.modes': ['human', 'rgb_array', 'depth_array'],
                'video.frames_per_second': 12
            }
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            return env
        return dummy_env_fn

    env = AsyncVectorEnv(env_fns, dummy_env_fn=gen_dummy_env())

    #Reset the environment
    try:
        obs, info = env.reset()
    except Exception as e:
        obs = env.reset()

    obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])

    def find_index_task(task, plan):
        for i, t in enumerate(plan):
            if t[0] == task[0] and t[1] == task[1]:
                return i+1
        return 0

    def valid_state_f(state):
        state = {k: state[k] for k in state if 'on' in k}
        # Filter only the values that are True
        state = {key: value for key, value in state.items() if value}
        # if state has not 3 keys, return None
        if len(state) != 3:
            return False
        # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
        pegs = []
        for relation, value in state.items():
            _, peg = relation.split('(')[1].split(',')
            pegs.append(peg)
        if len(pegs) != len(set(pegs)):
            #print("Two or more cubes are on the same peg")
            return False
        return True

    reset_gripper_pos = np.array([-0.14193391, -0.03391656,  1.05828137]) * 1000
    hanoi_successes = 0
    num_valid_pick_place_queries = 0
    pick_place_success = 0
    pick_place_successes = []
    percentage_advancement = []
    valid_pick_place_success = 0

    def reset_gripper(env):
        print("Resetting gripper")
        # First move up
        for _ in range(5):
            action = np.array([0, 0, 500, 0])
            obs, reward, done, info = env.step([[action, action, action, action]])
        # Second move to the initial position
        obs = obs[-1][-1]
        current_pos = obs[:3]
        delta = reset_gripper_pos - current_pos
        action = 5*np.array([delta[0], delta[1], delta[2], 0])
        while np.linalg.norm(delta) > 10:
            #print("Curent pos: ", current_pos)
            action = 5*np.array([delta[0], delta[1], delta[2], 0])
            action = action * 0.9
            obs, reward, done, info = env.step([[action, action, action, action]])
            obs = obs[-1][-1]
            current_pos = obs[:3]
            delta = reset_gripper_pos - current_pos
            #print(f"Delta: {delta}, Current pos: {current_pos}, Reset pos: {reset_gripper_pos}")


    ground_truth_plan = [("cube1","peg3"), ("cube2","peg2"), ("cube1","cube2"), ("cube3","peg3"), ("cube1","peg1"), ("cube2","cube3"), ("cube1","cube2")]

    for i in range(100):
        print("Episode: ", i)
        success = True
        valid_state = False
        np.random.seed(args.seed + i)
        # Reset the environment until a valid state is reached

        while not valid_state:
            # Reset the environment
            try:
                obs, info = env.reset()
            except Exception as e:
                obs = env.reset()
            obs, reward, done, info = env.step([[np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)]])
            state = info[0]['state'][-1]
            valid_state = valid_state_f(state)



        pickplace.load_policy()
        reasoner.load_policy()

        reset_gripper(env)
        pick_place_success = 0
        # Execute the first operator in the plan
        advancement = 0
        for j in range(7):
            if j == 0:
                task = ("cube1","peg3")
                previous_task = None
            else:
                previous_task = task
                task, success = reasoner.execute(env, None, None, None, render=args.render, info=info)
            print("Symgoal: ", task)

            valid_pick_place_querie = task == ground_truth_plan[j]
            if valid_pick_place_querie:
                num_valid_pick_place_queries += 1
            if task == previous_task:
                success = False

            if not success:
                break
            success = False
            goal = []
            obs, success = pickplace.execute(env, obs, goal, task, render=args.render)
            if success:
                pick_place_success += 1
                print("+++ Object successfully picked and placed.")
                # precentage advancement is the index of  the current task in the plan
                advancement = find_index_task(task, ground_truth_plan)
                print(f"Successfull pick_place: {pick_place_success}, Out of: {7}, Percentage advancement: {advancement/7}")
                if valid_pick_place_querie:
                    valid_pick_place_success += 1
            else:
                print("Execution failed.\n")
                # Print the number of operators that were successfully executed out of the total number of operators in the plan
                print("--- Object not picked and placed.")
                if previous_task is not None:
                    advancement = find_index_task(previous_task, ground_truth_plan)
                else:
                    advancement = 0
                print(f"Successfull pick_place: {pick_place_success}, Out of: {7}, Percentage advancement: {advancement/7}")
                break
            reset_gripper(env)
        if success:
            hanoi_successes += 1
            print("Hanoi Execution succeeded.\n")
        print("Success rate: ", hanoi_successes/(i+1))
        print("\n\n")

        pick_place_successes.append(pick_place_success)
        percentage_advancement.append(advancement/7)

        print("Successfull pick_place: ", pick_place_successes)
        print("Percentage advancement: ", percentage_advancement)
        print("Mean Successful pick_place: ", mean(pick_place_successes))
        print("Mean Percentage advancement: ", mean(percentage_advancement))
        print("Pick placce success rate: ", valid_pick_place_success/num_valid_pick_place_queries)

        print("Success rate: ", hanoi_successes/(100))
        os.makedirs("results", exist_ok=True)
        # Write the results to a file results_seed_{args.seed}.txt
        with open(f"results/results_hil_dense_{args.demos}_seed_{args.seed}.txt", 'w') as file:
            file.write("Success rate: {}\n".format(hanoi_successes/(100)))
            file.write("Mean Successful pick_place: {}\n".format(mean(pick_place_successes)))
            file.write("Mean Percentage advancement: {}\n".format(mean(percentage_advancement)))
            file.write("Pick placce success rate: {}\n".format(valid_pick_place_success/num_valid_pick_place_queries))


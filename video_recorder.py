# Description: This script is used to record a video of the trained agent performing a task in the environment.
import os
import robosuite as suite
import cv2
import imageio
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import SAC, HerReplayBuffer
from robosuite.wrappers import Wrapper, GymWrapper
from operator_wrapper import ReachWrapper
from HER_wrapper import HERWrapper

class ImageInfoWrapper(Wrapper):
    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        info["img_frames"] = obs_dict['birdview_image']
        return obs_dict, reward, done, info

controller_config = suite.load_controller_config(default_controller='OSC_POSITION')


#algs = ["ltl", "her_symbolic", "rapid_sac", "sac"]
algs = ["ltl"]
#Novelty_list = ['base', 'Elevated', 'Obstacle', 'Hole', 'Locked', 'Lightoff']
Novelty_list = ['Locked']

for alg in algs:
    for novelty in Novelty_list:

        env_nov = novelty if novelty != 'base' else 'PickPlaceCanNovelties'
        # Create the environment
        env = suite.make(
            env_nov,
            robots="Kinova3",
            controller_configs=controller_config,
            has_renderer=False,
            ignore_done=True,
            use_object_obs=True,
            has_offscreen_renderer=True,
            horizon=100000000,
            camera_names="birdview",
            camera_heights=512,
            camera_widths=512,
        )
        env = ImageInfoWrapper(env)
        env = GymWrapper(env, keys=['robot0_proprio-state', 'object-state'])
        env = ReachWrapper(env, dense_reward=True, render_init=False)

        path_nov = novelty if novelty != 'Lightoff' else 'Light'
        path_model = "/home/lorangpi/RoboSuite_RapidHER/operator_learners/models/" + alg + "/" + path_nov + "/best_model.zip"
        if alg == "sac":
            path_model = "/home/lorangpi/RoboSuite_RapidHER/operator_learners/models/sac/best_model.zip"
        if alg == "ltl":
            env = HERWrapper(env, symbolic_goal=False)
            model = SAC.load(path_model, env=env)
        elif alg == "her_symbolic":
            env = HERWrapper(env, symbolic_goal=True)
            model = SAC.load(path_model, env=env)
        else:
            model = SAC.load(path_model)


        path_to_video = "/home/lorangpi/RoboSuite_RapidHER/operator_learners/videos_bird/" + alg + "/" + path_nov + "/"
        os.makedirs(path_to_video, exist_ok=True)

        if env.metadata is None:
            env.metadata = {'render.modes': ['human', 'rgb_array', 'depth_array'], 'video.frames_per_second': 67}

        # Create a VideoWriter object
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
        #out = cv2.VideoWriter(path_to_video + 'output.mp4', fourcc, 20.0, (500, 500))  # adjust the size (500, 500) according to your needs

        try:
            obs, _ = env.reset()
        except:
            obs = env.reset()

        # create a video writer with imageio
        writer = imageio.get_writer(path_to_video+'video.mp4', fps=20)

        for i in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frame = info["img_frames"]
            # Reverse frame
            frame = cv2.flip(frame, 0)
            writer.append_data(frame)
            print("Saving frame #{}".format(i))
            if terminated or truncated or i == 499:
                #out.release()
                env.close()
                break

        writer.close()
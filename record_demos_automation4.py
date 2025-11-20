import os, argparse, time, zipfile, pickle, copy
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from datetime import datetime
import gymnasium as gym
#import gym
import numpy as np
from robosuite.wrappers.behavior_cloning.detector import Robosuite_Hanoi_Detector
from robosuite.wrappers.behavior_cloning.hanoi_pick_place import PickPlaceWrapper
from graph_learner import GraphLearner
from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import joblib
import pandas as pd
cv2.destroyAllWindows = lambda: None
yolo_model = YOLO("PDDL/yolo.pt")

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

#reset_gripper_pos = np.array([-0.14193391, -0.03391656,  1.20828137]) * 1000
reset_gripper_pos = np.array([-0.14193391, -0.03391656,  1.05828137]) * 1000

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class RecordDemos(gym.Wrapper):
    def __init__(self, env, args, randomize=False):
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
        self.gripper_body = self.env.sim.model.body_name2id('gripper0_eef')
        self.obj_to_pick = 'cube1_main'
        self.place_to_drop = 'cube2_main'
        self.count_step = 0
        self.randomize = randomize
        self.map_id_semantic = {
                "blue cube": "cube1",
                "red cube": "cube2",
                "green cube": "cube3",
                "yellow cube": "cube4",
        }
        # Environment parameters
        self.goal_mapping = {'cube1': 0, 'cube2': 1, 'cube3': 2, 'peg1': 3, 'peg2': 4, 'peg3': 5}
        self.obj_mapping = {'cube1': self.cube1_body, 'cube2': self.cube2_body, 'cube3': self.cube3_body, 'peg1': self.peg1_body, 'peg2': self.peg2_body, 'peg3': self.peg3_body}
        self.area_pos = {'peg1': self.env.pegs_xy_center[0], 'peg2': self.env.pegs_xy_center[1], 'peg3': self.env.pegs_xy_center[2]}

        self.ray_bins = {'peg': 0.15}
        self.detector = Robosuite_Hanoi_Detector(self)
        self.render_init = args.render
        self.max_distance = 10
        self.gripper_on = True
        self.picked = []
        self.placed = []
        self.action_steps = []

        # Set up the buffers
        self.symbolic_buffer = list()
        self.data_buffer = dict()

        self.counter_peg = 0
        self.counter_cube = 0
        self.counter_demos = min(self.counter_peg, self.counter_cube)
        self.df_metrics = pd.DataFrame(columns=['wx', 'wy', 'wz', 'pred_x', 'pred_y', 'pred_z', 'error'])
        self.bboxes_centers = []

    def cap(self, eps, max_val=0.07, min_val=0.01):
        # If the displacement is greater than the max value, cap it
        if np.linalg.norm(eps) > max_val:
            eps = eps / np.linalg.norm(eps) * max_val
        # If the displacement is smaller than the min value, cap it
        if np.linalg.norm(eps) < min_val:
            eps = eps / np.linalg.norm(eps) * min_val
        return eps

    def reset_gripper(self):
        self.reset_step_count = 0
        for _ in range(30):
            action = np.array([0, 0, 500, 0])
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render_init else None
            self.reset_step_count += 1
        # Second move to the initial position
        current_pos = self.env.sim.data.body_xpos[self.gripper_body][:3] * 1000
        delta = reset_gripper_pos - current_pos
        action = 5*np.array([delta[0], delta[1], delta[2], 0])
        while np.linalg.norm(delta) > 10:
            #print("Curent pos: ", current_pos)
            action = 5*np.array([delta[0], delta[1], delta[2], 0])
            action = action * 0.9
            next_obs, _, _, _, _  = self.env.step(action)
            self.env.render() if self.render_init else None
            self.reset_step_count += 1
            current_pos = self.env.sim.data.body_xpos[self.gripper_body][:3] * 1000
            delta = reset_gripper_pos - current_pos
        self.reset_step_count = 0
        self.env.time_step = 0

    def pick_reset(self, obs):
        """
        Resets the environment to a state where the gripper is holding the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        self.state_memory = state
        print("\n\n\t\t---------------------PICK GOAL IS: ", self.env.goal)
        print("\n\n")

        goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.env.goal]][:3]
        goal_quat = self.env.sim.data.body_xquat[self.obj_mapping[self.env.goal]]

        self.keypoint = np.concatenate([goal_pos, goal_quat])

        # Moving gripper to the front
        #print("Moving gripper to the front...")
        if not self.args.expert:
            for _ in range(np.random.randint(0, 50)):
                action = np.asarray([0.5,0,0,0])
                action = action * 1000
                next_obs, _, _, _, _  = self.env.step(action)
                self.env.render() if self.render_init else None

            # Moving randomly 0 to 200 steps
            for k in range(np.random.randint(1, 10)):
                generate_random_3d_action = np.random.uniform(-0.5, 0.5, 3)
                action = np.concatenate([generate_random_3d_action, [0]])
                action = action * 1000
                obs,_,_,_,_ = self.env.step(action)
                self.env.render() if self.render_init else None
        else:
            # Reset the gripper to the initial position
            self.reset_gripper()

        self.reset_step_count = 0
        #print("Moving up...")
        #print(len(obs))
        # for _ in range(np.random.randint(4, 15)):
        #     action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([np.random.normal(0, 0.2, 3), [0]])
        #     action = 5*self.cap(action)
        #     action = action * 1000
        #     next_obs, _, _, _, _  = self.env.step(action)
        #     next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        #     self.env.render() if self.render_init else None
        #     self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick")
        #     if self.state_memory is None:
        #         return False, obs
        #     obs, state = next_obs, next_state

        #print("Moving gripper over object...")
        while not state['over(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            #print("Distance to object: ", dist_xy_plan, " Norm: ", np.linalg.norm(dist_xy_plan))
            action = 7*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 7*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, 0.3*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="reach_pick", info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Opening gripper...")
        while not state['open_gripper(gripper)']:
            action = np.asarray([0,0,0,1])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="pick", info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 50:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down gripper to grab level...")
        while not state['at_grab_level(gripper,{})'.format(self.obj_to_pick)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            dist_z_axis = [object_pos[2] - gripper_pos[2]]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, 0.5*np.linalg.norm(dist_z_axis), 1), [0]])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="pick", info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Closing gripper...")
        while not state['grasped({})'.format(self.obj_to_pick)]:
            action = np.asarray([0,0,0,-1])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step="pick", info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        return True, obs

    def drop_reset(self, obs):
        """
        Resets the environment to a state where the gripper is holding the object
        """
        state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        print("\n\n\t\t---------------------DROP GOAL IS: ", self.env.goal)
        save_goal = copy.deepcopy(self.env.goal)
        print("\n\n")

        self.reset_step_count = 0

        action_step1 = "reach_place"
        action_step2 = "place"

        goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.env.goal]][:3]
        if 'peg' in self.env.goal:
            #action_step = "place_peg"
            goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.env.goal]][:3] - np.array([0.1, 0.04, 0])
        else:
            #action_step = "place_cube"
            goal_pos = self.env.sim.data.body_xpos[self.obj_mapping[self.env.goal]][:3]
        goal_quat = self.env.sim.data.body_xquat[self.obj_mapping[self.env.goal]]
        self.keypoint = np.concatenate([goal_pos, goal_quat])

        #print("Lifting object...")
        while not state['picked_up({})'.format(self.obj_to_pick)]:
            action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([np.random.normal(0, 0.1, 3), [0]])
            action = 5*self.cap(action)
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step=action_step1, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving gripper over place to drop...")
        while not state['over(gripper,{})'.format(self.place_to_drop)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            if 'peg' in self.place_to_drop:
                object_pos = self.area_pos[self.place_to_drop]
            else:
                object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]])
            dist_xy_plan = object_pos[:2] - gripper_pos[:2]
            dist_xy_plan = self.cap(dist_xy_plan)
            if self.args.expert:
                action = 7*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 7*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, 0.1*np.linalg.norm(dist_xy_plan), 3), [0]])
            else:            
                action = 7*np.concatenate([dist_xy_plan, [0, 0]]) if not(self.randomize) else 7*np.concatenate([dist_xy_plan, [0, 0]]) + np.concatenate([np.random.normal(0, 0.3*np.linalg.norm(dist_xy_plan), 3), [0]])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step=action_step1, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 200:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving down picked object on place to drop...")
        while not state['on({},{})'.format(self.obj_to_pick, self.place_to_drop)]:
            gripper_pos = np.asarray(self.env.sim.data.body_xpos[self.gripper_body])
            object_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]])
            place_pos = np.asarray(self.env.sim.data.body_xpos[self.obj_mapping[self.place_to_drop]])
            dist_z_axis = [- (object_pos[2] - place_pos[2])]
            dist_z_axis = self.cap(dist_z_axis)
            action = 5*np.concatenate([[0, 0], dist_z_axis, [0]]) if not(self.randomize) else 5*np.concatenate([[0, 0], dist_z_axis, [0]]) + np.concatenate([[0, 0], np.random.normal(0, 0.5*np.linalg.norm(dist_z_axis), 1), [0]])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step=action_step2, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 100:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("dropping object...")
        while not(state['open_gripper(gripper)']):#state['grasped({})'.format(self.obj_to_pick)]:
            action = np.asarray([0,0,0,1])
            action = action * 1000
            next_obs, _, _, _, info  = self.env.step(action)
            self.env.render() if self.render_init else None
            next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
            self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step=action_step2, info=info)
            if self.state_memory is None:
                return False, obs
            obs, state = next_obs, next_state
            self.reset_step_count += 1
            if self.reset_step_count > 30:
                return False, obs
        self.reset_step_count = 0
        self.env.time_step = 0

        #print("Moving up...")
        # for _ in range(np.random.randint(4, 15)):
        #     action = np.asarray([0,0,0.4,0]) if not(self.randomize) else [0,0,0.5,0] + np.concatenate([[0, 0], np.random.normal(0, 0.2, 1), [0]])
        #     action = 5*self.cap(action)
        #     action = action * 1000
        #     next_obs, _, _, _, _  = self.env.step(action)
        #     next_state = self.detector.get_groundings(as_dict=True, binary_to_float=False, return_distance=False)
        #     self.env.render() if self.render_init else None
        #     self.state_memory = self.record_demos(obs, action, next_obs, self.state_memory, next_state, action_step=action_step2)
        #     if self.state_memory is None:
        #         return False, obs
        #     obs, state = next_obs, next_state

        return True, obs

    def reset(self, seed=None):
        # Reset the environment
        self.episode_buffer = dict() # 1 episode here consists of a trajectory between 2 symbolic nodes
        self.symbolic_buffer = list()
        # if self.args.expert:
        #     self.place_to_drop = "peg1"
        #     while self.place_to_drop == "peg1":
        #         try:
        #             obs, _ = self.env.reset()
        #         except:
        #             obs = self.env.reset()
        #         self.sample_task()
        #else:
        try:
            obs, _ = self.env.reset()
        except:
            obs = self.env.reset()
        self.sample_task()
        self.sim.forward()
        return obs

    def save_csv_yolo(self, output_path="yolo_data.csv"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not self.bboxes_centers:
            print("No bounding boxes data to save.")
            return
        pd.DataFrame(self.bboxes_centers).to_csv(output_path, index=False)
        print(f"YOLO data saved at {output_path}")

    def next_episode(self):
        # Data buffer saves a tuple of (trajectory[obs, action, next_obs, done, reward], symbolic trajectory[state, "MOVE", next_state], task)
        for step in self.action_steps:
            if step in self.episode_buffer.keys():
                if step not in self.data_buffer.keys():
                    self.data_buffer[step] = [(self.episode_buffer[step], self.symbolic_buffer, "on({},{})".format(self.obj_to_pick, self.place_to_drop))]
                else:
                    self.data_buffer[step].append((self.episode_buffer[step], self.symbolic_buffer, "on({},{})".format(self.obj_to_pick, self.place_to_drop)))
                if 'peg' in step:
                    self.counter_peg += 1
                elif 'cube' in step:
                    self.counter_cube += 1
        self.counter_demos = min(self.counter_peg, self.counter_cube)
        self.save_buffer(self.data_buffer, self.args.traces)
        episode_nmb = len(list(self.data_buffer.keys()))
        self.save_csv_yolo(output_path=os.path.join(self.args.yolo_data, "yolo_data_{}.csv".format(episode_nmb)))
        obs = self.reset()
        return obs

    def step_episode(self, obs):
        done_pick, obs = self.pick_reset(obs)
        if not(done_pick):
            return done_pick
        done_drop, obs = self.drop_reset(obs)
        return done_drop

    def obs_mapping(self, obs, action_step="trace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3)}
        trace_obs_list = ["gripper_pos", "aperture", "place_to_drop_pos"]
        reach_pick_obs_list = ["gripper_pos"]
        pick_obs_list = ["gripper_z", "aperture"]
        reach_drop_obs_list = ["gripper_pos"]
        drop_obs_list = ["gripper_z", "aperture"]

        oracle = np.array([])
        if action_step == "trace":
            for key in trace_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "reach_pick":
            for key in reach_pick_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "pick":
            for key in pick_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "reach_place":
            for key in reach_drop_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        elif action_step == "place":
            for key in drop_obs_list:
                oracle = np.concatenate([oracle, obs[index_obs[key][0]:index_obs[key][1]]])
        return oracle

    def relative_obs_mapping(self, obs, action_step="trace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (9,10), "place_to_drop_z": (6,7)}
        # trace_obs_list = obj_to_pick_pos - gripper_pos, aperture, place_to_drop_pos - gripper_pos
        # reach_pick_obs_list = obj_to_pick_pos - gripper_pos
        # pick_obs_list = obj_to_pick_z - gripper_z, aperture
        # reach_drop_obs_list = place_to_drop_pos - gripper_pos
        # drop_obs_list = place_to_drop_z - gripper_z, aperture

        oracle = np.array([])
        if action_step == "trace":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "reach_pick":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "pick":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        elif action_step == "reach_place":
            oracle = np.concatenate([obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "place":
            oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        else:
            oracle = obs
        return oracle
    
    def keypoint_mapping(self, obs, action_step="trace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (9,10), "place_to_drop_z": (6,7)}
        trace_key = "obj_to_pick_pos"
        reach_pick_key = "obj_to_pick_pos"
        pick_key = "obj_to_pick_z"
        reach_drop_key = "place_to_drop_pos"
        drop_key = "place_to_drop_z"

        if action_step == "trace":
            keypoint = obs[index_obs[trace_key][0]:index_obs[trace_key][1]]
        elif action_step == "reach_pick":
            keypoint = obs[index_obs[reach_pick_key][0]:index_obs[reach_pick_key][1]]
        elif action_step == "pick":
            keypoint = obs[index_obs[pick_key][0]:index_obs[pick_key][1]]
        elif action_step == "reach_place":
            keypoint = obs[index_obs[reach_drop_key][0]:index_obs[reach_drop_key][1]]
        elif action_step == "place":
            keypoint = obs[index_obs[drop_key][0]:index_obs[drop_key][1]]
        return keypoint

    def pixel_to_world(self, px, py):
        # Load linear Regression models for cube positions
        models = joblib.load("calibration_models.pkl")
        #models = joblib.load("filtered_calibration_models0.pkl")
        reg_x, reg_y, reg_z = models["reg_x"], models["reg_y"], models["reg_z"]

        vec = np.array([px, py, 1.0])
        x = reg_x.predict([vec])[0]  # Subtract a small offset to the x coordinate
        y = reg_y.predict([vec])[0] # Subtract a small offset to the y coordinate
        z = reg_z.predict([vec])[0]  # Add a small offset to the z coordinate
        return x*1000., y*1000., z *1000.

    def pixel_to_world_dual(self, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z):
        # Load linear Regression models for cube positions
        models_dual = joblib.load("dual_cam_calibration_models.pkl")
        reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]

        features = np.array([[px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z]])
        x = reg_x_dual.predict(features)[0]
        y = reg_y_dual.predict(features)[0]
        z = reg_z_dual.predict(features)[0]
        return x*1000., y*1000., z*1000.

    def yolo_estimate(self, image1, image2=None, ee_pos=None, cubes_obs=None):
        # Resize the image to fit YOLO input requirements

        cubes_predicted_xyz = {}
        try:
            image1 = cv2.resize(image1, (256, 256))
        except Exception as e:
            print("Error resizing image: ", e, image1.shape, image1.dtype)
        if image2 is not None:
            try:
                image2 = cv2.resize(image2, (256, 256))
            except Exception as e:
                print("Error resizing image2: ", e, image2.shape, image2.dtype)
            # Concatenate the two images side by side
        # Mirror the image (top to bottom)
        image1 = cv2.flip(image1, 0)
        # Convert the image to BGR format if it is not already
        image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2BGR)
        # Run YOLO model on the image
        predictions1 = yolo_model.predict(image1, verbose=False)[0]
        if image2 is not None:
            image2 = cv2.flip(image2, 0)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            predictions2 = yolo_model.predict(image2, verbose=False)[0]

        # Draw bounding boxes from Roboflow JSON
        for pred in predictions1.boxes:#.get("predictions", []):
            # x, y = pred["x"], pred["y"]
            # w, h = pred["width"], pred["height"]
            # conf, cls = pred["confidence"], pred["class"]

            cls_id = int(pred.cls)
            cls = yolo_model.names[cls_id]
            x, y, w, h = pred.xywhn.tolist()[0]
            conf = pred.conf
            # Convert normalized coordinates to pixel coordinates
            x = int(x * image1.shape[1])
            y = int(y * image1.shape[0])
            w = int(w * image1.shape[1])
            h = int(h * image1.shape[0])

            ground_truth_xyz = cubes_obs[self.map_id_semantic[cls]]
            if image2 is not None:
                found_match = False
                for pred in predictions2.boxes:
                    cls_id2 = int(pred.cls)
                    if cls_id2 == cls_id:
                        found_match = True
                        x_cam2, y_cam2, w_cam2, h_cam2 = pred.xywhn.tolist()[0]
                        conf_cam2 = pred.conf
                        # Convert normalized coordinates to pixel coordinates
                        x_cam2 = int(x_cam2 * image2.shape[1])
                        y_cam2 = int(y_cam2 * image2.shape[0])
                        w_cam2 = int(w_cam2 * image2.shape[1])
                        h_cam2 = int(h_cam2 * image2.shape[0])
                        self.bboxes_centers.append({
                            "px_cam1": x,
                            "py_cam1": y,
                            "w_cam1": w,
                            "h_cam1": h,
                            "conf_cam1": float(conf),
                            "cls": cls,
                            "px_cam2": x_cam2,
                            "py_cam2": y_cam2,
                            "w_cam2": w_cam2,
                            "h_cam2": h_cam2,
                            "conf_cam2": float(conf_cam2),
                            "ee_x": ee_pos[0] if ee_pos is not None else None,
                            "ee_y": ee_pos[1] if ee_pos is not None else None,
                            "ee_z": ee_pos[2] if ee_pos is not None else None,
                            "world_x": ground_truth_xyz[0],
                            "world_y": ground_truth_xyz[1],
                            "world_z": ground_truth_xyz[2],
                        })
                if not found_match:
                    x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2 = 0, 0, 0, 0, 0
                    self.bboxes_centers.append({
                                "px_cam1": x,
                                "py_cam1": y,
                                "w_cam1": w,
                                "h_cam1": h,
                                "conf_cam1": float(conf),
                                "cls": cls,
                                "px_cam2": 0,
                                "py_cam2": 0,
                                "w_cam2": 0,
                                "h_cam2": 0,
                                "conf_cam2": 0,
                                "ee_x": ee_pos[0] if ee_pos is not None else None,
                                "ee_y": ee_pos[1] if ee_pos is not None else None,
                                "ee_z": ee_pos[2] if ee_pos is not None else None,
                                "world_x": ground_truth_xyz[0],
                                "world_y": ground_truth_xyz[1],
                                "world_z": ground_truth_xyz[2],
                            })
                # Use dual camera regression to estimate the position
                predicted_xyz = self.pixel_to_world_dual(x, y, w, h, conf, x_cam2, y_cam2, w_cam2, h_cam2, conf_cam2, ee_pos[0], ee_pos[1], ee_pos[2])
            else:
                # Use single camera regression to estimate the position
                predicted_xyz = self.pixel_to_world(x, y)
            cubes_predicted_xyz.update({self.map_id_semantic[cls]: predicted_xyz})
            # Print the predicted and ground truth positions
            #print(f"Predicted: {predicted_xyz}, Ground Truth: {ground_truth_xyz}", "Error: ", np.linalg.norm(np.array(predicted_xyz) - np.array(ground_truth_xyz)))

        return cubes_predicted_xyz

    def record_demos(self, obs, action, next_obs, state_memory, new_state, sym_action="MOVE", action_step="trace", reward=-1.0, done=False, info=None):
        # keypoint = last 3 values of obs
        if not(self.args.split_action):
            action_step = 'trace'
        keypoint = self.keypoint_mapping(obs, action_step)
        #print("Key point: ", keypoint, " Obs: ", obs)
        if self.args.use_yolo:
            image_agentview = info["agentview"]
            roboteye_images = info["robot0_eye_in_hand"]
            ee_pos = info["ee_pos"]
            cubes_obs = info["cubes_obs"]
            cubes_predicted_xyz = self.yolo_estimate(image_agentview, roboteye_images, ee_pos, cubes_obs)
            if self.obj_to_pick in cubes_predicted_xyz.keys():
                predicted_pos_to_pick = cubes_predicted_xyz[self.obj_to_pick]
                world_pos_to_pick = copy.deepcopy(obs[7:10])
                obs[7:10] = np.array(predicted_pos_to_pick)
                self.df_metrics = pd.concat(
                    [
                        self.df_metrics,
                        pd.DataFrame([{
                            'wx': world_pos_to_pick[0],
                            'wy': world_pos_to_pick[1],
                            'wz': world_pos_to_pick[2],
                            'pred_x': predicted_pos_to_pick[0],
                            'pred_y': predicted_pos_to_pick[1],
                            'pred_z': predicted_pos_to_pick[2],
                            'error': np.linalg.norm(np.array(predicted_pos_to_pick) - np.array(world_pos_to_pick))
                        }])
                    ],
                    ignore_index=True
                )
                self.df_metrics["diff_x"] = self.df_metrics["wx"] - self.df_metrics["pred_x"]
                self.df_metrics["diff_y"] = self.df_metrics["wy"] - self.df_metrics["pred_y"]
                self.df_metrics["diff_z"] = self.df_metrics["wz"] - self.df_metrics["pred_z"]
            if self.place_to_drop in cubes_predicted_xyz.keys():
                predicted_pos_to_drop = cubes_predicted_xyz[self.place_to_drop]
                world_pos_to_drop = copy.deepcopy(obs[4:7])
                obs[4:7] = np.array(predicted_pos_to_drop)
                # update self.df_metrics
                self.df_metrics = pd.concat(
                    [
                        self.df_metrics,
                        pd.DataFrame([{
                            'wx': world_pos_to_drop[0],
                            'wy': world_pos_to_drop[1],
                            'wz': world_pos_to_drop[2],
                            'px': predicted_pos_to_drop[0],
                            'py': predicted_pos_to_drop[1],
                            'pz': predicted_pos_to_drop[2],
                            'error': np.linalg.norm(np.array(predicted_pos_to_drop) - np.array(world_pos_to_drop))
                        }])
                    ],
                    ignore_index=True
                )
                self.df_metrics["diff_x"] = self.df_metrics["wx"] - self.df_metrics["pred_x"]
                self.df_metrics["diff_y"] = self.df_metrics["wy"] - self.df_metrics["pred_y"]
                self.df_metrics["diff_z"] = self.df_metrics["wz"] - self.df_metrics["pred_z"]
        obs = self.relative_obs_mapping(obs, action_step)
        #print("Action step: ", action_step, "Obs shape: ", obs.shape, "Key point shape: ", keypoint.shape)
        if self.args.goal_env:
            desired_goal = self.env.sim.data.body_xpos[self.obj_mapping[self.obj_to_pick]][:3]
            achieved_goal = self.env.sim.data.body_xpos[self.gripper_body][:3]
            transition = (obs, action, next_obs, reward, done, desired_goal, achieved_goal)
        elif self.args.keypoint:
            transition = (obs, action, next_obs, keypoint, reward, done)
        else:
            #print(len(obs), len(action), len(next_obs), reward, done)
            transition = (obs, action, next_obs, reward, done)
        #if obs.shape[0] != 15:
            #print("Obs shape: ", obs.shape)
        if action_step not in self.action_steps:
            self.action_steps.append(action_step)
        if action_step not in self.episode_buffer.keys():
            self.episode_buffer[action_step] = [transition]
        else:
            self.episode_buffer[action_step].append(transition)
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
                if (len(state) != 3 or len(new_state) != 3) and not(self.args.expert):
                    return None
                # Check if cubes have fallen from other subes, i.e., check if two or more cubes are on the same peg
                for test_state in [state, new_state]:
                    pegs = []
                    for relation, value in test_state.items():
                        _, peg = relation.split('(')[1].split(',')
                        pegs.append(peg)
                    if len(pegs) != len(set(pegs)) and not(self.args.expert):
                        return None
                if self.args.unique:
                    # Check if the transition is unique
                    if self.Graph.is_known_edge(state, sym_action, new_state):
                        print("Already known transition")
                        return None
                    else:
                        print("NEW UNIQUE TRANSITION")
                        print(env.Graph.state_mapping)
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
    parser.add_argument('--data_folder', type=str, default='./data/', help='Path to the data folder')
    parser.add_argument('--episodes', type=int, default=int(200), help='Number of episodes to train for')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--name', type=str, default=None, help='Name of the experiment')
    parser.add_argument('--render', action='store_true', help='Render the initial state')
    parser.add_argument('--split_action', action='store_true', help='Split the MOVE action into reach_pick, pick, reach_drop, drop')
    parser.add_argument('--goal_env', action='store_true', help='Use the goal environment')
    parser.add_argument('--keypoint', action='store_true', help='Store the keypoint')
    parser.add_argument('--unique', action='store_true', help='Unique transitions, 27 possible transitions')
    parser.add_argument('--expert', action='store_true', help='Expert mode')
    parser.add_argument('--use_yolo', action='store_true', help='Use YOLO to detect object positions')

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
    args.yolo_data = args.experiment_dir + '/yolo_data/'
    os.makedirs(args.experiment_dir, exist_ok=True)
    os.makedirs(args.graphs, exist_ok=True)
    os.makedirs(args.pddl, exist_ok=True)
    os.makedirs(args.traces, exist_ok=True)

    # Load the controller config
    controller_config = suite.load_controller_config(default_controller='OSC_POSITION')
    # Create the environment
    env = suite.make(
        "Hanoi",
        robots="Panda",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=True,
        horizon=100000000,
        use_camera_obs=True,
        use_object_obs=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        render_camera="robot0_eye_in_hand",#"robot0_eye_in_hand", # Available "camera" names = ('frontview', 'birdview', 'agentview', 'robot0_robotview', 'robot0_eye_in_hand')
        random_block_placement=True,
    )

    # Wrap the environment
    env = GymWrapper(env)
    env = PickPlaceWrapper(env)
    env = RecordDemos(env, args)
    # Reset the environment
    try:
        obs, _ = env.reset()
    except:
        obs = env.reset()
    # Run the environment
    done = False
    num_recorder_eps = 0
    episode = 1
    while num_recorder_eps < args.episodes: 
        print("Episode: {}".format(episode+1))
        done = env.step_episode(obs)
        if done:
            print("Saving episode {}...".format(episode))
            obs = env.next_episode()
        else:
            try:
                obs, _ = env.reset()
            except:
                obs = env.reset()
        done = False
        # if episode % 10 == 0 and episode > 0:
        #     try:
        #         obs, _ = env.reset()
        #     except:
        #         obs = env.reset()
        if episode % 27 == 0:
            print("\n Graph mapping: ", env.Graph.state_mapping)
        episode += 1
        keys = list(env.data_buffer.keys())
        if len(keys) > 0:
            num_recorder_eps = len(env.data_buffer[keys[0]])
            print("Number of recorded episodes: {}".format(num_recorder_eps))
            if args.use_yolo:
                print("Metric df: \n", env.df_metrics.describe())
                # Compute the error metrics
                mean_error_x = env.df_metrics["diff_x"].mean()
                mean_error_y = env.df_metrics["diff_y"].mean()
                mean_error_z = env.df_metrics["diff_z"].mean()
                std_error_x = env.df_metrics["diff_x"].std()
                std_error_y = env.df_metrics["diff_y"].std()
                std_error_z = env.df_metrics["diff_z"].std()
                print(f"Mean Error X: {mean_error_x}, Std Error X: {std_error_x}")
                print(f"Mean Error Y: {mean_error_y}, Std Error Y: {std_error_y}")
                print(f"Mean Error Z: {mean_error_z}, Std Error Z: {std_error_z}")
                print(f"Overall Mean Error: {(mean_error_x**2 + mean_error_y**2 + mean_error_z**2)**0.5}")
                print("\n\n")
    print("\n Graph mapping: ", env.Graph.state_mapping)
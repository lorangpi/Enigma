'''
# authors: Pierrick Lorang
# email: pierrick.lorang@tufts.edu

# This files implements the structure of the executor object used in this paper.

'''
import dill
import torch
import hydra
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.utils import set_random_seed
from omegaconf import DictConfig
from imitation_learning.environments import D4RLEnv
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace import TrainDiffusionTransformerLowdimWorkspace
import copy

# yolo imports
from ultralytics import YOLO
from roboflow import Roboflow
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import joblib
cv2.destroyAllWindows = lambda: None

# rf = Roboflow(api_key="tgB2r9aEPNuMRi0qB1cl")
# project = rf.workspace("cyclicproject").project("hannoi-cubes-agnft")
# version = project.version(2)
# yolo_model = version.model
yolo_model = YOLO("PDDL/yolo_dual_cam.pt")

set_random_seed(0, using_cuda=True)

class Executor():
	def __init__(self, id, mode, I=None, Beta=None, Circumstance=None, basic=False):
		super().__init__()
		self.id = id
		self.I = I
		self.Circumstance = Circumstance
		self.Beta = Beta
		self.basic = basic
		self.mode = mode
		self.policy = None

	def path_to_json(self):
		return {self.id:self.policy}

class Executor_RL(Executor):
    def __init__(self, id, alg, policy, I, Beta, Circumstance=None, basic=False, nulified_action_indexes=[], wrapper=None, horizon=None):
        super().__init__(id, "RL", I, Beta, Circumstance, basic)
        self.alg = alg
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.wrapper = wrapper
        self.horizon = horizon

    def execute(self, env, obs, goal, symgoal, render=False):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        horizon = self.horizon if self.horizon is not None else 500
        dummy_env = self.wrapper(env, nulified_action_indexes=self.nulified_action_indexes, horizon=horizon) if self.wrapper is not None else env
        print("\tTask goal: ", symgoal)
        print("\tLoading policy {}".format(self.policy))
        print("\tNumber of nulified indexes: ", len(self.nulified_action_indexes))
        print("\tAction space: ", dummy_env.action_space)
        if self.model is None:
            self.model = self.alg.load(self.policy, 
                                       env=dummy_env,
                                       custom_objects={'observation_space': dummy_env.observation_space, 
                                                       'action_space': dummy_env.action_space,
                                                       #'replay_buffer_class': None,
                                                       })
        step_executor = 0
        done = False
        success = False
        while not done:
            if goal is not None:
                #print("\tLow level goal: ", goal)
                #goal_copy = copy.deepcopy(goal)
                goal_copy = np.copy(goal)
                obs = np.concatenate((obs, goal_copy))
                #print("\tObservation shape: ", obs.shape)
                #print("\tObservation: ", obs)
            action, _states = self.model.predict(obs)
            #print("Input action: ", action)
            # if self.nulified_action_indexes is not empty, fill the action with zeros at the indexes
            if self.nulified_action_indexes != []:
                for index in self.nulified_action_indexes:
                    action = np.insert(action, index, 0)
            #print("Transformed action: ", action)        
            try: 
                obs, reward, terminated, truncated, info = env.step(action)
                #print(obs.shape)
                done = terminated or truncated
            except:
                obs, reward, done, info = env.step(action)
            step_executor += 1
            success = self.Beta(env, symgoal)
            if success:
                print("\tSuccess: Task completed in {} steps\n".format(step_executor))
                break
            done = success
            if step_executor > 500:
                done = True
            if render:
                env.render()
        return obs, success
    
class Executor_Diffusion(Executor):
    def __init__(self, 
                 id, 
                 policy, 
                 I, 
                 Beta, 
                 count=0,
                 Circumstance=None, 
                 basic=False, 
                 nulified_action_indexes=[], 
                 oracle=False, 
                 wrapper=None, 
                 horizon=None, 
                 use_yolo=False, 
                 save_data=False,
                 tracked_positions={}
                 ):
        super().__init__(id, "RL", I, Beta, Circumstance, basic)
        self.policy = policy
        self.model = None
        self.nulified_action_indexes = nulified_action_indexes
        self.wrapper = wrapper
        self.horizon = horizon
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.oracle = oracle
        self.use_yolo = use_yolo
        self.save_data = save_data
        self.image_buffer = []
        self.map_id_semantic = {
                "blue cube": "cube1",
                "red cube": "cube2",
                "green cube": "cube3",
                "yellow cube": "cube4",
        }
        self.tracked_positions = tracked_positions
        self.detected_positions = {}
        # Store bboxes centers and groundtruth positions of cubes
        self.bboxes_centers = []
        self.count = count

    def load_policy(self):
        path = self.policy
        # load checkpoint
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        #cls = hydra.utils.get_class(cfg._target_)
        #target = "diffusion_policy.diffusion_policy.workspace.train_diffusion_transformer_lowdim_workspace.TrainDiffusionTransformerLowdimWorkspace"
        #cls = hydra.utils.get_class(target)
        cls = TrainDiffusionTransformerLowdimWorkspace
        cfg.policy.num_inference_steps = 8
        #workspace = cls(cfg, output_dir="../data/")
        workspace = cls(cfg)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        #device = torch.device(self.device)
        policy.to(self.device)
        policy.eval()
        policy.reset()
        self.model = policy

    def pixel_to_world_dual(self, px1, py1, w1, h1, conf1, px2, py2, w2, h2, conf2, ee_x, ee_y, ee_z):
        # Load linear Regression models for cube positions
        models_dual = joblib.load("dual_cam_calibration_models.pkl")
        reg_x_dual, reg_y_dual, reg_z_dual = models_dual["reg_x"], models_dual["reg_y"], models_dual["reg_z"]

        features = np.array([[
            float(px1), float(py1), float(w1), float(h1), float(conf1),
            float(px2), float(py2), float(w2), float(h2), float(conf2),
            float(ee_x), float(ee_y), float(ee_z)
        ]])
        x = reg_x_dual.predict(features)[0]
        y = reg_y_dual.predict(features)[0]
        z = reg_z_dual.predict(features)[0]
        return x*1000., y*1000., z*1000.
    
    def pixel_to_world(self, px, py):
        # Load linear Regression models for cube positions
        models = joblib.load("calibration_models.pkl")
        #models = joblib.load("filtered_calibration_models0.pkl")
        reg_x, reg_y, reg_z = models["reg_x"], models["reg_y"], models["reg_z"]

        vec = np.array([px, py, 1.0])
        wx = reg_x.predict([vec])[0]  # Subtract a small offset to the x coordinate
        wy = reg_y.predict([vec])[0] # Subtract a small offset to the y coordinate
        wz = reg_z.predict([vec])[0]  # Add a small offset to the z coordinate
        return wx, wy, wz 
    
    # def pixel_to_world(self, px, py, w, h):

    #     # --- Load models ---
    #     models = joblib.load("poly_models.pkl")
    #     reg_x, reg_y, reg_z = models["reg_x"], models["reg_y"], models["reg_z"]
    #     features = np.array([[px, py, w, h]])
    #     x = reg_x.predict(features)[0]
    #     y = reg_y.predict(features)[0]
    #     z = reg_z.predict(features)[0]
    #     return x, y, z

    def yolo_estimate(self, image1, image2=None, save_video=False, cubes_obs=None, ee_pos=None):
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
        predictions1 = yolo_model.predict(image1, verbose=False, device="cpu")[0]
        if image2 is not None:
            image2 = cv2.flip(image2, 0)
            image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
            predictions2 = yolo_model.predict(image2, verbose=False, device="cpu")[0]
        #.json()#(image1, confidence=40, overlap=30).json()
        #print("Predictions: ", predictions)

        # Ensure ndarray
        if not isinstance(image1, np.ndarray):
            image1 = np.array(image1)

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

            # Convert to pixel coordinates
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)


            if self.map_id_semantic[cls] == "cube4":
                continue
            # Get the ground truth position of the cube
            #print("Cube obs keys: ", cubes_obs.keys())
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

            # if self.map_id_semantic[cls] == "cube2":
            #     predicted_xyz +=  np.asarray([-10,-10,0])
            #predicted_xyz = [predicted_xyz[0], predicted_xyz[1]*0.95, predicted_xyz[2]]
            self.detected_positions.update({self.map_id_semantic[cls]: predicted_xyz})
            cubes_predicted_xyz.update({self.map_id_semantic[cls]: predicted_xyz})

            # Print the predicted and ground truth positions
            #print(f"Predicted: {predicted_xyz}, Ground Truth: {ground_truth_xyz*1000.}", "Error: ", np.linalg.norm(np.array(predicted_xyz) - np.array(ground_truth_xyz*1000.)))


            if save_video:
                # Draw box + label
                cv2.rectangle(image1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image1, f"{cls}:{float(conf):.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if save_video:
            # Append to buffer for video saving
            if not hasattr(self, "image_buffer"):
                self.image_buffer = []

            #print("Image shape:", image1.shape, "dtype:", image1.dtype)
            self.image_buffer.append(image1.copy())

        return cubes_predicted_xyz

    def save_video(self, output_path="output.mp4", fps=10):
        if not self.image_buffer:
            print("No frames to save.")
            return
        
        height, width, _ = self.image_buffer[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in self.image_buffer:
            out.write(frame)

        out.release()
        print(f"Video saved at {output_path}")

    def save_csv_yolo(self, output_path="yolo_data.csv"):
        import pandas as pd
        if not self.bboxes_centers:
            print("No bounding boxes data to save.")
            return
        
        pd.DataFrame(self.bboxes_centers).to_csv(output_path, index=False)
        print(f"YOLO data saved at {output_path}")

    def relative_obs_mapping(self, obs, action_step="PickPlace"):
        index_obs = {"gripper_pos": (0,3), "aperture": (3,4), "place_to_drop_pos": (4,7), "obj_to_pick_pos": (7,10), "gripper_z": (2,3), "obj_to_pick_z": (9,10), "place_to_drop_z": (6,7)}
        # trace_obs_list = obj_to_pick_pos - gripper_pos, aperture, place_to_drop_pos - gripper_pos
        # reach_pick_obs_list = obj_to_pick_pos - gripper_pos
        # pick_obs_list = obj_to_pick_z - gripper_z, aperture
        # reach_drop_obs_list = place_to_drop_pos - gripper_pos
        # drop_obs_list = place_to_drop_z - gripper_z, aperture

        # Add 0.06m to the aperture (difference between kinova and panda)
        #obs[index_obs["aperture"][0]] += 60
        #obs[index_obs["aperture"][0]] += 60

        oracle = np.array([])
        if action_step == "PickPlace":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]], obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])
        elif action_step == "ReachPick":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_pos"][0]:index_obs["obj_to_pick_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]] ])# + [-10,-10,0]])
        elif action_step == "Grasp":
            oracle = np.concatenate([obs[index_obs["obj_to_pick_z"][0]:index_obs["obj_to_pick_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]]+18, obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        elif action_step == "ReachDrop":
            oracle = np.concatenate([obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]])# +[+0,25,0]]) #[-10,5,0]]) #best, 2nd
            oracle = np.concatenate([obs[index_obs["place_to_drop_pos"][0]:index_obs["place_to_drop_pos"][1]] - obs[index_obs["gripper_pos"][0]:index_obs["gripper_pos"][1]]+[+0,12,0]]) #[-10,5,0]]) #best, 2nd
        elif action_step == "Drop":
            oracle = np.concatenate([obs[index_obs["place_to_drop_z"][0]:index_obs["place_to_drop_z"][1]] - obs[index_obs["gripper_z"][0]:index_obs["gripper_z"][1]], obs[index_obs["aperture"][0]:index_obs["aperture"][1]]])
        else:
            oracle = obs
        return oracle

    def prepare_obs(self, obs, action_step="PickPlace"):
        #obs_dim = {"PickPlace": 10, "ReachPick": 6, "Grasp": 3, "ReachDrop": 6, "Drop": 3}
        obs_dim = {"PickPlace": 7, "ReachPick": 3, "Grasp": 2, "ReachDrop": 3, "Drop": 2}
        if action_step not in obs_dim.keys():
            return obs
        returned_obs = np.zeros((obs.shape[0], len(obs[0]), obs_dim[action_step]))
        for j, env_n_obs in enumerate(obs):
            for i in range(len(env_n_obs)):
                obs_step = env_n_obs[i]
                # Prepare the observation for the policy
                obs_policy = self.relative_obs_mapping(obs_step, action_step=action_step)
                #keypoint_policy = self.keypoint_mapping(obs_step, action_step=action_step)
                #concatenated_obs = np.concatenate([keypoint_policy, obs_policy], axis=-1)
                
                # Resize env_n_obs[i] to match the new shape
                #returned_obs[j][i] = concatenated_obs
                returned_obs[j][i] = obs_policy
        #print("Returned obs shape: ", returned_obs.shape)
        #print("Original obs shape: ", obs.shape)
        return returned_obs

    def insert_yolo_estimate(self, obs, cubes_xyz, obj_to_pick, place_to_drop):
        # Insert the yolo estimated positions in the obs
        # cubes_obs is a dictionary of shape {cube_id: [x, y, z]}

        for key in self.tracked_positions.keys():
                # try:
                #     z_position = cubes_xyz[key][2]
                #     #y_position = cubes_xyz[key][1]
                # except:
                #     pass
                #print("Using tracked position for cube: ", key, " position: ", self.tracked_positions[key])
                cubes_xyz[key] = self.tracked_positions[key]
                # try:
                #     cubes_xyz[key][2] = z_position
                #     #cubes_xyz[key][1] = y_position
                # except:
                #     pass

        if obj_to_pick in cubes_xyz:
            obj_to_pick_xyz = cubes_xyz[obj_to_pick]
            obs[7:10] = np.asarray(obj_to_pick_xyz) #+ np.array([0.0, 0.0, 20])
            obs[7:10] = np.asarray(obj_to_pick_xyz) #+ np.array([0.0, 0.0, 20])
        if place_to_drop in cubes_xyz:
            place_to_drop_xyz = cubes_xyz[place_to_drop]
            obs[4:7] = np.asarray(place_to_drop_xyz)#*1000.0
        return obs
                    
    def obs_base_from_info(self, info):
        obs_base = []
        for i in range(len(info)):
            obs_base.append(info[i]["obs_base"])
        return np.array(obs_base)

    def image_from_info(self, info, camera="agentview"):
        images = []
        for i in range(len(info)):
            if camera in info[i]:
                images.append(info[i][camera])
        return np.array(images)
    
    def track_positions(self, state, ee_pos, obs, symgoal):
        # Track the position of the cubes, if the cube is grasped its position is the ee position
        #print("State: ", state)
        for relation, value in state.items():
            if 'grasped' in relation and value:
                # Get the cube id
                cube_id = relation.split('(')[1].split(',')[0].split(')')[0]
                #print("Cube {} is grasped".format(cube_id))
                # Get the ee position
                # Update the tracked position of the cube
                ee_pos += np.array([0.00, 0.025, 0.00]) # Add an offset to the z position
                self.tracked_positions[cube_id] = np.asarray(ee_pos)*1000.0
                ee_pos += np.array([0.00, 0.025, 0.00]) # Add an offset to the z position
                self.tracked_positions[cube_id] = np.asarray(ee_pos)*1000.0
                if cube_id == symgoal[0]:
                    # If the cube is the one to pick, update the obs
                    obs[7:10] = np.asarray(ee_pos)*1000.0
        return obs


    def valid_state_f(self, state):
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

    def execute(self, env, obs, goal, symgoal, render=False, info = {}, setting="3x3"):
        '''
        This method is responsible for executing the policy on the given state. It takes a state as a parameter and returns the action 
        produced by the policy on that state. 
        '''
        self.image_buffer = []
        self.detected_positions = {}
        self.tracked_positions = {}
        horizon = self.horizon if self.horizon is not None else 500
        print("\tTask goal: ", symgoal)

        obs_base = False

        if isinstance(obs, np.ndarray):
            obs_base = np.any(obs == None)
        else:
            obs_base = obs == None

        step_executor = 0
        done = False
        success = False 
        while not done:
            # Prepare the observation for the policy
            if self.use_yolo or self.save_data:
                agentview_images = self.image_from_info(info, camera="agentview")
                roboteye_images = self.image_from_info(info, camera="robot0_eye_in_hand")
                obs_copy = np.copy(obs)
                if len(agentview_images) > 0:
                    #print("Number of images sequences: ", len(images))
                    for ni, image in enumerate(agentview_images[0]):
                        cubes_obs = info[0]['cubes_obs'][ni]
                        # Get the ground truth position of the cube
                        ground_truth_xyz = cubes_obs
                        ee_pos = info[0]['ee_pos'][ni]
                        #print("Image shape: ", image.shape)
                        if len(self.detected_positions) >= 3 and not(self.save_data) and self.id in ["Grasp", "Drop"]:
                            # If the detected positions are already set, skip the yolo estimation
                            cubes_xyz = copy.deepcopy(self.detected_positions)
                            #print("Using detected positions: ", cubes_xyz, "compared to ground truth: ", ground_truth_xyz)
                        else:
                            cubes_xyz = self.yolo_estimate(image1 = np.array(image), 
                                                           image2=np.array(roboteye_images[0][ni]), 
                                                           save_video=self.save_data, 
                                                           cubes_obs=cubes_obs,
                                                              ee_pos=ee_pos)
                        if len(self.tracked_positions) >= 3:
                            # If the tracked positions are already set, skip the yolo estimation
                            # Get z positions of the cubes from yolo estimations first
                            #z_positions = {}
                            #y_positions = {}
                            #for key in cubes_xyz.keys():
                                #z_positions[key] = cubes_xyz[key][2]
                                #y_positions[key] = cubes_xyz[key][1]
                            cubes_xyz = copy.deepcopy(self.tracked_positions)
                            #for key in cubes_xyz.keys():
                            #    cubes_xyz[key][2] = z_positions[key]
                            #    #cubes_xyz[key][1] = y_positions[key]
                            print("Using tracked positions: ", cubes_xyz, "compared to ground truth: ", ground_truth_xyz)
                        if len(cubes_xyz) > 0:
                            if self.use_yolo:
                                o = self.insert_yolo_estimate(obs_copy[0][ni], cubes_xyz=cubes_xyz, obj_to_pick=symgoal[0], place_to_drop=symgoal[1])
                                o = self.track_positions(state=info[0]['state'][ni], ee_pos=ee_pos, obs=o, symgoal=symgoal)
                                # insert o into obs
                                obs[0][ni] = o
            if self.oracle:
                obs = self.prepare_obs(obs, action_step=self.id)
            if obs_base:
                obs = self.obs_base_from_info(info)
            # create obs dict
            #print("Observation, ", obs)
            np_obs_dict = {
                'obs': obs.astype(np.float32)
            }
            # device transfer
            obs_dict = dict_apply(np_obs_dict, 
                lambda x: torch.from_numpy(x).to(
                    device=self.device))
            # run policy
            with torch.no_grad():
                action_dict = self.model.predict_action(obs_dict)
            # device_transfer
            np_action_dict = dict_apply(action_dict,
                lambda x: x.detach().to('cpu').numpy())
            action = np_action_dict['action']
            if obs_base:
                    #print("Action: ", action)
                    obj_to_pick = action[0][0][0]
                    obj_to_drop = action[0][0][1]
                    success = round(obj_to_drop) != round(obj_to_pick)
                    if not success:
                        action = (round(obj_to_pick), round(obj_to_drop))
                        print("Invalid task: ", action)
                        return None, success
                    obj_to_pick = "cube" + str(round(obj_to_pick))
                    obj_to_drop = "peg" + str(round(obj_to_drop)-3) if round(obj_to_drop) >= 4 else "cube" + str(round(obj_to_drop))
                    print("New task: ", (obj_to_pick, obj_to_drop))
                    env.set_task((obj_to_pick, obj_to_drop))
                    return (obj_to_pick, obj_to_drop), success
            # If the actions in action (array) do not have 4 elements, then concatenate [0] to the action array
            if len(action[0][0]) < 4:
                # Create a column of zeros
                zeros_column = np.zeros((action.shape[0], action.shape[1], 1))
                # Concatenate the zeros column to the original array
                #action = np.concatenate((action, zeros_column), axis=2)
                # Concatenate zeros_colum to action at self.nulified_action_indexes
                #print("Nulifying action at indexes: ", self.nulified_action_indexes)
                #print("Original action shape: ", action.shape)
                #print("Original action: ", action)
                for index in self.nulified_action_indexes:
                    action = np.insert(action, index, 0, axis=2)
            #print("Action: ", action)
            # step env
            #print(self.device)
            try: 
                obs, reward, terminated, truncated, info = env.step(action)
                #done = terminated or truncated
            except:
                obs, reward, done, info = env.step(action)
            #done = np.all(done)
            if done:
                print("Environment terminated")
            step_executor += 1
            state = info[0]['state'][-1]
            success = self.Beta(state, symgoal)
            success = success or info[0]['is_success'][-1]
            if success:
                done = True
            if step_executor > horizon:
                print("Reached executor horizon")
                done = True 
        if setting == "3x3":
            valid_state = self.valid_state_f(state)
            success = success and valid_state
            if not valid_state:
                print("Invalid HANOI state")
        #self.save_video(output_path=self.id+".mp4", fps=10)
        if self.save_data:
            #self.save_video(output_path=f"{self.id}_{self.count}.mp4", fps=10)
            self.save_csv_yolo(output_path=f"{self.id}_dualcam_{self.count}.csv")
            self.count += 1
        return obs, success, step_executor

import gymnasium as gym
from gymnasium import spaces
import pygame
import carla
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import warnings
import os
from collections import deque
import sys

# Suppress all warnings
warnings.filterwarnings("ignore")

from collections import deque
from gym_carlaRL.envs.utils.lane_detection.openvino_lane_detector import OpenVINOLaneDetector
from gym_carlaRL.envs.utils.lane_detection.lane_detector import LaneDetector
from gym_carlaRL.envs.utils.pid_controller import VehiclePIDController
from gym_carlaRL.envs.ufld.model.model_culane import parsingNet

from gym_carlaRL.envs.carla_util import *
from gym_carlaRL.envs.route_planner import RoutePlanner
from gym_carlaRL.envs.misc import *

from .global_route_planner2 import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class _plan():

  def __init__(self, world, start, goal):
     """
     Create an instance of a plan. 

     :param world: Carla Simulator world object
     :param start: start location
     :param goal: goal location
     """
     self.world = world
     self.map = self.world.get_map()
     self.path = []
     self.goal = goal
     self.start = start


  def get_high_level_plan(self):
    """
    Get a high level plan to get from the start to goal.
    Returns a list of [location, command] pairs, representing where to switch to the next command in the plan.
    """
    sampling_resolution = 1
    grp = GlobalRoutePlanner(self.map, sampling_resolution)
    route = grp.trace_route(self.start, self.goal) # get a list of [carla.Waypoint, RoadOption] to get from start to goal
    high_level_plan = []
    current_command = route[0][1]
    high_level_plan.append([route[0][0].transform.location, RoadOption.LANEFOLLOW])
    j=0

    for i in range(len(route)):
      waypoint, command = route[i]
      prev_loc, prev_command = high_level_plan[j-1]
    
      if command != RoadOption.CHANGELANELEFT and command != RoadOption.CHANGELANERIGHT:
        if prev_loc.distance(waypoint.transform.location) < 1:
         high_level_plan.pop(j-1)
         high_level_plan.append([waypoint.transform.location, command])

        else:
            if waypoint.is_junction and command == RoadOption.LANEFOLLOW and current_command != RoadOption.STRAIGHT:
             high_level_plan.append([waypoint.transform.location, RoadOption.STRAIGHT])
             current_command = RoadOption.STRAIGHT
             j = j+1
      
            elif current_command != command:
             if command == RoadOption.CHANGELANERIGHT or command == RoadOption.CHANGELANELEFT:
                high_level_plan.append([waypoint.transform.location, RoadOption.LANEFOLLOW])
                current_command = RoadOption.LANEFOLLOW
                j = j+1
             else:
                high_level_plan.append([waypoint.transform.location, command])
                current_command = command
                j = j+1

    high_level_plan.append([self.goal, "STOP"])

    return high_level_plan
  

class CarlaEnv(gym.Env):
    def __init__(self, params):
        super().__init__()

        self.params = params

        self.collision_sensor = None
        # TODO: self.lidar_sensor = None
        self.camera_rgb = None
        self.camera_windshield = None

        # Define observation space
        self.observation_space = spaces.Dict({
            'actor_input': spaces.Box(low=0, high=255, shape=(self.params['display_size'][1], self.params['display_size'][0], 3), dtype=np.uint8), 
            'vehicle_state': spaces.Box(np.array([-2, -1]), np.array([2, 1]), dtype=np.float64),  # lateral_distance, -delta_yaw
            'command': spaces.Discrete(3), #lane follow, right, left
            'next_command': spaces.Discrete(4), #lane follow, right, left, None
            })

        # Define action space
        self.action_space = spaces.Box(np.array(params['continuous_steer_range'][0]), 
                                       np.array(params['continuous_steer_range'][1]), 
                                       dtype=np.float32)  # steer

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        # self.total_step = 0
        # Initialize CARLA connection and environment setup
        self.setup_carla()

    def setup_carla(self):
        host = self.params.get('host', 'localhost')
        port = self.params.get('port', 2000)
        town = self.params.get('town', 'Town05')
        self.width, self.height = self.params['display_size'][0], self.params['display_size'][1]

        print(f'Connecting to the CARLA server at {host}:{port}...')
        time_start_connect = time.time()
        self.client = carla.Client(host, port)
        self.client.set_timeout(300.0)
        self.client.load_world(town)
        connection_time = time.time() - time_start_connect
        print(f'took {connection_time//60:.0f}m {connection_time%60:.0f}s to connect the server.')
        self.world = self.client.get_world()

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = self.params.get('dt', 0.1)
        self.world.apply_settings(settings)

        if self.params['display']:
            pygame.init()
            pygame.font.init()
            self.display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
                )
            self.display.fill((0,0,0))
            self.instance_display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF
                )
            self.instance_display.fill((0,0,0))
            pygame.display.flip()
            self.font = get_font()
            self.clock = pygame.time.Clock()

        weather_presets = find_weather_presets()
        self.world.set_weather(weather_presets[self.params.get('weather', 6)][0])
        # self.weather = Weather(self.world.get_weather())
        self.map = self.world.get_map()
        self.spawn_points = list(self.map.get_spawn_points())
        self.spawn_locs = [230, 341]  # specific locations to train for curve where lane detection is challenging
        self.spawn_loc = self.spawn_locs[0]
        self.straight_spawn_loc = 200

        # Base parameters for CARLA PID controller
        self.desired_speed = self.params['desired_speed']
        self._dt = self.params.get('dt', 0.1)
        self._target_speed = self.desired_speed * 3.6  # convert to km/h
        self._args_lateral_dict = {'K_P': 1.95, 'K_I': 0.05, 'K_D': 0.0, 'dt': self._dt}
        self._args_longitudinal_dict = {'K_P': 1.0, 'K_I': 0.05, 'K_D': 0, 'dt': self._dt}
        self._max_throt = 0.75
        self._max_brake = 0.3
        self._max_steer = 0.8
        self._offset = 0.0
        
        self.data_saver = DataSaver()

        # Initialize the lane detector
        if self.params['model'] == 'lanenet':
            # Two versions of lanenet for turning at intersections and for lanefollowing
            self.lane_detector = LaneDetector(model_path=self.params['model_path'])
            self.transform = A.Compose([
                A.Resize(256, 512),
                A.Normalize(),
                ToTensorV2()
            ])
            self.cg = self.lane_detector.cg

            self.left_lane_detector = LaneDetector(model_path=self.params['left_model_path'], intersection=True)
            self.right_lane_detector = LaneDetector(model_path=self.params['right_model_path'], intersection=True)

        elif self.params['model'] == 'ufld':
            self.image_width = 1280
            self.image_height = 720
            self.resize_width = 800
            self.resize_height = 320
            self.crop_ratio = 0.8
            self.num_row= 56
            self.num_col= 41
            self.num_cell_row= 100
            self.num_cell_col= 100
            self.row_anchor = np.linspace(0.42, 1, self.num_row)
            self.col_anchor = np.linspace(0, 1, self.num_col)

            self.lane_detector = parsingNet(
                pretrained = True,
                backbone = '18',
                num_grid_row = self.num_cell_row, num_cls_row = self.num_row,
                num_grid_col = self.num_cell_col, num_cls_col = self.num_col,
                num_lane_on_row = 4, num_lane_on_col = 4, 
                use_aux = False,
                input_height = self.resize_height, input_width = self.resize_width,
                fc_norm = False
            ).to(DEVICE)
            state_dict = torch.load(self.params['model_path'], map_location = 'cpu')['model']
            compatible_state_dict = {}
            for k, v in state_dict.items():
                if 'module.' in k:
                    compatible_state_dict[k[7:]] = v
                else:
                    compatible_state_dict[k] = v
            self.lane_detector.load_state_dict(compatible_state_dict, strict = True)
            self.lane_detector.eval()

            self.transform = A.Compose([
                A.Resize(int(self.resize_height / self.crop_ratio), self.resize_width),
                A.Normalize(),
                ToTensorV2()
            ])
        else:
            self.lane_detector = OpenVINOLaneDetector()
            self.cg = self.lane_detector.cg

        self.version = self.params['controller_version']
        if self.version >= 2:
            if self.params['algo'] == 'ppo':
                self.image_processor = ImageProcessor(controller_v=self.version, max_history_length=10, img_size=128)
            else:
                self.image_processor = ImageProcessor(controller_v=self.version, max_history_length=10, img_size=32)


    def step(self, action):
        target_wpt, target_wpt_opt = self.waypoints[0]
        control = self._vehicle_controller.run_step(self._target_speed, target_wpt)
        carla_pid_steer = control.steer
        if self.params['clip_action']:
            action = np.clip(action, -0.2, 0.2)
            carla_pid_steer = np.clip(carla_pid_steer, -0.2, 0.2)
        else:
            action = np.clip(action, -1.0, 1.0)
            carla_pid_steer = np.clip(carla_pid_steer, -1.0, 1.0)
        act = carla.VehicleControl(throttle=float(control.throttle), 
                                        steer=float(action), 
                                        brake=float(control.brake))
        self.ego.apply_control(act)

        self.world.tick()

        self.waypoints, self.lane_opt = self.routeplanner.run_step() 

        new_obs = self.get_observations()
        reward = self.get_reward(new_obs)
        done = self.is_done(new_obs)
        info = {
            'waypoints': self.waypoints,
            'road_option': target_wpt_opt,
            'guidance': carla_pid_steer,
        }

        # Update timesteps
        self.time_step += 1

        # # dynamic weather
        # self.weather.tick(0.1)
        # self.world.set_weather(self.weather.weather)

        return new_obs, reward, done, info

    def reset(self):
        self.reset_step+=1

        self.destroy_all_actors()

        # Disable sync mode
        self._set_synchronous_mode(False)

        rand_locations = True
        if rand_locations:
            self.start_type = 'random'
            start_pos = random.choice(self.spawn_points)
            index = np.random.randint(0, len(self.spawn_points))
            self.goal_pos = self.spawn_points[index]

        else:
            # Spawn the ego vehicle
            if self.params['mode'] == 'test':
                # get a random index for the spawn points
                index = np.random.randint(0, len(self.spawn_points))
                start_pos = self.spawn_points[index]
                index = np.random.randint(0, len(self.spawn_points))
                self.goal_pos = self.spawn_points[index]
                print(f'spawn location: {index}...')
            elif self.params['mode'] == 'train':
                if self.reset_step > 500:
                    start_pos = random.choice(self.spawn_points)
                    self.goal_pos = random.choice(self.spawn_points)
                else:
                    start_pos = self.spawn_points[self.straight_spawn_loc]
                    self.goal_pos = random.choice(self.spawn_points)
            elif self.params['mode'] == 'train_controller':
                if self.reset_step < 200:
                    self.start_type = 'straight'
                    start_pos = self.spawn_points[self.straight_spawn_loc]
                    index = np.random.randint(0, len(self.spawn_points))
                    self.goal_pos = self.spawn_points[index]
                elif self.reset_step < 2000:
                    self.start_type = 'random'
                    start_pos = random.choice(self.spawn_points)
                    index = np.random.randint(0, len(self.spawn_points))
                    self.goal_pos = self.spawn_points[index]
                else:
                    if np.random.rand() < 0.8:
                        self.start_type = 'random'
                        loc = np.random.randint(0, len(self.spawn_points))
                        start_pos = self.spawn_points[loc]
                        index = np.random.randint(0, len(self.spawn_points))
                        self.goal_pos = self.spawn_points[index]
                        print(f'\n ***random spawn location: {loc}...')
                    else:
                        self.start_type = 'challenge'
                        start_pos = self.spawn_points[self.spawn_loc]
                        index = np.random.randint(0, len(self.spawn_points))
                        self.goal_pos = self.spawn_points[index]
                        self.spawn_loc = self.spawn_locs[(self.spawn_locs.index(self.spawn_loc) + 1) % len(self.spawn_locs)]
                        print(f'\n ***challenge spawn location: {self.spawn_loc}...')

        #print("now getting plan for episode")
        goal_position = self.goal_pos.location
        path_plan = _plan(self.world, start_pos.location, goal_position)
        #based on plan (where you currently are, what steps to take to get to goal), receive command
        self.plan = path_plan.get_high_level_plan()
        self.steps_since_turn = 0 # record for done condition

        blueprint_library = self.world.get_blueprint_library()
        ego_vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        self.ego = self.world.spawn_actor(ego_vehicle_bp, start_pos)

        # CARLA PID controller
        self._vehicle_controller = VehiclePIDController(self.ego,
                                                        args_lateral=self._args_lateral_dict,
                                                        args_longitudinal=self._args_longitudinal_dict,
                                                        offset=self._offset,
                                                        max_throttle=self._max_throt,
                                                        max_brake=self._max_brake,
                                                        max_steering=self._max_steer)

        # add collision sensor
        self.collision_hist = deque(maxlen=1)
        collision_bp = blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
            self.collision_hist.append(intensity)

        # Initialize and attach camera sensor for display
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{self.width}')
        camera_bp.set_attribute('image_size_y', f'{self.height}')
        self.camera_rgb = self.world.spawn_actor(camera_bp,
                                                 carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)), 
                                                 attach_to=self.ego)
        self.camera_rgb.listen(lambda image: carla_img_to_array(image))
        self.image_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        def carla_img_to_array(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.image_rgb = array

        # Initialize the windshield camera to the ego vehicle
        # cg = CameraGeometry()
        cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=1.3), carla.Rotation(pitch=-1*5))
        bp = blueprint_library.find('sensor.camera.rgb')
        if self.params['model'] == 'ufld':
            bp.set_attribute('image_size_x', str(self.image_width))
            bp.set_attribute('image_size_y', str(self.image_height))
        else:
            bp.set_attribute('image_size_x', str(self.cg.image_width))
            bp.set_attribute('image_size_y', str(self.cg.image_height))
            bp.set_attribute('fov', str(self.cg.field_of_view_deg))
        self.camera_windshield = self.world.spawn_actor(bp, cam_windshield_transform, attach_to=self.ego)
        self.camera_windshield.listen(lambda image: carla_img_to_array_ws(image))
        if self.params['model'] == 'ufld':
            self.image_windshield = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
        else:
            self.image_windshield = np.zeros((self.cg.image_height, self.cg.image_width, 3), dtype=np.uint8)
        def carla_img_to_array_ws(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.image_windshield = array

        # Update timesteps
        self.time_step=0
        
        # Enable sync mode
        self._set_synchronous_mode(True)

        self.routeplanner = RoutePlanner(self.ego, self.params['max_waypt'])
        self.waypoints, self.lane_opt = self.routeplanner.run_step()

        return self.get_observations()

    def get_vehicle_speed(self):
        return np.linalg.norm(carla_vec_to_np_array(self.ego.get_velocity()))
    
    def get_intersection_waypoints(self, ego_loc, next_loc):
        sampling_resolution = 1
        grp = GlobalRoutePlanner(self.map, sampling_resolution)
        route = grp.trace_route(ego_loc, next_loc) # get a list of [carla.Waypoint, RoadOption] to get from current_loc to end of intersection
        next_waypoint = [route[0][0]]
        return next_waypoint
    
    def get_observations(self):
        speed = self.get_vehicle_speed()
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_z = ego_trans.location.z
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        
        command, next_command = self.get_command(ego_trans.location)

        if command == 1:
            lateral_dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
            delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        else:
            if len(self.plan) == 1:
                lateral_dis = 0
                w=0
                delta_yaw = 0
            else:
                intersection_waypoints = self.get_intersection_waypoints(ego_trans.location, self.plan[1][0])
                lateral_dis, w = get_lane_dis(intersection_waypoints, ego_x, ego_y)
                delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))

        v_state = np.array([lateral_dis, - delta_yaw, ego_x, ego_y, ego_z])

        obs = {}
        if self.params['model'] == 'ufld':
            image = self.process_image(self.image_windshield)
            pred, _ = self.lane_detector(image)
        else:
            if self.params['model'] == 'lanenet':
                image = self.process_image(self.image_windshield)
                # change which model depending on command
                if command == 1:
                    img = self.lane_detector(image)
                if command == 2:
                    img = self.right_lane_detector(image)
                else:
                    img = self.left_lane_detector(image)
            else:
                poly_left, poly_right, img = self.lane_detector(self.image_windshield)
                
            if np.max(img) > 1:
                max_val = np.max(img)
                min_val = np.min(img)
                if (max_val-min_val) == 0:
                    print("Divide by 0 in get_observations. Performing alternate normalization.")
                    if min_val < 0:
                        img = np.where(img <= 0, 0, 1)
                    else:
                        img = img
                else:
                    img = (img - np.min(img)) / (np.max(img) - np.min(img))
           
            img = img.astype(np.uint8)
            if self.version == 1:
                img_to_save = cv2.resize(img, (128,128))
                img = cv2.resize(img, (128,128))
                img = np.expand_dims(img, axis=0)
            elif self.version == 2:
                img, img_to_save = self.image_processor.process_image(img)
            elif self.version >= 3:
                img, img_to_save = self.image_processor.process_image(img)

            if self.params['display']:
                #cv2.imshow('Lane detector output', img)
                #cv2.waitKey(1)
                draw_image(self.display, self.image_rgb)
                pygame.display.flip()

                draw_image(self.instance_display, img_to_save)
                pygame.display.flip()

            if self.params['collect']:
                self.data_saver.save_image(self.image_windshield)
                self.data_saver.save_third_pov_image(self.image_rgb)
                self.data_saver.save_lane_image(img_to_save)
                self.data_saver.save_metrics(v_state)
                self.data_saver.step()

        #image type is numpy array
        obs = {
            'actor_input': pred if self.params['model'] == 'ufld' else img,
            'vehicle_state': v_state,
            'command': command, 'next_command': next_command,
        }

        return obs
    
    def get_command(self, ego_loc):
        """
        Get the current command to obey

        :param ego_loc: location of ego vehicle
        :return: command, next command in integer representation
        """
        if len(self.plan) == 1: # if the plan only has STOP, command is 2 (lane_follow) and next_command is 3 (None)
            command =  1
            next_command = 3
        
        else: # plan contains at least two commands
            current_objective = self.plan[0] # (location, command)
            next_objective = self.plan[1] 
            euclidean_dist = ego_loc.distance(next_objective[0]) # compare next plan location to ego location to see if need to switch command
            
            if euclidean_dist < 2: # close to next command, so finish current command
                command = next_objective[1] # command is now next item on list
                self.plan.pop(0) # remove current command from plan
                
                if len(self.plan) == 1: # if plan now only has one command, next command is None
                    next_command = 3
                else: # if plan still has at least two commands, next_command is next command on list
                    next_objective = self.plan[1] 
                    next_command = next_objective[1]

            else: # have not finished current part of plan, so plan list stays the same
                command = current_objective[1]
                next_command = next_objective[1]
            
            # convert command and next_command to integer representation for model
            if command == RoadOption.LANEFOLLOW or command == RoadOption.STRAIGHT:
                command = 1
            elif command == RoadOption.RIGHT:
                command = 2
            else:
                command = 0
            
            if next_command == RoadOption.LANEFOLLOW or next_command == RoadOption.STRAIGHT:
                next_command = 1
            elif next_command == RoadOption.RIGHT:
                next_command = 2
            else:
                next_command = 0

        return command, next_command


    def get_reward(self, obs):
        vehicle_state = obs['vehicle_state']
        current_command = obs['command'] #does the car steer according to the command? 
        steer = self.ego.get_control().steer
        r = 0  # current reward is 0     

        r_collision = 0
        if len(self.collision_hist) != 0:
            #print("collision has occurred")
            r_collision = -1
            r = r + r_collision

        # reward functions for different agents: lanefollowing agent, right turn agent, or left turn agent
        if current_command == 1: #lanefollow. Will affect lane_following agent
            self.steps_since_turn = 0
            r = self.steer_threshold_reward(steer, 0, r)
            if math.isnan(r):
                print("steer reward is nan")
            r = self.lane_threshold_reward(vehicle_state[0], r)
            if math.isnan(r):
                print("lane reward is nan")

        elif current_command == 2: # go right. Will affect right turn agent
            self.steps_since_turn = self.steps_since_turn + 1
            #print("must turn right")
            #print("steer: ", steer)
            r = self.steer_threshold_reward(steer, .9, r)
            if math.isnan(r):
                print("steer reward is nan")
            r = self.lane_threshold_reward(vehicle_state[0], r)
            if math.isnan(r):
                print("lane reward is nan")

        else: # go left. Will affect left turn agent
            self.steps_since_turn = self.steps_since_turn + 1
            #print("must turn left")
            #print("steer: ", steer)
            r = self.steer_threshold_reward(steer, -.9, r)
            if math.isnan(r):
                print("steer reward is nan")
            r = self.lane_threshold_reward(vehicle_state[0], r)
            if math.isnan(r):
                print("lane reward is nan")

        
        if math.isnan(r):
            print("reward is nan")
        return r

    
    def legal_lane_change(self, waypoint, direction):
        """
        Check if a right or left lane change is legal.
        
        :param waypoint: waypoint object of location to check
        :param direction: which lane change direction to check

        :return: True or False depending on lane type.
        """
        if direction == 0: #check right
            right_lane = waypoint.right_lane_marking
            if str(right_lane.type) == "Broken" or str(right_lane.type) == "NONE": 
                return True
            else:
                return False
            
        else:
            left_lane = waypoint.left_lane_marking
            if str(left_lane.type) == "Broken" or str(left_lane.type) == "NONE": 
                return True
            else:
                return False
            
    def lane_threshold_reward(self, lateral_distance, current_reward):
        """
        Calculate lane threshold reward
        
        :param lateral_distance: lateral distance from waypoints
        :param current_reward: current reward in reward function

        :return: Reward for lane following
        """
        if math.isnan(lateral_distance):
            print("distance is nan")
        dis = abs(lateral_distance)
        dis = -(dis / self.params['out_lane_thres'])  # normalize the lateral distance
        if math.isnan(lateral_distance):
            print("dis is nan")
        current_reward = current_reward + 1 + dis
        return current_reward
    
    def steer_threshold_reward(self, steer, target_steer, current_reward):
        """
        Calculate steer threshold reward
        
        :param steer: current steer of vehicle
        :param target_steer: target steer of vehicle
        :param current_reward: current reward in reward function

        :return: Reward for steering
        """
        r_steer = (1 - abs(steer - target_steer))
        
        return current_reward + r_steer

    def is_done(self, obs):

        # if collides
        if len(self.collision_hist)>0: 
            return True

        # If reach maximum timestep
        if self.time_step > self.params['max_time_episode']:
            return True
        
        # if close to goal
        ego_loc = self.ego.get_location()
        goal_loc = self.goal_pos.location
        euclidean_dist = ego_loc.distance(goal_loc)
        if euclidean_dist <= 4:
            return True
        
        #if lane_following and dis is too high
        current_command = obs['command']
        if current_command == 2:
            vehicle_state = obs['vehicle_state']
            dis = abs(vehicle_state[0])
            if dis >= 1.5:
                return True
            
        # if missing turn
        if current_command == 0:
            steer = self.ego.get_control().steer
            if self.steps_since_turn >= 40:
                #print("many steps since command")
                if (steer > -.1):
                    #print("not turning left")
                    #print("steer: ", steer)
                    return True
                else:
                    self.steps_since_turn = 0
            
        if current_command ==2:
            steer = self.ego.get_control().steer
            if self.steps_since_turn >= 40:
                #print("many steps since command")
                if (steer < .1):
                    #print("not turning right")
                    #print("steer: ", steer)
                    return True
                else:
                    self.steps_since_turn = 0
        
        return False
    
    def process_image(self, image):
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        image = self.transform(image=image)['image']
        if self.params['model'] == 'ufld':
            image = image[:, -self.resize_height:, :]
        image = image.unsqueeze(0).to(DEVICE)
        return image

    def start_record(self, episode):
        log_path = 'gym_carlaRL/envs/recording/ppo_imageOnly/'
        recording_file_name = os.path.join(log_path, f'episode_{episode}.log')
        self.client.start_recorder(recording_file_name, True)
        print(f'started recording and saving to {recording_file_name}...')

    def stop_record(self):
        # Stop the recording
        self.client.stop_recorder()

    def destroy_all_actors(self):
        # Clear sensor objects
        if self.collision_sensor is not None and self.collision_sensor.is_listening:
            self.collision_sensor.stop()
            # self.lidar_sensor.stop()
            self.camera_rgb.stop()
            self.camera_windshield.stop()

        self.collision_sensor = None
        # TODO: self.lidar_sensor = None
        self.camera_rgb = None
        self.camera_windshield = None

        self.trajectory = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors(['sensor.other.collision', 'sensor.lidar.ray_cast', 'sensor.camera.rgb', 'vehicle.*', 'controller.ai.walker', 'walker.*'])

    def _set_synchronous_mode(self, synchronous = True):
        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous
        self.world.apply_settings(settings)
        
    def _clear_all_actors(self, actor_filters):
        for actor_filter in actor_filters:
            if self.world.get_actors().filter(actor_filter):
                for actor in self.world.get_actors().filter(actor_filter):
                    try:
                        if actor.is_alive:
                            if actor.type_id == 'controller.ai.walker':
                                actor.stop()
                            actor.destroy()
                            # print(f'Destroyed {actor.type_id} {actor.id}')
                    except Exception as e:
                        print(f'Failed to destroy {actor.type_id} {actor.id}: {e}')

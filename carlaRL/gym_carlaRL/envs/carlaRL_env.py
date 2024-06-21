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

from agents.navigation.local_planner import RoadOption

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
from high_level_plan import *

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        town = self.params.get('town', 'Town04')
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
            pygame.display.flip()
            self.font = get_font()
            self.clock = pygame.time.Clock()

        weather_presets = find_weather_presets()
        self.world.set_weather(weather_presets[self.params.get('weather', 6)][0])
        # self.weather = Weather(self.world.get_weather())

        self.spawn_points = list(self.world.get_map().get_spawn_points())
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
            self.lane_detector = LaneDetector(model_path=self.params['model_path'])
            self.transform = A.Compose([
                A.Resize(256, 512),
                A.Normalize(),
                ToTensorV2()
            ])
            self.cg = self.lane_detector.cg
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

        # Spawn the ego vehicle
        if self.params['mode'] == 'test':
            # get a random index for the spawn points
            index = np.random.randint(0, len(self.spawn_points))
            start_pos = self.spawn_points[index]
            print(f'spawn location: {index}...')
        elif self.params['mode'] == 'train':
            if self.reset_step > 500:
                start_pos = random.choice(self.spawn_points)
            else:
                start_pos = self.spawn_points[self.straight_spawn_loc]
        elif self.params['mode'] == 'train_controller':
            if self.reset_step < 200:
                self.start_type = 'straight'
                start_pos = self.spawn_points[self.straight_spawn_loc]
            elif self.reset_step < 2000:
                self.start_type = 'random'
                start_pos = random.choice(self.spawn_points)
            else:
                if np.random.rand() < 0.8:
                    self.start_type = 'random'
                    loc = np.random.randint(0, len(self.spawn_points))
                    start_pos = self.spawn_points[loc]
                    print(f'\n ***random spawn location: {loc}...')
                else:
                    self.start_type = 'challenge'
                    start_pos = self.spawn_points[self.spawn_loc]
                    self.spawn_loc = self.spawn_locs[(self.spawn_locs.index(self.spawn_loc) + 1) % len(self.spawn_locs)]
                    print(f'\n ***challenge spawn location: {self.spawn_loc}...')

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
    
    def get_observations(self):
        obs = {}
        speed = self.get_vehicle_speed()
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_z = ego_trans.location.z
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        lateral_dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        v_state = np.array([lateral_dis, - delta_yaw, ego_x, ego_y, ego_z])
        
        
        path = plan(self.world, ego_trans.location)

        #based on plan (where you currently are, what steps to take to get to goal), receive command
        command = RoadOption(4) #default roadoption is lanefollow

        #if the next part of the path is a junction, change roadoption to direction of next edge
        if type(path[0]) is tuple:
            command = path[0][1]
        


        if self.params['model'] == 'ufld':
            image = self.process_image(self.image_windshield)
            pred, _ = self.lane_detector(image)
        else:
            if self.params['model'] == 'lanenet':
                image = self.process_image(self.image_windshield)
                img = self.lane_detector(image)
            else:
                poly_left, poly_right, img = self.lane_detector(self.image_windshield)
                
            if np.max(img) > 1:
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
                cv2.imshow('Lane detector output', img)
                cv2.waitKey(1)
                draw_image(self.display, self.image_rgb)
                pygame.display.flip()

            if self.params['collect']:
                self.data_saver.save_image(self.image_windshield)
                self.data_saver.save_third_pov_image(self.image_rgb)
                self.data_saver.save_lane_image(img_to_save)
                self.data_saver.save_metrics(v_state)
                self.data_saver.step()

        obs = {
            'actor_input': pred if self.params['model'] == 'ufld' else img,
            'command': command, 'vehicle_state': v_state,
        }

        return obs

    def get_reward(self, obs):
        # # Define how the reward is calculated based on the state
        # speed = self.get_vehicle_speed()
        # r_speed = -abs(speed - self.desired_speed)

        # reward for collision
        # r_collision = 0
        # if len(self.collision_hist) > 0:
        #     r_collision = -1

        vehicle_state = obs['vehicle_state']

        # reward for steering:
        r_steer = self.ego.get_control().steer**2  # squared steering to encourage small steering commands but penalize large ones
        r_steer = -(r_steer / 0.04)  # normalize the steering

        # reward for out of lane
        dis = abs(vehicle_state[0])
        r_out = 0
        if dis > self.params['out_lane_thres']:
            r_out = -1
        dis = -(dis / self.params['out_lane_thres'])  # normalize the lateral distance
        # ego_trans = self.ego.get_transform()
        # ego_x = ego_trans.location.x
        # ego_y = ego_trans.location.y
        # ego_yaw = ego_trans.rotation.yaw/180*np.pi
        # dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        # delta_yaw = np.arcsin(np.cross(w, np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        # print(f'dis: {dis}; delta_yaw: {delta_yaw}')
        

        # # longitudinal speed
        # v = self.ego.get_velocity()
        # lspeed = np.array([v.x, v.y])
        # lspeed_lon = np.dot(lspeed, w)

        # # cost for too fast
        # r_fast = 0
        # if lspeed_lon > self.desired_speed:
        #     r_fast = -1

        # # cost for lateral acceleration
        # r_lat = - abs(self.ego.get_control().steer) * lspeed_lon**2  # discourage high steering commands at high speeds

        r = 1 + dis

        return r

    def is_done(self, obs):
        ego_x, ego_y = get_pos(self.ego)

        # if collides
        if len(self.collision_hist)>0: 
            return True

        # If reach maximum timestep
        if self.time_step > self.params['max_time_episode']:
            return True
        
        # If out of lane
        vehicle_state = obs['vehicle_state']
        dis = abs(vehicle_state[0])
        if abs(dis) > self.params['out_lane_thres']:
            return True
        
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

#!/usr/bin/env python

import gymnasium as gym
import numpy as np
import torch

from gym_carlaRL.envs.carlaRL_env import CarlaEnv
from gym_carlaRL.agent.ppo_agent import ActorCritic

import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {DEVICE}')



def main():
    params = {
        'host': 'localhost',  # '104.51.58.17',
        'port': 2000,  # The port where your CARLA server is running
        'town': 'Town05',  # The map to use
        'mode': 'test',  # The mode to run the environment in
        'algo' : 'ppo',  # this decides how the image is processed
        'controller_version': 1,  # The version of the controller to use
        'dt': 0.1,  # time interval between two frames
        'desired_speed': 3.6,  # Desired speed (m/s)
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'clip_action': False,  # clip the steering angle
        'max_waypt': 12,  # maximum number of waypoints
        'out_lane_thres': 1.5,  # threshold for out of lane
        'display_size': [256, 256],  # Screen size for the pygame window
        'display' : False,  # Whether to display the pygame window
        'max_time_episode': 300,  # Maximum time for each episode
        'weather': 6,  # Weather preset (6 is sunny)
        'fps_sim': 20,  # Simulation FPS
        'model': 'lanenet',  # Lane detection model to use
        'model_path': 'C:/carla/WindowsNoEditor/PythonAPI/v-e2e-rl-ad/carlaRL/gym_carlaRL/envs/lanenet_lane_detection_pytorch/log/loss=0.1223_miou=0.5764_epoch=73.pth',  # Path to the lane detection model
        'record_interval': 10,  # The interval in which to record the episode
        'collect': True,  # Whether to collect the data
    }
    print("test.py started")

    # Initialize the CarlaEnv environment with the specified parameters
    env = gym.make('CarlaRL-v0', params=params)

    # Create an instance of the agent
    ppoAgent = ActorCritic().to(DEVICE)
    model_path = 'C:/carla/WindowsNoEditor/PythonAPI/v-e2e-rl-ad/verification/models/ppo_his/random_epi=4283_r=288.pth' #change path to where you have your lane following model
    ppoAgent.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) #add map location if you just want to use CPU
    ppoAgent.eval()

    # Run episodes
    final_v_states = []
    for i in range(20):
        save_pos_counter = 0
        v_states = []
        print('\n Episode:', i+1)
        state = env.reset()
        done = False
        for s in range(params['max_time_episode']):
            with torch.no_grad():
                action, _, _ = ppoAgent(state['actor_input'])
                # action = controller(state['actor_img'])
            action_to_perform = action.item()
            pos = state['vehicle_state'][-3:]
            v_states.append(pos)
            save_pos_counter += 1
            if save_pos_counter % 30 == 0:
                final_v_states.append(v_states)
                # save final vehicle states
                np.save('veh_pos.npy', final_v_states)
                save_pos_counter = 0
                v_states = []
                print(f'saved a serires of vehicle states...')
            next_state, reward, done, info = env.step(action_to_perform)
            lane_option = info['road_option']
            if lane_option != 1:
                done = True
            state = next_state
            if done:
                break

if __name__ == '__main__':
    main()

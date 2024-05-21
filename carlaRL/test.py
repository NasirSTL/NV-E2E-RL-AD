#!/usr/bin/env python

import gymnasium as gym
import numpy as np
import torch

from gym_carlaRL.envs.carlaRL_env import CarlaEnv
# from gym_carlaRL.agent.random_agent import RandomAgent
from gym_carlaRL.agent.ppo_agent import ActorCritic
from gym_carlaRL.agent.controller_agent import ControllerNet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE: {DEVICE}')


def main():
    params = {
        'host': 'localhost',  # '104.51.58.17',
        'port': 2000,  # The port where your CARLA server is running
        'town': 'Town04',  # The map to use
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
        'model': 'openvino',  # Lane detection model to use
        'record_interval': 10,  # The interval in which to record the episode
        'collect': True,  # Whether to collect the data
    }

    # Initialize the CarlaEnv environment with the specified parameters
    env = gym.make('CarlaRL-v0', params=params)

    # Create an instance of the agent
    ppoAgent = ActorCritic().to(DEVICE)
    model_path = '/storage1/yvorobeychik/Active/aml/carla/verification/models/ppo_single_img/episode=4330_reward=287.671.pth'  # this is single image agent
    # model_path = '/storage1/yvorobeychik/Active/aml/carla/verification/models/ppo_his/random_epi=4283_r=288.pth'  # this is agent with history of 10 weighted images
    # f1tenth_path = '/storage1/yvorobeychik/Active/aml/carla/carlaRL/log/ppo/imgOnly/f1tenth/1/random_epi=1703_r=570.pth'
    ppoAgent.load_state_dict(torch.load(model_path))
    ppoAgent.eval()
    # agent = RandomAgent(env.action_space)
    # controller = ControllerNet().to(DEVICE)
    # model_path = '/storage1/yvorobeychik/Active/aml/carla/carlaRL/log/controller/2/random_r=286.13_len=300.pth'
    # controller.load_state_dict(torch.load(model_path))
    # controller.eval()

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

    # while True:
    #     # Placeholder for the action selection logic
    #     # For now, we'll just use a dummy action
    #     # Update this with a meaningful action based on your action space definition
    #     obs = env.reset()
    #     action = agent.act(obs)
    #     done = False
    #     # Take a step in the environment with the chosen action
    #     obs, reward, done, info = env.step(action)

    #     # Check if the episode is done, and reset the environment if so
    #     if done:
    #         obs = env.reset()

if __name__ == '__main__':
    main()

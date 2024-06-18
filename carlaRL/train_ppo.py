import torch
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
import gymnasium as gym
import argparse

from enum import Enum
from carlaRL.utils import blockPrint, enablePrint
from gym_carlaRL.envs.carlaRL_env import CarlaEnv
from gym_carlaRL.agent.ppo_agent import ActorCritic
from utils import *

import sys, os


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'DEVICE: {DEVICE}')


def main(args):

    save_dir = "./log/ppo/imgOnly/f1tenth/ufld/{}".format(args.id)
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters
    EPISODES = args.max_episodes
    STEPS = args.steps
    BATCH_SIZE = args.batch_size
    pi_lr = 1e-5
    vf_lr = 5e-4
    saving_model = args.saving_model
    update_interval = args.update_interval

    params = {
        'host': 'localhost',  # '104.51.58.17',
        'port': args.port,  # The port where your CARLA server is running
        'town': 'Town04',  # The map to use
        'mode': 'train_controller',  # The mode to run the environment in: train is for RL algorithms only
        'algo' : 'ppo',  # this decides how the image is processed
        'controller_version': 3,  # The version of the controller to use
        'dt': 0.1,  # time interval between two frames
        'desired_speed': 3.6,  # Desired speed (m/s)
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-1.0, 1.0],  # continuous steering angle range
        'clip_action': args.clip_action,  # clip the steering angle
        'max_waypt': 12,  # maximum number of waypoints
        'out_lane_thres': 1.5,  # threshold for out of lane
        'display_size': [256, 256],  # Screen size for the pygame window
        'display' : args.display,  # Whether to display the pygame window
        'max_time_episode': STEPS,  # Maximum time for each episode
        'weather': 6,  # Weather preset (6 is sunny)
        'fps_sim': 20,  # Simulation FPS
        'model': 'lanenet',  # Lane detection model to use
        'model_path': 'gym_carlaRL/envs/lanenet_lane_detection_pytorch/log/loss=0.1223_miou=0.5764_epoch=73.pth',  # Path to the lane detection model
        'record_interval': 10,  # The interval in which to record the episode
        'restriction': 40 if args.curriculum else STEPS, 
        'collect': False,
    }
    """
    controller version 1: simple agent that takes one 128x128 image as input
    controller version 2: simple agent that takes one 32x32 image as input
    controller version 3: history of 10 128x128 weighted images as input
    """

    env = gym.make('CarlaRL-v0', params=params)
    input_f = 'dict' if params['model'] == 'ufld' else 'image'
    agent = ActorCritic(obs_dim=1,
                        pi_lr=pi_lr,
                        v_lr=vf_lr,
                        pi_epochs=5,
                        v_epochs=5).to(DEVICE)
    if args.load_model is not None:
        agent.load_state_dict(torch.load(args.load_model))
        print(f'Model loaded from {args.load_model}')
    agent.train()
    training_log = {'episode': [], 'policy_loss': [], 'value_loss': [], 'episode_return': [], 'episode_length': [], 'policy_mean': [], 'policy_std': [], 'policy_entropy': []}
    best_return = -np.inf

    blockPrint()

    restrict = params['restriction']
    for episode in range(EPISODES):
        state, ep_return, ep_len = env.reset(), 0, 0
        bootstrap_value = 0.0
        if (episode + 1) % update_interval == 0:
            restrict = restrict + 20

        with tqdm(total=STEPS, desc=f"Episode {episode + 1}/{EPISODES}", leave=True) as pbar:
            for step in range(STEPS):
                action, value, logp = agent(state['actor_input'])
                action = action.item()
                next_state, reward, done, info = env.step(action)
                steer_guidance = info['guidance']
                road_opt = info['road_option']
                ep_return += reward
                value = value.item()
                logp = logp.item()

                if road_opt == 1:
                    """Only store the data if the guiding steer is going straight or lanefollowing
                    """
                    ep_len += 1
                    agent.memory.add(state, action, steer_guidance, reward, done, value, logp)
                else:
                    enablePrint()
                    print(f'Road option: {road_opt} \n')
                    blockPrint()

                state = next_state

                pbar.set_postfix(r_s=reward,
                                r_ep=ep_return,
                                len_ep=ep_len,
                                v=value,
                                a=action,
                                a_t=steer_guidance,)
                pbar.update(1)

                reach_restrict = step == restrict - 1
                max_steps = step == STEPS - 1

                if done or reach_restrict or max_steps:
                    if done:
                        bootstrap_value = float(0)
                    else:
                        _, value, _ = agent(state['actor_input'])
                        bootstrap_value = value.item()
                    break
                        
        agent.finish_path(bootstrap_value)
        agent.compute_gae(gamma=args.gamma, lam=0.95)
        num_samples = ep_len
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        num_batches = int(np.ceil(num_samples / BATCH_SIZE))
        policy_losses, value_losses, ents = [], [], []
        if ((episode+1) % update_interval) < (update_interval / 2):
            b = 0.01
        else:
            b = 0.0
        for i in range(num_batches):
            begin = i * BATCH_SIZE
            end = min((i + 1) * BATCH_SIZE, num_samples)
            batch_indices = indices[begin:end]
            policy_loss, value_loss, ent = agent.update(batch_indices=batch_indices, clip_param=0.2, beta=b)
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            ents.append(ent)

            enablePrint()
            policy_mean = agent.pi.mu.detach().cpu().numpy()[0].item()
            policy_std = torch.exp(agent.pi.log_std).item()
            print(f'policy mean: {policy_mean:.3f}; policy std: {policy_std:.5f} \n episode losses: policy_loss: {np.mean(policy_losses):.4f}, value_loss: {np.mean(value_losses):.4f} \n')
            blockPrint()

        agent.memory.clear()

        training_log['episode'].append(episode+1)
        training_log['episode_return'].append(ep_return)
        training_log['episode_length'].append(ep_len)
        training_log['policy_loss'].append(np.mean(policy_losses))
        training_log['value_loss'].append(np.mean(value_losses))
        training_log['policy_mean'].append(policy_mean)
        training_log['policy_std'].append(policy_std)
        training_log['policy_entropy'].append(np.mean(ents))
        df = pd.DataFrame(training_log)
        train_log_save_filename = os.path.join(save_dir, f'log.csv')
        df.to_csv(train_log_save_filename, index=False, encoding='utf-8')

        if saving_model and ep_return > best_return:
            best_return = ep_return
            save_filename = '{}/{}_epi={}_r={}.pth'.format(save_dir, env.start_type, episode+1, int(ep_return))
            torch.save(agent.state_dict(), save_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--id', type=int, default=0, help='Experiment ID')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--clip_action', type=bool, default=False, help='Whether to clip the steering angle')
    parser.add_argument('--gamma', type=float, default=0.90, help='Discount factor')
    parser.add_argument('--update_interval', type=int, default=20, help='Update interval for changing curriculum')
    parser.add_argument('--batch_size', '-bs', type=int, default=64, help='Batch size for learning')
    parser.add_argument('--max_episodes', type=int, default=5000, help='Maximum number of episodes to train')
    parser.add_argument('--steps', type=int, default=600, help='Maximum number of steps per episode')
    parser.add_argument('--saving_model', type=bool, default=True, help='Whether to save the model')
    parser.add_argument('--load_model', type=str, default=None, help='Pre-trained model to load')
    parser.add_argument('--display', type=bool, default=False, help='Whether to display the pygame window')
    parser.add_argument('--curriculum', type=bool, default=True, help='Whether to use curriculum learning')

    args = parser.parse_args()

    main(args)

        

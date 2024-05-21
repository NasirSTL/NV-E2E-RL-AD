import torch
import torch.nn as nn
from torch.distributions import Normal
import copy
from tqdm import tqdm
import numpy as np
import pandas as pd
import os

from BC_model_controller.utils.general import init_log, get_optimizer, get_scheduler, get_minibatch
from BC_model_controller.bc_model.loss import calculate_loss


def train_bc_controller(cfg, G, bc_controller):
    """
    Train the controller model based on bicycle dynamical model and generator model
    :param cfg: configurations
    :param G: generator model
    :param bc_controller: controller model

    :return: None
    """
    optimizer = get_optimizer(cfg, bc_controller)
    schedular = get_scheduler(optimizer, cfg, 1)
    log = init_log(cfg)
    best_model_wts = copy.deepcopy(bc_controller.state_dict())

    positions = np.load("veh_pos.npy")
    pbar = tqdm(total=cfg.epochs)
    best_loss = np.inf
    for epoch in range(cfg.epochs):
        # initial state for generator
        pos = np.random.choice(len(positions))
        d = np.random.normal(0, 0.25)
        theta = np.random.normal(0, 0.12)

        rewards, images, actions, lateral_errors, thetas = generate_trajectory(d, theta, positions[pos], bc_controller, G, steps=cfg.steps_per_epoch)
        true_actions = actions[0]
        lateral_errors_dist = Normal(torch.tensor([np.mean(lateral_errors)], dtype=torch.float).cuda(), torch.tensor([np.std(lateral_errors)], dtype=torch.float).cuda())

        # calculate the loss
        optimizer.zero_grad()
        num_batches, indices = get_minibatch(len(images), cfg.batch_size)
        for i in range(num_batches):
            begin = i * cfg.batch_size
            end = min((i + 1) * cfg.batch_size, len(images))
            batch_indices = indices[begin:end]
            loss = calculate_loss(cfg, bc_controller, images[batch_indices], true_actions[batch_indices], lateral_errors_dist)
            loss.backward()
        schedular.step()

        pbar.set_postfix({'l': loss.item(), 'r': np.mean(rewards), 'd': np.mean(lateral_errors)})
        pbar.update(1)

        # dynamically save the log
        log['epoch'].append(epoch+1)
        log['loss'].append(loss.item())
        log['rewards'].append(np.mean(rewards))
        log['lateral_error'].append(np.mean(lateral_errors))
        pd.DataFrame(log).to_csv(os.path.join(cfg.save_dir, 'log.csv'))

        # save the best model
        if loss.item() < best_loss:
            best_model_wts = copy.deepcopy(bc_controller.state_dict())
            best_loss = loss.item()
            # save actions and true actions
            best_epoch_path = os.path.join(cfg.save_dir, 'best_epoch')
            os.makedirs(best_epoch_path, exist_ok=True)
            np.save(os.path.join(best_epoch_path, 'true_actions.npy'), actions[0])
            np.save(os.path.join(best_epoch_path, 'computed_actions.npy'), actions[1])
            np.save(os.path.join(best_epoch_path, 'actions.npy'), actions[2])
            np.save(os.path.join(best_epoch_path, 'lat_errors.npy'), lateral_errors[0])
            np.save(os.path.join(best_epoch_path, 'pred_lat_errors.npy'), lateral_errors[1])
            np.save(os.path.join(best_epoch_path, 'thetas.npy'), thetas[0])
            np.save(os.path.join(best_epoch_path, 'pred_thetas.npy'), thetas[1])

    # save the best model
    bc_controller.load_state_dict(best_model_wts)
    torch.save(bc_controller.state_dict(), os.path.join(cfg.save_dir, 'bc_controller.pth'))
    pbar.close()


def generate_trajectory(d, theta, pos, controller, generator, v=3.6, dt=0.05, L=2.5, steps=30):
    z_noise = torch.randn(1, 100).cuda()
    rewards = []
    images = []

    action_array = [[], [], []]
    lateral_error_array = [[], []]
    theta_array = [[], []]

    for s in range(steps):
        d = float(d)
        theta = float(theta)
        deg = np.rad2deg(theta)

        if abs(d) > 1.5:
            break

        # get the position of the vehicle
        x, y, z = pos[s % len(pos)]
        # add some noise to the position
        x += np.random.normal(0, 0.1)
        y += np.random.normal(0, 0.1)
        z += np.random.normal(0, 0.1)

        # get the observation
        observation = torch.tensor([d, deg, x, y, z], dtype=torch.float).cuda()
        image = generator(z_noise, observation)
        # image = image.expand(-1, 3, -1, -1)

        pred_d, pred_theta, pred_steer = controller(image)
        # ground truth action based on d and theta
        true_action = -0.74 * d - 0.44 * theta
        compute_action = -0.74 * pred_d.item() - 0.44 * pred_theta.item()

        action_array[0].append(true_action)
        action_array[1].append(compute_action)
        action_array[2].append(pred_steer.item())
        lateral_error_array[0].append(d)
        lateral_error_array[1].append(pred_d.item())
        theta_array[0].append(theta)
        theta_array[1].append(pred_theta.item())
        rewards.append(calc_reward(d))
        images.append(image.cpu().detach().numpy().squeeze(0))

        # update the state
        d, theta = dynamics(d, theta, v, dt, L, pred_steer.item())

    return np.array(rewards), np.array(images), np.array(action_array), np.array(lateral_error_array), np.array(theta_array)

def dynamics(d, theta, v, dt, L, steering_angle):
    d = d + v * dt * np.sin(theta)
    theta = theta + (v / L) * dt * np.tan(steering_angle)

    return d, theta

def calc_reward(d):
    return 1 - abs(d)
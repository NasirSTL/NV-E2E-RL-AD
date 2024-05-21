import torch
import torch.nn as nn
import numpy as np


def calculate_loss(cfg, model, images, true_actions, lateral_errors_dist):
    """
    Calculate the loss
    :param cfg: configurations
    :param model: controller model
    :param images: generator images
    :param true_actions: ground truth actions
    :param lateral_errors_dist: lateral errors distribution
    :return: loss
    """
    images = torch.as_tensor(images, dtype=torch.float).cuda()
    _, _, actions = model(images)
    actions = actions.squeeze()
    true_actions = torch.as_tensor(true_actions, dtype=torch.float).cuda()
    loss = imitation_loss(actions, true_actions)
    for index, l in enumerate(cfg.loss):
        function_name = l + '_loss'
        if l == 'imitation':
            continue
        elif l == 'lateral':
            loss_l = function_name(loss, lateral_errors_dist)
        else:
            raise NotImplementedError
        loss = loss + cfg.loss_weights[index] * loss_l

    return loss

def imitation_loss(y_pred, y_true):
    """
    Compute the imitation loss
    :param y_pred: predicted values
    :param y_true: ground truth values
    :return: imitation loss
    """
    return nn.MSELoss()(y_pred, y_true)
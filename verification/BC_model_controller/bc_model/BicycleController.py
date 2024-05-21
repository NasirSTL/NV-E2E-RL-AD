import torch
import torch.nn as nn
import numpy as np

from BC_model_controller.bc_model.backbone import *
from BC_model_controller.utils.general import initialize_weights


class BC_controller(torch.nn.Module):
    def __init__(self, 
                 backbone='18', 
                 pretrained=True, 
                 input_height=128,
                 input_width=128):
        super(BC_controller, self).__init__()

        self.input_dim = input_height // 16 * input_width // 16 * 8

        # self.backbone = resnet(backbone, pretrained)
        # self.pool = nn.Conv2d(512,8,1) if backbone in ['34','18', '34fca'] else nn.Conv2d(2048,8,1)
        
        self.backbone = featExtractor()
        initialize_weights(self.backbone)
        self.pool = nn.Conv2d(512,8,1)

        self.model_d = nn.Linear(self.input_dim, 1)
        self.model_theta = nn.Linear(self.input_dim, 1)
        self.model_steer = nn.Linear(self.input_dim, 1)

        initialize_weights(self.model_d, self.model_theta, self.model_steer)

    def forward(self, x):
        fea = self.backbone(x)
        fea = self.pool(fea)

        fea = fea.view(-1, self.input_dim)
        d, theta, steer = self.model_d(fea), self.model_theta(fea), self.model_steer(fea)

        return d, theta, steer
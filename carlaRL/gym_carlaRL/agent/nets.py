import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.05)
        nn.init.kaiming_normal_(tensor=m.weight, mode='fan_out', nonlinearity='relu')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.05)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.05)
        m.bias.data.fill_(0)


class ConvPPO(nn.Module):
    def __init__(self, obs_dim=1, action_dim=1):
        super(ConvPPO, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(obs_dim, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(1600, 512),
            nn.PReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.PReLU(),
            nn.Dropout(0.2),
            nn.Linear(10, action_dim),
        )

        self.model = nn.Sequential(
            self.conv_layers,
            self.fc_layers
        )
        self.model.apply(weights_init)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x
    

class MLP(nn.Module):
    def __init__(self,
                 num_grid_row = 100,
                 num_cls_row = 56,
                 num_grid_col = 100,
                 num_cls_col = 41,
                 num_lane_on_row = 4,
                 num_lane_on_col = 4):
        super(MLP, self).__init__()

        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col

        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col

        self.m1 = nn.Sequential(
            nn.Linear(self.dim1, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        self.m2 = nn.Sequential(
            nn.Linear(self.dim2, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 128),
            nn.ReLU(),
        )

        self.m3 = nn.Sequential(
            nn.Linear(self.dim3, 32),
            nn.ReLU(),
        )

        self.m4 = nn.Sequential(
            nn.Linear(self.dim4, 32),
            nn.ReLU(),
        )

        self.final = nn.Sequential(
            nn.Linear(128 + 128 + 32 + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x1, x2, x3, x4 = self.process_input(x)

        x1 = self.m1(x1)
        x2 = self.m2(x2)
        x3 = self.m3(x3)
        x4 = self.m4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.final(x)

        return x
    
    def process_input(self, x):
        if isinstance(x, dict):
            # If the input is a single dictionary, convert it to a batch of size 1
            x = np.array([x])

        loc_row_list = []
        loc_col_list = []
        exist_row_list = []
        exist_col_list = []

        for pred_dict in x:
            loc_row_list.append(pred_dict['loc_row'].flatten())
            loc_col_list.append(pred_dict['loc_col'].flatten())
            exist_row_list.append(pred_dict['exist_row'].flatten())
            exist_col_list.append(pred_dict['exist_col'].flatten())

        x1 = torch.stack(loc_row_list, dim=0)
        x2 = torch.stack(loc_col_list, dim=0)
        x3 = torch.stack(exist_row_list, dim=0)
        x4 = torch.stack(exist_col_list, dim=0)

        return x1, x2, x3, x4
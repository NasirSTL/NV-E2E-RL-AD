import pandas as pd
import numpy as np
import torch
import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cGAN.sn_cGAN.generator import Generator
from BC_model_controller.bc_model.train_model import train_bc_controller
from BC_model_controller.bc_model.BicycleController import BC_controller
from BC_model_controller.bc_model.train_model import train_bc_controller

from BC_model_controller.utils.cfg_helper import parse_args


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # Load the generator model
    G = Generator(version=cfg.generator_version)
    G.load_state_dict(torch.load(cfg.generator_path))
    G.cuda()
    G.eval()

    # Load the controller model
    bc_controller = BC_controller().cuda()

    # Train the controller model
    train_bc_controller(cfg, G, bc_controller)


if __name__ == '__main__':
    main()
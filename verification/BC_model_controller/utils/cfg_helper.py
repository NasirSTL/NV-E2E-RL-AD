import argparse


def parse_args():
    SAVE_DIR = 'BC_model_controller/logs/test/'
    GEN_VERSION = 'v2'
    GEN_PATH = 'models/G_epoch=99_loss=0.069.pth'

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--log_cat', default=['epoch', 'loss', 'rewards', 'lateral_error'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=384)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_dir', default=SAVE_DIR)

    # training
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--loss', default=['imitation'])
    parser.add_argument('--loss_weights', default=[1.0])
    parser.add_argument('--scheduler', default='multi')
    parser.add_argument('--steps', default=[50, 80])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--warmup', default='linear')
    parser.add_argument('--warmup_iters', type=int, default=100)

    # generator model
    parser.add_argument('--generator_version', default=GEN_VERSION)
    parser.add_argument('--generator_path', default=GEN_PATH)

    return parser.parse_args()
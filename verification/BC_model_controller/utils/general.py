import torch
import numpy as np


def initialize_weights(*models):
    for model in models:
        real_init_weights(model)

def real_init_weights(m):
    if isinstance(m, list):
        for mini_m in m:
            real_init_weights(mini_m)
    else:
        if isinstance(m, torch.nn.Conv2d):    
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.normal_(0.0, std=0.01)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m,torch.nn.Module):
            for mini_m in m.children():
                real_init_weights(mini_m)
        else:
            print('unkonwn module', m)

def get_optimizer(cfg, model):
    training_params = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, cfg, iters_per_epoch):
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup, iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
    else:
        raise NotImplementedError
    return scheduler

def init_log(cfg):
    log = {}
    for key in cfg.log_cat:
        log[key] = []
    return log

def get_minibatch(total, batch_size):
    num_batches = total // batch_size
    indices = np.arange(total)
    np.random.shuffle(indices)
    return num_batches, indices


class MultiStepLR:
    def __init__(self, optimizer, steps, gamma = 0.1, iters_per_epoch = None, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # print(self.iters, self.iters_per_epoch, self.steps, power)
            
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)
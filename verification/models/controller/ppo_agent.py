import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import numpy as np
import scipy.signal


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.autograd.set_detect_anomaly(True)


class ReplayBuffer:
    def __init__(self):
        self.buffer = []
        self.advantages = None
        self.returns = None
        self.position = 0

    def add(self, obs, action, steer_guide, reward, done, value, logp):
        self.buffer.append(None)  # Expand the buffer if not at capacity
        
        # Store each component of the observation dictionary separately
        state = (obs['actor_img'], obs['actor_speed'], obs['actor_target_speed'], obs['vehicle_state'])
        
        # Store the transition in the buffer
        self.buffer[self.position] = (state, action, steer_guide, reward, done, value, logp)
        self.position = self.position + 1

    def store_adv_and_return(self, advantages, returns):
        self.advantages = advantages
        self.returns = returns

    def get(self, batch_indices):
        # Get a batch of experiences from the buffer
        if isinstance(batch_indices, int):
            batch_indices = [batch_indices]
        states, actions, steer_guides, rewards, dones, values, logps = zip(*[self.buffer[i] for i in batch_indices])

        # states, actions, rewards, dones, values, logps = zip(*self.buffer)
        
        actor_imgs, _, _, vehicle_states = zip(*states)
        
        # Convert each component to a numpy array (or Tensor, depending on your framework)
        actor_imgs = np.array(actor_imgs)
        vehicle_states = np.array(vehicle_states)
        actions = np.array(actions)
        steer_guides = np.array(steer_guides)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        logps = np.array(logps)

        return actor_imgs, vehicle_states, actions, steer_guides, rewards, dones, values, logps

    def get_advantages_and_returns(self, batch_indices):
        if isinstance(batch_indices, int):
            batch_indices = [batch_indices]
        advantages = [self.advantages[i] for i in batch_indices]
        returns = [self.returns[i] for i in batch_indices]

        return advantages, returns
    
    def clear(self):
        self.buffer.clear()
        self.advantages = None
        self.returns = None
        self.position = 0

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)
    

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, learning_rate):
        super().__init__()

        log_std = -3.0 * np.ones(action_dim, dtype=np.float32)  # Log standard deviation
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std, device=DEVICE))
        # log_std = -3.0 * np.ones(action_dim, dtype=np.float32)  # Log standard deviation
        # self.log_std = torch.as_tensor(log_std, dtype=torch.float32, device=DEVICE)
        
        self.mu = torch.as_tensor(0.0, device=DEVICE)

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),
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

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, image, actions=None):
        x = torch.as_tensor(image).float().to(DEVICE)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Ensure it has batch dimension
        # print(f'input shape to conv_layers: {x.shape}')
        x = self.conv_layers(x)
        # print(f'output shape after conv_layers: {x.shape}')
        # print(f'input shape to fc_layers: {x.shape}')
        x = self.fc_layers(x)

        mu = torch.tanh(x)
        self.mu = mu
        std = torch.exp(self.log_std)  # Standard deviation
        # print(f'mu: {mu}; std: {std}')
        pi = Normal(mu, std)

        if actions is None:
            if self.training:
                action = pi.sample()
                logp = pi.log_prob(action)
            else:
                action = mu
                logp = None
            return action, logp
        else:
            actions = actions.unsqueeze(-1)  # Reshape actions to [10, 1] to match mu
            logps = pi.log_prob(actions).squeeze(-1)  # Compute log_prob and then squeeze back to [10]

            # Compute the entropy
            entropy = pi.entropy().mean()
            # print(f'entropy: {entropy}')
            return logps, mu, entropy


class Critic(nn.Module):
    def __init__(self, obs_dim, learning_rate):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1),
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
            nn.Linear(10, 1),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, image):
        x = torch.as_tensor(image).float().to(DEVICE)
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        value = torch.squeeze(x, -1)

        return value



class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, pi_lr=1e-4, v_lr=1e-3, pi_epochs=5, v_epochs=5):
        super(ActorCritic, self).__init__()

        self.memory = ReplayBuffer()
        # self.last_value = torch.tensor(0.0).float().to(DEVICE)  # Last value of the trajectory
        self.last_value = 0.0  # Last value of the trajectory
        self.append_last_value = False  # Whether to append the last value to the trajectory
        
        # The number of epochs over which to optimize the PPO loss for each batch of data
        self.pi_epochs = pi_epochs
        self.v_epochs = v_epochs
        
        # Initialize the actor and critic networks
        self.pi = Actor(obs_dim, action_dim, pi_lr).to(DEVICE)
        self.v = Critic(obs_dim, v_lr).to(DEVICE)


    def forward(self, image):
        action, logp = self.pi(image)
        value = self.v(image)
        
        return action, value, logp
    
    # def store_transition(self, state, action, reward, done, value, logp):
    #     # print(f'sanity check before storing experience: \n value: {value} \n logp: {logp}')
    #     self.memory.add(state, action, reward, done, value, logp)

    def finish_path(self, last_value=0, bootstrap=True, v_index=0):
        if bootstrap:
            self.last_value = last_value
        else:
            _, _, _, _, _, value, _ = self.memory.get(v_index)
            self.last_value = value

    def compute_pi_loss(self, images, vehicle_states, actions, steer_guides, advantages, logps_old, clip_ratio=0.2, beta=0.01):
        print(f'\n sanity check at computing pi loss:')
        logps, means, entropy = self.pi(images, actions)
        # print(f'means: \n {means.squeeze(-1)} \n guide steers: \n {steer_guides}')
        ratio = torch.exp(logps - logps_old)
        # print(f'actions: \n {actions}')
        print(f'logps: \n {logps} \n logps_old: \n {logps_old}')
        print(f'ratio: \n {ratio}')
        surr1 = ratio * advantages
        # print(f'adv: \n {advantages}')
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        # print(f'surr1: \n {surr1} \n surr2: \n {surr2}')
        loss_ppo = -torch.min(surr1, surr2).mean()
        loss_imitation = ((means - steer_guides)**2).mean()
        loss_ent = - beta * entropy
        loss_pi = loss_ppo + loss_imitation * 10
        print(f'loss_ppo: {loss_ppo.item():.4f}; loss_imitation: {loss_imitation.item():.4f}; loss_pi: {loss_pi.item():.4f}')
        return loss_pi, entropy, loss_ppo
    
    def compute_v_loss(self, images, vehicle_states, returns):
        value = self.v(images)
        # print(f'sanity check at compute_v_loss: \n value: {value} \n returns: {returns}')
        loss_v = ((value - returns)**2).mean()
        return loss_v

    def update(self, batch_indices, clip_param=0.2, beta=0.01):
        policy_loss = []
        value_loss = []
        entropy_list = []
        # if batch_indices is None:
        #     # If batch_indices is None, then the entire buffer is used to get adv and returns for this episode
        #     batch_indices = np.arange(len(self.memory))
        #     _, _, _, rewards, dones, values, _ = self.memory.get(batch_indices)
        #     advantages, returns = self.compute_gae(rewards, values, dones, gamma, lambda_gae)
        #     self.memory.store_adv_and_return(advantages, returns)
        #     return
        
        # Sample a batch of experiences
        images, vehicle_states, action, steer_guides, _, _, _, logps = self.memory.get(batch_indices)
        # convert to tensor
        actions = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        steer_guides = torch.as_tensor(steer_guides, dtype=torch.float32, device=DEVICE)
        # rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
        # dones = torch.tensor(dones, dtype=torch.float32, device=DEVICE)
        # values = values.squeeze()
        # values = torch.tensor(values, dtype=torch.float32, device=DEVICE)
        logps = torch.as_tensor(logps, dtype=torch.float32, device=DEVICE)

        # print(f'sanity check after sampling from buffer: \n values: {values} \n logps: {logps}')

        # compute returns and advantages
        advantages, returns = self.memory.get_advantages_and_returns(batch_indices)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

        # train policy
        for _ in range(self.pi_epochs):
            self.pi.optimizer.zero_grad()
            pi_loss, entropy, ppo_loss = self.compute_pi_loss(images, vehicle_states, actions, steer_guides, advantages, logps, clip_param, beta=beta)
            # print(f'policy_loss: {pi_loss.item()}')
            pi_loss.backward()
            self.pi.optimizer.step()
            policy_loss.append(ppo_loss.item())
            entropy_list.append(entropy.item())

        # train value function
        for _ in range(self.v_epochs):
            self.v.optimizer.zero_grad()
            v_loss = self.compute_v_loss(images, vehicle_states, returns)
            # print(f'value_loss: {v_loss.item()}')
            v_loss.backward()
            self.v.optimizer.step()
            value_loss.append(v_loss.item())


        # values = torch.stack(values)
        # logps = torch.stack(logps).squeeze()

        # if self.append_last_value:
        #     # values = torch.concat([values, self.last_value.unsqueeze(0)], dim=0)  # Add the last value to the trajectory
        #     # rewards = torch.concat(rewards, self.last_value)  # Add the last reward to the trajectory
        #     values = np.append(values, self.last_value)  # Add the last value to the trajectory
        #     # self.last_value = self.last_value.detach().cpu().numpy()
        #     rewards = np.append(rewards, self.last_value)  # Add the last reward to the trajectory
        #     self.append_last_value = False

        # Clear the memory after updating
        # self.memory.clear()

        # print(f'sanity check after update: \n policy_loss: {policy_loss} \n value_loss: {value_loss}')

        return np.mean(policy_loss), np.mean(value_loss), np.mean(entropy_list)

    def compute_gae(self, gamma=0.99, lam=0.95):
        # compute generalized advantage estimation
        batch_indices = np.arange(len(self.memory))
        _, _, _, _, rewards, dones, values, _ = self.memory.get(batch_indices)

        values4adv = np.append(values, self.last_value)  # Add the last value to the trajectory
        # print(f'values4adv: {values4adv}')

        deltas = rewards + (1-dones) * gamma * values4adv[1:] - values4adv[:-1]
        # print(f'deltas: {deltas}')

        advantages = scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]
        advantages = advantages.copy()
        
        # advantages = []
        # advantage = 0.0
        # for delta, done in zip(reversed(deltas), reversed(dones)):
        #     if done:
        #         advantage = 0  # Reset advantage at the end of an episode
        #     advantage = delta + gamma * lambda_gae * advantage
        #     advantages.insert(0, advantage)  # Insert to maintain the original order

        # Compute returns as the sum of advantages and values
        # returns = advantages + values
        returns = scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]
        returns = returns.copy()

        # print(f'sanity check at gae calculation: \n values: \n {values4adv} \n advantages: \n {advantages} \n returns: \n {returns}')

        # returns = torch.stack(returns)
        # advantages = torch.stack(advantages)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # print(f'sanity check at gae calculation: \n values: {values} \n returns: {returns}')

        self.memory.store_adv_and_return(advantages, returns)

        return advantages, returns


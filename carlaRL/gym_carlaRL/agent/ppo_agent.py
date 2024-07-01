import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
import numpy as np
import scipy.signal


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class ReplayBuffer:
    def __init__(self, input_format='image'):
        self.buffer = []
        self.advantages = None
        self.returns = None
        self.position = 0
        self.input_format = input_format

    def add(self, obs, action, steer_guide, reward, done, value, logp):
        self.buffer.append(None)  # Expand the buffer if not at capacity

        if self.input_format == 'image':
            actor_input_np = obs['actor_input']
        elif self.input_format == 'dict':
            actor_input_np = {k: v.cpu().numpy() for k, v in obs['actor_input'].items()}
        state = (actor_input_np, obs['vehicle_state'])
        
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
        actor_inputs, vehicle_states = zip(*states)
        
        # Convert each component to a numpy array (or Tensor, depending on your framework)
        actor_inputs = np.array(actor_inputs)
        vehicle_states = np.array(vehicle_states)
        actions = np.array(actions)
        steer_guides = np.array(steer_guides)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        logps = np.array(logps)

        return actor_inputs, vehicle_states, actions, steer_guides, rewards, dones, values, logps

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

        # self.model = nn.Sequential(
        #     self.conv_layers,
        #     self.fc_layers,
        # )
        # self.model.apply(weights_init)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, image, actions=None):
        x = torch.as_tensor(image).float().to(DEVICE)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Ensure it has batch dimension
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        mu = torch.tanh(x)
        self.mu = mu
        std = torch.exp(self.log_std)
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

        # self.model = nn.Sequential(
        #     self.conv_layers,
        #     self.fc_layers,
        # )
        # self.model.apply(weights_init)
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
    def __init__(self, obs_dim=1, action_dim=1, pi_lr=1e-4, v_lr=1e-3, pi_epochs=5, v_epochs=5):
        super(ActorCritic, self).__init__()

        self.memory = ReplayBuffer()
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

    def finish_path(self, last_value=0, bootstrap=True, v_index=0):
        if bootstrap:
            self.last_value = last_value
        else:
            _, _, _, _, _, value, _ = self.memory.get(v_index)
            self.last_value = value

    def compute_pi_loss(self, images, vehicle_states, actions, steer_guides, advantages, logps_old, clip_ratio=0.2, beta=0.01):
        logps, means, entropy = self.pi(images, actions)
        ratio = torch.exp(logps - logps_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

        loss_ppo = -torch.min(surr1, surr2).mean()
        loss_imitation = ((means - steer_guides)**2).mean()
        loss_ent = - beta * entropy
        loss_pi = loss_ppo + loss_imitation * 10

        return loss_pi, entropy, loss_ppo
    
    def compute_v_loss(self, images, vehicle_states, returns):
        value = self.v(images)
        loss_v = ((value - returns)**2).mean()
        return loss_v

    def update(self, batch_indices, clip_param=0.2, beta=0.01):
        policy_loss = []
        value_loss = []
        entropy_list = []
        
        # Sample a batch of experiences
        images, vehicle_states, action, steer_guides, _, _, _, logps = self.memory.get(batch_indices)
        # convert to tensor
        actions = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        steer_guides = torch.as_tensor(steer_guides, dtype=torch.float32, device=DEVICE)
        logps = torch.as_tensor(logps, dtype=torch.float32, device=DEVICE)

        # compute returns and advantages
        advantages, returns = self.memory.get_advantages_and_returns(batch_indices)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

        # train policy
        for _ in range(self.pi_epochs):
            self.pi.optimizer.zero_grad()
            pi_loss, entropy, ppo_loss = self.compute_pi_loss(images, vehicle_states, actions, steer_guides, advantages, logps, clip_param, beta=beta)
            pi_loss.backward()
            self.pi.optimizer.step()
            policy_loss.append(ppo_loss.item())
            entropy_list.append(entropy.item())

        # train value function
        for _ in range(self.v_epochs):
            self.v.optimizer.zero_grad()
            v_loss = self.compute_v_loss(images, vehicle_states, returns)
            v_loss.backward()
            self.v.optimizer.step()
            value_loss.append(v_loss.item())

        return np.mean(policy_loss), np.mean(value_loss), np.mean(entropy_list)

    def compute_gae(self, gamma=0.99, lam=0.95):
        batch_indices = np.arange(len(self.memory))
        _, _, _, _, rewards, dones, values, _ = self.memory.get(batch_indices)

        """
        print("IN compute_gae")
        print("batch_indices: ", self.memory.get(batch_indices))
        print("rewards: ", rewards)
        print("dones: ", dones)
        print("values: ", values)
        """

        values4adv = np.append(values, self.last_value)  # Add the last value to the trajectory

        """
        print("values4adv: ", values4adv)
        dones = np.where(dones == None, False, dones) #in case of Nonetype
        print("dones: ", dones)
        """

        deltas = rewards + (1-dones) * gamma * values4adv[1:] - values4adv[:-1]

        advantages = scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]
        advantages = advantages.copy()
        
        returns = scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]
        returns = returns.copy()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.memory.store_adv_and_return(advantages, returns)

        return advantages, returns
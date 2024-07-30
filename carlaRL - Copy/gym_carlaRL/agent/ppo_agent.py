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
        state = (actor_input_np, obs['vehicle_state'], obs['command'], obs['next_command'])
        
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
        actor_inputs, vehicle_states, commands, next_commands = zip(*states)
        
        # Convert each component to a numpy array (or Tensor, depending on your framework)
        actor_inputs = np.array(actor_inputs)
        vehicle_states = np.array(vehicle_states)
        actions = np.array(actions)
        steer_guides = np.array(steer_guides)
        rewards = np.array(rewards)
        dones = np.array(dones)
        values = np.array(values)
        logps = np.array(logps)

        return commands, next_commands, actor_inputs, vehicle_states, actions, steer_guides, rewards, dones, values, logps

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

        # Fully connected layers for command processing
        self.command_fc = nn.Sequential(
            nn.Embedding(3, 10),  # Assuming 5 is the number of discrete commands
            nn.Linear(10, 30),
            nn.PReLU()
        )

        # Fully connected layers for command processing
        self.next_command_fc = nn.Sequential(
            nn.Embedding(4, 10),  # Assuming 6 is the number of discrete commands
            nn.Linear(10, 40),
            nn.PReLU()
        )

        # Fully connected layers for combined features
        self.fc_layers = nn.Sequential(
            nn.Linear(1600 + 30 + 40, 512),  # Adjust input dimension to include command features
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

    def forward(self, command, next_command, image, actions=None):
        
        #add command arg
        x = torch.as_tensor(image).float().to(DEVICE)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Ensure it has batch dimension

        #print(f"Image tensor shape after unsqueeze: {x.shape} (Expected: [N, 1, H, W])")

        # Process image
        image_features = self.conv_layers(x)
        #print(f"Image features shape after conv layers: {image_features.shape} (Expected: [N, C', H', W'])")
        
        # Flatten image features for concatenation
        image_features = image_features.view(image_features.size(0), -1)
        #print(f"Image features shape after flatten: {image_features.shape} (Expected: [N, 1600])")
        
        # Process command
        command = torch.as_tensor(command).int().to(DEVICE)
        command_features = self.command_fc(command)
        #print(f"Command features shape after command_fc: {command_features.shape} (Expected: [N, 50])")
        
        # Ensure command_features has the same batch dimension
        if command_features.dim() == 1:
            command_features = command_features.unsqueeze(0)
        
        # Expand command_features to match the batch size of image_features
        command_features = command_features.expand(image_features.size(0), -1)
        #print(f"Command features shape after expand: {command_features.shape} (Expected: [N, 50])")

        # Process next command
        next_command = torch.as_tensor(next_command).int().to(DEVICE)
        next_command_features = self.next_command_fc(next_command)
        
        # Ensure command_features has the same batch dimension
        if next_command_features.dim() == 1:
            next_command_features = next_command_features.unsqueeze(0)
        
        # Expand command_features to match the batch size of image_features
        next_command_features = next_command_features.expand(image_features.size(0), -1)

        # Combine features
        combined_features = torch.cat([image_features, command_features, next_command_features], dim=1)
        
        # Forward pass through fully connected layers
        x = self.fc_layers(combined_features)
        #print(f"Output shape after fc layers: {x.shape} (Expected: [N, 1])")
        
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

        # Fully connected layers for command processing
        self.command_fc = nn.Sequential(
            nn.Embedding(3, 10),  # Assuming command_dim is the number of discrete commands
            nn.Linear(10, 30),
            nn.PReLU()
        )

         # Fully connected layers for command processing
        self.next_command_fc = nn.Sequential(
            nn.Embedding(4, 10),  # Assuming command_dim is the number of discrete commands
            nn.Linear(10, 40),
            nn.PReLU()
        )

        # Fully connected layers for combined features
        self.fc_layers = nn.Sequential(
            nn.Linear(1600 + 30 + 40, 512),  # Adjust input dimension to include command features
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

    def forward(self, command, next_command, image):
        x = torch.as_tensor(image).float().to(DEVICE)
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Ensure it has batch dimension

        image_features = self.conv_layers(x)
        
        # Flatten image features for concatenation
        image_features = image_features.view(image_features.size(0), -1)
        
        # Process command
        command = torch.as_tensor(command).int().to(DEVICE)
        command_features = self.command_fc(command)
        
        # Ensure command_features has the same batch dimension
        if command_features.dim() == 1:
            command_features = command_features.unsqueeze(0)
        
        # Expand command_features to match the batch size of image_features
        command_features = command_features.expand(image_features.size(0), -1)

        # Process next command
        next_command = torch.as_tensor(next_command).int().to(DEVICE)
        next_command_features = self.next_command_fc(next_command)
        
        # Ensure command_features has the same batch dimension
        if next_command_features.dim() == 1:
            next_command_features = next_command_features.unsqueeze(0)
        
        # Expand command_features to match the batch size of image_features
        next_command_features = next_command_features.expand(image_features.size(0), -1)

        # Combine features
        combined_features = torch.cat([image_features, command_features, next_command_features], dim=1)
        
        # Forward pass through fully connected layers
        x = self.fc_layers(combined_features)

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

    def forward(self, image, command, next_command):
        action, logp = self.pi(command, next_command, image)
        value = self.v(command, next_command, image)
        
        return action, value, logp

    def finish_path(self, last_value=0, bootstrap=True, v_index=0):
        if bootstrap:
            self.last_value = last_value
        else:
            _, _, _, _, _, _, _, _, value, _ = self.memory.get(v_index)
            self.last_value = value

    def compute_pi_loss(self, commands, next_commands, images, vehicle_states, actions, steer_guides, advantages, logps_old, clip_ratio=0.2, beta=0.01):
        logps, means, entropy = self.pi(commands, next_commands, images, actions)
        ratio = torch.exp(logps - logps_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages

        loss_ppo = -torch.min(surr1, surr2).mean()
        loss_imitation = ((means - steer_guides)**2).mean()
        loss_ent = - beta * entropy
        loss_pi = loss_ppo + loss_imitation * 10

        return loss_pi, entropy, loss_ppo
    
    def compute_v_loss(self, commands, next_commands, images, vehicle_states, returns):
        value = self.v(commands, next_commands, images)
        loss_v = ((value - returns)**2).mean()
        return loss_v

    def update(self, batch_indices, beta, clip_param=0.2):
        policy_loss = []
        value_loss = []
        entropy_list = []
        
        # Sample a batch of experiences
        command, next_command, images, vehicle_states, action, steer_guides, _, _, _, logps = self.memory.get(batch_indices)
        # convert to tensor
        actions = torch.as_tensor(action, dtype=torch.float32, device=DEVICE)
        steer_guides = torch.as_tensor(steer_guides, dtype=torch.float32, device=DEVICE)
        logps = torch.as_tensor(logps, dtype=torch.float32, device=DEVICE)

        # compute returns and advantages
        advantages, returns = self.memory.get_advantages_and_returns(batch_indices)
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=DEVICE)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=DEVICE)

        # train policy add command for pi and v loss
        for _ in range(self.pi_epochs):
            self.pi.optimizer.zero_grad()
            pi_loss, entropy, ppo_loss = self.compute_pi_loss(command, next_command, images, vehicle_states, actions, steer_guides, advantages, logps, clip_param, beta=beta)
            pi_loss.backward()
            self.pi.optimizer.step()
            policy_loss.append(ppo_loss.item())
            entropy_list.append(entropy.item())

        # train value function
        for _ in range(self.v_epochs):
            self.v.optimizer.zero_grad()
            v_loss = self.compute_v_loss(command, next_command, images, vehicle_states, returns)
            v_loss.backward()
            self.v.optimizer.step()
            value_loss.append(v_loss.item())

        return np.mean(policy_loss), np.mean(value_loss), np.mean(entropy_list)

    def compute_gae(self, gamma=0.99, lam=0.95):
        batch_indices = np.arange(len(self.memory))
        _, _, _, _, _, _, rewards, dones, values, _ = self.memory.get(batch_indices)

        values4adv = np.append(values, self.last_value)  # Add the last value to the trajectory

        deltas = rewards + (1-dones) * gamma * values4adv[1:] - values4adv[:-1]

        advantages = scipy.signal.lfilter([1], [1, -gamma * lam], deltas[::-1], axis=0)[::-1]
        advantages = advantages.copy()
        
        returns = scipy.signal.lfilter([1], [1, -gamma], rewards[::-1], axis=0)[::-1]
        returns = returns.copy()

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.memory.store_adv_and_return(advantages, returns)

        return advantages, returns
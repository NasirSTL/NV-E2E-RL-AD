import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def add(self, obs, action, reward, next_obs, done):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)  # Expand the buffer if not at capacity
        
        # Store each component of the observation dictionary separately
        state = (obs['actor_img'], obs['actor_speed'], obs['actor_target_speed'], obs['vehicle_state'])
        next_state = (next_obs['actor_img'], next_obs['actor_speed'], next_obs['actor_target_speed'], next_obs['vehicle_state'])
        
        # Store the transition in the buffer
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Randomly sample a batch of experiences from memory."""
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack and organize the components of each experience
        states, actions, rewards, next_states, dones = zip(*batch)
        
        actor_imgs, actor_speeds, actor_target_speeds, vehicle_states = zip(*states)
        next_actor_imgs, next_actor_speeds, next_actor_target_speeds, next_vehicle_states = zip(*next_states)
        
        # Convert each component to a numpy array (or Tensor, depending on your framework)
        actor_imgs = np.array(actor_imgs)
        actor_speeds = np.array(actor_speeds)
        actor_target_speeds = np.array(actor_target_speeds)
        vehicle_states = np.array(vehicle_states)

        next_actor_imgs = np.array(next_actor_imgs)
        next_actor_speeds = np.array(next_actor_speeds)
        next_actor_target_speeds = np.array(next_actor_target_speeds)
        next_vehicle_states = np.array(next_vehicle_states)

        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        return (actor_imgs, actor_speeds, actor_target_speeds, vehicle_states), actions, rewards, (next_actor_imgs, next_actor_speeds, next_actor_target_speeds, next_vehicle_states), dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.buffer)

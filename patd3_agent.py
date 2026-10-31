"""Actor, twin critics and the TD3 update used by PATD3 (Algorithm 1)."""
import numpy as np
import random
import copy
import os
import torch
import torch.nn.functional as F
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(torch.nn.Module):
    def __init__(self, state_size, action_size, max_action=1.0):
        super(Actor, self).__init__()
        self.max_action = float(max_action)
        self.fc1 = torch.nn.Linear(state_size, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.mu = torch.nn.Linear(64, action_size)
        self.act = torch.nn.LeakyReLU()
        self.tanh = torch.nn.Tanh()
        self.layernorm = torch.nn.ModuleList([torch.nn.LayerNorm(64) for _ in range(3)])

    def forward(self, state):
        x = self.act(self.fc1(state))
        x = self.layernorm[0](x)
        x = self.act(self.fc2(x))
        x = self.layernorm[1](x)
        x = self.act(self.fc3(x))
        x = self.layernorm[2](x)
        return self.max_action * self.tanh(self.mu(x))

class Critic(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()

        self.fc1 = torch.nn.Linear(state_size , 64)
        self.fc2 = torch.nn.Linear(64 + action_size, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.q = torch.nn.Linear(64, 1)
        self.fc4 = torch.nn.Linear(state_size , 64)
        self.fc5 = torch.nn.Linear(64 + action_size, 64)
        self.fc6 = torch.nn.Linear(64, 64)
        self.q2 = torch.nn.Linear(64, 1)

        self.act = torch.nn.LeakyReLU()
        self.layernorm = torch.nn.ModuleList([torch.nn.LayerNorm(64) for _ in range(6)])

    def forward(self, state, action):
        x = self.act(self.fc1(state))
        x = self.layernorm[0](x)
        x = torch.cat((x, action), dim=-1)
        x = self.act(self.fc2(x))
        x = self.layernorm[1](x)
        x = self.act(self.fc3(x))
        x = self.layernorm[2](x)

        x2 = self.act(self.fc4(state))
        x2 = self.layernorm[3](x2)
        x2 = torch.cat((x2, action), dim=-1)
        x2 = self.act(self.fc5(x2))
        x2 = self.layernorm[4](x2)
        x2 = self.act(self.fc6(x2))
        x2 = self.layernorm[5](x2)
        return self.q(x), self.q2(x2)

    def Q1(self, state, action):
        x = self.act(self.fc1(state))
        x = self.layernorm[0](x)
        x = torch.cat((x, action), dim=-1)
        x = self.act(self.fc2(x))
        x = self.layernorm[1](x)
        x = self.act(self.fc3(x))
        x = self.layernorm[2](x)

        return self.q(x)

def build_agent(params):
    return PATD3Agent(params['state_size'], params['action_size'], params['MEMORY_SIZE'],
                      params['actor_lr'], params['critic_lr'],
                      params['policy_freq'], params['tau'],
                      params['policy_noise'], params['noise_clip'], params['expl_noise'],
                      params['max_action'])


class PATD3Agent():
    def __init__(self, state_size, action_size, mem_maxlen, actor_lr, critic_lr,
                 policy_freq, tau, policy_noise, noise_clip, expl_noise,
                 max_action=1.0):
        self.action_dim = action_size
        self.max_action = float(max_action)
        self.actor = Actor(state_size, action_size, max_action).to(device)
        self.target_actor = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = Critic(state_size, action_size).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.memory = deque(maxlen=mem_maxlen)
        self.policy_freq = policy_freq
        self.tau = tau
        self.total_it = 0
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.expl_noise = expl_noise

    def select_action(self, state):
        action = self.actor(torch.FloatTensor(state).to(device)).cpu().detach().numpy()
        return [action]

    def append_sample(self, state, action, reward, next_state, not_done):
        self.memory.append((state, action, reward, next_state, not_done))

    def train_model(self, batch_size, discount_factor):
        self.total_it += 1
        batch = random.sample(self.memory, batch_size)

        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)

        not_done   = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, not_done = map(lambda x: torch.FloatTensor(x).to(device),
                                                          [state, action, reward, next_state, not_done])

        noise = (torch.randn_like(action) * self.policy_noise * self.max_action).clamp(
            -self.noise_clip * self.max_action, self.noise_clip * self.max_action)

        with torch.no_grad():
            proto_next_actions = (
                    self.target_actor(next_state) + noise).clamp(-self.max_action, self.max_action)
            next_q1, next_q2 = self.target_critic(next_state, proto_next_actions)
            next_q = torch.min(next_q1, next_q2)
        target_q = reward + not_done * discount_factor * next_q

        q1, q2 = self.critic(state, action)
        # Huber loss on both critics (Algorithm 1, line 16)
        critic_loss = F.smooth_l1_loss(target_q, q1) + F.smooth_l1_loss(target_q, q2)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            action_pred = self.actor(state)
            actor_loss = -self.critic.Q1(state, action_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update_target(self.tau)

    def soft_update_target(self, tau):
        for target_param, local_param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "actor" : self.actor.state_dict(),
            "actor_optimizer" : self.actor_optimizer.state_dict(),
            "critic" : self.critic.state_dict(),
            "critic_optimizer" : self.critic_optimizer.state_dict(),
        }, save_path+'/ckpt')

    def load_model(self, load_path):
        print(f"... Load Model from {load_path}/ckpt ...")
        checkpoint = torch.load(load_path+'/ckpt', map_location=device, weights_only=True)
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["actor"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["critic"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

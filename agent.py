import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, state_dim=8, action_dim=2):
        super().__init__()
        # Standard SOTA Actor Network (64 units per RL-Zoo / CleanRL)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

        # Standard SOTA Critic Network (64 units per RL-Zoo / CleanRL)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def get_value(self, state):
        return self.critic(state)

    def get_action_and_value(self, state, action=None):
        action_mean = self.actor_mean(state)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action).sum(dim=-1)
        entropy = probs.entropy().sum(dim=-1)
        value = self.critic(state)

        return action, log_prob, entropy, value


class PPOBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

    def add(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_gae(self, next_value, gamma=0.999, gae_lambda=0.98):
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        values = np.array(self.values + [next_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + values[:-1]
        return advantages, returns


class PPOAgent:
    def __init__(
        self,
        state_dim=8,
        action_dim=2,
        lr=3e-4,
        gamma=0.999,      # RL-Zoo LunarLanderContinuous gold standard
        gae_lambda=0.98,   # RL-Zoo LunarLanderContinuous gold standard
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,   # RL-Zoo gold standard: 4 epochs to prevent policy collapse
        batch_size=64,
        device=None,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.network = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = PPOBuffer()

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def select_action(self, state):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.network.get_action_and_value(state_t)
            return (
                action.cpu().numpy()[0],
                log_prob.cpu().numpy()[0],
                value.cpu().numpy()[0][0],
            )

    def select_deterministic_action(self, state):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_mean = self.network.actor_mean(state_t)
            return np.clip(action_mean.cpu().numpy()[0], -1.0, 1.0)

    def get_value(self, state):
        with torch.no_grad():
            state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return float(self.network.get_value(state_t).cpu().numpy()[0][0])

    def update(self, next_value):
        advantages, returns = self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)

        b_states = torch.as_tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        b_actions = torch.as_tensor(np.array(self.buffer.actions), dtype=torch.float32, device=self.device)
        b_logprobs = torch.as_tensor(np.array(self.buffer.log_probs), dtype=torch.float32, device=self.device)
        b_advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        b_returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        b_values = torch.as_tensor(np.array(self.buffer.values), dtype=torch.float32, device=self.device)

        # Advantage Normalization
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        dataset_size = len(self.buffer.states)
        indices = np.arange(dataset_size)

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]

                _, new_logprob, entropy, new_value = self.network.get_action_and_value(
                    b_states[batch_idx], b_actions[batch_idx]
                )

                log_ratio = new_logprob - b_logprobs[batch_idx]
                ratio = torch.exp(log_ratio)

                # Policy Loss with Clipped Objective
                mb_advantages = b_advantages[batch_idx]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1.0 - self.clip_coef, 1.0 + self.clip_coef)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value Loss with Value Clipping
                new_value = new_value.view(-1)
                v_loss_unclipped = (new_value - b_returns[batch_idx]) ** 2
                v_clipped = b_values[batch_idx] + torch.clamp(
                    new_value - b_values[batch_idx], -self.clip_coef, self.clip_coef
                )
                v_loss_clipped = (v_clipped - b_returns[batch_idx]) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                # Entropy Regularization Loss
                entropy_loss = entropy.mean()

                loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.buffer.clear()

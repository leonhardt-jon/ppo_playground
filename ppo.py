import torch
import torch.nn as nn
import torch.optim as optim
#from torch.distributions import Normal
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class PPO:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, clip_ratio, gamma, lam):
        self.action_dim = action_dim
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.clip_ratio = clip_ratio
        self.gamma = gamma
        self.lam = lam

    def select_action(self, state):
        state = torch.FloatTensor(state)
        mean_act = self.actor(state)
        #dist = MultivariateNormal(mean_act, torch.eye(self.action_dim))
        dist = Categorical(mean_act)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def _eval(self, state):
        state = torch.FloatTensor(state)
        mean_act = self.actor(state)
        #dist = MultivariateNormal(mean_act, torch.eye(self.action_dim))
        dist = Categorical(mean_act)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy


    #def update(self, states, actions, log_probs, rewards, dones, next_states):
        #states = torch.FloatTensor(states)
        #actions = torch.FloatTensor(actions)
        #log_probs = torch.FloatTensor(log_probs)
        #rewards = torch.FloatTensor(rewards)
        #dones = torch.FloatTensor(dones)
        #next_states = torch.FloatTensor(next_states)

        #values = self.critic(states)
        #next_values = self.critic(next_states)
        #advantages = []
        #returns = []
        #for i in range(len(rewards)):
            #td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            #advantage = td_error
            #for j in range(i + 1, len(rewards)):
                #advantage += self.gamma * self.lam ** (j - i - 1) * rewards[j] * (1 - dones[j])
            #advantages.append(advantage)
            #returns.append(advantage + values[i])
        #advantages = torch.stack(advantages)
        #returns = torch.stack(returns)

        ##new_log_probs = self.actor(states).split(self.out_features // 2, dim=-1)[1].unsqueeze(1)#.transpose(1,0)
        #mean1, mean2, log_std1, log_std2 = torch.split(self.actor(states), self.out_features // 4, dim=-1)
        #new_log_probs = torch.stack([log_std1, log_std2], dim=1)
        #new_log_probs = new_log_probs.gather(1, actions.long())

        ##mean, log_std = torch.split(self.actor(states), self.out_features // 2, dim=-1)
        ##std = torch.exp(log_std)
        ##distribution = Normal(mean, std)
        ##new_log_probs = distribution.log_prob(new_actions).sum(-1, keepdim=True)

        ##print(new_log_probs.shape)
        ##print(log_probs.shape)
        #ratio = torch.exp(new_log_probs - log_probs)
        #surr1 = ratio * advantages
        #surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
        #actor_loss = -torch.min(surr1, surr2).mean()
        #critic_loss = nn.MSELoss()(values, returns)

        #self.actor_optimizer.zero_grad()
        #actor_loss.backward()
        #self.actor_optimizer.step()

        #self.critic_optimizer.zero_grad()
        #critic_loss.backward()
        #self.critic_optimizer.step()

        #return actor_loss.item(), critic_loss.item()



    def update(self, states, actions, log_probs, rewards, dones, next_states, num_updates=10, kl_target=0.01):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        log_probs = torch.FloatTensor(log_probs)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones)
        next_states = torch.FloatTensor(next_states)
        log_probs = log_probs.unsqueeze(1)
        for _ in range(num_updates):
            values = self.critic(states)
            next_values = self.critic(next_states)
            advantages, returns = self.compute_advantages_and_returns(rewards, dones, values, next_values)
            critic_loss = nn.MSELoss()(values, rewards) * 0.5

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        for _ in range(num_updates):

            with torch.no_grad():
                values = self.critic(states)
                next_values = self.critic(next_states)
                _, returns = self.compute_advantages_and_returns(rewards, dones, values, next_values)
                advantages = rewards - returns

            #kl_div = torch.distributions.kl_divergence(
                #torch.distributions.Normal(new_mean, torch.exp(new_log_probs)),
                #torch.distributions.Normal(actions, torch.exp(log_probs))
            #).mean()
            #print(new_log_probs.mean())
            #print(log_probs.mean())
            #new_mean, new_log_std = torch.split(self.actor(states), self.out_features // 2, dim=-1)
            #new_log_probs = torch.sum(new_log_std + (actions - new_mean) ** 2 / (2 * torch.exp(new_log_std) ** 2), dim=-1, keepdim=True)
            #_log_probs = torch.sum(log_probs, dim=-1, keepdim=True )
            new_log_probs, entropy = self._eval(states)

            ratio = torch.exp(new_log_probs - log_probs)
            advantages = torch.abs(advantages)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            kl_div = torch.log(torch.abs(new_log_probs.mean()) / torch.abs(log_probs.mean()))

            if kl_div > 2 * kl_target:
                print(kl_div)
                break

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()


        return actor_loss.item(), critic_loss.item()

    def compute_advantages_and_returns(self, rewards, dones, values, next_values):
        advantages = []
        returns = []
        for i in range(len(rewards)):
            td_error = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            advantage = td_error
            for j in range(i + 1, len(rewards)):
                advantage += self.gamma * self.lam ** (j - i - 1) * rewards[j] * (1 - dones[j])
            advantages.append(advantage)
            returns.append(advantage + values[i])
        return torch.stack(advantages), torch.stack(returns)
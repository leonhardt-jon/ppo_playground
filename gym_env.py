import gymnasium as gym
from gymnasium.utils.save_video import save_video 
from ppo import PPO
import numpy as np

env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
_ = env.reset()

class Normalizer():
    def __init__(self, min_values, max_values):
        self.min_values = np.array(min_values)
        self.max_values = np.array(max_values)
    
    def normalize(self, values):
        values = np.array(values)
        return (values - self.min_values) / (self.max_values - self.min_values)

normalizer = Normalizer(env.observation_space.low, env.observation_space.high)

state_dim = 8
action_dim = 4 
actor_lr = 1e-3
critic_lr = 1e-3
clip_ratio = 1.2
gamma = 0.9
lam = 0.9
ppo_ac = PPO( state_dim, action_dim, actor_lr, critic_lr, clip_ratio, gamma, lam)

episodes = 500
for episode in range(episodes):
    observation, info = env.reset(seed=42)
    step_starting_index = 0 
    episode_index = 0
    observations = [normalizer.normalize(observation)]
    rewards = []
    actions = []
    log_probs = []
    next_states = []
    dones = []
    for step_index in range(1000):
        norm_obs = normalizer.normalize(observation)
        action, log_prob = ppo_ac.select_action(norm_obs)  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(norm_obs)
        rewards.append(reward)
        actions.append(action)
        log_probs.append(log_prob)
        next_states.append(norm_obs)
        dones.append(int(terminated))

        if terminated or truncated:
                save_video(
                    env.render(),
                    "videos",
                    fps=16,
                    step_starting_index=step_starting_index,
                    episode_index=episode
                ) 
                break
    actor_loss, critic_loss = ppo_ac.update(np.array(observations[:-1]), np.array(actions), log_probs, np.array(rewards), np.array(dones), np.array(next_states), num_updates=50)
    print(episode)
    print(actor_loss, critic_loss)
    print(np.array(rewards).mean())
env.close()
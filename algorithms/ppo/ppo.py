import numpy as np
np.bool8 = bool
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from .network import FeedForwardNN
from torch.distributions import MultivariateNormal
from torch.optim import Adam
import matplotlib.pyplot as plt
import gymnasium as gym

class PPO:
    def __init__(self, env):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]

        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)

#Contencion por si el espacio es discreto y no continuo: se agrega otro 
# camino sin multivariate normal 
        if self.is_discrete:
            self.act_dim = env.action_space.n
        else:
            self.act_dim = env.action_space.shape[0]
            self.cov_var = torch.full(size=(self.act_dim,),fill_value=0.5)
            self.cov_mat = torch.diag(self.cov_var) 

        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        self._init_hyperparameters()

        self.actor_optim = Adam(self.actor.parameters(), lr = self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0 # The discounted reward so far
            for rew in reversed(ep_rews):
              discounted_reward = rew + discounted_reward * self.gamma
              batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def _init_hyperparameters(self):
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.99 #antes 0.95
        self.n_updates_per_iteration = 10
        self.clip = 0.2 # Para el clipping de la función de perdida
        self.lr = 3e-4 # antes 0.005
        self.max_grad_norm = 0.5
        self.target_kl = 0.02 #antes 0.01
        self.num_minibatches = 32 #antes 4
        self.ent_coef = 0.01
        self.lam = 0.95
        self.render = False

    def get_action(self, obs):

        out = self.actor(obs)
        if self.is_discrete:
            dist = torch.distributions.Categorical(logits=out)
        else:
            dist = MultivariateNormal(out, self.cov_mat)

       # mean = self.actor(obs)
       # dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.detach().numpy(), log_prob.detach()

    def learn(self, total_timesteps):
        t_so_far = 0 #timesteps hasta ahora
        ep_rewards_history = []

        while t_so_far < total_timesteps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_vals, batch_dones, batch_rews = self.rollout()
            t_so_far +=np.sum(batch_lens)

            #plot ---------------------------------------------------------------------------
            for ep_rew in batch_rews:
                ep_rewards_history.append(sum(ep_rew))
            
            avg_reward = np.mean([sum(ep) for ep in batch_rews])
            print(f'Timesteps: {t_so_far}, reward promedio del batch: {avg_reward:.1f}')
            #--------------------------------------------------------------------------------
            
# Calculate V_{phi, k}
            A_k = self.calculate_gae(batch_rews, batch_vals, batch_dones)

            V = self.critic(batch_obs).squeeze()
# ALG STEP 5
# Calculate advantage
            batch_rtgs = A_k + V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            #setea minibatches --------------------------------
            step = batch_obs.size(0)
            inds = np.arange(step)
            minibatch_size = step // self.num_minibatches
            # -------------------------------------------------

            for _ in range(self.n_updates_per_iteration):
                frac = (t_so_far - 1.0) / total_timesteps
                new_lr = self.lr * (1.0 - frac)
                new_lr = max(new_lr, 0.0001)
                self.actor_optim.param_groups[0]['lr'] = new_lr
                self.critic_optim.param_groups[0]['lr'] = new_lr
            
                np.random.shuffle(inds)
                for start in range(0, step, minibatch_size):
                    end = start + minibatch_size
                    idx = inds[start:end]
                    mini_obs = batch_obs[idx]
                    mini_acts = batch_acts[idx]
                    mini_log_probs = batch_log_probs[idx]
                    mini_advantage = A_k[idx]
                    mini_rtgs = batch_rtgs[idx]

                    # pi_tita (a_t | s_t)
                    V, curr_log_probs, entropy = self.evaluate(mini_obs, mini_acts)
                    logratios = curr_log_probs - mini_log_probs
                    ratios = torch.exp(logratios)
                    surr1 = ratios * mini_advantage
          #          surr1 = ratios * A_k 
                    surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * mini_advantage
                    
                    actor_loss = (-torch.min(surr1, surr2)).mean()
                    entropy_loss = entropy.mean()
                    actor_loss = actor_loss - self.ent_coef * entropy_loss
                    critic_loss = nn.MSELoss()(V.squeeze(), mini_rtgs.squeeze())
                    


                    # Update the actor network
                    self.actor_optim.zero_grad()
                    actor_loss.backward(retain_graph=True)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.actor_optim.step()

                    self.critic_optim.zero_grad()   
                    critic_loss.backward()    
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.critic_optim.step()

                aprox_kl = ((ratios - 1) - logratios).mean()
                if aprox_kl > self.target_kl:
                    break


        plt.figure(figsize=(10, 5))
        plt.plot(ep_rewards_history)
    # media movil de 20 episodios para ver la tendencia
        if len(ep_rewards_history) >= 20:
            moving_avg = np.convolve(ep_rewards_history, np.ones(20)/20, mode='valid')
            plt.plot(range(19, len(ep_rewards_history)), moving_avg, 
            color='red', linewidth=2, label='Media movil (20 ep)')
    
        plt.xlabel('Episodio')
        plt.ylabel('Reward total')
        plt.title('PPO - LunarLandingContinuous-v3')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/ppo_rewards.png')   # guarda el grafico como imagen
        plt.show()                




    def rollout(self):
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []
        batch_vals = []
        batch_dones = []

        # para GAE--------
        ep_rews = []
        ep_vals = []
        ep_dones = []
        #-----------------
        obs = self.env.reset()
        done = False
        t = 0

        while t < self.timesteps_per_batch:
            ep_rews = []
            ep_vals = []
            ep_dones = []
            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
               
                if self.render:
                    self.env.render()
                ep_dones.append(done)

                t += 1

                #tomamos observaciones
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs_tensor = torch.tensor(obs, dtype=torch.float)

                val = self.critic(obs_tensor).detach()

                obs, rew, terminated, trucanted, _ = self.env.step(action)
                done = terminated or trucanted

                ep_rews.append(rew)
                ep_vals.append(val.flatten())
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)
            batch_vals.append(ep_vals)
            batch_dones.append(ep_dones)
            # Reshape data as tensors in the shape specified before returning
        batch_obs = torch.tensor(np.array(batch_obs), dtype=torch.float)
        batch_acts = torch.tensor(np.array(batch_acts), dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()
        # ALG STEP #4
        batch_rtgs = self.compute_rtgs(batch_rews)
        # Return the batch data
        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_vals, batch_dones, batch_rews

    def evaluate(self, batch_obs, batch_acts):
      # Query critic network for a value V for each obs in batch_obs.
      V = self.critic(batch_obs).squeeze()
      out = self.actor(batch_obs)
      if self.is_discrete:
          dist = torch.distributions.Categorical(logits=out)
      else:
          dist = MultivariateNormal(out, self.cov_mat)
      log_probs = dist.log_prob(batch_acts)
      # Return predicted values V and log probs log_probs
      return V, log_probs, dist.entropy()
    
    def calculate_gae(self, rewards, values, dones):
        batch_advantages = []
        for ep_rews, ep_vals, ep_dones in zip(rewards, values, dones):
            advantages = []
            last_advantage = 0

            for t in reversed(range(len(ep_rews))):
                if t + 1 < len(ep_rews):
                    delta = ep_rews[t] + self.gamma * ep_vals[t+1] * (1 - ep_dones[t+1]) - ep_vals[t]
                else:
                    delta = ep_rews[t] - ep_vals[t]

                advantage = delta + self.gamma * self.lam * (1 - ep_dones[t]) * last_advantage
                last_advantage = advantage
                advantages.insert(0, advantage)

            batch_advantages.extend(advantages)

        return torch.tensor(batch_advantages, dtype=torch.float)
        



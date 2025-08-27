#!/usr/bin/env python3
# ==========================================================
# WEPS3-EPTS Spiral RL Trainer v1.0
# Institutional-Grade Training Orchestrator for WEPS RL Pipeline
# Author: Ola Bode (WEPS Creator)
# ==========================================================
import os
import time
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Tuple

class WEPSRLTrainer:
    """
    WEPS3-EPTS RL Trainer

    Coordinates:
      - environment interactions (WEPSFESISpiralEnv)
      - agent learning (RainbowLTCAgent)
      - prioritized replay buffer (PrioritizedReplayBuffer)
      - epsilon exploration scheduling
      - checkpointing and logging

    Features:
      - Supports multi-episode training with max steps per episode
      - Tracks episodic returns, losses, and diagnostics
      - Epsilon decay with min epsilon threshold
      - Checkpoint save/load for resuming training
      - Designed for scalability and extensibility to multi-asset or multi-agent
    """

    def __init__(
        self,
        env,
        agent,
        replay_buffer,
        device: torch.device,
        max_episodes: int = 10000,
        max_steps_per_episode: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.05,
        epsilon_decay_frames: int = 500000,
        checkpoint_dir: str = "./checkpoints",
        save_every_episodes: int = 100,
        log_every_episodes: int = 10,
    ):
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.device = device

        self.max_episodes = max_episodes
        self.max_steps = max_steps_per_episode

        # Exploration parameters
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_frames = epsilon_decay_frames
        self.total_steps = 0

        # Checkpoint and logging
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.save_every = save_every_episodes
        self.log_every = log_every_episodes

        # Tracking
        self.episode_rewards: List[float] = []
        self.episode_losses: List[float] = []

    def epsilon_by_step(self, step: int) -> float:
        """Linear epsilon decay."""
        epsilon = max(
            self.epsilon_final,
            self.epsilon_start - step * (self.epsilon_start - self.epsilon_final) / self.epsilon_decay_frames
        )
        return epsilon

    def run_episode(self) -> Tuple[float, float]:
        """Run a single training episode."""
        state = self.env.reset()
        done = False
        episode_reward = 0.0
        episode_loss = 0.0
        step_count = 0

        while not done and step_count < self.max_steps:
            epsilon = self.epsilon_by_step(self.total_steps)
            action = self.agent.select_action(state, epsilon)
            next_state, reward, done, info = self.env.step(action)

            transition = {
                'states': state,
                'actions': action,
                'rewards': reward,
                'next_states': next_state,
                'dones': done
            }
            self.replay_buffer.push(transition)

            loss_info = self.agent.learn()
            if 'loss' in loss_info:
                episode_loss += loss_info['loss']

            state = next_state
            episode_reward += reward
            self.total_steps += 1
            step_count += 1

        avg_loss = episode_loss / step_count if step_count > 0 else 0.0
        return episode_reward, avg_loss

    def save_checkpoint(self, episode: int):
        """Save agent and replay buffer states."""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.pth")
        torch.save({
            'episode': episode,
            'agent_state_dict': self.agent.online_net.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
            'total_steps': self.total_steps,
        }, checkpoint_path)
        print(f"[Trainer] Checkpoint saved at episode {episode} → {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load agent and replay buffer from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.online_net.load_state_dict(checkpoint['agent_state_dict'])
        self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.replay_buffer = checkpoint['replay_buffer']
        self.total_steps = checkpoint['total_steps']
        print(f"[Trainer] Loaded checkpoint from {path}, starting at episode {checkpoint['episode']}")

    def train(self):
        """Run full training loop."""
        print(f"[Trainer] Starting training for {self.max_episodes} episodes...")
        start_time = time.time()

        for episode in range(1, self.max_episodes + 1):
            ep_reward, ep_loss = self.run_episode()
            self.episode_rewards.append(ep_reward)
            self.episode_losses.append(ep_loss)

            # Logging
            if episode % self.log_every == 0:
                avg_reward = np.mean(self.episode_rewards[-self.log_every:])
                avg_loss = np.mean(self.episode_losses[-self.log_every:])
                elapsed = time.time() - start_time
                print(
                    f"[Trainer][Episode {episode}] Avg Reward: {avg_reward:.4f}, Avg Loss: {avg_loss:.6f}, "
                    f"Epsilon: {self.epsilon_by_step(self.total_steps):.4f}, Elapsed: {elapsed:.1f}s"
                )

            # Checkpointing
            if episode % self.save_every == 0:
                self.save_checkpoint(episode)

        print("[Trainer] Training complete.")


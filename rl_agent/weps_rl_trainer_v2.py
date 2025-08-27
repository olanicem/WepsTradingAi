#!/usr/bin/env python3
# ================================================================
# WEPSRLTrainerV2 — Institutional-Grade Multi-Asset Trainer for WEPS Spiral RL Agent
# Logs detailed episode summaries, uses prioritized replay,
# checkpointing, full FESI-compliant reward, and advanced logging
# Author: Ola Bode (WEPS Creator)
# ================================================================

import os
import torch
import logging
from typing import List, Dict, Any
from weps.rl_env.weps_fesi_spiral_env_v2 import WEPSFESISpiralEnv
from weps.rl_agent.rainbow_ltc_agent import RainbowLTCAgent
from weps.rl_agent.prioritized_replay import PrioritizedReplayBuffer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger("WEPSRLTrainerV2")

class WEPSRLTrainerV2:
    def __init__(
        self,
        organisms: List[str],
        device: torch.device,
        replay_capacity: int = 100_000,
        batch_size: int = 32,
        gamma: float = 0.99,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        multi_step: int = 3,
        max_steps_per_episode: int = 10_000,
        checkpoint_dir: str = "./checkpoints",
        save_every: int = 50,
    ) -> None:
        self.organisms = [org.upper() for org in organisms]
        self.device = device
        self.batch_size = batch_size
        self.max_steps = max_steps_per_episode
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.agent = RainbowLTCAgent(
            device=device,
            batch_size=batch_size,
            gamma=gamma,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            multi_step=multi_step,
            replay_buffer=None,  # assigned after replay buffer init
        )
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=replay_capacity,
            alpha=alpha,
            beta_start=beta_start,
            beta_frames=beta_frames,
            multi_step=multi_step,
            gamma=gamma,
        )
        self.agent.replay_buffer = self.replay_buffer

    def _calculate_reward(
        self,
        state: Any,
        action: int,
        next_state: Any,
        info: Dict[str, Any]
    ) -> float:
        """
        FESI-compliant reward combining:
         - Realized P&L on trade exit
         - Wave confidence bonus
         - Entropy and mutation penalties
         - Phase-aligned trading rewards
        """
        reward = 0.0
        if info.get("position_open") and action == 3:  # Exit closes position
            entry_price = info.get("entry_price", 0.0)
            exit_price = info.get("price", 0.0)
            pos = info.get("position")
            if pos == "long":
                reward += exit_price - entry_price
            elif pos == "short":
                reward += entry_price - exit_price

        reward += 0.05 * info.get("wave_confidence", 0.0)
        reward -= 0.1 * max(0.0, info.get("entropy", 0.0) - 0.5)
        reward -= 0.2 * info.get("mutation", 0.0)

        phase = info.get("phase", "neutral")
        if phase in ["rebirth", "growth"] and action == 1:
            reward += 0.1
        if phase == "decay" and action == 2:
            reward += 0.1
        if phase == "death" and action == 3:
            reward += 0.2

        return reward

    def _get_epsilon(self, episode: int, max_episodes: int) -> float:
        """
        Linear epsilon decay: 1.0 → 0.1 over max_episodes
        """
        epsilon_start = 1.0
        epsilon_final = 0.1
        return max(
            epsilon_final,
            epsilon_start - (episode / max_episodes) * (epsilon_start - epsilon_final),
        )

    def _save_checkpoint(self, organism: str, episode: int) -> None:
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{organism}_episode_{episode}.pth")
        torch.save(
            {
                "model_state_dict": self.agent.online_net.state_dict(),
                "optimizer_state_dict": self.agent.optimizer.state_dict(),
                "replay_buffer": self.replay_buffer,
                "episode": episode,
                "organism": organism,
            },
            checkpoint_path,
        )
        logger.info(f"[CHECKPOINT] Saved checkpoint: {checkpoint_path}")

    def train(self, episodes_per_asset: int = 500) -> None:
        for organism in self.organisms:
            logger.info("==============================================")
            logger.info(f"Starting training for organism: {organism}")
            logger.info("==============================================")

            env = WEPSFESISpiralEnv(organism=organism, max_steps=self.max_steps)
            for episode in range(1, episodes_per_asset + 1):
                state = env.reset()
                done = False
                total_reward = 0.0
                step_count = 0

                while not done and step_count < self.max_steps:
                    epsilon = self._get_epsilon(episode, episodes_per_asset)
                    action = self.agent.select_action(state, epsilon=epsilon)
                    next_state, info = env.step(action)

                    # Enhance info for reward calc
                    info_enhanced = info.copy()
                    info_enhanced["position_open"] = env.trade_open
                    info_enhanced["position"] = env.position
                    info_enhanced["entry_price"] = env.entry_price
                    info_enhanced["price"] = info.get("price", 0.0)

                    reward = self._calculate_reward(state, action, next_state, info_enhanced)

                    self.replay_buffer.push(
                        {
                            "states": state,
                            "actions": action,
                            "rewards": reward,
                            "next_states": next_state,
                            "dones": done,
                        }
                    )

                    state = next_state
                    total_reward += reward
                    step_count += 1

                    if len(self.replay_buffer) > self.batch_size:
                        learn_info = self.agent.learn()
                        # Optionally log learn_info['loss']

                logger.info(
                    f"[TRAIN] Episode {episode:04d} | Organism {organism} | "
                    f"Total Reward: {total_reward:.4f} | Steps: {step_count}"
                )

                if episode % self.save_every == 0:
                    self._save_checkpoint(organism, episode)

            logger.info("==============================================")
            logger.info(f"Finished training for organism: {organism}")
            logger.info("==============================================")

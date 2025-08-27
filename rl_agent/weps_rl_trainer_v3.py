#!/usr/bin/env python3
# ================================================================
# WEPSRLTrainerV3 — Institutional-Grade Dual-Phase RL Trainer
# Phase 1: Historical data batch training (local CSV)
# Phase 2: Live polling with retry & wait
# Author: Ola Bode (WEPS Creator)
# ================================================================

import logging
import numpy as np
import torch
import threading
import time
import os
import pandas as pd
from typing import Dict, Any, Optional, List
from weps.rl_agent.rainbow_ltc_agent import RainbowLTCAgent
from weps.rl_agent.prioritized_replay import PrioritizedReplayBuffer
from weps.utils.live_data_feeder import LiveDataFeeder
from weps.utils.live_data_polling import LiveDataPolling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("WEPS.RLTrainerV3")

class WEPSRLTrainerV3:
    def __init__(
        self,
        organisms: List[str],
        device: torch.device,
        episodes: int = 500,
        max_steps_per_episode: int = 10000,
        replay_capacity: int = 65536,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-4,
        multi_step: int = 3,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay_frames: int = 100000,
        timeframes: Optional[List[str]] = None,
        polling_interval_sec: int = 60,
        max_retry_attempts: int = 10,
        retry_delay_sec: int = 60,
        api_key: Optional[str] = None,
    ):
        self.device = device
        self.organisms = [org.upper() for org in organisms]
        self.episodes = episodes
        self.max_steps = max_steps_per_episode
        self.batch_size = batch_size
        self.timeframes = timeframes or ["1h", "4h", "1d"]

        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay_frames = epsilon_decay_frames
        self.frame_idx = 0

        self.polling_interval_sec = polling_interval_sec
        self.max_retry_attempts = max_retry_attempts
        self.retry_delay_sec = retry_delay_sec
        self.api_key = api_key

        # Initialize RL agent and replay buffer
        self.agent = RainbowLTCAgent(
            device=device,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            multi_step=multi_step,
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

        self.state = None
        self.position = None
        self.entry_price = None

    def _epsilon_by_frame(self) -> float:
        epsilon = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
                  np.exp(-1. * self.frame_idx / self.epsilon_decay_frames)
        return epsilon

    def _calculate_reward(self, action: int, price: float) -> float:
        reward = 0.0
        if action == 3 and self.position is not None and self.entry_price is not None:
            if self.position == 'long':
                reward = price - self.entry_price
            elif self.position == 'short':
                reward = self.entry_price - price
            self.position = None
            self.entry_price = None
        elif action == 1 and self.position is None:
            self.position = 'long'
            self.entry_price = price
        elif action == 2 and self.position is None:
            self.position = 'short'
            self.entry_price = price
        return reward

    def _safe_step_with_retry(self, live_feeder: LiveDataFeeder, organism: str, episode: int, step: int) -> Dict[str, Any]:
        retry_count = 0
        while retry_count < self.max_retry_attempts:
            try:
                return live_feeder.step()
            except RuntimeError as e:
                if "No new data to step" in str(e):
                    logger.info(f"[{organism}] Waiting for new candles (episode {episode}, step {step}), retry {retry_count+1}/{self.max_retry_attempts}...")
                    time.sleep(self.retry_delay_sec)
                    retry_count += 1
                else:
                    logger.error(f"[{organism}] Unexpected error during step: {e}", exc_info=True)
                    raise
        logger.error(f"[{organism}] Max retries exceeded waiting for new data.")
        raise RuntimeError("Max retries exceeded.")

    def _load_historical_data_for_organism(self, organism: str) -> dict:
        """
        Load historical OHLCV data from local CSV files for all timeframes.
        Expects CSV files named like 'aapl_ohlcv.csv' under ~/weps/data/raw_ohlcv/
        Returns dict: { '1h': df_1h, '4h': df_4h, '1d': df_1d }
        """
        base_path = os.path.expanduser("~/weps/data/raw_ohlcv")
        organism_lower = organism.lower()

        # Only daily CSVs available locally, fallback to daily if missing others
        timeframe_to_file = {
            "1d": f"{organism_lower}_ohlcv.csv",
        }

        data = {}
        for tf, filename in timeframe_to_file.items():
            filepath = os.path.join(base_path, filename)
            if not os.path.isfile(filepath):
                raise FileNotFoundError(f"Historical CSV not found: {filepath}")

            df = pd.read_csv(filepath, parse_dates=["date"])
            df = df.sort_values("date").reset_index(drop=True)

            # FIXED: Set datetime index properly
            df.set_index("date", inplace=True)
            df.index = pd.to_datetime(df.index)

            data[tf] = df

        # Fallback missing timeframes to daily data
        for tf in ["1h", "4h", "1d"]:
            if tf not in data:
                data[tf] = data.get("1d").copy()

        return data

    def _train_on_feeder(self, live_feeder: LiveDataFeeder, organism: str, live_mode: bool):
        for episode in range(1, self.episodes + 1):
            try:
                # Start with step 0 or retry logic based on live_mode
                if live_mode:
                    state_info = self._safe_step_with_retry(live_feeder, organism, episode, 0)
                else:
                    state_info = live_feeder.step()

                self.state = state_info["state_vector"]
                info = state_info["info"]
                done = state_info["done"]

                total_reward = 0.0
                step_count = 0

                for step in range(self.max_steps):
                    epsilon = self._epsilon_by_frame()
                    action = self.agent.select_action(self.state, epsilon)

                    price = info.get("price", 0.0)
                    reward = self._calculate_reward(action, price)
                    total_reward += reward

                    if live_mode:
                        try:
                            next_state_info = self._safe_step_with_retry(live_feeder, organism, episode, step)
                        except RuntimeError:
                            logger.warning(f"[{organism}] Skipping rest of episode {episode} due to no new data.")
                            break
                    else:
                        next_state_info = live_feeder.step()

                    next_state = next_state_info["state_vector"]
                    done = next_state_info["done"]
                    info = next_state_info["info"]

                    transition = {
                        "states": self.state,
                        "actions": action,
                        "rewards": reward,
                        "next_states": next_state,
                        "dones": done,
                    }
                    self.agent.replay_buffer.push(transition)

                    self.state = next_state
                    self.frame_idx += 1

                    if len(self.agent.replay_buffer) > self.batch_size:
                        learn_info = self.agent.learn()
                        if "loss" in learn_info:
                            logger.debug(f"Episode {episode} Step {step} Loss: {learn_info['loss']:.6f}")

                    if done:
                        break

                    step_count += 1

                logger.info(f"[TRAIN] Organism {organism} Episode {episode:04d} Complete | Total Reward: {total_reward:.4f} | Steps: {step_count} | Epsilon: {epsilon:.4f}")

            except Exception as e:
                logger.error(f"Exception during training at organism {organism}, episode {episode}: {e}", exc_info=True)

    def train(self):
        for organism in self.organisms:
            logger.info("=" * 60)
            logger.info(f"Starting WEPS RL Training on organism: {organism} for {self.episodes} episodes.")
            logger.info("=" * 60)

            # PHASE 1: Historical batch training (local CSV)
            try:
                hist_data = self._load_historical_data_for_organism(organism)
                live_feeder = LiveDataFeeder(organism, timeframes=self.timeframes)
                live_feeder.initialize(hist_data)
                self._train_on_feeder(live_feeder, organism, live_mode=False)
            except Exception as e:
                logger.error(f"[{organism}] Failed historical data training: {e}", exc_info=True)
                continue

            # PHASE 2: Live polling with retry/wait
            live_feeder_live = LiveDataFeeder(organism, timeframes=self.timeframes)
            poller = LiveDataPolling(
                organism,
                live_feeder_live,
                polling_interval_sec=self.polling_interval_sec,
                api_key=self.api_key,
            )
            poll_thread = threading.Thread(target=poller.start_polling_loop, daemon=True)
            poll_thread.start()

            logger.info(f"[{organism}] Started live data polling thread for live training.")

            self._train_on_feeder(live_feeder_live, organism, live_mode=True)

        logger.info("All organism trainings completed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="WEPS RL Trainer v3 dual-phase training")
    parser.add_argument("--organisms", nargs="+", required=True, help="List of organisms (assets) to train on")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes per organism")
    parser.add_argument("--steps", type=int, default=10000, help="Max steps per episode")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run training on")
    parser.add_argument("--polling_interval_sec", type=int, default=60, help="Polling interval for live data in seconds")
    parser.add_argument("--api_key", type=str, default=None, help="FMP API key for live data polling")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    trainer = WEPSRLTrainerV3(
        organisms=args.organisms,
        device=device,
        episodes=args.episodes,
        max_steps_per_episode=args.steps,
        polling_interval_sec=args.polling_interval_sec,
        api_key=args.api_key,
    )
    trainer.train()

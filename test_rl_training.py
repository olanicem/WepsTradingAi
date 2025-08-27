import torch
from weps.rl_agent.weps_rl_trainer_v3 import WEPSRLTrainerV3

def run_test_training():
    trainer = WEPSRLTrainerV3(
        organisms=["AAPL"],
        device=torch.device("cpu"),
        episodes=5,
        max_steps_per_episode=50,
        polling_interval_sec=30,
        api_key="oemCdaq9J01EUH6VkoGrAisCdTYZLXfI"  # Your real FMP API key
    )
    trainer.train()

if __name__ == "__main__":
    run_test_training()

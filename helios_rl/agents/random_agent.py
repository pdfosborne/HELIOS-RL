import random
from helios_rl.agents.agent_abstract import Agent
import torch
from torch import Tensor

class RandomAgent(Agent):
    """This is simply a random decision maker, does not learn."""
    def __init__(self):
        super().__init__()

    def policy(self, state: Tensor, legal_actions: list) -> str:
        action = str(random.choice(legal_actions))
        
        return action
    def learn(self, state: Tensor, next_state: Tensor, r_p: float, action_code: str) -> float:
        # Do nothing.
        return None

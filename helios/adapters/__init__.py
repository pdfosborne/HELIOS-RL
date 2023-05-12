from abc import ABC, abstractmethod
from typing import List, Any

class StateAdapter(ABC):
    def __init__(self, raw_state):
        super().__init__()
        # Define the fields that describe the state features:
        self.state: list = self._read(raw_state)

    @abstractmethod
    def _read(raw_state) -> list:
        # Read the data.
        # fill in the feature fields
        raise NotImplementedError

        

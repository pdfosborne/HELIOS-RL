import random
import numpy as np
import pandas as pd
from typing import List

from similarity_measurements.similarity_abstract import SimilarityMeasure

class CosineSimilarity(SimilarityMeasure):
    def similarity(self, legal_moves, state, known_states, sim_threshold) -> str:
        # # NEW SIMILARITY METRIC
        # Parameter to set minimum similarity required
        # Init output
        best_state = None
        best_state_score = 0
        # Check for each known state the sim score
        for known_state in known_states:
            score = np.sum(state == known_state)/len(state)
            if (score > sim_threshold)&(score > best_state_score):
                best_state = known_state
                best_state_score = score
        # If nothing better than threshold, random action. Else adopt policy of most similar action
        if best_state is not None:
            move_uci = max(known_states[tuple(best_state)], key=known_states[tuple(best_state)].get)                
        else:
            move_uci = str(random.choice(legal_moves))
        return move_uci

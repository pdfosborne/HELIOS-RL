import os
import sys
import platform
import warnings
from typing import List
from itertools import product
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch
from torch import tensor

import matplotlib.pyplot as plt

# ------ Experiment Import --------------------------------------
from helios.analysis import Analysis
# ------ Evaluation Metrics -----------------------------------------
from helios.evaluation.convergence_measure import Convergence_Measure
# ------ Agent Imports -----------------------------------------
# Universal Agents
from helios.agents.agent_abstract import Agent, QLearningAgent
from helios.agents.table_q_agent import TableQLearningAgent
from helios.agents.neural_q_agent import NeuralQLearningAgent

AGENT_TYPES = {
    "Qlearntab": TableQLearningAgent,
    "Neural_Q": NeuralQLearningAgent,
    "Neural_Q_2": NeuralQLearningAgent,
    "Neural_Q_language": NeuralQLearningAgent
}

PLAYER_PARAMS = {
    "Qlearntab": ["alpha", "gamma", "epsilon"],
    "Neural_Q": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Neural_Q_2": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"],
    "Neural_Q_language": ["input_type", "input_size", "sent_hidden_dim", "hidden_dim", "num_hidden", "sequence_size", "memory_size"]
}

from helios.encoders.sentence_transformer_MiniLM_L6v2 import LanguageEncoder

# This is the main run functions for HELIOS to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead

class HELIOS():
    def __init__(self, Config:dict, LocalConfig:dict, Environment, 
                 number_exploration_episodes:int = 100, sim_threshold:float = 0.9, feedback_increment=0.5, num_repeats:int=1,
                 observed_states:dict=None, instruction_results:dict=None):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig

        self.env = Environment
        self.setup_info:dict = self.ExperimentConfig['data'] | vars(self.LocalConfig) # TODO: configs aren't consistent formatting 
        self.training_setups: dict = {}

        # New instruction learning
        if not observed_states:
            self.observed_states:dict = {}
        else:
            self.observed_states = observed_states   
        if not instruction_results:
            self.instruction_results:dict = {}
        else:
            self.instruction_results = instruction_results
            
        # Unsupervised search parameters
        self.enc = LanguageEncoder()
        self.number_exploration_episodes: int = number_exploration_episodes
        self.sim_threshold: float = sim_threshold
        self.cos = torch.nn.CosineSimilarity(dim=0)
        # Helios inits
        self.feedback_layer_form: tensor = torch.zeros(self.enc.encode(['']).size()) # Init base sizing of tensor produced by language encoder
        self.feedback_increment: float = feedback_increment
        self.feedback_results: dict = {}
        self.num_repeats: int = num_repeats

    def search(self, instruction:str='', instr_description:str='', action_cap:int=5, re_search_override:bool=False,
               simulated_instr_goal:any=None):
        # Trigger re-search
        if re_search_override:
            self.observed_states:dict = {}
        print(" ")
        print("==========")
        print("Instruction: ", instruction)        
        # Create tensor vector of description
        instruction_vector = self.enc.encode(' '.join(instr_description))
        # Seen sub_goal before & sim above threshold
        if (instruction in self.instruction_results):
            if (agent_type+'_'+adapter) in self.instruction_results[instruction]['env_code']:
                # We use feedback layer even if sub_goal not a good match
                feedback_layer = self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['feedback_layer']
                if (self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sim_score']>=self.sim_threshold):
                    sub_goal = self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sub_goal']
            else:
                feedback_layer = self.feedback_layer_form.clone() # init null feedback layer for current instr
        else:
            feedback_layer = self.feedback_layer_form.clone() # init null feedback layer for current instr
        # ---
        sim_delta = 1 # Parameter that stops updates when change to feedback is small
        for repeat in range(0,self.num_repeats): # Arbitrary repeat to further reinforce matching state to instr
            print("===")
            print("Repeated search num ", repeat+1)
            #instruction_vector = torch.rand(observed_states[list(observed_states.keys())[0]].size()) # NEED TO FIND A METHOD TO VECTORIZE OBSERVED STATES AND INSTRUCTIONS TO COMPARE
            for n, agent_type in enumerate(self.setup_info['agent_select']):
                # We are adding then overriding some inputs from general configs for experimental setups
                train_setup_info = self.setup_info.copy()
                # Override action cap for shorter term sub-goals for faster learning
                train_setup_info['training_action_cap'] = action_cap 
                # ----- State Adapter Choice
                adapter = train_setup_info["adapter_select"][n]
                # ----- Agent parameters
                agent_parameters = train_setup_info["agent_parameters"][agent_type]
                train_setup_info['agent_type'] = agent_type
                train_setup_info['agent_name'] = str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
                train_setup_info['adapter_select'] = adapter
                # ----- init agent
                player = AGENT_TYPES[agent_type](**agent_parameters)
                train_setup_info['agent'] = player
                # -----
                # Set env function to training# Repeat training
                train_setup_info['train'] = True
                # --- 
                # Set exploration parameters
                train_setup_info['number_training_episodes'] = self.number_exploration_episodes # Override 
                # ---------------------------------HELIOS-----------------------------------------
                # EXPLORE TO FIND LOCATION OF SUB-GOAL
                sub_goal = None
                # Train on Live system for limited number of total episodes
                train_setup_info['training_results'] = False
                if not self.observed_states:
                    train_setup_info['observed_states'] = False
                else:
                    train_setup_info['observed_states'] = self.observed_states
                train_setup_info['experience_sampling'] = False
                train_setup_info['live_env'] = True 
                
                search_count=0
                while not sub_goal:
                    # If no description -> no sub-goal (i.e. envs terminal goal position)
                    if not instr_description: 
                        sub_goal = None
                        # If no sub-goal -> find best match of description from env 
                    else:
                        search_count+=1
                        print("------") 
                        print("Search: ", search_count)
                        
                        # Only run on live env if observed states empty
                        if not self.observed_states:
                            train_setup_info['sub_goal'] = sub_goal
                            # ---
                            # Explore env with limited episodes
                            # Environment now init here and called directly in experimental setup loop
                            # - setup helios info
                            # Train on Live system for limited number of total episodes
                            train_setup_info['live_env'] = True                        
                            live_env = self.env(train_setup_info)
                            explore_results = live_env.episode_loop()
                            train_setup_info['training_results'] = explore_results
                            train_setup_info['observed_states'] = live_env.helios.observed_states
                            train_setup_info['experience_sampling'] = live_env.helios.experience_sampling
                            # Extract visited states from env
                            self.observed_states = live_env.helios.observed_states

                        # Compare to instruction vector                            
                        max_sim = -1
                        sim_tracker = []
                        # TODO: Any states that are above threshold 
                        sub_goal_list = []
                        for obs_state in self.observed_state:
                            str_state = self.observed_states[obs_state]
                            str_state_stacked = ' '.join(str_state)
                            t_state = self.enc.encode(str_state_stacked)
                            # ---
                            total_sim = 0
                            for instr_sentence in instruction_vector:
                                for state_sentence in t_state:
                                    total_sim+=self.cos(torch.add(state_sentence, feedback_layer), instr_sentence)
                            sim = total_sim.item()/(len(instruction_vector)*len(t_state))
                            if sim > max_sim:
                                max_sim  = sim
                                sub_goal_max = obs_state
                                sub_goal_max_t = t_state
                            if sim > self.sim_threshold:
                                sub_goal = obs_state # Sub-Goal code
                                sub_goal_list.append(sub_goal)
        
                        # TODO: OR if none above threshold within (1-threshold%) of max sim
                        if max_sim < self.sim_threshold:
                            for obs_state in self.observed_states:
                                str_state = self.observed_states[obs_state]
                                str_state_stacked = ' '.join(str_state)
                                t_state = self.enc.encode(str_state_stacked)
                                # ---
                                total_sim = 0
                                # Average sim across each sentence in instruction vs state
                                for instr_sentence in instruction_vector:
                                    for state_sentence in t_state:
                                        total_sim+=self.cos(torch.add(state_sentence, feedback_layer), instr_sentence) 
                                sim = total_sim.item()/(len(instruction_vector)*len(t_state))
                                if sim > max_sim*(self.sim_threshold):
                                    sub_goal = obs_state # Temp Sub-Goal as most known similar
                                    sub_goal_list.append(sub_goal)

                        sub_goal = sub_goal_list
                        if max_sim < self.sim_threshold:
                            print("Minimum sim for observed states to match instruction not found, using best match instead. Best match sim value = ", max_sim )
                    # If adapter is poor to match to instruction vector none of them observed states match
                    if (max_sim<-1)|(max_sim>1):
                        print("All observed states result in similarity outside bounds (i.e. strongly opposite vectors to instruction). Re-starting Search.")
                        print(sim_tracker)
                        sub_goal = None
                        self.observed_states = {}
                    elif (sim_delta<0)&(max_sim<0):
                        print("Change in sim less than or equal to delta cap, assume goal-state not observed. Re-starting Search.")
                        print("-- Known sub_goal position: ", simulated_instr_goal)
                        print("-- Best match: ", sub_goal_max)
                        sub_goal = None
                        self.observed_states = {}
                    else:
                        if not simulated_instr_goal:
                            print(" ")
                            print("- Best match state for instruction: ", sub_goal)
                            feedback = input("-- Does this match the expectation instruction outcome? (Y/N)")
                        else:
                            if type(simulated_instr_goal[0]) != type(sub_goal_max):
                                print("- ERROR: Typing of simulated sub-goal check does not match typing of state from environment, please correct this.")
                                print(type(simulated_instr_goal[0])," - ", type(sub_goal_max))
                                #print("Observed States Examples: ", self.observed_states(list(self.observed_states.keys())[0:5]))
                                exit()
                            else:
                                if sub_goal_max in simulated_instr_goal:
                                    print("- Simulated sub-goal match found.")
                                    print("-- Known sub_goal position: ", simulated_instr_goal)
                                    print("-- Unsupervised state best match: ", sub_goal_max)
                                    feedback = 'Y'
                                else:
                                    print("- Match NOT found.")
                                    print("-- Known sub_goal position: ", simulated_instr_goal)
                                    print("-- Unsupervised state best match: ", sub_goal_max)
                                    feedback = 'N'
                        
                        if (feedback.lower() == 'y')|(feedback.lower() == 'yes'):
                            for sentence in sub_goal_max_t:
                                feedback_layer = torch.add(feedback_layer, self.feedback_increment*(torch.sub(instruction_vector, sentence))) 
                            total_sim = 0
                            # Average sim across each sentence in instruction vs state
                            for instr_sentence in instruction_vector:
                                for state_sentence in sub_goal_max_t:
                                    total_sim+=self.cos(torch.add(state_sentence, feedback_layer), instr_sentence)
                            sim = total_sim.item()/(len(instruction_vector)*len(t_state))
                            sim_delta = sim-max_sim
                            print("--- Change in sim results with POSITIVE reinforcement of correct state match =", sim_delta)
                            # log results of feedback loop
                            if instruction not in self.feedback_results:
                                self.feedback_results[instruction] = {}
                            if (agent_type+'_'+adapter) not in self.feedback_results[instruction]:
                                self.feedback_results[instruction][agent_type+'_'+adapter] = {}

                            self.feedback_results[instruction][agent_type+'_'+adapter][repeat] = [search_count, max_sim, np.median(sim_tracker)]
                        else:
                            for sentence in sub_goal_max_t:
                                feedback_layer = torch.sub(feedback_layer, self.feedback_increment*(torch.sub(instruction_vector, sentence)))
                            total_sim = 0
                            # Average sim across each sentence in instruction vs state
                            for instr_sentence in instruction_vector:
                                for state_sentence in sub_goal_max_t:
                                    total_sim+=self.cos(torch.add(state_sentence, feedback_layer), instr_sentence)
                            sim = total_sim.item()/(len(instruction_vector)*len(t_state))
                            sim_delta = sim-max_sim
                            print("--- Change in sim results with NEGATIVE reinforcement for NO MATCH =", sim_delta)
                            sub_goal = None

        # Log matching sub_goal with instruction        
        if instruction not in self.instruction_results:
            self.instruction_results[instruction] = {}    
            self.instruction_results[instruction]['env_code'] = {} 
            self.instruction_results[instruction]['action_cap'] = action_cap
            
        if (agent_type+'_'+adapter) not in self.instruction_results[instruction]['env_code']:
            self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter] = {}
            
        self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sub_goal'] = sub_goal_list
        self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['sim_score'] = max_sim
        self.instruction_results[instruction]['env_code'][agent_type+'_'+adapter]['feedback_layer'] = feedback_layer
        # --------------------------------------------------------------------------------
        # Quick Visual Analysis of Feedback results
        searches = []
        correct_sims = []
        avg_sims = []
        for repeat in self.feedback_results[instruction][agent_type+'_'+adapter]:
            failed_search_count = self.feedback_results[instruction][agent_type+'_'+adapter][repeat][0]
            correct_state_sim = self.feedback_results[instruction][agent_type+'_'+adapter][repeat][1]
            avg_state_sim = self.feedback_results[instruction][agent_type+'_'+adapter][repeat][2]

            searches.append(failed_search_count)
            correct_sims.append(np.round(correct_state_sim,5))
            avg_sims.append(np.round(avg_state_sim,5))
            
        plt.plot(searches, label='Num Searches')
        plt.title('Number of Searches before Correct State Found')
        plt.ylabel('Count')
        plt.xlabel('Repeat Number')
        plt.show()
        plt.savefig('./output/'+'/reinforcement_results.png', dpi=100)
        plt.close()

        # plt.plot(correct_sims, 'k-', label='Correct State')
        # plt.plot(avg_sims, 'r--', label='Median for All States')
        # plt.title('State Sim with Instruction')
        # plt.ylabel('Cosine Similarity')
        # plt.xlabel('Repeat Number')
        # plt.show()

        return self.observed_states, self.instruction_results
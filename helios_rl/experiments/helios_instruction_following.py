import os
import pandas as pd
import json

# ------ Experiment Import --------------------------------------
from helios_rl.analysis import Analysis
# ------ Agent Imports -----------------------------------------
# Universal Agents
from helios_rl.agents.agent_abstract import Agent, QLearningAgent
from helios_rl.agents.table_q_agent import TableQLearningAgent
from helios_rl.agents.neural_q_agent import NeuralQLearningAgent

# TODO: Enable any number of the same agent types with varying parameters
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

# This is the main run functions for HELIOS to be imported
# Defines the train/test operators and imports all the required agents and experiment functions ready to be used
# The local main.py file defines the [adapters, configs, environment] to be input

# This should be where the environment is initialized and then episode_loop (or train/test) is run
# -> results then passed down to experiment to produce visual reporting (staticmethod)
# -> instruction following approach then becomes alternative form of this file to be called instead
# -> This means we have multiple main.py types (e.g. with/without convergence measure) so should create a directory and finalise naming for this

class HeliosOptimize:
    def __init__(self, Config:dict, LocalConfig:dict, Environment, 
                 save_dir:str, show_figures:str, window_size:float,
                 instruction_path: dict, predicted_path: dict=None, instruction_episode_ratio:float=0.1):
        self.ExperimentConfig = Config
        self.LocalConfig = LocalConfig
        
        if not predicted_path:
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir+'/Supervised_Instr_Experiment'
        else:
            save_dir_extra = save_dir.split("/")[-1]
            save_dir = '/'.join(save_dir.split("/")[:-1])
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            self.save_dir = save_dir+'/'+save_dir_extra

        self.show_figures = show_figures
        if (self.show_figures.lower() != 'y')|(self.show_figures.lower() != 'yes'):
            print("Figures will not be shown and only saved.")

        self.env = Environment
        self.setup_info = self.ExperimentConfig['data'] | self.LocalConfig['data']  
        self.training_setups: dict = {}
        # New instruction learning
        # New for when we have predicted goal states for training
        if predicted_path:
            self.instruction_path = predicted_path
            if instruction_path:
                self.testing_path = instruction_path
            else:
                self.testing_path = predicted_path
        else:
            self.instruction_path = instruction_path
            self.testing_path = False
        self.known_instructions = list(self.instruction_path.keys())
        self.known_instructions_dict = {}
        for instr in self.known_instructions:
            start = instr.split("---")[0]
            end = instr.split("---")[1]

            for n, agent_type in enumerate(self.setup_info['agent_select']):
                adapter = self.setup_info["adapter_select"][n]
                if (agent_type+'_'+adapter) not in self.known_instructions_dict:
                        self.known_instructions_dict[(agent_type+'_'+adapter)] = {}

                if (agent_type+'_'+adapter) in self.instruction_path[instr]:
                    count = self.instruction_path[instr][(agent_type+'_'+adapter)]['count']
                    if start not in self.known_instructions_dict[(agent_type+'_'+adapter)]:
                        self.known_instructions_dict[(agent_type+'_'+adapter)][start] = {}
                        if end not in self.known_instructions_dict[(agent_type+'_'+adapter)][start]:
                            self.known_instructions_dict[(agent_type+'_'+adapter)][start][end] = count


        print("-----")
        print("Known human instruction inputs. ")
        print(self.known_instructions_dict)
        # new - store agents cross training repeats for completing the same start-end goal
        self.trained_agents: dict = {}
        self.num_training_seeds = self.setup_info['number_training_seeds']
        self.analysis = Analysis(window_size=window_size)
        # Defines the number of episodes used for sub-instructions
        self.instruction_episode_ratio = instruction_episode_ratio
        
    def train(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        for n, agent_type in enumerate(self.setup_info['agent_select']):
            # We are adding then overriding some inputs from general configs for experimental setups
            train_setup_info = self.setup_info.copy()
            # TODO: fix experience sampling
            if train_setup_info['experience_sample_batch_ratio']>0:
                print("NOTE - Experience Sampling feature not currently implemented and will not be used")
                train_setup_info['experience_sample_batch_ratio'] = 0
            # ----- State Adapter Choice
            adapter = train_setup_info["adapter_select"][n]
            # ----- Agent parameters
            agent_parameters = train_setup_info["agent_parameters"][agent_type]
            train_setup_info['agent_type'] = agent_type
            train_setup_info['agent_name'] = str(agent_type) + '_' + str(adapter) + '_' + str(agent_parameters)
            train_setup_info['adapter_select'] = adapter
            # ----- Sub-Goal
            # - If we have setup dict to include agent_adapter specific location of sub-goals
            #   i.e. {instr:{env_code:{agent_adapter:{sub_goal:'ENV_CODE', sim_score:0.8}}, action_cap:5}}
            #   Otherwise is standard user defined input {instr:{env_code:'ENV_CODE', action_cap:5}}
            # -----
            # Repeat training
            train_setup_info['train'] = True
            number_training_episodes = train_setup_info['number_training_episodes']
            number_training_repeats = self.ExperimentConfig["number_training_repeats"]
            print("Training Agent " + str(agent_type) + " for " + str(number_training_repeats) + " repeats")
            if str(agent_type) + '_' + str(adapter) not in self.trained_agents:
                self.trained_agents[str(agent_type) + '_' + str(adapter)] = {}

            seed_recall = {}
            seed_results_connection = {}
            for seed_num in range(0,self.num_training_seeds):
                if self.num_training_seeds > 1:
                    print("------")
                    print("- Seed Num: ", seed_num)
                # -------------------------------------------------------------------------------
                # Initialise Environment
                # Environment now init here and called directly in experimental setup loop
                # - Observed states, experience sampling passed over seeds but not training repeats
                if seed_num==0:
                    train_setup_info['training_results'] = False
                    train_setup_info['observed_states'] = False
                    train_setup_info['experience_sampling'] = False
                else:
                    train_setup_info['training_results'] = False
                    train_setup_info['observed_states'] = observed_states_stored.copy()
                    train_setup_info['experience_sampling'] = experience_sampling_stored.copy()
                # --- 
                setup_num:int = 0
                temp_agent_store = {}
                for training_repeat in range(1,(number_training_repeats+1)):
                    if number_training_repeats > 1:
                        print("------")
                        print("- Repeat Num: ", training_repeat)
                    setup_num+=1
                    
                    # ----- init agent
                    player = AGENT_TYPES[agent_type](**agent_parameters)
                    train_setup_info['agent'] = player
                    
                    # init live environment
                    train_setup_info['live_env'] = True
                    live_env = self.env(train_setup_info)
                    # ----------------
                    #  Start obs reset between seeds but fixed for repeat
                    if training_repeat > 1:
                        live_env.start_obs = env_start

                    env_start = live_env.start_obs
                    goal = str(env_start).split(".")[0] + "---" + "GOAL"
                    print("Long-term Goal: ", goal)
                    if goal in seed_recall:
                        setup_num = seed_recall[goal]
                    else:
                        seed_recall[goal] = 1
                    # Load results table from previous seeds to continue output graphs
                    if goal in seed_results_connection:
                        live_env.results.load(seed_results_connection[goal])
                
                    # - Results save dir -> will override for same goal if seen in later seed
                    if self.num_training_seeds > 1:
                        agent_save_dir = self.save_dir+'/'+agent_type+'_'+adapter+'__training_results_'+str(goal)+'_'+str(setup_num) 
                    else:
                        agent_save_dir = self.save_dir+'/'+agent_type+'_'+adapter+'__training_results_'+str(setup_num)
                    if not os.path.exists(agent_save_dir):
                        os.mkdir(agent_save_dir)
                    
                    if train_setup_info['experience_sample_batch_ratio']>0:
                        live_sample_batch_size = int(number_training_episodes*train_setup_info['experience_sample_batch_ratio'])
                        live_sample_batch_count = int(1/train_setup_info['experience_sample_batch_ratio'])
                        # Train on Live system for limited number of total episodes   
                        live_env.num_train_episodes = live_sample_batch_size
                        print("-- Training with Simulated Batches, ", live_sample_batch_count, " total...")
                        # init simulated environment
                        train_setup_info['live_env'] = False
                        simulated_env = self.env(train_setup_info)
                        simulated_env.start_obs = env_start

                        sample_batch_agent_store = {} # Need to continue learning across batches
                        sample_batch_results_store = {} # Pass over results table to next batch for the same instr TODO not working
                        for live_sample_batch in range(0, live_sample_batch_count-1):
                            print("--- Live Interaction Batch Num", live_sample_batch+1)
                            # ---- 
                            # Train for sub instr plan
                            start = str(env_start).split(".")[0]
                            i=0
                            while True:
                                # Only allow insutrction up until total limi
                                # - Prevents it being given more episodes than flat
                                # - Prevents cyclic instruction paths
                                if int(number_training_episodes*self.instruction_episode_ratio)<=(number_training_episodes-total_instr_episodes):
                                    # Override with trained agent if goal seen previously
                                    if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                        live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                        # Reset exploration parameter between seeds so to not get 'trapped'
                                        live_env.agent.exploration_parameter_reset()
                                    break
                                
                                elif start in self.known_instructions_dict[(agent_type+'_'+adapter)]:
                                    i+=1
                                    max_count = 0
                                    for end in self.known_instructions_dict[(agent_type+'_'+adapter)][start]:
                                        if self.known_instructions_dict[(agent_type+'_'+adapter)][start][end] > max_count:
                                            max_count = self.known_instructions_dict[(agent_type+'_'+adapter)][start][end]
                                            instr = start + "---" + end
                                            print("Sub-instr: ", instr)
                                    # ---
                                    # Override trained agent with known instruction agent 
                                    if instr in sample_batch_agent_store:
                                        live_env.agent = sample_batch_agent_store[instr]
                                    elif instr in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                        live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][instr].clone()
                                    if instr in sample_batch_results_store:
                                        live_env.results.load(sample_batch_results_store[instr])
                                    
                                    # elif prior_instr:
                                    #     live_env.agent = live_env.agent.clone() # Start from agent from previous task
                                    # TODO: ADOPT AGENT OF MOST SIMILAR POLICY
                                    # ---
                                    sub_goal = self.instruction_path[instr][agent_type+'_'+adapter]['sub_goal']
                                    live_env.sub_goal = sub_goal
                                    live_env.agent.exploration_parameter_reset()
                                    training_results = live_env.episode_loop()
                                    sample_batch_agent_store[instr] = live_env.agent.clone()
                                    # Override episode numbers and store for next batch
                                    training_results['episode'] = training_results.index
                                    sample_batch_results_store[instr] = live_env.results.copy()
                                    # ---
                                    # Store instruction results
                                    instr_save_dir = agent_save_dir+'/'+str(i)+"-"+instr.replace(" ","").replace("/","_")
                                    if not os.path.exists(instr_save_dir):
                                        os.mkdir(instr_save_dir)

                                    # Produce training report with Analysis.py
                                    Return = self.analysis.train_report(training_results, instr_save_dir, self.show_figures)
                                    if instr not in temp_agent_store:
                                        temp_agent_store[instr] = {}
                                    temp_agent_store[instr][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                    # ---
                                    start = end
                                    live_env.results.reset() # Force reset so we don't get overlapping outputs
                                    #live_env.start_obs = sub_goal[0] # Set start position of env for next -> pick first sub-goal
                                else:
                                    # if at least one sub-instr start with that for goal agent
                                    # if i > 1:
                                    #     live_env.agent = live_env.agent.clone() # Start from agent from previous task

                                    # if no instructions and first batch then adopt known agent
                                    # - Will then use this for next batches
                                    if i == 0 :
                                        if live_sample_batch == 0:
                                            # Override with trained agent if goal seen previously
                                            if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                                live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                    break
                            
                            # train for entire path 
                            #live_env.start_obs = env_start
                            live_env.sub_goal = None 
                            print("Goal: ", goal)
                            if goal in sample_batch_results_store:
                                live_env.results.load(sample_batch_results_store[goal])
                            live_env.agent.exploration_parameter_reset()
                            training_results = live_env.episode_loop()
                            training_results['episode'] = training_results.index
                            sample_batch_results_store[goal] = live_env.results.copy()

                            # Train based on simulated exp             
                            simulated_env.num_train_episodes = number_training_episodes
                            # Connect live agent -> This should be linked so continues learning
                            simulated_env.agent = live_env.agent
                            simulated_env.episode_loop()
                        # Final batch doesn't require simulated exp after and we need result output
                        print("--- Final batch")
                        train_setup_info['live_env'] = True
                        train_setup_info['number_training_episodes'] = live_sample_batch_size
                        live_env.agent.exploration_parameter_reset()
                        training_results = live_env.episode_loop()

                        # Have to 'fix' output
                        training_results['episode'] = training_results.index
                        cumulative_r = 0
                        cumulative_r_lst = []
                        for r in training_results['episode_reward']:
                            cumulative_r+=r
                            cumulative_r_lst.append(cumulative_r)
                        training_results['cumulative_reward'] = cumulative_r_lst
                    else:
                        # ---- 
                        # Train for sub instr plan
                        # Set start position of env for next -> pick first sub-goal env label
                        start = str(env_start).split(".")[0]
                        i=0
                        total_instr_episodes = 0
                        instr_results = None
                        while True:
                            i+=1
                            max_count = 0
                            # Only allow insutrction up until total limi
                            # - Prevents it being given more episodes than flat
                            # - Prevents cyclic instruction paths
                            if int(number_training_episodes*self.instruction_episode_ratio)<=(number_training_episodes-total_instr_episodes):
                                # Override with trained agent if goal seen previously
                                if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                    live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                    # Reset exploration parameter between seeds so to not get 'trapped'
                                    live_env.agent.exploration_parameter_reset()
                                break
                            
                            elif start in self.known_instructions_dict[(agent_type+'_'+adapter)]:
                                for end in self.known_instructions_dict[(agent_type+'_'+adapter)][start]:
                                    if self.known_instructions_dict[(agent_type+'_'+adapter)][start][end] > max_count:
                                        max_count = self.known_instructions_dict[(agent_type+'_'+adapter)][start][end]
                                        instr = start + "---" + end
                                        print("Sub-instr: ", instr)
                                        
                                # Instructions use fewer epispdes, lower bound to 10
                                number_instr_episodes = int(number_training_episodes*self.instruction_episode_ratio)
                                if number_instr_episodes<10:
                                    number_instr_episodes=10
                                total_instr_episodes+=number_instr_episodes
                                live_env.num_train_episodes = number_instr_episodes
                                # ---
                                # Override trained agent with known instruction agent
                                if instr in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                    live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][instr].clone()
                                # TODO: ADOPT AGENT OF MOST SIMILAR POLICY
                                # ---
                                sub_goal = self.instruction_path[instr][agent_type+'_'+adapter]['sub_goal']
                                live_env.sub_goal = sub_goal
                                live_env.agent.exploration_parameter_reset()
                                if type(instr_results)==type(pd.DataFrame()):
                                    live_env.results.load(instr_results)
                                training_results = live_env.episode_loop()
                                training_results['episode'] = training_results.index
                                # ---
                                # Store instruction results
                                instr_save_dir = agent_save_dir+'/'+str(i)+"-"+instr.replace(" ","").replace("/","_")
                                if not os.path.exists(instr_save_dir):
                                    os.mkdir(instr_save_dir)

                                # Produce training report with Analysis.py
                                Return = self.analysis.train_report(training_results, instr_save_dir, self.show_figures)
                                if instr not in temp_agent_store:
                                    temp_agent_store[instr] = {}
                                temp_agent_store[instr][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                # ---
                                start = end
                                # New: Dont reset results for each sub-instr so we show the training results with this included
                                #live_env.results.reset() # Force reset so we don't get overlapping outputs
                                instr_results =  live_env.results.copy()
                            else:
                                # If no instructions then train for full goal
                                if i == 1:
                                    # Override with trained agent if goal seen previously
                                    if goal in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                                        live_env.agent = self.trained_agents[str(agent_type) + '_' + str(adapter)][goal].clone()
                                        # Reset exploration parameter between seeds so to not get 'trapped'
                                        live_env.agent.exploration_parameter_reset()
                                break
                        # train for entire path 
                        #live_env.start_obs = env_start
                        # Number of episodes used reduced by those used for instructions (lower bounded)
                        if (number_training_episodes-total_instr_episodes)<int(number_training_episodes*self.instruction_episode_ratio):
                            if int(number_training_episodes*self.instruction_episode_ratio) < 10:
                                live_env.num_train_epispdes = 10
                            else:
                                live_env.num_train_epispdes = int(number_training_episodes*self.instruction_episode_ratio)
                        else:
                            live_env.num_train_episodes = number_training_episodes - total_instr_episodes
                        # Remove sub-goal
                        live_env.sub_goal = None
                        print("Goal: ", goal)
                        # Add instruction training to output chart
                        if type(instr_results)==type(pd.DataFrame()):
                            live_env.results.load(instr_results)
                        
                        training_results = live_env.episode_loop()
                        training_results['episode'] = training_results.index
                        
                    # Opponent now defined in local setup.py
                    # ----- Log training setup      
                    training_results.insert(loc=0, column='Repeat', value=setup_num)                    
                    # Produce training report with Analysis.py
                    Return = self.analysis.train_report(training_results, agent_save_dir, self.show_figures)
                    if goal not in temp_agent_store:
                        temp_agent_store[goal] = {}
                    temp_agent_store[goal][setup_num] = {'Return':Return,'agent':live_env.agent.clone()}
                                                            
                    # Extract trained agent from env and stored for re-call
                    # - Observed states and experience sampling from best repeat stored for next seed
                    if training_repeat == 1:
                        max_Return = Return
                        training_results_stored =  live_env.results.copy()
                        observed_states_stored = live_env.helios.observed_states
                        experience_sampling_stored = live_env.helios.experience_sampling
                    if Return > max_Return:
                        max_Return = Return
                        training_results_stored =  live_env.results.copy()
                        observed_states_stored = live_env.helios.observed_states
                        experience_sampling_stored = live_env.helios.experience_sampling
                    
                    seed_recall[goal] = seed_recall[goal] + 1
                seed_results_connection[goal] = training_results_stored

                for instr in temp_agent_store:
                    start_repeat_num = list(temp_agent_store[instr].keys())[0]
                    end_repeat_num = list(temp_agent_store[instr].keys())[-1]
                    # Only save the best agent from repeated training
                    best_return = temp_agent_store[instr][start_repeat_num]['Return']
                    best_agent = temp_agent_store[instr][start_repeat_num]['agent']
                    for repeat in range(start_repeat_num+1,end_repeat_num+1):
                        if temp_agent_store[instr][repeat]['Return']>best_return:
                            best_return = temp_agent_store[instr][repeat]['Return']
                            best_agent = temp_agent_store[instr][repeat]['agent']
            
                    if instr not in self.trained_agents[str(agent_type) + '_' + str(adapter)]:
                        self.trained_agents[str(agent_type) + '_' + str(adapter)][instr] = {}
                    self.trained_agents[str(agent_type) + '_' + str(adapter)][instr] = best_agent      
                
            # Store last train_setup_info as collection of observed states and experience sampling
            self.training_setups['Training_Setup_'+str(agent_type) + '_' + str(adapter)] = train_setup_info
        if (number_training_repeats>1)|(self.num_training_seeds):
            self.analysis.training_variance_report(self.save_dir, self.show_figures)
                
        #json.dump(self.training_setups) # TODO: Won't currently serialize this output to a json file
        return self.training_setups

    # TESTING PLAY
    def test(self, training_setups:str=None):
        # Override input training setups with previously saved 
        if training_setups is None:
            training_setups = self.training_setups
        else:
            json.load(training_setups)

        for training_key in list(training_setups.keys()):    
            test_setup_info = training_setups[training_key]
            test_setup_info['train'] = False # Testing Phase
            print("----------")
            print("Testing results for trained agents in saved setup configuration:")
            number_training_repeats = test_setup_info['number_test_repeats']
            agent_adapter = test_setup_info['agent_type'] + "_" + test_setup_info['adapter_select']

            for testing_repeat in range(0, test_setup_info['number_test_repeats']):  
                # Re-init env for testing
                env = self.env(test_setup_info)
                # ---
                start_obs = env.start_obs
                goal = str(start_obs).split(".")[0] + "---" + "GOAL"
                print("Flat agent Goal: ", goal)
                # Override with trained agent if goal seen previously
                if goal in self.trained_agents[test_setup_info['agent_type']+ '_' +test_setup_info['adapter_select']]:
                    print("Trained agent available for testing.")
                    env.agent = self.trained_agents[test_setup_info['agent_type']+'_'+test_setup_info['adapter_select']][goal]
                else:
                    print("NO agent available for testing position.")
                env.agent.epsilon = 0 # Remove random actions
                # ---
                # Testing generally is the agents replaying on the testing ENV
                testing_results = env.episode_loop() 
                test_save_dir = (self.save_dir+'/'+agent_adapter+'__testing_results_'+str(goal).split("/")[0]+"_"+str(testing_repeat) )
                if not os.path.exists(test_save_dir):
                    os.mkdir(test_save_dir)
                # Produce training report with Analysis.py
                Return = self.analysis.test_report(testing_results, test_save_dir, self.show_figures)

        # Path is the experiment save dir + the final instruction
        if number_training_repeats>1:
            self.analysis.testing_variance_report(self.save_dir, self.show_figures)
import matplotlib.pyplot as plt
from typing import List

# Define convergence evaluation function
class Convergence_Measure:
    def __init__(self, total_num_episodes):
        # --- PARAMETERS ---
        self.conv_threshold_perc = 5
        self.num_prior_epi = int(total_num_episodes/10)
        self.num_prior_epi_points = 5
        self.plot_convergence_figures = False
        # ------------------
        # Ploy display time
        self.display_plot_time = 10

    # Analyse repeat results & pick best agent that has converged
    # - Not going to be fast but is shouldn't cause huge issues
    # - Need to check for convergence as well, best might not be stable
    # conv_measure = Convergence_Measure(self.ExperimentConfig["num_episodes"])

    # R_max = 0
    # convergence_results_met = []
    # convergence_results_first_epi = []
    # # Whites Convergence
    # for agent_num in range(0,setup_num):
    #     visual_save_dir = agent_save_dir+'/'+str(self.ExperimentConfig['name']+'/Setup_'+str(agent_num+1))
    #     # Whites Convergence
    #     value_list_w = training_results_full[training_results_full['Setup']==(agent_num+1)]['Whites_total_reward'].values
    #     convergence_met_w = conv_measure.convergence_check(value_list_w, 'Whites', visual_save_dir)
    #     # Blacks Convergence
    #     value_list_b = training_results_full[training_results_full['Setup']==(agent_num+1)]['Blacks_total_reward'].values
    #     convergence_met_b = conv_measure.convergence_check(value_list_b, 'Blacks', visual_save_dir)
    #     # Log for output file
    #     convergence_results_met.append([convergence_met_w[0],convergence_met_b[0]])
    #     convergence_results_first_epi.append([convergence_met_w[1],convergence_met_b[1]])
    #     if convergence_met_w[0]==True:
    #         Final_R_w = training_results_full[training_results_full['Setup']==(agent_num+1)]['Whites_total_reward'].iloc[-1]
    #         if Final_R_w >= R_max:
    #             best_agent = agents_temp[agent_num]
    #             R_max = Final_R_w
    #         else:
    #             continue
    #     else:
    #         continue
    #     if convergence_met_b[0]==True:
    #         Final_R_w = training_results_full[training_results_full['Setup']==(agent_num+1)]['Blacks_total_reward'].iloc[-1]
    #         if Final_R_w >= R_max:
    #             best_agent = agents_temp[agent_num]
    #             R_max = Final_R_w
    #         else:
    #             continue
    #     else:
    #         continue

    # with open(self.save_dir+'/'+str(self.ExperimentConfig['name'])+'_convergence_results.txt', 'w') as f:
    #     f.write('Agent list: ' + str(list(agents_summary_info.keys()))+'\n'+ 
    #             '--- \n' +
    #             'Converegence Met: ' + str(convergence_results_met)+'\n'+
    #             'Converegence First Episode: ' + str(convergence_results_first_epi)+'\n')            

    # if R_max == 0:
    #     print("No agents convereged...Selecting first agent for testing...")
    #     agents.append(agents_temp[0])
    # else:
    #     agents.append(best_agent)


    def convergence_check(self, value_list: List[float], player_side: str, visual_save_dir: str):
        """ CONVERGENCE CHECK METHODOLOGY
        - Goes through each Q value by episode and calculates the percentage change from the previous result
        - Because a single point can not provide accurate results, we introduce a system in which N prior output points are checked
        - We set the prior check points by setting a range of episodes and evenly space N points between the current episode and the fist check point defined by the range
        - We accept that the output has converged if ALL the prior N outputs percentage change is less than our threshold
        - The episode for which the output has converged in then the first check point of this providing a systematic numeric convergence evaluation
        """
        perc_change_tracker = []
        prior_change_long_term_tracker = []
        conv_met_check = []
        conv_met = False
        for n in range(0,len(value_list)):
            value = value_list[n]
            # First row fixed value
            if n == 0:
                perc_change = 100
            else:
                perc_change = abs((value - prior_row_value)/prior_row_value)*100
            perc_change_tracker.append(perc_change)

            prior_epi_points_tracker = []
            if n<self.num_prior_epi:
                prior_change_long_term_epi_point = 100
                for prior_epi_point in range(0, self.num_prior_epi_points):
                    prior_epi_points_tracker.append(prior_change_long_term_epi_point)
            else:
                for prior_epi_point in range(0, self.num_prior_epi_points):
                    prior_Q_value_point = value_list[(n - self.num_prior_epi)+int(prior_epi_point*self.num_prior_epi/self.num_prior_epi_points)]
                    prior_change_long_term_epi_point = abs((value - prior_Q_value_point)/prior_Q_value_point)*100
                    prior_epi_points_tracker.append(prior_change_long_term_epi_point)
            prior_change_long_term_tracker.append(max(prior_epi_points_tracker))
            
            # Find first instance where all check points are below threshold and log the prior point check as well
            if (max(prior_epi_points_tracker)<self.conv_threshold_perc)&(conv_met==False):
                conv_met_check_points = []
                for prior_epi_point in range(0,self.num_prior_epi_points):
                    prior_Q_value_point = value_list[(n - self.num_prior_epi)+int(prior_epi_point*self.num_prior_epi/self.num_prior_epi_points)]
                    conv_met_check_points.append(prior_Q_value_point)
                conv_met_check_points.append(value)
                conv_met_check.append([conv_met_check_points])
                conv_met = True
                # We take the first check point where output has converged
                convergence_epi_first = (n - self.num_prior_epi)
                convergence_epi = n
                conv_met_check_plot_points = conv_met_check[convergence_epi][0]
            else:
                conv_met_check.append(0)
                conv_met = conv_met

            # current row becomes prior for next iteration
            prior_row_value = value


        print("Producing visual analysis...")
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.plot(value_list, alpha=0.5)
        if conv_met == True:
            print("Output converges after ", convergence_epi_first, "episodes")
            for prior_epi_point in range(0, self.num_prior_epi_points+1):
                # Plot line in addition to data points for first as this is where the system converges
                if prior_epi_point == 0:
                    Y = conv_met_check_plot_points[prior_epi_point]
                    plt.scatter((convergence_epi - self.num_prior_epi)+int(prior_epi_point*self.num_prior_epi/self.num_prior_epi_points), Y, c='red', marker='+')
                    plt.plot([0, len(conv_met_check)], [Y, Y], 'r--', linewidth=2)
                else:
                    Y = conv_met_check_plot_points[prior_epi_point]
                    plt.scatter((convergence_epi - self.num_prior_epi) + int(prior_epi_point * self.num_prior_epi / self.num_prior_epi_points), Y, c='red', marker='+')
        else:
            print("Convergence not found!")
            print("-- If oscillating, reduce alpha. If still learning, increase the number of episodes.") 
            convergence_epi_first = "None"

        plt.title("Mean Value by Episode")
        plt.ylabel("Mean Value")
        plt.xlabel("Episode")

        plt.subplot(1, 2, 2)
        plt.plot(prior_change_long_term_tracker, linewidth=2)
        plt.plot([0, len(prior_change_long_term_tracker)], [self.conv_threshold_perc, self.conv_threshold_perc], 'k--', linewidth=1)
        plt.title("Max Percentage Difference between Values from \n Current Episode and the "+str(self.num_prior_epi_points)+" Prior Convergence Points")
        plt.xlabel("Episode")
        plt.ylabel("Max Percentage Change")
        plt.ylim(0,105)
        
        if self.plot_convergence_figures == 'Y':
            plt.show(block=False)
            plt.pause(self.display_plot_time)
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(visual_save_dir+'/Convergence_results_'+str(player_side)+'.png', dpi=100)
            plt.close()
        else:
            fig = plt.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.savefig(visual_save_dir+'/Convergence_results_'+str(player_side)+'.png', dpi=100)
            plt.clf()

        return conv_met, convergence_epi_first
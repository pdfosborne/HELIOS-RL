import matplotlib.pyplot as plt
import numpy as np
import os 


def combined_variance_analysis_graph(variance_results:dict={}, save_dir:str='', show_figures:str='N'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    cycol = 'brgcmyk'
    line_styles = ['solid','dotted','dashed','dashdot', 'solid','dotted','dashed','dashdot']
    col = 0
    fig, axs = plt.subplots(2,2)
    for experiment in list(variance_results.keys()):
        results = variance_results[experiment]['results']
        num_episode = np.max(results['episode'])
        avg_r_mean_sorted = np.sort(results['avg_R_mean'])
        cdf_mean = 1. * np.arange(len(avg_r_mean_sorted)) / (len(avg_r_mean_sorted) - 1)

        # Plot RL Reward results for each approach
        # 1.1 Summary of total REWARD
        c = cycol[col]
        l = line_styles[col]
        if col <= len(cycol):
            col+=1
        else:
            c = np.random.rand(len(x),3)
            l = 'solid'                
        x =  results['episode']
        avg_R = np.array(results['avg_R_mean'])
        avg_R_SE = np.array(results['avg_R_se'])
        cum_R = np.array(results['cum_R_mean'])
        cum_R_SE = np.array(results['cum_R_se'])
        time_mean = np.array(results['time_mean'])
        
        axs[0,0].plot(x,avg_R, color=c, linestyle=l, label=str(experiment))
        axs[0,0].fill_between(x,avg_R-avg_R_SE, avg_R+avg_R_SE, color=c, alpha = 0.2)
        axs[0,1].plot(avg_r_mean_sorted,cdf_mean, color=c, linestyle=l)
        axs[1,0].plot(x,cum_R, color=c, linestyle=l)
        axs[1,0].fill_between(x,cum_R-cum_R_SE, cum_R+cum_R_SE, color=c, alpha = 0.2)
        axs[1,1].hist(time_mean, color=c, alpha=0.25)

    axs[0,0].set_xlabel("Episode")
    axs[0,0].set_ylabel('Reward')
    axs[0,0].axes.get_xaxis().set_ticks([0, num_episode])
    axs[0,0].set_title("Mean and Std. Err. of Rolling Avg. R epi)")
    
    axs[0,1].set_ylabel("Cumulative Probability")
    axs[0,1].set_xlabel("Mean Reward per Episode Window")
    axs[0,1].set_title("CDF of Rolling Average R")
    
    axs[1,0].set_xlabel("Episode")
    axs[1,0].set_ylabel('Cumulative Reward')
    axs[1,0].axes.get_xaxis().set_ticks([0, num_episode])
    axs[1,0].set_title("Cumulative R with Std. Err.")
    
    axs[1,1].set_ylabel("Occurence")
    axs[1,1].set_xlabel("Time")
    axs[1,1].set_title("Dist of Time per Episode")

    #ax1.legend(loc=2, bbox_to_anchor=(-0.05, 0), fancybox=True, shadow=True, framealpha=1)
    #ax2.legend(loc=2, bbox_to_anchor=(0, 1), fancybox=True, shadow=True, framealpha=1)
    fig.suptitle("Variance Results for: "+str(variance_results[experiment]['env_name'])+' Variance over '+str(variance_results[experiment]['num_repeats'])+" runs \n with random & independent seeds")
    fig.legend(loc='upper right', fancybox=True, shadow=True, framealpha=1)
    fig.set_size_inches(12, 8)
    fig.tight_layout()

    if show_figures == 'Y':
        plt.show(block=False)
        plt.pause(5)
        fig.savefig(save_dir+'/variance_comparison.png', dpi=100)
        plt.close()
    else:
        fig.savefig(save_dir+'/variance_comparison.png', dpi=100)
        plt.close()


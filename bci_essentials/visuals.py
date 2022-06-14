# Visualization toolbox for BCI
# Written by: Brian Irvine
# 03/28/2022

import matplotlib.pyplot as plt
import numpy as np

# Visualization for P300 ERP
def decision_vis(decision_block, f_sample, label, channel_labels = [], ylims=(-100,100)):
    """
    decision_block shape in N,M,P
        N is the number of channels
        M is the number of samples
        P is the number of possible selections

    f_sample is the sampling rate

    label identifies the ERP from training label

    channel_labels identify the names of the channels according to 10-20 system
    """
    P,N,M = decision_block.shape

    # If no channel label then assign one based on its position
    if channel_labels == []:
        for n in range(N):
            channel_labels.append(n)

    # Make time vector
    t = np.ndarray((M))
    for m in range(M):
        t[m] = m / f_sample


    # Initialize subplot
    ax = [0] * P
    for p in range(P):
        ax[p] = plt.subplot(P+1,1,p+1)

    # Plot the ERP in the first subplot
    # for n in range(N):
    #     ax[0].plot(t, decision_block[label,n,:], label=channel_labels[n])
    #     ax[0].legend()
    #     ax[0].set_ylim(ylims)


    ind = 0
    # Plot non ERP in the subsequent subplots
    for p in range(P):
        if p == label:
            for n in range(N):
                ax[ind].plot(t, decision_block[label,n,:], label=channel_labels[n])

            ax[ind].legend()
            ax[ind].set_ylim(ylims) 
            ax[ind].set_ylabel("ERP")

            

            ind += 1

        else:
            for n in range(N):
                ax[ind].plot(t, decision_block[ind,n,:], label=channel_labels[n])

            ax[ind].set_ylim(ylims)
            ax[ind].legend()
            ax[ind].set_ylabel("Not ERP")

            ind += 1

            

    

    plt.show()

def plot_big_decision_block(big_decision_block, f_sample, channel_labels = [], erp_targets=None, ylims=(-100,100)):
    """
    decision_block shape in N,M,P
        N is the number of channels
        M is the number of samples
        P is the number of possible selections

    f_sample is the sampling rate

    label identifies the ERP from training label

    channel_labels identify the names of the channels according to 10-20 system
    """
    D,O,W,N,M = big_decision_block.shape

    D = 9

    # Give default channel names if none are given
    if channel_labels == []:
        for n in range(N):
            channel_labels.append(n)

    # Make time vector
    t = np.ndarray((M))
    for m in range(M):
        t[m] = m / f_sample

    fig = [None] * D
    # for d in D create a figure
    for d in range(D):
        fig[d], ax = plt.subplots(nrows=2,ncols=1)

        # for ERP in O and one non ERP in O, each in own subplot
        
        # plot the ERP
        erp_label = erp_targets[d]
        # for w in W plot the signal
        for w in range(W):
            sum_range = big_decision_block[d,erp_label,w,0,0:10].sum()
            if sum_range != 0:
                for n in range(N):
                    color_string = "C{}".format(int(n))
                    ax[0].plot(t, big_decision_block[d,erp_label,w,n,:], label=channel_labels[n], color=color_string)

                # plot the average of all W in bold
                color_string = "C{}".format(int(n))
            else:
                wmax = w-1
                break
        
        win_mean_bdb = np.mean(big_decision_block[d,erp_label,:,:,:], axis=0)
        for n in range(N):
            color_string = "C{}".format(int(n))
            ax[0].plot(t, win_mean_bdb[n,:], label=channel_labels[n], color=color_string, linewidth=1.0)

        ax[0].set_title("ERP on object{}".format(erp_label))
        ax[0].set_ylim(ylims)
        ax[0].legend()

        
        # plot any non ERP
        non_erp_label = 0
        # for w in W plot the signal
        for w in range(W):
            sum_range = big_decision_block[d,non_erp_label,w,0,0:10].sum()
            if sum_range != 0:
                for n in range(N):
                    color_string = "C{}".format(int(n))
                    ax[1].plot(t, big_decision_block[d,non_erp_label,w,n,:], label=channel_labels[n], color=color_string)

                # plot the average of all W in bold
                color_string = "C{}".format(int(n))
            else:
                wmax = w-1
                break
        
        win_mean_bdb = np.mean(big_decision_block[d,non_erp_label,:,:,:], axis=0)
        for n in range(N):
            color_string = "C{}".format(int(n))
            ax[1].plot(t, win_mean_bdb[n,:], label=channel_labels[n], color=color_string, linewidth=1.0)

        ax[1].set_title("Non-ERP on object{}".format(non_erp_label))
        ax[1].set_ylim(ylims)
        ax[1].legend()
            # for w in W plot the signal

            # plot the average of all W in bold

        fig[d].show()


# Plot window
def plot_window(window, f_sample, channel_labels = []):
    N,M = window.shape

    # If no channel label then assign one based on its position
    if channel_labels == []:
        for n in range(N):
            channel_labels.append(n)

    # Make time vector
    t = np.ndarray((M))
    for m in range(M):
        t[m] = m / f_sample


    fig, axs = plt.subplots(N)

    for n in range(N):
        axs[n].plot(t,window[n,:])
        axs[n].ylabel = channel_labels[n]

    fig.show()
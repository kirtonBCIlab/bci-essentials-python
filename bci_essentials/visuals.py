"""
Visualization toolbox for BCI Essentials

The EEG data inputs for each function are either windows or
decision blocks.
- For windows, inputs are of the shape `N x M x P`, where:
    - N = number of channels
    - M = number of samples
    - P = number of windows (for a single window `P = 1`)
- For decision blocks, inputs are of the shape `N x M x P`, where:
    - N = number of channels
    - M = number of samples
    - P = number of possible selections

"""
import matplotlib.pyplot as plt
import numpy as np


def decision_vis(decision_block, f_sample, label, channel_labels=[], ylims=(-100, 100)):
    """Visualization for P300 ERP.

    Creates plots of the P300 ERP and non-ERP for each channel.

    Parameters
    ----------
    decision_block : numpy.ndarray
        A decision block of EEG data.
        3D array containing data with `float` type.

        shape = (`N_channels`,`M_samples`,`P_selections`)
    f_sample : float
        Sampling rate of the signal.
    label : int
        Identifies the ERP from training label.
    channel_labels : list of `str`, *optional*
        Identity of the names of the channels according to 10-20 system
        - Default is `[]` and assigns labels based on the channel's index.
    ylims : tuple of `(int, int)`, *optional*
        Y-axis limits for the plots
        - Default is `(-100, 100)`.

    Returns
    -------
    `None`

    """
    P, N, M = decision_block.shape

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
        ax[p] = plt.subplot(P + 1, 1, p + 1)

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
                ax[ind].plot(t, decision_block[label, n, :], label=channel_labels[n])

            ax[ind].legend()
            ax[ind].set_ylim(ylims)
            ax[ind].set_ylabel("ERP")

            ind += 1

        else:
            for n in range(N):
                ax[ind].plot(t, decision_block[ind, n, :], label=channel_labels[n])

            ax[ind].set_ylim(ylims)
            ax[ind].legend()
            ax[ind].set_ylabel("Not ERP")

            ind += 1

    plt.show()


def plot_big_decision_block(
    big_decision_block, f_sample, channel_labels=[], erp_targets=None, ylims=(-100, 100)
):
    """Plots the big decision block.

    Creates plots of the P300 ERP and non-ERP big decision blocks for each
    channel.

    Parameters
    ----------
    decision_block : numpy.ndarray
        A decision block of EEG data.
        3D array containing data with `float` type.

        shape = (`N_channels`,`M_samples`,`P_selections`)
    f_sample : float
        Sampling rate of the signal.
    label : int
        Identifies the ERP from training label.
    channel_labels : list of str, *optional*
        Identity of the names of the channels according to 10-20 system.
        - Default is `[]` and assigns labels based on the channel's index).
    erp_targets : list, *optional*
        List of the ERP targets
        - Default is `None`.
    ylims : tuple of `(int, int)`, *optional*
        Y-axis limits for the plots
        - Default is `(-100, 100)`.

    Returns
    -------
    `None`

    """
    D, O, W, N, M = big_decision_block.shape

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
        fig[d], ax = plt.subplots(nrows=2, ncols=1)

        # for ERP in O and one non ERP in O, each in own subplot

        # plot the ERP
        erp_label = erp_targets[d]
        # for w in W plot the signal
        for w in range(W):
            sum_range = big_decision_block[d, erp_label, w, 0, 0:10].sum()
            if sum_range != 0:
                for n in range(N):
                    color_string = "C{}".format(int(n))
                    ax[0].plot(
                        t,
                        big_decision_block[d, erp_label, w, n, :],
                        label=channel_labels[n],
                        color=color_string,
                    )

                # plot the average of all W in bold
                color_string = "C{}".format(int(n))
            else:
                break

        win_mean_bdb = np.mean(big_decision_block[d, erp_label, :, :, :], axis=0)
        for n in range(N):
            color_string = "C{}".format(int(n))
            ax[0].plot(
                t,
                win_mean_bdb[n, :],
                label=channel_labels[n],
                color=color_string,
                linewidth=1.0,
            )

        ax[0].set_title("ERP on object{}".format(erp_label))
        ax[0].set_ylim(ylims)
        ax[0].legend()

        # plot any non ERP
        non_erp_label = 0
        # for w in W plot the signal
        for w in range(W):
            sum_range = big_decision_block[d, non_erp_label, w, 0, 0:10].sum()
            if sum_range != 0:
                for n in range(N):
                    color_string = "C{}".format(int(n))
                    ax[1].plot(
                        t,
                        big_decision_block[d, non_erp_label, w, n, :],
                        label=channel_labels[n],
                        color=color_string,
                    )

                # plot the average of all W in bold
                color_string = "C{}".format(int(n))
            else:
                break

        win_mean_bdb = np.mean(big_decision_block[d, non_erp_label, :, :, :], axis=0)
        for n in range(N):
            color_string = "C{}".format(int(n))
            ax[1].plot(
                t,
                win_mean_bdb[n, :],
                label=channel_labels[n],
                color=color_string,
                linewidth=1.0,
            )

        ax[1].set_title("Non-ERP on object{}".format(non_erp_label))
        ax[1].set_ylim(ylims)
        ax[1].legend()
        # for w in W plot the signal

        # plot the average of all W in bold

        fig[d].show()


# Plot window
def plot_window(window, f_sample, channel_labels=[]):
    """Plots a window of EEG data.

    Parameters
    ----------
    window : numpy.ndarray
        A window of EEG data.
        2D array containing data with `float` type.

        shape = (`N_channels`,`M_samples`)
    f_sample : float
        Sampling rate of the signal.
    channel_labels : list, *optional*
        Identity of the names of the channels according to 10-20 system
        - Default is `[]` and assigns labels based on the channel's index.

    Returns
    -------
    `None`

    """
    N, M = window.shape

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
        axs[n].plot(t, window[n, :])
        axs[n].ylabel = channel_labels[n]

    fig.show()

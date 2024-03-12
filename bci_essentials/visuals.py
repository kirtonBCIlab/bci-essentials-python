"""
Visualization toolbox for BCI Essentials

The EEG data inputs for each function are either trials or
decision blocks.
- For trials, inputs are of the shape `n_trials x n_channels x n_samples`, where:
    - n_trials = number of trials (for a single trial `n_trials = 1`)
    - n_channels = number of channels
    - n_samples = number of samples
- For decision blocks, inputs are of the shape `n_decisions x n_channels x n_samples`, where:
    - n_decisions = number of possible decisions to select from
    - n_channels = number of channels
    - n_samples = number of samples

"""

import matplotlib.pyplot as plt
import numpy as np

from .utils.logger import Logger  # Logger wrapper

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


def decision_vis(decision_block, f_sample, label, channel_labels=[], ylims=(-100, 100)):
    """Visualization for P300 ERP.

    Creates plots of the P300 ERP and non-ERP for each channel.

    Parameters
    ----------
    decision_block : numpy.ndarray
        A decision block of EEG data.
        3D array containing data with `float` type.

        shape = (`n_decisions`,`n_channels`,`n_samples`)
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
    n_decisions, n_channels, n_samples = decision_block.shape

    # If no channel label then assign one based on its position
    if channel_labels == []:
        channel_labels = [str(channel) for channel in range(n_channels)]

    # Make time vector
    t = np.ndarray((n_samples))
    for sample in range(n_samples):
        t[sample] = sample / f_sample

    # Initialize subplot
    ax = [0] * n_decisions
    for decision in range(n_decisions):
        ax[decision] = plt.subplot(n_decisions + 1, 1, decision + 1)

    # Plot the ERP in the first subplot
    # for channel in range(n_channels):
    #     ax[0].plot(t, decision_block[label,channel,:], label=channel_labels[channel])
    #     ax[0].legend()
    #     ax[0].set_ylim(ylims)

    ind = 0
    # Plot non ERP in the subsequent subplots
    for decision in range(n_decisions):
        if decision == label:
            decision_slice = decision_block[decision, :, :]
            for channel in range(n_channels):
                channel_data = decision_slice[channel, :]
                ax[ind].plot(t, channel_data, label=channel_labels[channel])

            ax[ind].legend()
            ax[ind].set_ylim(ylims)
            ax[ind].set_ylabel("ERP")

            ind += 1

        else:
            decision_slice = decision_block[ind, :, :]
            for channel in range(n_channels):
                channel_data = decision_slice[channel, :]
                ax[ind].plot(t, channel_data, label=channel_labels[channel])

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

        shape = (`n_channels`,`n_samples`,`P_selections`)
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
    n_decisions, O, n_trials, n_channels, n_samples = big_decision_block.shape

    n_decisions = 9

    # Give default channel names if none are given
    if channel_labels == []:
        channel_labels = [str(channel) for channel in range(n_channels)]

    # Make time vector
    t = np.ndarray((n_samples))
    for sample in range(n_samples):
        t[sample] = sample / f_sample

    fig = [None] * n_decisions
    # for decision in n_decisions create a figure
    for decision in range(n_decisions):
        fig[decision], ax = plt.subplots(nrows=2, ncols=1)

        # for ERP in O and one non ERP in O, each in own subplot

        # plot the ERP
        erp_label = erp_targets[decision]
        # for trial in n_trials plot the signal
        for trial in range(n_trials):
            sum_range = big_decision_block[decision, erp_label, trial, 0, 0:10].sum()
            if sum_range != 0:
                for channel in range(n_channels):
                    color_string = "C{}".format(int(channel))
                    ax[0].plot(
                        t,
                        big_decision_block[decision, erp_label, trial, channel, :],
                        label=channel_labels[channel],
                        color=color_string,
                    )

                # plot the average of all n_trials in bold
                color_string = "C{}".format(int(channel))
            else:
                break

        trial_mean_bdb = np.mean(
            big_decision_block[decision, erp_label, :, :, :], axis=0
        )
        for channel in range(n_channels):
            color_string = "C{}".format(int(channel))
            ax[0].plot(
                t,
                trial_mean_bdb[channel, :],
                label=channel_labels[channel],
                color=color_string,
                linewidth=1.0,
            )

        ax[0].set_title("ERP on object{}".format(erp_label))
        ax[0].set_ylim(ylims)
        ax[0].legend()

        # plot any non ERP
        non_erp_label = 0
        # for trial in n_trials plot the signal
        for trial in range(n_trials):
            sum_range = big_decision_block[
                decision, non_erp_label, trial, 0, 0:10
            ].sum()
            if sum_range != 0:
                for channel in range(n_channels):
                    color_string = "C{}".format(int(channel))
                    ax[1].plot(
                        t,
                        big_decision_block[decision, non_erp_label, trial, channel, :],
                        label=channel_labels[channel],
                        color=color_string,
                    )

                # plot the average of all n_trials in bold
                color_string = "C{}".format(int(channel))
            else:
                break

        trial_mean_bdb = np.mean(
            big_decision_block[decision, non_erp_label, :, :, :], axis=0
        )
        for channel in range(n_channels):
            color_string = "C{}".format(int(channel))
            ax[1].plot(
                t,
                trial_mean_bdb[channel, :],
                label=channel_labels[channel],
                color=color_string,
                linewidth=1.0,
            )

        ax[1].set_title("Non-ERP on object{}".format(non_erp_label))
        ax[1].set_ylim(ylims)
        ax[1].legend()
        # for trial in n_trials plot the signal

        # plot the average of all n_trials in bold

        fig[decision].show()


# Plot trial
def plot_trial(eeg_data_trial, f_sample, channel_labels=[]):
    """Plots a trial of EEG data.

    Parameters
    ----------
    eeg_data_triala : numpy.ndarray
        A trial of EEG data.
        2D array containing data with `float` type.

        shape = (`n_channels`,`n_samples`)
    f_sample : float
        Sampling rate of the signal.
    channel_labels : list, *optional*
        Identity of the names of the channels according to 10-20 system
        - Default is `[]` and assigns labels based on the channel's index.

    Returns
    -------
    `None`

    """
    n_channels, n_samples = eeg_data_trial.shape

    # If no channel label then assign one based on its position
    if channel_labels == []:
        channel_labels = [str(channel) for channel in range(n_channels)]

    # Make time vector
    t = np.ndarray((n_samples))
    for sample in range(n_samples):
        t[sample] = sample / f_sample

    fig, axs = plt.subplots(n_channels)

    for channel in range(n_channels):
        channel_data = eeg_data_trial[channel, :]
        axs[channel].plot(t, channel_data)
        axs[channel].ylabel = channel_labels[channel]

    fig.show()

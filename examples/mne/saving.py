import mne
from bci_essentials.utils.logger import Logger
from bci_essentials.eeg_data import EegData
from bci_essentials.erp_data import ErpData

# Instantiate a logger for the module at the default level of logging.INFO
# Logs to bci_essentials.__module__) where __module__ is the name of the module
logger = Logger(name=__name__)


def mne_export_as_raw(eeg_data_object):
    """MNE export EEG as RawArray.

    Exports the EEG data as a MNE RawArray object.

    **Requires MNE**

    Returns
    -------
    raw_array : mne.io.RawArray
        MNE RawArray object.

    """
    
    logger.error("mne_export_as_raw has not been implemented yet")

    assert isinstance(eeg_data_object, EegData)

    # create info from metadata
    info = mne.create_info(
        ch_names=eeg_data_object.channel_labels,
        sfreq=eeg_data_object.fsample,
        ch_types="eeg",
    )

    # create the MNE epochs, pass in the raw

    # make sure that units match
    raw_data = eeg_data_object.eeg_data.transpose()
    raw_array = mne.io.RawArray(data=raw_data, info=info)

    # change the last column of epochs array events to be the class labels
    # raw_array.events[:, -1] = eeg_data_object.labels

    return raw_array


def mne_export_as_epochs(eeg_data_object):
    """MNE export EEG as EpochsArray.

    Exports the EEG data as a MNE EpochsArray object.

    **Requires MNE**

    Returns
    -------
    epochs_array : mne.EpochsArray
        MNE EpochsArray object.

    """

    logger.error("mne_export_as_raw has not been implemented yet")
    
    assert isinstance(eeg_data_object, EegData)

    # create info from metadata
    info = mne.create_info(
        ch_names=eeg_data_object.channel_labels,
        sfreq=eeg_data_object.fsample,
        ch_types=eeg_data_object.ch_type,
    )

    # create the MNE epochs, pass in the raw

    # make sure that units match
    epoch_data = eeg_data_object.raw_eeg_windows.copy()
    for i, u in enumerate(eeg_data_object.ch_units):
        if u == "microvolts":
            # convert to volts
            epoch_data[:, i, :] = epoch_data[:, i, :] / 1000000

    epochs_array = mne.EpochsArray(data=epoch_data, info=info)

    # change the last column of epochs array events to be the class labels
    epochs_array.events[:, -1] = eeg_data_object.labels

    return epochs_array


def mne_export_resting_state_as_raw(eeg_data_object):
    """MNE export resting state EEG as RawArray.

    Exports the resting state EEG data as a MNE RawArray object.

    **Requires MNE**

    Returns
    -------
    raw_array : mne.io.RawArray
        MNE RawArray object.

    """

    logger.error("mne_export_as_raw has not been implemented yet")

    assert isinstance(eeg_data_object, EegData)

    # Check for mne
    try:
        import mne
    except Exception:
        logger.critical(
            "Could not import mne, you may have to install (pip install mne)"
        )

    # create info from metadata
    info = mne.create_info(
        ch_names=eeg_data_object.channel_labels,
        sfreq=eeg_data_object.fsample,
        ch_types="eeg",
    )

    try:
        # create the MNE epochs, pass in the raw

        # make sure that units match
        raw_data = eeg_data_object.rest_windows[0, :, :]
        raw_array = mne.io.RawArray(data=raw_data, info=info)

        # change the last column of epochs array events to be the class labels
        # raw_array.events[:, -1] = eeg_data_object.labels

    except Exception:
        # could not find resting state data, sending the whole collection instead
        logger.warning(
            "NO PROPER RESTING STATE DATA FOUND, SENDING ALL OF THE EEG DATA INSTEAD"
        )
        raw_data = eeg_data_object.eeg_data.transpose()
        raw_array = mne.io.RawArray(data=raw_data, info=info)

    return raw_array


def mne_export_erp_as_epochs(erp_data_object):
    """MNE export EEG as EpochsArray.

    Exports the EEG data as a MNE EpochsArray object.

    **Requires MNE**

    Returns
    -------
    epochs_array : mne.EpochsArray
        MNE EpochsArray object.

    """

    assert isinstance(erp_data_object, ErpData)

    # create info from metadata
    info = mne.create_info(
        ch_names=erp_data_object.channel_labels,
        sfreq=erp_data_object.fsample,
        ch_types=erp_data_object.ch_type,
    )

    # create the MNE epochs, pass in the raw

    # make sure that units match
    # This only works because ErpData.erp_windows_processed is not private, like it probable should be
    epoch_data = erp_data_object.erp_windows_processed[
        : len(erp_data_object.target_index), :, :
    ].copy()
    for i, u in enumerate(erp_data_object.ch_units):
        if u == "microvolts":
            # convert to volts
            epoch_data[:, i, :] = epoch_data[:, i, :] / 1000000

    epochs_array = mne.EpochsArray(data=epoch_data, info=info)

    # change the last column of epochs array events to be the class labels
    epochs_array.events[:, -1] = erp_data_object.target_index.astype(int)

    return epochs_array


def mne_export_erp_as_evoked(erp_data_object):
    """MNE Export evoked EEG data as EpochsArray.

    Exports the evoked EEG data as a MNE EpochsArray object.

    **Requires MNE**

    **HAS NOT BEEN IMPLEMENTED YET.**

    Returns
    -------
    evoked_array : mne.EpochsArray
        MNE EpochsArray object.

        **NOTE: NOT ACTUALLY THE CASE AT THE MOMENT**.
        This is what the code will return once it has been implemented.

    """
    logger.error("mne_export_as_evoked has not yet been implemented")

    assert isinstance(erp_data_object, ErpData)
    # # Check for mne
    # try:
    #     import mne
    # except Exception:
    #     logger.critical(
    #         "Could not import mne, you may have to install (pip install mne)"
    #     )

    # # create info from metadata
    # info = mne.create_info(ch_names=erp_data_object.channel_labels, sfreq=erp_data_object.fsample, ch_types=erp_data_object.ch_type)

    # # create the MNE epochs, pass in the raw

    # # make sure that units match
    # evoked_data = erp_data_object.raw_eeg_windows.copy()
    # for i, u in enumerate(erp_data_object.ch_units):
    #     if u == "microvolts":
    #         # convert to volts
    #         evoked_data[:,i,:] = evoked_data[:,i,:] / 1000000

    # evoked_array = mne.EpochsArray(data=evoked_data, info=info, tmin=erp_data_object.window_start)

    # # change the last column of epochs array events to be the class labels
    # evoked_array.events[:, -1] = erp_data_object.labels

    # return evoked_array

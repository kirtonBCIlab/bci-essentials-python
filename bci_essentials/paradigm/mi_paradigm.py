import numpy as np

from .base_paradigm import BaseParadigm


class MiParadigm(BaseParadigm):
    """
    MI paradigm.
    """

    def __init__(
        self,
        filters=[5, 30],
        channel_subset=None,
        iterative_training=False,
        live_update=False,
    ):
        """
        Parameters
        ----------
        filters : list of floats, *optional*
            Filter bands.
            - Default is `[5, 30]`.
        channel_subset : list of str, *optional*
            Channel subset to use.
            - Default is `None`.
        iterative_training : bool, *optional*
            Flag to indicate if the classifier will be updated iteratively.
            - Default is `False`.
        live_update : bool, *optional*
            Flag to indicate if the classifier will be used to provide
            live updates on trial classification.
            - Default is `False`.
        """
        super().__init__(filters, channel_subset)

        self.live_update = live_update
        self.iterative_training = iterative_training

        if self.live_update:
            self.classify_each_epoch = True

        if self.iterative_training:
            self.classify_each_trial = True

    def process_markers(self, markers, marker_timestamps, eeg, eeg_timestamps):
        """
        This takes 
        """
        X = None
        y = None
        return X, y

    def check_compatibility(self):
        pass

    def process_trial(self):
        pass

    def process_event_marker(self):
        pass

    def update_classifier(self):
        pass

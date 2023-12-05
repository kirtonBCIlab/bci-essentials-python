
import numpy as np
from ..classification import FeatureExtractorSSVEP
# import ..classification.FeatureExtractorSSVEP as FeatureExtractorSSVEP
# from ..classification.FeatureExtractorSSVEP import FeatureExtractor
# import FeatureExtractorSSVEP
from ..classification.generic_classifier import Generic_classifier

class SSVEP_NCAN_classifier(Generic_classifier):
    """ 
    Interfacing classifier to use NCAN SSVEP classifiers 
    with BCI-essentials-python 
    """

    def set_ssvep_settings(
        self,
        sampling_freq:int,
        target_freqs:np.ndarray,
        classifier_name:str="MEC",
        harmonics_count:int=4,
        subbands=None,
        voters_count:int=1,
        use_gpu:bool=False,
        max_batch_size:int=16,
        explicit_multithreading:int=0,
    ):
        """
        Parameters
        ----------
        `sampling_frequency` : int
            Sampling frequency of the signal [Hz].
        `target frequencies` : list[int] 
            Stimulation frequencies of each target. Must be a 1D array.
            This must be a 1D array,  where the first element is the stimulation
            frequency of the first target, the second element is the stimulation
            frequency of the second target, and so on.  The length of this array
            indicates the number of targets.  The user must determine the 
            targets_frequencies but the number of targets (targets_count) is 
            extracted automatically from the length of targets_frequencies. 
        `classifier_name` : str
            Name of the classifier to be used. Options are: CCA, MEC, MSI.
        `harmonics_count` : int
            The number of harmonics to be used in constructing the template
            signal.  This variable must be a positive integer number
            (typically a value from 3 to 5).  
        `subbands` : np.ndarray

        `voters_count` : int=1,
        use_gpu:bool=False,
        max_batch_size:int=16,
        explicit_multithreading:int=0
        `samples_count` : int
            If provided, the class performs precomputations that
            only depend on the number of samples, e.g., computing the template
            signal.  If not provided, the class does not perform precomputations.
            Instead, it does the computations once the input signal was provided 
            and the class learns the number of samples from the input signal. 
            Setting samples_count is highly recommended.  If the feaure extraction
            method is being used in loop (e.g., BCI2000 loop), setting this 
            parameter eliminates the need to compute the template matrix each
            time. It also helps the class to avoid other computations in each
            iteration. samples_count passed to this function must be the same 
            as the third dimension size of the signal passed to extract_features().
            If that is not the case, the template and input signal will have 
            different dimensions.  The class should issue an error in this case
            and terminate the execution. 
        """
        self.sampling_freq = sampling_freq
        self.target_freqs = target_freqs
        self.classifier_name = classifier_name
        self.harmonics_count = harmonics_count
        self.subbands = subbands
        self.voters_count = voters_count
        self.use_gpu = use_gpu
        self.max_batch_size = max_batch_size
        self.explicit_multithreading = explicit_multithreading
        
        # Create classifier object
        classifiers = {
            "CCA": FeatureExtractorSSVEP.FeatureExtractorCCA,
            "MEC": FeatureExtractorSSVEP.FeatureExtractorMEC,
            "MSI": FeatureExtractorSSVEP.featureExtractorMSI,
        }
        self.clf = classifiers[classifier_name]()

        # Flag to determine if the classifier has been created and set up
        self.setup = False
        
    def fit(self, print_fit=True, print_performance=True):
        """ Fit the model. """

    def predict(self, X, print_predict):
        # Get the shape of the data
        (nwindows, nchannels, nsamples) = X.shape

        # Check that the classifier has not been created, to create it only once
        if self.setup == False:
            # self.clf = exec(f"import FeatureExtractorSSVEP.FeatureExtractor{self.classifier_name}()")
            self.clf.setup_feature_extractor(
                harmonics_count = self.harmonics_count,
                targets_frequencies = self.target_freqs,
                sampling_frequency = self.sampling_freq,
                voters_count = self.voters_count,
                use_gpu = self.use_gpu,
                max_batch_size = self.max_batch_size,
                explicit_multithreading = self.explicit_multithreading,
                samples_count = nsamples,
                subbands = self.subbands,
            )

            self.setup = True
        else: 
            # Convert data to 32-bit for better GPU performance
            X = np.float32(X)

            prediction = self.clf.extract_features(X)

        return prediction
        

        
# featureExtractorTemplateMatching.py
"""Definition for the parent class for CCA, MEC, and MSI"""
from .featureExtractor import FeatureExtractor
import numpy as np

class FeatureExtractorTemplateMatching(FeatureExtractor):
    """A parent class for CCA, MEC, and MSI"""
    
    __targets_frequencies_setup_guide = (
        "The frequencies of targets must be a one dimensional array of real "
        + "positive numbers, where the first element represents the "
        + "frequency of the first target, the second element is the "
        + "frequency of the scond target, and so on. All frequencies must "
        + "be in Hz. ")
    
    def __init__(self):
        """Setting all attributes to valid initiali values"""
        super().__init__()
        
        # Hold the pre-computed template signals for SSVE.
        # This is a 3D array, with dimensions of [number of targets, 2*Nh, 
        # number of samples].  Each target has a unique template signal that 
        # consistes of a linear combination of 2*Nh sinusoidal signals, where. 
        # Nh is the number of harmonics.  For each harmonic a (sine, cosine)
        # pair is computed, hence the midle dimension has the size of 2*Nh.
        self.template_signal = None
        
        # The number of harmonics or Nh as explained in the previous comment.
        # This must be a natural number (generally less than 5).  The user
        # must set this value.
        self.harmonics_count = 0
        
        # The stimulation freqeuency of each target.  This must be a 1D vector,
        # where the first element is the stimulation frequency of the first
        # target, the second element is the stimulation frequency of the 
        # second target, and so on.  The length of this vector indicates the
        # number of targets.  The user must determine the targets_frequencies
        # but the number of targets (targets_count) is extracted automatically
        # from the length of targets_frequencies. 
        self.targets_frequencies = None
        self.targets_count = 0
          
    def compute_templates(self):
        """Pre-compute the template signals for all target frequencies"""
        # Template must have the same length as the signal.
        t = np.arange(1, self.samples_count+1)
        
        # Dividing by the sampling frequency returns the actual
        # time in seconds.
        t = t/self.sampling_frequency
        
        # Pre-compute the template signal.  This generates a 4D array.  
        # Index the first dimension to access the target.  Index the 
        # second dimension to access harmonics.  Index the third dimension
        # to access either sine or cosine.  Index the fourth dimension to
        # access samples. 
        template_signal = np.array(
            [[(np.sin(2*np.pi*t*f*h), np.cos(2*np.pi*t*f*h))
             for h in range(1, self.harmonics_count+1)]
             for f in self.targets_frequencies])
        
        # Reshape template_signal into a 3D array, where the first dimension
        # indexes targets, the second dimension indexes harmonics and 
        # sine/cosine, and the third dimension indexes samples. 
        # The first enetry of the second dimension is the sine wave of the 
        # fundamental frequency, the second entry is the cosine wave of the 
        # fundamental frequency, the third entry is the sine wave of the first
        # harmonic, the fourth entry is the cosine wave of the first harmonic,
        # and so on. 
        self.template_signal = np.reshape(
            template_signal, 
            (self.targets_count,
             self.harmonics_count*2,
             self.samples_count))
        
        self.template_signal = np.transpose(
            self.template_signal, axes=(0, 2, 1))
        
    @property
    def  template_signal(self):
        """Getter function for the template signals"""
        return self._template_signal
    
    @template_signal.setter
    def template_signal(self, template_signal):
        """Setter function for the template signals"""
        error_message = "template_signal must be a 3D array of floats."
        
        if template_signal is None:
            self._template_signal = 0
            return 
        
        try:
            template_signal = template_signal.astype(np.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(error_message)
        
        if not template_signal.ndim == 3:
            self.quit(error_message)
        
        self._template_signal = template_signal            
    
    @property
    def harmonics_count(self):
        """Getter function for the number of harmonics"""
        if self._harmonics_count == 0:
            self.quit("The number of harmonics is not set properly. "
                      + "To set the number of harmonics use the "
                      + "harmonics_count option of method "
                      + "setup_feature_extractor. ")
            
        return self._harmonics_count
    
    @harmonics_count.setter
    def harmonics_count(self, harmonics_count):
        """Setter method for the number of harmonics"""
        error_message = "Number of harmonics must be a positive integer."
        
        try:
            harmonics_count = int(harmonics_count)
        except (ValueError, TypeError):
            self.quit(error_message)
        
        if harmonics_count < 0:
            self.quit(error_message)      
          
        self._harmonics_count = harmonics_count 
        
    @property 
    def targets_frequencies(self):
        """Getter function for the frequencies of stimuli"""
        if self._targets_frequencies is None:
            self.quit("The frequencies of targets is not specified. To set "
                      + "this variable, use the targets_frequencies option "
                      + "of setup_feature_extractor. " 
                      + self.__targets_frequencies_setup_guide)
            
        return self._targets_frequencies
    
    @targets_frequencies.setter
    def targets_frequencies(self, stimulation_frequencies):
        """Setter function for the frequencies of stimuli"""
        error_message = ("Target frequencies must be an array of positive "
                         + "real numbers. ")
        error_message += self.__targets_frequencies_setup_guide
        
        if stimulation_frequencies is None:
            self._targets_frequencies = None
            self.targets_count = 0
            return
            
        try:
            stimulation_frequencies = np.array(stimulation_frequencies)
            stimulation_frequencies.astype(np.float32)
        except (ValueError, TypeError, AttributeError):
            self.quit(error_message)
            
        if stimulation_frequencies.ndim == 0:
            stimulation_frequencies = np.array([stimulation_frequencies])
        
        if np.array([stimulation_frequencies <= 0]).any():
           self.quit(error_message)
           
        self._targets_frequencies = stimulation_frequencies
        self.targets_count = stimulation_frequencies.size
        
    @property
    def targets_count(self):
        """Getter function for the number of targets"""
        if self._targets_count == 0:
            self.quit("The number of targets is not set. This happens "
                      + "because the target frequencies is not specified. "
                      + "To specify the target frequencies use the "
                      + "targets_frequencies option of the method "
                      + "setup_feature_extractor. ")
        return self._targets_count
    
    @targets_count.setter
    def targets_count(self, targets_count):
        """Setter function for the number of targets"""
        error_message = "Number of targets must be a positive integer."
        
        try:
            targets_count = int(targets_count)
        except (ValueError, TypeError):
            self.quit(error_message)
            
        if targets_count < 0:
            self.quit(error_message)
            
        self._targets_count = targets_count
# featureExtractorMEC.py
"""
Implementation of MEC feature extraction method
# Feature extraction method using minimum energy combination based on:
# Friman, Ola, Ivan Volosyak, and Axel Graser. "Multiple channel
# detection of steady-state visual evoked potentials for brain-computer
# interfaces." IEEE transactions on biomedical engineering 54.4 (2007).
"""
# Import the definition of the parent class.  Make sure the file is in the
# working directory.  
from .featureExtractorTemplateMatching import FeatureExtractorTemplateMatching

# Needed for many matrix computations
import numpy as np

try:
    import cupy as cp
    cupy_available_global = True
except:
    cupy_available_global = False
    cp = np

# # A custom CUDA kernel for sum of products.
# @cp.fuse(kernel_name='sum_of_products')
# def sum_of_products(x, y):
#     return cp.sum(x * y, axis = -1)

class FeatureExtractorMEC(FeatureExtractorTemplateMatching):
    """Class of minimum energy combination feature extractor"""
    
    def __init__(self):
        """MEC feature extractor class constructor"""
        super().__init__()
        
        # The order of the AR model used for estimating noise energy.
        # This must be a single positive integer.  The order of the AR cannot
        # be more than the signal length. 
        self.ar_order = 15
        
        # The ratio of noise energy remained in the projected signal.
        # This must be a number between 0 and 1. 
        self.energy_ratio = 0.05
        
        # A temporary pre-computed value. 
        self.xplus = 0
        
        # The psudo-inverse of each [sine, cosine] pair for each harmonic
        self.sub_template_inverse = 0   
        
        if cupy_available_global == True:
            # CUDA kernel for computing sum of products. 
            self.sum_product_raw = cp.RawKernel(
                r'''
                extern "C" __global__
                void sum_product_raw(const float* const timeSeries,
                    const int batchSize, 
                    const int signalSize,
                    const int arraySize,
                    const int offset,
                    float* const results)
                {     
                  const int p = blockIdx.y;
                  const int batchId = blockIdx.x * blockDim.x + threadIdx.x;
                  const int index = batchId*batchSize - blockIdx.x*offset;
                  
                  if (index >= arraySize)
                      return;
                    
                  const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
                  const int threadId = blockId * blockDim.x + threadIdx.x;                   

                  for (int i = 0; i < batchSize; i++) 
                  {
                      if (threadIdx.x*batchSize+i+p >= signalSize)
                          break;          
            
                       results[threadId] += 
                           timeSeries[index+i] * timeSeries[index+i+p];           
                  } // end for i
                  
                } // end kernel sum_product_raw
                ''', 'sum_product_raw')
       
    def setup_feature_extractor(
            self, 
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            ar_order=15,
            energy_ratio=0.05,
            embedding_dimension=0,
            delay_step=0,
            filter_order=0,
            filter_cutoff_low=0,
            filter_cutoff_high=0,
            subbands=None,
            voters_count=1,
            random_seed=0,
            use_gpu=False,
            max_batch_size=16,
            explicit_multithreading=0,
            samples_count=0):
        """
        Setup the feature extractor parameters (MEC).
        
        Mandatory Parameters:
        ---------------------
        harmonics_count: The number of harmonics to be used in constructing
        the template signal.  This variable must be a positive integer 
        number (typically a value from 3 to 5).  
        
        targets_frequencies: The stimulation freqeuency of each target.  
        This must be a 1D array,  where the first element is the stimulation
        frequency of the first target, the second element is the stimulation
        frequency of the second target, and so on.  The length of this array
        indicates the number of targets.  The user must determine the 
        targets_frequencies but the number of targets (targets_count) is 
        extracted automatically from the length of targets_frequencies. 
        
        sampling_frequency: The sampling rate of the signal 
        (in samples per second).  It must be a real positive value. 
        
        
        Optional Parameters:
        --------------------
        ar_order: This defines the order of the auto-regressive model, which
        is used to compute the expected energy of the noise.  This must be a
        positive integer. 
        
        energy_ratio: This define the ratio of the nuisance signal that is 
        kept in the projected channels.  This must be a real number between
        zero and one.  Reducing this ratio, reduces the energy of the nuisance
        signal but also leads to the loss of information. 
        
        embedding_dimension: This is the dimension of time-delay embedding. 
        This must be a non-negative integer.  If set to zero, no time-dely
        embedding will be used.  If there are E electrodes and we set the 
        embedding_dimension to n, the class expands the input signal as if we
        had n*E channels.  The additional channels are generated by shift_left
        operator.  The number of samples that we shift each signal is 
        controlled by delay_step.  Embedding delays truncates the signal. 
        Make sure the signal is long enough. 
        
        delay_step: The number of samples that are shifted for each delay
        embedding dimension.  For example, assume we have ten channels, 
        embedding_dimension is two, and delay_step is three.  In this case, the
        class creates 30 channels.  The first ten channels are the original
        signals coming from the ten electrodes.  The second ten signals are
        obtained by shifting the origianl signals by three samples.  The third
        ten signals are obtained by shifting the original signals by six 
        samples.  The signals are truncated accordingly. 
        
        filter_order: The order of the filter used for filtering signals before
        analysis.  If filter_order is zero (the default value), no filtering
        is performed.  Otherwise, the class creates a filter of order 
        filter_order.  This must be positive integer. 
        
        cutoff_frequency_low: The first cutoff frequency of the bandpass 
        filter.  This must be a single real positive number.  If filter_order
        is zero, this attribute is ignored.  
        
        cutoff_frequency_high: The second cutoff frequency of the bandpass
        filter. This must be a single real positive number.  If filter_order
        is zero, this attribute is ignored.  
        
        subbands: This is the primary way to instruct the classifier whether 
        to use filterbank or not.  The default value is None.  If set to None, 
        the classifier uses none-fitlerbank implementation.  To use
        filterbanks, subbands must be set to a 2D array, whith exactly two 
        columns.  Each row of this matrix defines a subband with two 
        frequencies provided in two columns.  The first column is the first
        cutoff frequency and the second column is the second cutoff frequency
        of that subband.  Filterbank filters the signal using a bandpass
        filter with these cutoff frequencies to obtain a new subband.  The
        number of rows in the matrix defines the number of subbands. All
        frequencies must be in Hz.  For each row, the second column must
        always be greater than the first column. 
        
        voters_count: The number of electrode-selections that are used for
        classification.  This must be a positive integer.  This is the 
        same as the number of voters.  If voters_count is larger that the 
        cardinality of the power set of the current selected electrodes, 
        then at least one combination is bound to happen more than once. 
        However, because the selection is random, even if voters_count is
        less than the cardinality of the power set, repettitions are still
        possible (although unlikely). If not specified or 1, no 
        voting will be used. 
        
        random_seed: This parameter control the seed for random selection 
        of electrodes.  This must be set to a non-negative integer.  The 
        default value is zero.
        
        use_gpu: When set to 'True,' the class uses a gpu to extract features.
        The host must be equipped with a CUDA-capable GPU.  When set to
        'False,' all processing will be on CPU. 
        
        max_batch_size: The maximum number of signals/channel selections
        that are processed in one batch.  Increasing this number improves
        parallelization at the expense of more memory requirement.  
        This must be a single positve integer. 
        
        explicit_multithreading: This parameter determines whether to use 
        explicit multithreading or not.  If set to a non-positive integer, 
        no multithreading will be used.  If set to a positive integer, the 
        class creates multiple threads to process signals/voters in paralle.
        The number of threads is the same as the value of this variable. 
        E.g., if set to 4, the class distributes the workload among four 
        threads.  Typically, this parameter should be the same as the number
        of cores the cput has, if multithreading is to be used. 
        Multithreading cannot be used when use_gpu is set to True.
        If multithreading is set to a positive value while used_gpu is 
        set to True or vice versa, the classes raises an error and the 
        program terminates. 
        
        samples_count: If provided, the class performs precomputations that
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
        self.build_feature_extractor(
            harmonics_count,
            targets_frequencies,
            sampling_frequency,
            subbands=subbands,           
            embedding_dimension=embedding_dimension,
            delay_step=delay_step,
            filter_order=filter_order,
            filter_cutoff_low=filter_cutoff_low,
            filter_cutoff_high=filter_cutoff_high,
            voters_count=voters_count,
            random_seed=random_seed,
            use_gpu=use_gpu,
            max_batch_size=max_batch_size,
            explicit_multithreading=explicit_multithreading,
            samples_count=samples_count)
        
        self.ar_order = ar_order
        self.energy_ratio = energy_ratio
            
    def get_features(self, device):
        """Extract MEC features (SNRs) from signal"""    
        # Extract current batch of data
        (signal, y_bar_squared) = self.get_current_data_batch()
                
        xp = self.get_array_module(signal)
        
        # Swap the dimensions for samples and electrodes to make the 
        # implementation consistent with the reference.
        signal = xp.transpose(signal, axes=(0, 2, 1))         
        
        # Extract SNRs                   
        features = self.compute_snr(signal, y_bar_squared, device)
               
        batch_size = self.channel_selection_info_bundle[1]
        
        # De-bundle the results.
        features = xp.reshape(features, (
            features.shape[0]//batch_size,
            batch_size,
            self.targets_count,
            self.features_count)
            )
        
        return features
    
    def get_features_multithreaded(self, signal):
        """Extract MEC features (SNRs) from signal"""        
        # signal is an E by T 2D array, where T is the number
        # of samples and E is the number of electrodes.  Thus, we must
        # transpose to make it T by E. 
        signal -= np.mean(signal, axis=-1)[:, None]
        signal /= np.std(signal, axis=-1)[:, None]
        signal = np.transpose(signal)
        signal = signal[None, :, :]
        
        # Compute Ybar per Eq. (9)
        y_bar = signal - np.matmul(self.xplus, signal)
        
        y_bar_squared = np.matmul(
            np.transpose(y_bar, axes=(0, 2, 1)), y_bar)
        
        y_bar_squared = y_bar_squared[None, :, :, :]                             
        features = self.compute_snr(signal, y_bar_squared, device=0)    

        # De-bundle the results.
        features = np.reshape(features, (
            1, 
            1,
            1,
            self.targets_count,
            1))        
        return features
       
    def compute_snr(self, signal, y_bar_squared, device):
        """Compute the SNR"""         
        xp = self.get_array_module(signal)
        
        # Project the signal to minimize the power of nuisance signal
        (projected_signal, n_s) = self.project_signal(
            signal, y_bar_squared, device)    
        
        # Computer the signal power
        template = self.template_signal_handle[device][None, :, :, :]
        template = xp.transpose(template, axes=(0, 1, 3, 2))
        power = xp.matmul(template, projected_signal)
        power = xp.square(power)
        
        # The following is a trick to add each row of the matrix to the row 
        # below it and save it in the top row. One row is for the sine and
        # the other is for the cosine.
        power = power + xp.roll(power, -1, axis=2)
        power = power[:, :, 0:-1:2]
        power = xp.transpose(power, axes=(2, 0, 1, 3))    

        x_inverse_signal = xp.matmul(
            self.sub_template_inverse_handle[device][:, None, :, :, :], 
            projected_signal[None, :, :, :, :])
        
        x = xp.reshape(
            self.template_signal_handle[device], 
            self.template_signal_handle[device].shape[0:-1] + (-1, 2))
        
        x = xp.transpose(x, (2, 0, 1, 3))
        s_bar = xp.matmul(x[:, None, : , :, :], x_inverse_signal)
        s_bar = projected_signal[None, :, :, :, :] -  s_bar
        s_bar = xp.transpose(s_bar, axes=(0, 1, 2, 4, 3))        
        
        # Extract the noise energy
        coefficients, noise_energy = self.yule_walker(s_bar, device)
        sigma_bar = self.k2 * noise_energy        
        denominator = xp.zeros(coefficients.shape[0:-1], dtype=np.cdouble)
        coefficients = xp.transpose(coefficients, axes=(3, 0, 1, 2, 4))
        coefficients *= -1
        
        coefficients = xp.multiply(
            coefficients, self.k3_handle[device][None, :, None, :, :])
        
        denominator = xp.sum(coefficients, axis=-1)
        denominator = xp.transpose(denominator, axes=(1, 2, 3, 0))
        denominator = xp.abs(1 + denominator)
        sigma_bar /= denominator
        power = power / sigma_bar
        
        # For each signal, only keep the first n_s number of channels.
        snrs = xp.sum(power, axis=0)
        snrs = xp.cumsum(snrs, axis=-1)    

        snrs_reshaped = xp.reshape(snrs, (-1, snrs.shape[2]))
        ns = 1 + xp.arange(snrs_reshaped.shape[-1])
        ns = xp.multiply(xp.ones(snrs_reshaped.shape), ns[None, :])
        ns = (ns == (n_s.flatten())[:, None])
        snrs_reshaped = snrs_reshaped[ns]
        snrs = xp.reshape(snrs_reshaped, snrs.shape[0:2])
        
        a = cp.asnumpy(snrs)
        if np.isnan(a).any():
            b = 3
        
        return snrs        
                            
    def project_signal(self, signal, y_bar_squared, device):
        """Project the signal such that noise has the minimum energy"""  
        xp = self.get_array_module(signal)
          
        # Compute eigen values and eigen vectors, which give us the
        # solution to the optimization problem of Eq. (10).
        # The matrix is symmetric, thus we use eigh function.
        eigen_values, eigen_vectors = xp.linalg.eigh(y_bar_squared)
        
        # Compute how many channes we need to keep based on the desired 
        # energy of the retained noise. See Eq. (12)
        n_s = self.compute_channels_count(eigen_values, device)
        
        # The following manipulations are simply to normalize eigen vectors
        eigen_values = xp.sqrt(eigen_values)
        eigen_values = xp.expand_dims(eigen_values, axis=3)       
        eigen_vectors = xp.transpose(eigen_vectors, axes=(0, 1, 3, 2))
        eigen_vectors = xp.divide(eigen_vectors, eigen_values)
        eigen_vectors = xp.transpose(eigen_vectors, axes=(0, 1, 3, 2))
        
        # It is possible that each signal in the batch needs a different
        # number of channels.  This prevents us from keeping everything in a 
        # single matrix to batch-process them.  To overcome this, we compute
        # the maximum number of channels in the batch and keep that many
        # channels.  Later on in the code, we will discard the remaining
        # channels that were supposed to be discarded here.  
        max_index = xp.max(n_s)
        eigen_vectors = eigen_vectors[:, :, :, 0:max_index]
        
        # Compute the projected signal per Eq. (7).
        projected_signal = xp.matmul(signal[:, None, :, :], eigen_vectors)            
        return (projected_signal, n_s)
    
    def compute_channels_count(self, eigen_values, device):
        """Compute how many channels we need based on ratio of energy."""        
        # The following is a pythonic implementation of Eq. (12)   
        xp = self.get_array_module(eigen_values)
        running_sum = xp.cumsum(eigen_values, axis=-1)
        total_energy = xp.expand_dims(running_sum[:, :, -1], axis=-1)
        energy_ratio = xp.divide(running_sum, total_energy)
        flags = (energy_ratio <= self.energy_ratio_handle[device])        
        n_s = xp.sum(flags, axis=-1)        
        n_s[n_s == 0] = 1
        return n_s
        
    def yule_walker(self, time_series, device):       
        "Yule-Walker AR model estimation, based on the sm models"            
        xp = self.get_array_module(time_series)
        
        # A short hand for ar_order
        p = self.ar_order

        if self.use_gpu == True: 
            batch_size = 5
            
            # The custom kernel works with 3D arrays only.
            shape = time_series.shape
            time_series = xp.reshape(time_series, (-1, shape[-1]))
            
            # One is added in case the samples count is not divisible by batch_size
            # Adding extra zeros does not affect the summation, so it is safe.
            batch_count = shape[-1]//batch_size + 1
                         
            # Helps us keep track of indexing time_series in the kernel 
            # considering that the sizes of time_series and r do not mach any more
            # (because of the +1 we have in the previous statement)
            offset = batch_count * batch_size - shape[-1]
            
            r = xp.zeros(
                (p+1,) + shape[0:-1] + (batch_count,), dtype=xp.float32) 
            
            r = xp.reshape(r, (p+1, -1, batch_count))    
            
            # Kernel settings
            grid_size = (r.shape[1], r.shape[0])        
            block_size = (batch_count,)
                       
            self.sum_product_raw(grid_size, block_size, (
                time_series, batch_size, shape[-1],
                time_series.size, offset, r))
            
            r = xp.sum(r, axis=-1)
            r = xp.reshape(r, (p+1,) + shape[:-1])   
            
        else:            
            r = xp.zeros((p+1,) + time_series.shape[0:-1], dtype=xp.float32)                
            r[0] = xp.sum(xp.square(time_series), axis=-1)
        
            # By far, the slowest part of the entire algorithm.        
            for k in xp.arange(1, p+1): 
                r[k] = xp.sum(xp.multiply(
                    time_series[:, :, :, :, :-k], 
                    time_series[:, :, :, :,  k:]),
                    axis=-1) 
           
        r = xp.transpose(r, axes=(1, 2, 3, 4, 0))
        r = r / self.samples_count_handle[device]   
        G = xp.zeros(r.shape[:-1] + (p, p), dtype=xp.float32) 
        G[:, :, :, :, 0, :] = r[:, :, :, :, :-1]
        
        for i in np.arange(1, p):
            G[:, :, :, :, i, i:] = r[:, :, :, :, :-i-1]
          
        # Construct a toeplitz matrix.
        # Such matrix can be constructed using the following transformation:
        # (1/a0)A = GGT - (G - I)(G - I)T = G + GT - I
        A = xp.divide(G, (G[:, :, :, :, 0, 0])[:, :, :, :, None, None])
        A = A + xp.transpose(A, axes=(0, 1, 2, 3, 5, 4))
        A = xp.subtract(A, xp.eye(A.shape[-1], dtype=xp.float32))
        R = xp.multiply(A, (G[:, :, :, :, 0, 0])[:, :, :, :, None, None])
        rho = xp.linalg.solve(R, r[:, :, :, :, 1:])
        sigmasq = xp.sum(xp.multiply(r[:, :, :, :, 1:], rho), axis=-1)
        sigmasq =  r[:, :, :, :, 0] - sigmasq     
        
        # Returns the coefficients and noise energy
        return rho, sigmasq
    
    def perform_voting_initialization(self, device):
        """Perform initialization and precomputations common to all voters"""              
        # Normalize all data
        self.all_signals -= np.mean(self.all_signals, axis=-1)[:, :, None]
        self.all_signals /= np.std(self.all_signals, axis=-1)[:, :, None]    
        
        # Generate handles to normalized data
        self.all_signals_handle = self.handle_generator(self.all_signals)
        
        # The rest is only needed for gpu-based and single-threaded only!
        if self.explicit_multithreading > 0:
            return
        
        signal = self.all_signals_handle[device]      
        xp = self.get_array_module(signal)

        # Pre-compute y_bar per Eq. (9).
        signal = xp.transpose(signal, axes=(0, 2, 1))
        
        # An artifical for loop is ensuing!
        # Although it could be avoided, using a for loop eases up memory usage.
        y_bar = xp.zeros(
            (self.signals_count,
              self.targets_count,
              self.samples_count,
              self.electrodes_count), dtype=xp.float32)
        
        for i in xp.arange(self.signals_count):
            y_bar[i] = xp.matmul(
                self.xplus_handle[device],
                signal[i, :, :])    
        
        # Equivalent to the For loop above but needs more memory to compute
        # everything in one shot. 
        # y_bar = xp.matmul(
        #     self.xplus_handle[device][None, :, :, :],
        #     signal[:, None, :, :])           
        
        y_bar = xp.subtract(signal[:, None, :, :], y_bar)
                  
        self.y_bar_squared = xp.matmul(
            xp.transpose(y_bar, axes=(0, 1, 3, 2)), y_bar)  
        
        # Generate handles
        self.y_bar_squared_handles = self.handle_generator(self.y_bar_squared)
        
    def class_specific_initializations(self):
        """Perform necessary initializations and precomputations"""
        # Perform some percomputations only in the first run.  
        # These computations only rely on the template signal and can thus
        # be pre-computed to improve performance. 
        self.compute_templates()  
        
        # Get the inverse of sine and cosine pairs of each harmonic. 
        # Need for computing SNR later on. 
        self.precompute_each_harmonic_inverse()
        
        self.xplus = np.matmul(
            self.template_signal,
            np.linalg.pinv(self.template_signal))
                
        # Compute some constants. We need them later for computing SNR. 
        k1 = ((-2 * 1j * np.pi / self.sampling_frequency)
              * self.targets_frequencies)
        k1 = np.multiply(
            k1[:, None], (np.arange(1, self.ar_order+1)[:, None]).T)
        k2 = np.pi * self.samples_count / 4
        harmonics_scaler = np.arange(1, self.harmonics_count+1)
        k3 = np.multiply(harmonics_scaler[:, None, None], k1)
        k3 = np.exp(k3)        
        self.k3 = k3
        self.k2 = k2
        
        # Create handles
        self.template_signal_handle = self.handle_generator(
            self.template_signal) 
        
        self.sub_template_inverse_handle = self.handle_generator(
            self.sub_template_inverse)
        
        self.xplus_handle = self.handle_generator(self.xplus)
        self.k2_handle = self.handle_generator(self.k2)
        self.k3_handle = self.handle_generator(self.k3)
        self.energy_ratio_handle = self.handle_generator(self.energy_ratio)           
        self.samples_count_handle = self.handle_generator(self.samples_count)
        
        # Force compile the kernel by running some dummy analysis. 
        if self.use_gpu == True:
            dummy = np.random.rand(8, 300)
            dummy = cp.asarray(dummy)
            r = cp.zeros((8, 8))
            self.sum_product_raw(
                (8, 8), (64,), (dummy, 10, 300, dummy.size, 0, r)
                )
                   
    def precompute_each_harmonic_inverse(self):
        """"pre-compute the iverse of each harmonics."""        
        # This saves up like 5% performance.
        # Extract sine and cosine pair of each harmonic and compute its
        # inverse.  This is needed for computing the SNRs.  It needs to be done
        # only once. 
        self.sub_template_inverse = np.zeros(
            (self.harmonics_count,
              self.template_signal.shape[0],
              2,
              self.template_signal.shape[1]))
        
        for h in np.arange(0, self.harmonics_count*2, 2):
            x = self.template_signal[:, :, (h, h+1)]
            self.sub_template_inverse[np.int(h/2)] = np.linalg.pinv(x)
            
    def get_current_data_batch(self):
        """Bundle all data so they can be processed toegher"""
        # Bundling helps increase GPU and CPU utilization. 
       
        # Extract bundle information. 
        # Helps with the code's readability. 
        batch_index = self.channel_selection_info_bundle[0]        
        batch_population = self.channel_selection_info_bundle[1]
        batch_electrodes_count = self.channel_selection_info_bundle[2]
        first_signal = self.channel_selection_info_bundle[3]
        last_signal = self.channel_selection_info_bundle[4]
        signals_count = last_signal - first_signal
        
        # Pre-allocate memory for the batch
        signal = np.zeros(
            (signals_count, batch_population,
             batch_electrodes_count, self.samples_count),
            dtype=np.float32)        
                
        y_bar_squared = np.zeros(
            (signals_count, batch_population, self.targets_count, 
             batch_electrodes_count, batch_electrodes_count),
            dtype=np.float32)
        
        selected_signals = self.all_signals_handle[0][first_signal:last_signal]  
        selected_ybar = self.y_bar_squared_handles[0][first_signal:last_signal]
    
        for j in np.arange(batch_population):
            current_selection = self.channel_selections[batch_index]
            signal[:, j] = selected_signals[:, current_selection, :]     
            ybar2 = selected_ybar[:, :, current_selection, :]
            ybar2 = ybar2[:, :, :, current_selection]     
            y_bar_squared[:, j] = ybar2
            batch_index += 1
            
        signal = np.reshape(signal, (-1,) + signal.shape[2:])
        
        y_bar_squared = np.reshape(
            y_bar_squared, (-1,) + y_bar_squared.shape[2:])
          
        # Move the extracted batches to the device memory if need be. 
        if self.use_gpu == True:      
            signal = cp.asarray(signal)
            y_bar_squared = cp.asarray(y_bar_squared)
            
        return (signal, y_bar_squared)
                      
    @property
    def ar_order(self):
        """Getter function for the order of the autoregressive model"""
        return self.__ar_order
    
    @ar_order.setter
    def ar_order(self, order):
        """Setter function for the order of the autoregressive model"""
        error_message = "Oorder of the AR model must be a positive integer."
        
        try:
            order = int(order)
        except (ValueError, TypeError):
            self.quit(error_message)
            
        if order <= 0:
            self.quit(error_message)     
            
        self.__ar_order = order
        
    @property
    def energy_ratio(self):
        """Getter function for energy ratio"""
        return self.__energy_ratio
    
    @energy_ratio.setter
    def energy_ratio(self, energy_ratio):
        """Setter function for energy ratio"""
        error_message = "Energy ratio must be a real number between 0 and 1"
        
        try:
            energy_ratio = float(energy_ratio)
        except (ValueError, TypeError):
            self.quit(error_message)
            
        if not 0 < energy_ratio < 1:
            self.quit(error_message)    
            
        self.__energy_ratio = energy_ratio

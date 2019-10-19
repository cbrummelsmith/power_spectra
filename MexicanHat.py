import numpy as np
from scipy.ndimage.filters import gaussian_filter
import sys

class MexicanHat:
    def __init__(self, Nk=20, kspacing='log', mode='nearest', eps=1e-3):
        """
        Mexican Hat Filter used to filter out features in data at a certain length scale.
        This class sets up the basic parameters needed to run the convolutions and filter
        the data. Other parameters such as wavenumbers are passed to the necessary functions.
        Parameters
        ----------
        Nk : int
            The number of k values i.e. wavenumbers. This is the number of filtered
            ndarrays that are returned.
            Default: 20
        mode : ('nearest' | 'constant' | 'mirror' | 'reflect' | 'wrap')
            How to handle the edges when convolving. 
            See scipy.ndimage.filters.gaussian_filter for more info.
            Default: 'nearest'
        kspacing : ('log' | 'linear')
            The type of spacing between each wavenumber
            Default: 'log'       
        eps : float
            A small value to use with mexican hat filter. Do not change without good reason.
            Default: 0.001             
        """
        self.eps = eps
        self.Nk = Nk
        self.mode = mode
        self.kspacing = kspacing

    def mexican_hat_convolve(self, data, sig, mask=None):
        """
        A filter which filters out features in data that are larger or smaller 
        than a certain length scale l = ~0.255 * sig

        Parameters
        ----------
        data : np.ndarray
            input data to be filtered
        sig : float
            standard deviation of the gaussian kernal in units of pixels 
        mask : np.ndarray (optional)
            A mask the same shape as data used to remove regions of data            
        
        Returns
        -------        
        The filtered image : np.ndarray (same shape as data)
        """
        sig1 = sig / np.sqrt(1 + self.eps)
        sig2 = sig * np.sqrt(1 + self.eps)

        if mask is None:
            I1 = gaussian_filter(data, sig1, mode=self.mode)
            I2 = gaussian_filter(data, sig2, mode=self.mode)
            return I1 - I2

        M = mask.astype(float)  
        M1 = gaussian_filter(M, sig1, mode=self.mode)
        M2 = gaussian_filter(M, sig2, mode=self.mode)  
        data = M*data
           
        I1 = gaussian_filter(data, sig1, mode=self.mode)
        I2 = gaussian_filter(data, sig2, mode=self.mode)


        # Valid cells are where the mask is non-zero
        M1_valid = M1 != 0
        M2_valid = M2 != 0
        M1v = M1[M1_valid]
        M2v = M2[M2_valid]
        I1v = I1[M1_valid]
        I2v = I2[M2_valid]

        # I_o_M = I/M   
        I1_o_M1 = np.zeros_like(data)
        I2_o_M2 = np.zeros_like(data)

        # Use valid cells to avoid dividing by zero with the mask
        I1_o_M1[M1_valid] = I1v/M1v
        I2_o_M2[M2_valid] = I2v/M2v

        # mexican hat filter
        return (I1_o_M1 - I2_o_M2) * M

    # -- End mexican_hat_convolve --    

    def filter_data_k(self, data, mask=None):
        """
        Use mexican hat convolution to filter fluctuations at different length
        scales ranging from the full size of the array to the the cell size.
    
        Parameters
        ----------
        data : np.ndarray
            input data to be filtered
        mask : np.ndarray (optional)
            A mask the same shape as data used to remove regions of data
        
        Returns
        -------
        tuple (a, b)
            a) filtered_data : An array of arrays where each element is the
               filtered data at length scale 1/k. filtered_data.shape = (Nk, data.shape)
            b) k: wavenumbers
        """
        kmin = 1./max(data.shape)
        kmax = 0.5 # Nyquist frequency
        
        # wavenumbers
        if self.kspacing == 'log':
            k = np.logspace(np.log10(kmin), np.log10(kmax), self.Nk)
        elif self.kspacing == 'linear':
            k = np.linspace(kmin, kmax, self.Nk)
        else:
            print('Error: %s spacing not implemented. Exiting now.')
            sys.exit()
    
        sigma = (2 * np.pi**2)**(-0.5)/k
    
        filtered_data = []
    
        print("Using Mexican Hat Filter with %d wavenumbers in range k = [1/%d, 0.5] cells^-1" % (self.Nk,self.Nk))
        for i, sig in enumerate(sigma):
            sys.stdout.write("\r%d/%d"%(i+1,self.Nk))
            sys.stdout.flush()
            Sk = self.mexican_hat_convolve(data, sig, mask)
            filtered_data.append(Sk)
        print("")
        return np.array(filtered_data), np.array(k)                                                                        

    # -- End filter_data_k --

    def power_spectrum(self, data_a, data_b=None, mask=None, phys_width=None, return_filtered_data=False):
        """
        Use Mexican Hat to compute power spectrum of data_a, 
        or cross spectra between data_a and data_b.
    
        Parameters
        ----------
        data_a : np.ndarray
            input data
        data_b : np.ndarray (optional)
            second input data. Only used when computing cross spectrum.
            Must be same shape as data_a.
        mask : np.ndarray (optional)
            A mask the same shape as data used to remove regions of data
        phys_width : float (optional)
            physical width of data (e.g. 5 arcminutes, 1 Mpc) Assumes square data in 2d and cubic in 3d.
            Used for unit conversion.
        
        Returns
        -------
        tuple (a, b)
            a) filtered_data : An array of arrays where each element is the
               filtered data at length scale 1/k. filtered_data.shape = (Nk, data.shape)
            b) k: wavenumbers
        """        
        if data_b is not None:
            if data_a.shape != data_b.shape:
                print('data_a.shape = (%d,%d,%d) must equal\n\
                       data_b.shape = (%d,%d,%d)')
                sys.exit()
        
        nside = max(data_a.shape)
        n = len(data_a.shape)
        upsilon = n * (n/2. + 1) * 2**(-n/2. - 1) * np.pi**(n/2.)  
        size = float(data_a.size)

        # Filter data at all length scales. 
        # Handle both regualar and cross spectrum cases
        S_all_k_a, k = self.filter_data_k(data_a, mask) 
        if data_b is not None:
            # filter data_b for cross spectrum
            print("cross")
            S_all_k_b, k = self.filter_data_k(data_b, mask)
        else:
            S_all_k_b = S_all_k_a

        # scale dependent normalization, accounting for masked cells
        Nrat = size/mask.sum() if mask is not None else 1.
        norm = Nrat / (self.eps*self.eps * upsilon * k**n)  

        # power given by spatially averaged, filtered, data.
        sumaxes = tuple(range(1,n+1))
        power = (S_all_k_a * S_all_k_b).sum(axis=sumaxes) / size

        # normalize
        power *= norm

        # convert wavenumber from units of 1/pixels to physical units
        if phys_width is not None:
            len_per_pix = phys_width / nside
            k /= len_per_pix

        print('Finished power spectrum.')
        if return_filtered_data:
            return power, k, S_all_k_a, S_all_k_b
        else:
            return power, k

        # -- End power_spectrum --

    def power_spectrum_amplitude(self, power, k, n):
        if n == 3:
            return np.sqrt(4*np.pi * (k/k.max()/2.)**3 * power)
        if n == 2:            
            return np.sqrt(2*np.pi * (k/k.max()/2.)**2 * power)
        if n == 1:
            return np.sqrt((k/k.max()/2.) * power)            

    def coherence(self, Pa, Pb, Pab):
        return Pab / np.sqrt(Pa * Pb)

    def ratio(self, Pa, Pab):
        return Pab / Pa

# === End Mexican Hat Class ===
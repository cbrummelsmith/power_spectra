import numpy as np

def fft_pos_freq(data):
    """
    FFTs on real data.
    Note that since our data is real, there will be
    too much information here.  fftn puts the positive freq terms in
    the first half of the axes -- that's what we keep.  Our
    normalization has an 2^n to account for this clipping to one
    quadrent (n=2) or octant (n=3). 
    """   
    nside = data.shape
    ndims = len(data.shape)
    ncells = np.prod(nside)
    if ndims == 3:
        ru = np.fft.fftn(data)[0:nside[0]//2+1, 0:nside[1]//2+1, 0:nside[2]//2+1]
    if ndims == 2:
        ru = np.fft.fftn(data)[0:nside[0]//2+1, 0:nside[1]//2+1]    
    if ndims == 1:
        ru = np.fft.fftn(data)[0:nside[0]//2+1]                
    
    return ru * 2.**ndims / ncells

 
def power_spectrum_FFT(data_a, data_b=None, unit_length=None):

    nside = np.array(data_a.shape)
    ndims = len(nside)

    # do the FFTs -- note that since our data is real, there will be
    # too much information here.  fftn puts the positive freq terms in
    # the first half of the axes -- that's what we keep.  Our
    # normalization has an 2^n to account for this clipping to one
    # quadrent (n=2), octant (n=3).
    print('FFT')
    Fk_a = fft_pos_freq(data_a)  # Fk_a = Fourier transformed field
    if data_b is not None:
        # FFT of data_b for cross spectra
        Fk_b = fft_pos_freq(data_b)
    else:
        # for normal power spectrum
        Fk_b = Fk_a 


    # Physical length of box sides (i.e. in cm, Mpc, etc.)
    L = np.array(nside, dtype=np.float64)
    if unit_length is not None:
        L = L * unit_length

    # Compute the wavenumber grid and the wavevector 'lengths' kr
    all_k = [np.fft.rfftfreq(nside[i]) * nside[i] / L[i] for i in range(ndims)]  # in 3d, all_k = [kx,ky,kz]
    kgrid = np.meshgrid(*all_k, indexing='ij')
    kr = np.sqrt(np.sum(np.power(kgrid, 2), axis=0))

    # Physical limits to the wavenumbers
    kmin = np.min(1.0 / L)
    kmax = np.min(0.5*nside / L)

    # 'radial' k bins
    kbins = np.arange(kmin, kmax, kmin) # linear bins 
    whichbin = np.digitize(kr.flat, kbins)
    ncount = np.bincount(whichbin)    
    N = len(kbins)

    # Energy spectrum density (or cross spectra for different signals, i.e. Fk_a != Fk_b).
    # Here, the term energy is used in the generalized sense of signal processing.
    E_density = np.abs(Fk_a * np.conj(Fk_b))
    Espectrum = np.zeros(len(ncount)-1)

    # Integrate over k shell, to get Energy spectrum
    for n in range(1,len(ncount)):
        Espectrum[n-1] = np.sum(E_density.flat[whichbin==n])

    k = 0.5*(kbins[0:N-1] + kbins[1:N])[:-1]
    Espectrum = Espectrum[1:N-1]

    # Convert energy spectrum to power spectrum
    Pspectrum = Espectrum / (k/k.max()/2.)**(ndims-1)

    return Pspectrum, k   


def power_spectrum_amplitude(power, k, n):
    if n == 3:
        return np.sqrt(4*np.pi * (0.5 * k/k.max())**3 * power)
    if n == 2:
        return np.sqrt(np.pi * (0.5 * k/k.max())**2 * power)
    if n == 1:
        return np.sqrt((0.5 * k/k.max()) * power)        

def coherence(Pa, Pb, Pab):
    return Pab / np.sqrt(Pa * Pb)

def ratio(Pa, Pab):
    return Pab / Pa



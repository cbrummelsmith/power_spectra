import numpy as np
from scipy import ndimage
from scipy.interpolate import CubicSpline


def radial_profile(data, mask=None, nbins=None):
    """
    data: ndarray, to compute radial mean
    mask: ndarray (same shape as data), 1 where data should be excluded, 0 otherwise
    nbins: int, number of radial bins
    """
    cent = np.array(data.shape)/2.
    coordinates = np.indices(data.shape) + 0.5
    position = np.array([(coordinates[i] - cent[i]) for i in range(len(data.shape))])
    r = np.sqrt(np.sum(np.power(position, 2), axis=0))
    if nbins == None:
        nbins = r.max()
    rbin = (nbins * r/r.max()).astype(np.int)

    # If data has a mask, set radius bin value to -1 so it is
    # not included in the bin average for the radial profile
    if mask is not None:
        mask = mask.astype(bool)
        rbin[mask] = -1
        
    binIndex = np.arange(0, rbin.max() +1)    
    radial_mean = ndimage.mean(data, labels=rbin, index=binIndex)
    r = 0.5*(binIndex[1:] + binIndex[:-1])
    dr = r[1]-r[0]
    r = np.append(r, [r[-1] + dr]) # add one last bin center to radius list
    return r, radial_mean   

def radial_profile_field(data, mask=None):
    cent = np.array(data.shape)/2. # assume all side lengths same
    pr, pv = radial_profile(data, mask)
    pfield = np.zeros_like(data)    
    coordinates = np.indices(data.shape) + 0.5
    position = np.array([(coordinates[i] - cent[i]) for i in range(len(data.shape))])
    radius = np.sqrt(np.sum(np.power(position, 2), axis=0))
    pfield[radius < pr[0]] = pv[0]
    for i in range(len(pr)-1):
        rbin = np.logical_and(radius >= pr[i], radius < pr[i+1])
        pfield[rbin] = np.interp(radius[rbin], [pr[i], pr[i+1]], [pv[i],pv[i+1]])
    pfield[radius >= pr[-1]] = pv[-1]
    return pfield, pr, pv           
          
def radial_profile_field_scipy(data, mask=None):
    cent = np.array(data.shape)/2. # assume all side lengths same
    pr, pv = radial_profile(data, mask)    
    coordinates = np.indices(data.shape) + 0.5
    position = np.array([(coordinates[i] - cent[i]) for i in range(len(data.shape))])
    radius = np.sqrt(np.sum(np.power(position, 2), axis=0))
    pfield = np.zeros_like(data)    
    interp_profile = CubicSpline(pr, pv)
    pfield = interp_profile(radius)    
    return pfield, pr, pv          


def residual_field(data, mask=None, method='ratio'):
    """
    Compute the residual of a field, which is how much the field fluctuates about the radially
    averaged value. There are three different definitions of the residual field, 
    specified my the method parameter.
    data: input data - ndarray
    
    Parameters
    ----------
    data : ndarray
        The input data used to compute the residual of
    method : string ('ratio' | 'difference' | 'fractional')
        How the residual is defined. If the data is x, x_avg is the mean, and R is the residual,
        ratio:      R = x / x_avg
        difference: R = x - x_avg
        fractional: R = (x - x_avg) / x_avg
        (Default: 'ratio')
    """
    #mean, trash1, trash2 = radial_profile_field(data)
    mean, trash1, trash2 = radial_profile_field_scipy(data, mask)

    if method == 'ratio':
        return data / mean
    elif method == 'difference':
        return data - mean
    elif method == 'fractional':    
        return (data - mean)

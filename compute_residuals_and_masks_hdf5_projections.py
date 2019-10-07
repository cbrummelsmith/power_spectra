import yt
import h5py
import numpy as np
from python_modules import RadialProfile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.io import fits

def scale_plot_size(factor=1.5):
    import matplotlib as mpl
    default_dpi = mpl.rcParamsDefault['figure.dpi']
    mpl.rcParams['figure.dpi'] = default_dpi*factor
scale_plot_size(1.5)

#snapshots = [114]
snapshots = [46,114]
kpc_per_arcmin=21.21
width_arcmin=7. 
mock_pixels_per_arcmin = 1 * 60.
nside = int(width_arcmin*mock_pixels_per_arcmin)

datadir = '/Users/coreybrummel-smith/GT/Kavli_Summer_Program/results_data'
dsfn = '/Users/coreybrummel-smith/simulations/AGN_FB/DD%04d/stest_%04d'
mockfn = '/Users/coreybrummel-smith/GT/Kavli_Summer_Program/AGN_FB_mock_xray_img_fits_SOXS/\
subsample_1e5K_cut/nside1200_1.0_arcsec_res/10Ms/mock_xray_stest_%04d_acisi_cy19_square__diagonal_0010Ms_%s_img.fits'

fields = ['number_density', 'pressure', 'temperature']
h5file = h5py.File('%s/residuals_2d.h5' % datadir, 'w')
h5file.attrs['width_arcmin'] = width_arcmin
h5file.attrs['kpc_per_arcmin'] = kpc_per_arcmin


for snap in snapshots:
    print snap
    group = h5file.create_group('s%03d' % snap)   
    mask_group = group.create_group('masks')

    ds = yt.load(dsfn % (snap,snap))
    center = ds.arr([0.5,0.5,0.5], 'code_length')
    radius = ds.quan(width_arcmin/2. * kpc_per_arcmin, 'kpc')
    width_kpc=radius*2
    le = center - radius
    re = center + radius
    box = ds.box(le,re)

    # temperature cut mask (same used for mock images 1e5 K)
    min_temperature = 1e5
    box = box.cut_region('obj["temperature"] > %f' % min_temperature)

    
    for field in fields:
        print field
        # density weighted projection same normal vector as mock images
        proj = yt.off_axis_projection(box, center=center, normal_vector=[1,1,1], width=width_kpc,
                                      resolution=nside, item=field, weight='number_density')
        proj = proj.v.T # transpose to get same orientation as mock image        
        residual = RadialProfile.residual_field(proj)
        group.create_dataset('residual_%s' % field, data=residual)

        # Make mask for large density fluctuations associated with small cold clumps
        if field == 'number_density':
            max_fluc = 5.
            mask = residual < max_fluc
            mask_group.attrs['large_density_fluctuation_volume_fraction'] = mask[mask==0].size/float(mask.size)
            mask_group.attrs['max_dens_fluc'] = max_fluc
            mask_group.create_dataset('no_large_density_fluctuation', data=mask)        

    if snap != 46:
        for band in ['hard', 'soft']:
            print band
            # load mock image
            mock = fits.open(mockfn % (snap,band))[0].data
            # trim mock image to get cool core region
            c = mock.shape[0]/2 # image center
            mock = mock[c-nside/2:c+nside/2, c-nside/2:c+nside/2]         
            residual = RadialProfile.residual_field(mock)
            group.create_dataset('residual_mock_%s' % band, data=residual)


h5file.close()
print('Finished.')
import yt
import h5py
import numpy as np
from python_modules import RadialProfile


gridLevel=7
#snapshots = [114]
snapshots = [46,114]
kpc_per_arcmin=21.21
width_arcmin=7. 

datadir = '/Users/coreybrummel-smith/GT/Kavli_Summer_Program/results_data'
dsfn = '/Users/coreybrummel-smith/simulations/AGN_FB/DD%04d/stest_%04d'

h5file = h5py.File('%s/residuals_3d_gridLevel_%02d.h5' % (datadir, gridLevel), 'w')
h5file.attrs['width_arcmin'] = width_arcmin
h5file.attrs['gridLevel'] = gridLevel
h5file.attrs['kpc_per_arcmin'] = kpc_per_arcmin

fields = ['number_density', 'pressure', 'temperature']

for snap in snapshots:
    print snap
    group = h5file.create_group('s%03d' % snap)   
    mask_group = group.create_group('masks')


    ds = yt.load(dsfn % (snap,snap))
    center = ds.arr([0.5,0.5,0.5], 'code_length')
    radius = ds.quan(width_arcmin/2. * kpc_per_arcmin, 'kpc')
    le = center - radius
    re = center + radius
    # 
    # make uniform grid from AMR data with left edge at le = (x, y, z)
    # nside is the resolution of the cube (nx, ny, nz)
    #
    nside = np.round((re-le) * ds.domain_dimensions * 2**gridLevel).astype(int)
    for i in range(len(nside)):
        if nside[i]%2 == 0:
            nside[i] += 1
            
    group.attrs['nside'] = nside

    covGrid = ds.covering_grid(level=gridLevel, left_edge=le, dims=nside)
    covGrid.set_field_parameter('center', center)

    # temperature cut mask (same used for mock images 1e5 K)
    min_temperature = 1e5
    temp = covGrid['temperature']
    temp_mask = covGrid['temperature'] > min_temperature
    mask_group.attrs['cold_gas_volume_fraction'] = temp_mask[temp_mask==0].size/float(temp_mask.size)
    mask_group.attrs['min_temperature'] = min_temperature
    mask_group.create_dataset('no_cold_gas', data=temp_mask)

    for field in fields:
        print field
        residual = RadialProfile.residual_field(covGrid[field] * temp_mask) # mask out cold gas before computing residual
        group.create_dataset('residual_%s' % field, data=residual)

        # Make mask for large density fluctuations associated with small cold clumps
        if field == 'number_density':
            max_fluc = 5.
            mask = residual < max_fluc
            mask_group.attrs['large_density_fluctuation_volume_fraction'] = mask[mask==0].size/float(mask.size)
            mask_group.attrs['max_dens_fluc'] = max_fluc
            mask_group.create_dataset('no_large_density_fluctuation', data=mask)



h5file.close()
print('Finished.')
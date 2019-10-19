import yt
import h5py
import numpy as np
import sys
sys.path.append('/Users/Takeo/Kavli_Summer_Program/code/python_modules')
import RadialProfile


def pad_zeros_on_edges(data, npad):
    data_padded = np.zeros(np.array(data.shape) + 2*npad)
    data_padded[npad : npad+data.shape[0],
                    npad : npad+data.shape[1],
                    npad : npad+data.shape[2]] = data

    return data_padded

def padded_data_mask(data, npad):
    pad_mask = np.zeros(np.array(data.shape) + 2*npad)
    pad_mask[npad : npad+data.shape[0],
             npad : npad+data.shape[1],
             npad : npad+data.shape[2]] = np.ones_like(data) 

    return pad_mask

padded = False
npad = 3

gridLevel=9
#snapshots = [114]
snapshots = [150, 137, 114, 103, 82, 59, 147, 46]
kpc_per_arcmin=21.21
width_arcmin=7. 

#datadir = '/Users/Takeo/Kavli_Summer_Program/results_data/pad_testing'
datadir = '/Users/Takeo/Kavli_Summer_Program/results_data/all_samples'
dsfn = '/Volumes/Mac-500GB/simulations/AGN_FB/DD%04d/stest_%04d'

if padded:
    h5file = h5py.File('%s/padded_residuals_3d_gridLevel_%02d.h5' % (datadir, gridLevel), 'w')
else:
    h5file = h5py.File('%s/not_padded_residuals_3d_gridLevel_%02d.h5' % (datadir, gridLevel), 'w')
h5file.attrs['gridLevel'] = gridLevel
h5file.attrs['kpc_per_arcmin'] = kpc_per_arcmin

fields = ['number_density', 'pressure', 'temperature']

for snap in snapshots:
    print(snap)
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

    arcmin_per_cell = width_arcmin/max(nside)

    covGrid = ds.covering_grid(level=gridLevel, left_edge=le, dims=nside)
    covGrid.set_field_parameter('center', center)

    # temperature cut mask (same used for mock images 1e5 K)
    min_temperature = 1e5
    temp = covGrid['temperature']
    temp_mask = covGrid['temperature'] > min_temperature
    mask_group.attrs['cold_gas_volume_fraction'] = temp_mask[temp_mask==0].size/float(temp_mask.size)
    mask_group.attrs['min_temperature'] = min_temperature

    # spherical mask
    cent = nside/2.
    coordinates = np.indices(nside) + 0.5
    position = np.array([(coordinates[i] - cent[i]) for i in range(len(nside))])
    r_grid = np.sqrt(np.sum(np.power(position, 2), axis=0))
    r_grid *= arcmin_per_cell
    sphere_mask = r_grid < width_arcmin/2.
    mask_group.create_dataset('sphere', data=sphere_mask)

    for field in fields:
        print(field)
        residual = RadialProfile.residual_field(covGrid[field] * temp_mask) # mask out cold gas before computing residual

        if padded:
            # Pad zeros on edges
            residual_padded = pad_zeros_on_edges(residual, npad)
            residual = residual_padded

        group.create_dataset('residual_%s' % field, data=residual)
    

        # Make mask for large density fluctuations associated with small cold clumps
        if field == 'number_density':
            max_fluc = 5.
            mask = residual < max_fluc
            mask_group.attrs['large_density_fluctuation_volume_fraction'] = mask[mask==0].size/float(mask.size)
            mask_group.attrs['max_dens_fluc'] = max_fluc
            mask_group.create_dataset('no_large_density_fluctuation', data=mask)

    if padded:
        group.attrs['padded'] = True
        mask_group.attrs['npad'] = npad
        pad_mask = padded_data_mask(temp_mask, npad)
        temp_mask = pad_zeros_on_edges(temp_mask, npad)
        mask_group.create_dataset('pad', data=pad_mask)
        nside += 2*npad
        arcmin_per_cell = width_arcmin/max(nside)
        width_arcmin += arcmin_per_cell * 2*npad
    else:
        group.attrs['padded'] = False


    h5file.attrs['width_arcmin'] = width_arcmin
    group.attrs['nside'] = nside
    mask_group.create_dataset('no_cold_gas', data=temp_mask)        

h5file.close()
print('Finished.')
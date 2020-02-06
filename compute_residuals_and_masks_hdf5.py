machine = 'macbook-pro'
#machine = 'Takeo'

import sys
if machine == 'macbook-pro':
    kav = '/Users/coreybrummel-smith/GT/Kavli_Summer_Program'
    dsfn = '/Users/coreybrummel-smith/simulations/AGN_FB/DD%04d/stest_%04d'
elif machine == 'Takeo':
    kav = '/Users/Takeo/Kavli_Summer_Program'
    dsfn = '/Volumes/Mac-500GB/simulations/AGN_FB/DD%04d/stest_%04d'
else:
    print('machine must be "macbook-pro" or "Takeo"')
sys.path.append(kav+'/code/python_modules')

import yt
import h5py
import numpy as np
import sys
import RadialProfile

def coarse_grain_3d_array(data, factor, mask=None):
    if mask is not None:
        data = np.ma.array(data)
        # masked array masks remove data where mask = 1. 
        #  I defined mask to remove data where mask = 0.
        data.mask = np.logical_not(mask)
    shape = data.shape[0] # assume all dimensions same length
    new_shape = res//factor
    if res % factor:
        print 'Cube side length (%d) must be divisible by factor (%d)' % (res, factor) 
        return None
    data = data.reshape(new_shape, factor,  
                        new_shape, factor, 
                        new_shape, factor)
    return data.mean(1).mean(2).mean(3)  


def irina_region_to_unigrid_mask(grid_shape, snapshot, exclude_bright):

    from astropy import wcs
    from astropy.io import fits
    import regions as Regions

    # Irina's fits image and ds9 region paths 
    regions_path='/Users/coreybrummel-smith/GT/Kavli_Summer_Program/Irina_feedback_simulations_stuff'
    img_temp = "/images/icc30.s%d_hcc.imaged1.fits"
    if snapshot == 114:
        reg_temp = "/region_files/s%d_hcc_nocl_nofeat_v1.reg"
    else:
        reg_temp = "/region_files/s%d_hcc_nofeat_v1.reg"
    img_name = regions_path + img_temp % snapshot
    reg_name = regions_path + reg_temp % snapshot

    regions = Regions.read_ds9(reg_name)
    hdulist = fits.open(img_name)
    img = hdulist[0].data
    w = wcs.WCS(hdulist[0].header)

    #plt.imshow(img, origin='lower')

    # physical units and positions of image and region
    kpc_per_arcmin = 21.21 
    kpc_per_arcsec = kpc_per_arcmin / 60
    pixels_per_side = img.shape[0]
    img_c = np.array([img.shape[0]/2., img.shape[1]/2.])
    full_reg = regions[0]
    width_arcsec = full_reg.radius.value * 2
    kpc_per_pix = (width_arcsec/60 * kpc_per_arcmin)/pixels_per_side

    # Coordiante grid 
    coord = np.indices(grid_shape)
    i = coord[0]
    j = coord[1]
    k = coord[2]
    indicies = dict(i = (i, "dimensionless"),
                    j = (j, "dimensionless"),
                    k = (k, "dimensionless"))
    bbox = np.array([[-width_kpc/2, width_kpc/2], 
                     [-width_kpc/2, width_kpc/2], 
                     [-width_kpc/2, width_kpc/2]]) # kpc
    ds = yt.load_uniform_grid(indicies, grid_shape, length_unit="kpc", bbox=bbox, periodicity=[False, False, False])    

    # x and y images axes for diagonal projection along normal = [1,1,1]
    # imaage oriented so img_y = x, so img_x = img_y cross normal
    normal = np.array([1.,1.,1.])
    img_y = np.array([1.,0.,0.])
    img_x = np.cross(img_y, normal)
    img_x = img_x / np.sqrt((img_x**2).sum()) # normalize  

    if not exclude_bright:  
        regions = [regions[0]]

    for i, reg in enumerate(regions):
        ### offset = center of region relative to center of image in pixels ###
        reg_c = reg.center.to_pixel(w, origin=1)
        offset_pix = reg_c - img_c
        offset_kpc = offset_pix * kpc_per_pix
        kpc_rad = ds.quan(reg.radius.value * kpc_per_arcsec, 'kpc')

        print('img_c', img_c)
        print('reg_c', reg_c)
        print('offset_pix', offset_pix)
        print('offset_kpc', offset_kpc)
        print('kpc_rad', kpc_rad)

        img_xoff = ds.quan(offset_kpc[0], 'kpc')
        img_yoff = ds.quan(offset_kpc[1], 'kpc')

        reg_off = img_x*img_xoff + img_y*img_yoff
        reg_center = center + reg_off
        height = ds.quan(width_kpc, 'kpc')

        # cylinders corresponding to Irina's ds9 deprojected circles
        yt_irina_reg = ds.disk(reg_center, normal=normal, radius=kpc_rad, height=height)

        if i == 0:
            yt_region = yt_irina_reg # half cool-core region
        else:
            yt_region = yt_region - yt_irina_reg # subtract cylinders spanning bright regions

        # End for reg in regions

    # Valid cells within half cool-core not masked out by cylinders
    ci = yt_region['i'].v.astype(int)
    cj = yt_region['j'].v.astype(int)
    ck = yt_region['k'].v.astype(int)
    index = ci + grid_shape[0]*(cj + grid_shape[1]*ck)
    mask = np.zeros(coord.size)
    for n in index:
        mask[n-1] = 1
    mask = mask.reshape(grid_shape)

    # End irina_region_to_unigrid_mask

#-------------------------------------------------------------------------------------    

exclude_bright = True
if exclude_bright:
    exclude_bright_str = '_exclude_bright'
else:
    exclude_bright_str = ''

maxLevel = 10
gridLevel = 8
factor = 2**(maxLevel-gridLevel)
#snapshots = [114]
snapshots = [150, 137, 114, 103, 82, 59, 147, 46]
kpc_per_arcmin=21.21
width_arcmin=7. 
width_kpc = width_arcmin * kpc_per_arcmin

#datadir = kav + '/results_data/testing_truncate'
datadir = kav + '/results_data/all_samples'

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
    # make max resolution uniform grid from AMR data with left edge at le = (x, y, z)
    # nside is the resolution of the cube (nx, ny, nz)
    # ensure even side length for so we can coarse grain by factor of 2^n
    #
    nside = np.round((re-le) * ds.domain_dimensions * 2**maxLevel).astype(int)
    for i in range(len(nside)):
        if nside[i]%2 != 0:
            nside[i] += 1
    covGrid = ds.covering_grid(level=maxLevel, left_edge=le, dims=nside)
    covGrid.set_field_parameter('center', center)
    
    # make second lower resolution covering grid only for masking out Irina's bright regions
    nside = np.round((re-le) * ds.domain_dimensions * 2**gridLevel).astype(int)
    nside = nside//factor

    # temperature cut mask (same used for mock images 1e5 K)
    min_temperature = 1e5
    temp = covGrid['temperature']
    temp_mask = covGrid['temperature'] > min_temperature
    mask_group.attrs['cold_gas_volume_fraction'] = temp_mask[temp_mask==0].size/float(temp_mask.size)
    mask_group.attrs['min_temperature'] = min_temperature

    # half-cool-core excluding bright region cylinders mask 
    region_mask = irina_region_to_unigrid_mask(nside, snap, exclude_bright=True)

    for field in fields:
        print(field)
        coarse_field = coarse_grain_3d_array(covGrid[field], 
                                            factor=factor,
                                            mask=temp_mask)
        # set region mask for the coarse grained field. 
        # numpy.ma uses convention where unwanted/invalid data = 1 in mask
        # this is the opposite of how I defined the region mask so we need to negate it.
        mask_group.create_dataset('half-cool-core%s'%exclude_bright_str, data=region_mask)
        coarse_field.mask = np.logical_not(region_mask) 


        # ******** NEED TO TEST THAT NUMPY.MASKED ARRAYS WORK WITH RADIAL PROFILE FUNCTION *******
        residual = RadialProfile.residual_field(coarse_field)
        # ****************************************************************************************

        group.create_dataset('residual_%s' % field, data=residual)

        ### Make mask for large density fluctuations associated with small cold clumps
        ##if field == 'number_density':
        ##    max_fluc = 5.
        ##    overdens_mask = residual < max_fluc
        ##    mask_group.attrs['large_overdensity_volume_fraction']  \
        ##            = overdens_mask[overdens_mask==0].size/float(overdens_mask.size)
        ##    mask_group.attrs['max_dens_fluc'] = max_fluc
        ##    mask_group.create_dataset('no_large_overdensity', data=overdens_mask)
        ##
        ### Recompute radial profile after removing gas in large overdensity regions
        ##residual = RadialProfile.residual_field(covGrid[field] * temp_mask * overdens_mask) 
        ##group.create_dataset('residual_%s_large_overdensity_masked' % field, data=residual)

    h5file.attrs['width_arcmin'] = width_arcmin
    group.attrs['nside'] = nside
    mask_group.create_dataset('no_cold_gas', data=temp_mask)        

h5file.close()
print('Finished.')
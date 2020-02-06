def load_corey_power_spectrum_data(level, mask_names, projection=False):
    if not projection:
        gridLevel=level
        ndims = 3
        dim_str = '3d_gridLevel_%02d' % gridLevel
    else:
        ndims = 2
        dim_str = '2d'   
        
    mask_str = ''
    if mask_names is not None:
        for name in mask_names:
            mask_str += '_' + name
            
    datadir = '/Users/coreybrummel-smith/GT/Kavli_Summer_Program/results_data/all_samples'
    spectra_fn = '%s/spectra_%s%s.h5' % (datadir, dim_str, mask_str)
    data = D2H5.load_dict_from_hdf5(spectra_fn)
    return data

#-------------------------------------------------------------------------------------------

projection = False
padded = False
no_large_overdensity = True

gridLevel = 9

snapshots = [150, 137, 114, 103, 82, 59, 147]
snapshots = [150]

field2letter = {'number_density':'n', 'pressure':'P', 'temperature':'T', 'mock_hard':'H', 'mock_soft':'S'}
colors = {'nn':'b', 'TT':'r', 'PP':'g', 'nT':'m', 'nP':'c', 'TP':'gold', 'HH':'tab:red', 'SS':'tab:blue', 'HS':'k'}
field_pairs = [('number_density', 'number_density'), 
               ('pressure', 'pressure'), 
               ('temperature', 'temperature'), 
               ('number_density', 'pressure'),
               ('number_density', 'temperature'),
               ('temperature', 'pressure')]
field_pairs_short = ['%s%s' % (field2letter[a],field2letter[b]) for (a,b) in field_pairs]
cross_pairs = ['nP', 'nT', 'TP']

#mask_names = None
#mask_names = ['no_cold_gas']
#mask_names = ['no_large_density_fluctuation']
#mask_names = ['no_large_density_fluctuation', 'no_cold_gas']
mask_names = ['no_cold_gas', 'sphere']

thermo_data = load_corey_power_spectrum_data(gridLevel, mask_names)

for snap in snapshots:
    for pair in field_pairs_short:
        k = thermo_data['k']
        a3d = thermo_data[snap][pair]['amplitude']
        axis.plot(k, a3d, marker='o', markersize=3, color=colors[pair], label=pair)#, linestyle='none')
    axis.set_title('s%s'%(snap))
    axis.set_xscale('log')
    axis.set_yscale('log')    
    axis.set_ylim(1e-2,1)
    axis.set_xlim(k.min(),1)    
    axis.set_title('s%03d'% (snap))
    axis.set_xlabel(r"$k\ (kpc^{-1})$")
    axis.set_ylabel(r"$A(k)$")
    axis.legend()

machine = 'macbook-pro'
#machine = 'Takeo'

import sys
if machine == 'macbook-pro':
    kav = '/Users/coreybrummel-smith/GT/Kavli_Summer_Program'
elif machine == 'Takeo':
    kav = '/Users/Takeo/Kavli_Summer_Program'
else:
    print('machine must be "macbook-pro" or "Takeo"')
sys.path.append(kav+'/code/python_modules')

import h5py
import numpy as np
import yt
import json
from astropy.io import fits
import MexicanHat as mh
import fft_power_spectrum as my_fft_ps
import math
import matplotlib
import matplotlib.pyplot as plt
import Dictionary_to_HDF5 as D2H5
import RadialProfile


def scale_plot_size(factor=1.5):
    import matplotlib as mpl
    default_dpi = mpl.rcParamsDefault['figure.dpi']
    mpl.rcParams['figure.dpi'] = default_dpi*factor
scale_plot_size(2.5)

def nearest_index(data, target):
    return np.argmin(np.abs(data-target))

def compute_power_spectra(h5File, MexicanHat, mask_names, snapshots, akpc, width_arcmin, field_pairs, field2letter, ndims):
    spectra_data = {}

    width = width_arcmin * akpc

    for snap in snapshots:
        if mask_names is not None:
            nside = h5File['s%03d' % snap].attrs['nside']
            mask = np.ones(nside)
            for name in mask_names:
                single_mask = h5File['s%03d' % snap]['masks'][name].value
                mask = np.logical_and(mask, single_mask)
        else:
            mask = None

        spectra_data[snap] = {}
        for fa, fb in field_pairs:
            print('fa, fb', fa, fb)
            pair = '%s%s' %(field2letter[fa], field2letter[fb])      
            spectra_data[snap][pair] = {}
            residual_a = h5File['s%03d' % snap]['residual_%s' % fa].value

            if fa == fb:
                #print('fa=fb EQUAL')
                p_MH, k_MH = MexicanHat.power_spectrum(residual_a, mask=mask, phys_width=width)
            else:   
                #print('fa != fb cross spectra')
                residual_b = h5File['s%03d' % snap]['residual_%s' % fb].value
                p_MH, k_MH = MexicanHat.power_spectrum(residual_a, residual_b, mask=mask, phys_width=width)  
            
            a_MH = MexicanHat.power_spectrum_amplitude(p_MH, k_MH, n=ndims)
 
            spectra_data[snap][pair]['power_spec'] = p_MH    
            spectra_data[snap][pair]['amplitude']  = a_MH
    spectra_data['k'] = k_MH

    return spectra_data


def compute_C_and_R(MexicanHat, snapshots, spectra_data, cross_pairs):
    for snap in snapshots:
        for pair in cross_pairs:
            print('s%d %s'%(snap, pair))
            a,b = pair[0],pair[1]
            print(a,b)
            power_a  = spectra_data[snap]['%s%s'%(a,a)]['power_spec']
            power_b  = spectra_data[snap]['%s%s'%(b,b)]['power_spec']
            power_ab = spectra_data[snap]['%s%s'%(a,b)]['power_spec']

            coh = MexicanHat.coherence(power_a, power_b, power_ab)
            rat = MexicanHat.ratio(power_a, power_ab)

            spectra_data[snap][pair]['C'] = coh
            spectra_data[snap][pair]['R'] = rat

    return spectra_data

def C_and_R_stats(spectra_data, snapshots, ranges, cross_pairs):
    # Get statistics for C and R in different length scale bins
    stats = {}
    k    = spectra_data['k']
    for snap in snapshots:
        stats[snap] = {'C': {}, 'R': {}}
        for pair in cross_pairs:
            coh  = spectra_data[snap][pair]['C']
            rat  = spectra_data[snap][pair]['R']
    
            # find C and R stats at different length scales
            l = 1./k
            lscale_ind = {'%s-%s'%(a,b): range(nearest_index(l,b),nearest_index(l,a)+1)  for a, b in ranges}      
            
            stats[snap]['C'][pair] = {}
            stats[snap]['R'][pair] = {}
            
            for a, b in ranges:
                scale = '%s-%s'%(a,b)
                ind = lscale_ind[scale]
                stats[snap]['C'][pair][scale] = (coh[ind].min(),coh[ind].max(),coh[ind].mean(),coh[ind].std())
                stats[snap]['R'][pair][scale] = (rat[ind].min(),rat[ind].max(),rat[ind].mean(),rat[ind].std())
    return stats
            
#----------------------------------------------------------------------

def coherence_and_ratio_nP(aQ, aP, cross_fields):
    gQ, gP, gT = 5./3., 0., 1. # adiabatic indicies
    aQsq = aQ * aQ
    aPsq = aP * aP
    aTsq = 1 - aQsq - aPsq
    
    alphaSq = [aQsq, aPsq, aTsq]
    gamma   = np.array([gQ, gP, gT])
    w = {'n': [1.,1.,1.], 'P': gamma, 'T': gamma - 1}

    
    top = np.zeros_like(aQ)
    bot0 = np.zeros_like(aQ)
    bot1 = np.zeros_like(aQ)


    for i in range(len(alphaSq)):
        top  += alphaSq[i] * w[cross_fields[0]][i] * w[cross_fields[1]][i] 
        bot0 += alphaSq[i] * w[cross_fields[0]][i]**2
        bot1 += alphaSq[i] * w[cross_fields[1]][i]**2

    R = top / bot0
    C = top / (np.sqrt(bot0) * np.sqrt(bot1))

    mask = np.logical_not(np.sqrt(aTsq) <= 1)    
    R[mask] = np.nan
    C[mask] = np.nan    
    
    return C, R

def make_analytic_C_R_maps(res, cross_pairs):

    map_data = {'C': {}, 'R': {}}

    # Parameter space: alpha_adiabatic (aQ) and alpha_isobaric (aP)
    res = 800
    x = y = np.linspace(0, 1, 800)
    aQ, aP = np.meshgrid(x, y)

    # isothermal quarter circle line
    xnew = x[x < 1/np.sqrt(2)]
    ynew = np.sqrt(1./2. - xnew**2)
    yT = ynew*res
    xT = xnew*res
    
    # outer quarter circle line
    xEdge = x[x < 1]
    yEdge = np.sqrt(1 - xEdge**2)
    yMax = xEdge*res
    xMax = yEdge*res
    
    # dividing line between isobaric and adiabatic
    xQP = np.linspace(xT.max()/np.sqrt(2), res/np.sqrt(2), 2)
    
    for pair in cross_pairs:
        Cmap, Rmap = coherence_and_ratio_nP(aQ, aP, pair)
        map_data['C'][pair] = {}
        map_data['R'][pair] = {}
        map_data['C'][pair]['map'] = Cmap
        map_data['R'][pair]['map'] = Rmap

    map_data['isothermal_line'] = (xT,  yT)
    map_data['isob_adiab_line'] = (xQP, xQP)
    map_data['max_line']        = (xMax, yMax)   
    map_data['res'] = res
    
    #----------------------------------------------------------------
    
    # Set up custom uniform colormaps
    colors_w = [(1,1,1),(1,1,1)]  #RGBA
    colors_t = [(0,1,1),(0,1,1)] # teal 
    colors_m = [(1,0,1),(1,0,1)] # magenta
    
    n_bin = 1 
    cm_w = matplotlib.colors.LinearSegmentedColormap.from_list(
        'white', colors_w, N=n_bin)
    cm_t = matplotlib.colors.LinearSegmentedColormap.from_list(
        'teal', colors_t, N=n_bin)
    cm_m = matplotlib.colors.LinearSegmentedColormap.from_list(
        'magenta', colors_m, N=n_bin)

    map_data['C']['cmap'] = cm_m
    map_data['R']['cmap'] = cm_t   

    return map_data

def add_CR_bands_to_CR_map(map_data, snapshots, stats, scales, cross_pairs):
    '''
     Create masks for plotting bands of Cp and Rp ontop analytic Ca and Ra maps
     These bands are also used for determining the mean and error on alpha parameters
     characterizing the nature of perturbations.
     Cp, Rp: Coherence and ratio determined from power spectra
     Ca, Ra, analytical Coherence and ratio as function of alpha parameters
    
     We determine the max delta_Ca and delta_Ra between neighboring cells in analytic map.
     Due to the non uniform distribution of Ca and Ra in the analytic maps, the same delta
     is not good for all values of Cp and Rp. This means we have to calculate deltas for
     each length scale. This delta is used to plot very thin bands representing mean values.
    '''
    for letter in ['C', 'R']:
        for pair in cross_pairs:               
            vmap = map_data[letter][pair]['map']
            vmap[np.isnan(vmap)] = np.inf
            res = vmap.shape[0]            
            for snap in snapshots:
                map_data[letter][pair][snap] = {}  
                for scale in scales:
                    map_data[letter][pair][snap][scale] = {}               
                    avg = stats[snap][letter][pair][scale][2]  # mean value at given length scale
                    std = stats[snap][letter][pair][scale][3]

                    # find delta_Ca and delta_Ra
                    near = nearest_index(vmap, avg)
                    j0,i0 = np.unravel_index(near, vmap.shape)
                    v = vmap[j0,i0]
                    delta = -1
                    for j in [-1,0,1]:
                        for i in [-1,0,1]:
                            if (j0+j < 0) or (i0+i < 0) or (j0+j >= res) or (i0+i >= res): continue
                            dv = np.abs(v - vmap[j0+j, i0+i])
                            if dv > delta: delta = dv
                    map_data[letter][pair][snap][scale]['max_delta'] = 3*delta  # make band thicker than 1 cell

                    # masks for bands showing mean +/- 1 stdev, and mean +/- delta_analytic 
                    # for both Cp and Rp.
                    for band in ['err_mask', 'mean_mask']:
                        if band == 'err_mask':
                            vmin = avg-std
                            vmax = avg+std
                        elif band == 'mean_mask':    
                            vmin = avg-delta
                            vmax = avg+delta                 
                        mask = np.logical_and(vmap > vmin, vmap < vmax)
                        map_data[letter][pair][snap][scale][band] = mask 

                    # end for band
                # end for scale
            # end for pair in cross_pairs
        # end for snap
    # end for letter
    return map_data    



#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
projection = False
padded = False

datadir = kav + '/results_data/testing_truncate'
plotdir = kav + '/AGN_FB_plots/corey_MH_results/testing_truncate'
#datadir = kav + '/results_data/all_samples'
#plotdir = kav + '/AGN_FB_plots/corey_MH_results/all_samples'
print('datadir:', datadir) 
print('plotdir:', plotdir) 

snapshots = [114]
#snapshots = [150, 137, 114, 103, 82, 59, 147]

cr_map_resolution = 800
map_letters = ['R','C']
           
field2letter = {'number_density':'n', 'pressure':'P', 'temperature':'T', 'mock_hard':'H', 'mock_soft':'S'}
colors = {'nn':'b', 'TT':'r', 'PP':'g', 'nT':'m', 'nP':'c', 'TP':'gold', 'HH':'tab:red', 'SS':'tab:blue', 'HS':'k'}

field_pairs = [('number_density', 'number_density'), 
               ('pressure', 'pressure'), 
               ('temperature', 'temperature'), 
               ('number_density', 'pressure'),
               ('number_density', 'temperature'),
               ('temperature', 'pressure')]
field_pairs_short = ['%s%s' % (field2letter[a],field2letter[b]) for (a,b) in field_pairs]
print("\n\n",field_pairs_short)
cross_pairs = ['nP', 'nT', 'TP']

"""
field_pairs = [('number_density', 'number_density'), 
               ('pressure', 'pressure'), 
               ('number_density', 'pressure')]
field_pairs_short = ['%s%s' % (field2letter[a],field2letter[b]) for (a,b) in field_pairs]
cross_pairs = ['nP']
"""

ranges = [(0,5), (5,15), (15,30), (30,50)] # length scales in kpc
scales = ['%s-%s'%(a,b) for a,b in ranges]


if not projection:
    gridLevel=8
    ndims = 3
    dim_str = '3d_gridLevel_%02d' % gridLevel
else:
    ndims = 2
    dim_str = '2d' 
    #field_pairs = [('mock_hard', 'mock_hard'),
                    #('mock_soft', 'mock_soft'),
                    #('mock_hard', 'mock_soft')]     
    #field_pairs_short = ['HH', 'SS', 'HS']
    #cross_pairs = ['HS']

residual_file = 'residuals_%s.h5' % dim_str  
if padded:
    residual_file = 'padded_' + residual_file
else:
    residual_file = 'not_padded_' + residual_file  
residual_file = "%s/%s" % (datadir, residual_file)

print(residual_file)


h5file = h5py.File(residual_file, 'r')
#h5file = h5py.File('/Users/Takeo/Kavli_Summer_Program/code/all_data_s114_new2.h5', 'r')
#h5file = h5py.File('/Users/coreybrummel-smith/GT/Kavli_Summer_Program/code/all_data_s114_new2.h5', 'r')
kpc_per_arcmin = h5file.attrs['kpc_per_arcmin'] 
width_arcmin   = h5file.attrs['width_arcmin']

kpts=25
conv_method = 'nearest'
#conv_method = 'constant'
#mask_names = None
#mask_names = ['no_cold_gas']
#mask_names = ['no_large_density_fluctuation']
#mask_names = ['no_large_density_fluctuation', 'no_cold_gas']
mask_names = ['no_cold_gas', 'sphere']

if padded:
    mask_names.append('pad')

MH = mh.MexicanHat(Nk=kpts, mode=conv_method)

spectra_data = compute_power_spectra(h5file, MH, mask_names, snapshots, 
    kpc_per_arcmin, width_arcmin, field_pairs, field2letter, ndims)
h5file.close()
spectra_data = compute_C_and_R(MH, snapshots, spectra_data, cross_pairs)
stats = C_and_R_stats(spectra_data, snapshots, ranges, cross_pairs)
cr_map_data = make_analytic_C_R_maps(cr_map_resolution, cross_pairs)
cr_map_data = add_CR_bands_to_CR_map(cr_map_data, snapshots, stats, scales, cross_pairs)


# ------- PLOTTING --------  

res = cr_map_resolution

# pixel boundaries of our map
xmin, xmax, ymin, ymax = (0, res, 0, res)
# We'll also create a grey background into which the pixels will fade
white = np.array([255,255,255]*res**2).reshape(res,res,3)


for snap in snapshots:

    # Power spectra amplitude figure
    fig1, axis = plt.subplots(1,1)

    # C(k) and R(k) figure
    fig2, (ax1,ax2) = plt.subplots(2,1, sharex=True)
    fig2.subplots_adjust(hspace=0)  

    for pair in field_pairs_short:
        k = spectra_data['k']
        a3d = spectra_data[snap][pair]['amplitude']
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

    for pair in cross_pairs:
        print("pair", pair)
        # C and R map overlap figure
        fig, axes = plt.subplots(2,2, figsize=(8,8))
        fig.subplots_adjust(hspace=0.05, wspace=0.05, right=0.88)    
        fig.suptitle('s%s: %s'%(snap,pair), x=0.5, y=0.92, fontsize=14)
        axes = axes.flatten()

        k = spectra_data['k']
        coh = spectra_data[snap][pair]['C']
        rat = spectra_data[snap][pair]['R']
        ax1.set_title('s%s'%(snap))
        ax1.plot(k, coh, alpha=0.8, marker='.', label='%s'%pair) 
        ax2.plot(k, rat, alpha=0.8, marker='.', label='%s'%pair)

        # loop over 4 C and R overlap axes
        for ax in axes:
            ax.imshow(white, origin='lower')
            ax.plot(cr_map_data['isothermal_line'][0], cr_map_data['isothermal_line'][1], 'k--')
            ax.plot(cr_map_data['isob_adiab_line'][0], cr_map_data['isob_adiab_line'][1], 'k--')
            ax.plot(cr_map_data['max_line'][0], cr_map_data['max_line'][1], 'k--')
            ax.set_xlabel(r'$\alpha_{adiab.}$', fontsize=14)
            ax.set_ylabel(r'$\alpha_{isob.}$', fontsize=14)  
            ax.set(xticks=[0, res/2, res], yticks=[0, res/2, res],
                   xticklabels=[0, 0.5, 1], yticklabels=[0, 0.5, 1])
            ax.text(0.7*res,  0.25*res, 'adiabatic',  fontsize=10)
            ax.text(0.1*res,  0.8*res,  'isobaric',   fontsize=10)
            ax.text(0.05*res, 0.20*res, 'isothermal', fontsize=10)
            
        for letter in map_letters:
            #print('\n',letter)
            data = cr_map_data[letter][pair]['map']
            cmap = cr_map_data[letter]['cmap']
            for i, (scale, (a,b)) in enumerate(zip(scales,ranges)):
                stat = stats[snap][letter][pair][scale]

                avg = stat[2]
                std = stat[3]
                vmin = avg-std
                vmax = avg+std  
                #print(snap, letter, avg)
                
                # Create an alpha channel based measured C or R values
                mask = cr_map_data[letter][pair][snap][scale]['err_mask']
                alphas = np.clip(mask, 0, .5)  # alpha set to 
                
                # Make uniform MxM pixel value array
                highlight = np.ones_like(data)
                # turn our pixel values into MxNx4 color array
                highlight = cmap(highlight)
                # Now set the alpha channel to the one we created above
                highlight[..., -1] = alphas

                axes[i].imshow(highlight, extent=(xmin, xmax, ymin, ymax), origin='lower')
                axes[i].tick_params(labelsize=10)
                axes[i].text(0.7, 0.85, '%s-%s kpc'%(a,b), color='k', 
                             transform=axes[i].transAxes, fontsize=11,
                             bbox=dict(facecolor='white', boxstyle='round', lw=0.5))
                
                # mask outside of circle in grey
                greyVal = 0.5
                n_bin = 1 
                colors_gr = [(greyVal,greyVal,greyVal),(greyVal,greyVal,greyVal)] # grey 
                cm_gr = matplotlib.colors.LinearSegmentedColormap.from_list(
                    'grey', colors_gr, N=n_bin)                
                x = y = np.linspace(0, 1, 800)
                aQ, aP = np.meshgrid(x, y)
                r = np.sqrt(aQ**2 + aP**2)
                mask = r > 1
                alphas = np.clip(mask, 0, .5)
                highlight = np.ones_like(r)
                highlight = cm_gr(highlight)
                # Now set the alpha channel to the one we created above
                highlight[..., -1] = alphas
                axes[i].imshow(highlight, extent=(xmin, xmax, ymin, ymax), origin='lower')  

        ## END for letter in map_letters                          
            
        axes[0].get_xaxis().set_visible(False)
        axes[1].get_xaxis().set_visible(False)
        axes[1].get_yaxis().set_visible(False) 
        axes[3].get_yaxis().set_visible(False)  

        # C(k) and R(k) axes specs
        ax2.set_xscale('log')
        #ax2.set_xlim(k.min(),1)    
        ax2.set_xlabel(r"$k,\ kpc^{-1}$")
        ax1.set_ylabel(r"$C$")
        ax2.set_ylabel(r"$R$")
        ax2.legend(loc=8, ncol=1, bbox_to_anchor=(1.08,-0.03), fontsize=9, handlelength=1)
    
        mask_str = ''
        if mask_names is not None:
            for name in mask_names:
                mask_str += '_' + name
        fig2.savefig("%s/truncate4_MH_%03d_%s_CR_%s%s"           % (plotdir, snap, pair, dim_str, mask_str))
        fig.savefig( "%s/truncate4_MH_%03d_%s_CR_map_%s%s"       % (plotdir, snap, pair, dim_str, mask_str))        
    
    ## END for pair in cross_pairs

    fig1.savefig("%s/truncate4_MH_%03d_power_spectra_%s%s"% (plotdir, snap, dim_str, mask_str))

    #plt.show()  

### END for snap in snapshots

spectra_fn = '%s/truncate4_spectra_%s%s.h5' % (datadir, dim_str, mask_str)
D2H5.save_dict_to_hdf5(spectra_data, spectra_fn)

##*** Remove cr_map_data[letter]['cmap'] because hdf5 Cannot save <class 'matplotlib.colors.LinearSegmentedColormap'>  ***
#cr_map_fn = '%s/cr_map_data_%s%s.h5' % (datadir, dim_str, mask_str)
#D2H5.save_dict_to_hdf5(cr_map_data, cr_map_fn)

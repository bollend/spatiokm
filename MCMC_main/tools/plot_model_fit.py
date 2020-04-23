import sys
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main')
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main/tools')
import os
import shutil
import argparse
import numpy as np
import pickle
import Cone
import geometry_binary
from astropy import units as u
import parameters_DICT
import uncertainty_DICT
import MCMC
import matplotlib.pylab as plt
import time

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   Plots the model spectra and the observed spectra
    ============================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import matplotlib as mpl
fts  = 20
fts_tick = fts*0.75
# mpl.rcParams['text.usetex'] = True
mpl.rc('xtick', labelsize=fts_tick)
mpl.rc('ytick', labelsize=fts_tick)
plt.rc('font', weight='bold')
plt.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}\boldmath'] #for \text command
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']
plt.rcParams['axes.linewidth']      = 1.

"""
==================================================
Command line input
==================================================
"""
args = sys.argv

parser = argparse.ArgumentParser()

parser.add_argument('-o', dest='object_id',
                    help='Object identifier')

parser.add_argument('-d', dest='datafile',
                    help='data file with all the input parameters and specifics')

args          = parser.parse_args()
object_id     = str(args.object_id)
datafile      = str(args.datafile)
line='halpha'

"""
================================
Binary system and jet properties
================================
"""

AU              = 1.496e+11     # 1AU in m
AU_to_km        = 1.496e+08     # 1AU in km
days_to_sec     = 24*60*60      # 1day in seconds
degr_to_rad     = np.pi/180.    # Degrees to radians


###### Read in the object specific and model parameters ########################
parameters = {}
InputDir   = '../../input_data/'+object_id+'/'
InputFile = datafile

###### Create the parameter dictionary with all jet, binary, and model parameters
parameters = parameters_DICT.read_parameters(InputDir+InputFile)
parameters['BINARY']['T_inf'] = geometry_binary.T0_to_IC(parameters['BINARY']['omega'],
                                                         parameters['BINARY']['ecc'],
                                                         parameters['BINARY']['period'],
                                                         parameters['BINARY']['T0'])
pars_model = parameters_DICT.read_model_parameters(InputDir+InputFile)

pars_model_array = np.zeros( len(pars_model.keys()) )

for n,param in enumerate(parameters['MODEL'].keys()):

    parameters['MODEL'][param]['id'] = n
    pars_model_array[n] = pars_model[param]

###### additional parameters ###################################################

pars_add = MCMC.calc_additional_par(parameters, pars_model_array)


"""
===============
Stellar spectra
===============
"""

###### Observed spectra, background spectra, and wavelength region #############

spectra_observed    = {}
spectra_wavelengths = {}
spectra_background  = {}

with open(InputDir+'halpha/'+object_id+'_observed_'+line+'.txt', 'rb') as f:
    spectra_observed    = pickle.load(f)
with open(InputDir+'halpha/'+object_id+'_wavelength_'+line+'.txt', 'rb') as f:
    spectra_wavelengths = pickle.load(f)
with open(InputDir+'halpha/'+object_id+'_init_'+line+'.txt', 'rb') as f:
    spectra_background  = pickle.load(f)

phases  = list()
spectra = list()
phases_dict = {}

for ph in spectra_observed.keys():

    phases.append(ph)
    phases_dict[ph] = []

    for spec in spectra_observed[ph].keys():

        spectra.append(spec)
        phases_dict[ph].append(spec)


"""
=======================
Uncertainty of the data
=======================
"""

standard_deviation = uncertainty_DICT.create_uncertainties(object_id, parameters, InputDir+'halpha/', phases_dict)

"""
======================================
Cut the wavelength region if necessary
======================================
"""

if parameters['OTHER']['cut_waveregion']==True:

    wavmin = min(range(len(spectra_wavelengths)), key = lambda j: abs(spectra_wavelengths[j]- parameters['OTHER']['wave_begin']))
    wavmax = min(range(len(spectra_wavelengths)), key = lambda j: abs(spectra_wavelengths[j]- parameters['OTHER']['wave_end']))
    spectra_wavelengths = spectra_wavelengths[wavmin:wavmax]

    for phase in spectra_observed.keys():

        for spectrum in spectra_observed[phase].keys():

            spectra_observed[phase][spectrum]   = spectra_observed[phase][spectrum][wavmin:wavmax]
            spectra_background[phase][spectrum] = spectra_background[phase][spectrum][wavmin:wavmax]

            if parameters['OTHER']['uncertainty_back']==True:

                standard_deviation[phase][spectrum] = standard_deviation[phase][spectrum][wavmin:wavmax]

###### Calculate the total number of datapoints
datapoints = 0
for phase in spectra_observed.keys():
    for spectrum in spectra_observed[phase].keys():
        datapoints += len(spectra_observed[phase][spectrum])

parameters['OTHER']['total_datapoints'] = datapoints

"""
===========================
Calculate the model spectra
===========================
"""
time0 = time.time()
chi_squared, probability, spectra_model = MCMC.ln_likelihood(parameters, pars_model_array, pars_add, spectra_observed, spectra_background, spectra_wavelengths, standard_deviation, return_intensity=True)

print('number of datapoints is ', datapoints)
print(len(pars_model_array))
print('The chi-squared is %f and the loglikelihood is %f' % (chi_squared, probability) )
print('The BIC is %f' % (np.log(datapoints)*len(pars_model_array) - 2*probability))
"""
==================================================
Create an array for the observed and model spectra
==================================================
"""

spectra_observed_array = np.zeros([100, len(spectra_wavelengths)])
spectra_model_array    = np.zeros([100, len(spectra_wavelengths)])

for phase in spectra_observed.keys():

    sum_spectra = np.zeros(len(spectra_wavelengths))

    for spectrum in spectra_observed[phase].keys():

        sum_spectra += spectra_background[phase][spectrum]/len(spectra_observed[phase])

    for spectrum in spectra_observed[phase].keys():

        spectra_observed_array[phase,:] = spectra_observed[phase][spectrum]

        spectra_model_array[phase,:] = spectra_model[phase][spectrum]

"""
=================================================================
Plot the model spectra vs the observed spectra (not interpolated)
=================================================================
"""

ratio = 1


## Set limits
# tsmin = 4.028
# tsmax = 184.78
# y_min  = 0
# y_max  = 110

## Plots here
plt.rc('font', weight='bold')
# plt.rc('xtick', labelsize=ratio*15)
# mpl.rc('ytick', labelsize=ratio*15)
# mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}\boldmath'] #for \text command
plt.rc('text', usetex=True)

## Fontsize
fts  = ratio*36
lbsz = ratio*10
lbs  = ratio*16
lgds = ratio*12

## Figure
# fig, axes = plt.subplots(1, 4, figsize=(ratio*210/30,ratio*0.8*297/30) )

## Ticks
# minor_yticks = phases[1::2]
# major_yticks = phases[0::2]

# Color
# num_plots = len(spectra)
colormap = plt.cm.seismic


# fig = plt.figure()
###### The observed spectra
# ax1  = fig.add_subplot(121)
fig, axes = plt.subplots(1, 2, figsize=(11,5) )
cax1 = axes[0].imshow(spectra_observed_array, extent=((spectra_wavelengths[0] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, (spectra_wavelengths[-1] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, 1, 0),  cmap=colormap,  vmin=0.2, vmax=1.8, aspect='auto')

title = " Period = " + str(parameters['BINARY']['period'])
# cb1   = fig.colorbar(cax1)

# cb1.set_label('Flux', fontsize=18)
#
# for ticks in cb1.ax.get_yticklabels():
#
#      ticks.set_fontsize(14)
#
# fig.autofmt_xdate()
plt.xlabel('RV (km/s)', fontsize=18)
plt.ylabel('Phase', fontsize=18)

###### The model spectra



cax2 = axes[1].imshow(spectra_model_array, extent=((spectra_wavelengths[0] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, (spectra_wavelengths[-1] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, 1, 0), cmap=colormap,  vmin=0.2, vmax=1.8, aspect='auto')
# axes[1].set_xlim((data_wavelength_mcmc[0] - central_wavelength)/central_wavelength*3.e5,\
#                     (data_wavelength_mcmc[-1] - central_wavelength)/central_wavelength*3.e5)
# cb2 = fig.colorbar(cax2)
#
# cb2.set_label('Flux', fontsize=18)
#
# for ticks in cb2.ax.get_yticklabels():
#
#      ticks.set_fontsize(14)
#
# fig.autofmt_xdate()

cb=fig.colorbar(cax2, ax=axes[1], pad=0.01, aspect=50)
cb.set_label('Flux', fontsize=15)
for t in cb.ax.get_yticklabels():
     t.set_fontsize(12)

plt.xlabel ('RV (km/s)', fontsize=18)
fig.tight_layout()
plt.show()

"""
=============================================================
Plot the model spectra vs the observed spectra (interpolated)
=============================================================
"""
###### Interpolation ###########################################################

observed_spectra_dict = spectra_observed
n_phases = len(observed_spectra_dict.keys())

list_phases = []

for count,ph in enumerate(observed_spectra_dict.keys()):
    list_phases.append(ph)

list_phases.sort()
array_phases = 0.01 * np.array(list_phases)
n_phases = len(array_phases)
observed_spectra = np.zeros((len(spectra_wavelengths), n_phases))
triple_observed_spectra = np.zeros((len(spectra_wavelengths), 3*n_phases))

synthetic_spectra = np.zeros((len(spectra_model_array[0,:]), n_phases))
triple_synthetic_spectra = np.zeros((len(spectra_model_array[0,:]), 3*n_phases))


triple_phases = np.zeros(3*n_phases)
for i in range(3):
    triple_phases[i*n_phases:(i+1)*n_phases] = array_phases + (i-1)

for count, key in enumerate(list_phases):
    mean_data = np.zeros(len(spectra_wavelengths))
    for spectr in observed_spectra_dict[key]:

        mean_data += observed_spectra_dict[key][spectr]

    mean_data /= len(observed_spectra_dict[key])
    observed_spectra[:,count] = mean_data
    synthetic_spectra[:,count] = spectra_model_array[key,:]
    triple_observed_spectra[:,count] = mean_data
    triple_observed_spectra[:,count+n_phases] = mean_data
    triple_observed_spectra[:,count+2*n_phases] = mean_data
    triple_synthetic_spectra[:,count] = spectra_model_array[key,:]
    triple_synthetic_spectra[:,count+n_phases] = spectra_model_array[key,:]
    triple_synthetic_spectra[:,count+2*n_phases] = spectra_model_array[key,:]


interpolated_data = np.zeros((len(spectra_wavelengths), 100))
interpolated_synthetic = np.zeros((len(spectra_model_array[0,:]), 100))
phases = np.arange(0, 1.0, 0.01)

from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.interpolate import spline
import matplotlib.gridspec as gridspec

###### observed spectra ########################################################

gs1 = gridspec.GridSpec(1, 2)
my_dpi = 100
fig = plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)
axes[0] = fig.add_subplot(gs1[0])
axes[1] = fig.add_subplot(gs1[1])
for w, wave in enumerate(spectra_wavelengths):
    interp_wave = spline(triple_phases, triple_observed_spectra[w,:], phases, order=1, kind='smoothest')
    interpolated_data[w,:] = interp_wave
    central_wavelength = 6562.8


axes[0].imshow(interpolated_data.T, cmap=colormap, \
        extent=((spectra_wavelengths[0] - central_wavelength)/\
                        central_wavelength*3.e5, (spectra_wavelengths[-1] - central_wavelength)/\
                        central_wavelength*3.e5, 1, 0),\
                        origin='upper', vmin=0.2, vmax=1.8, aspect='auto')
# plt.suptitle("Observations vs model")
axes[0].set_xlabel('RV (km/s)', fontsize=18)
axes[0].set_ylabel('Phase', fontsize=18)

###### Model spectra ###########################################################

for w, wave in enumerate(spectra_model_array[0,:]):
    interp_synth = spline(triple_phases, triple_synthetic_spectra[w,:], phases, order = 1, kind='smoothest')
    interpolated_synthetic[w,:] = interp_synth
    central_wavelength = 6562.8

cax2 = axes[1].imshow(interpolated_synthetic.T,cmap= colormap, \
        extent=((spectra_wavelengths[0] - parameters['OTHER']['wave_center'])/\
                        parameters['OTHER']['wave_center']*3.e5, (spectra_wavelengths[-1] - parameters['OTHER']['wave_center'])/\
                        parameters['OTHER']['wave_center']*3.e5, 1, 0), vmin=0.2, vmax=1.8, aspect='auto')
# axes[1].set_xlim((data_wavelength_mcmc[0] - central_wavelength)/central_wavelength*3.e5,\
#                     (data_wavelength_mcmc[-1] - central_wavelength)/central_wavelength*3.e5)
cb=fig.colorbar(cax2, ax=axes.ravel().tolist(), pad=0.01, aspect=50)
cb.set_label('Flux', fontsize=18)
for t in cb.ax.get_yticklabels():
     t.set_fontsize(12)

# pl.title("H_alpha")
axes[1].set_xlabel('RV (km/s)', fontsize=18)
# fig.tight_layout()
plt.savefig('obsvsmodel.png',dpi=300)
plt.show()

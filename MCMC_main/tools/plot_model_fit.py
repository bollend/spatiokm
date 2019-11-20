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
import emcee
from emcee.utils import MPIPool
import MCMC
import matplotlib.pylab as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   Plots the model spectra and the observed spectra
    ============================================================================
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

chi_squared, probability, spectra_model = MCMC.ln_likelihood(parameters, pars_model_array, pars_add, spectra_observed, spectra_background, spectra_wavelengths, standard_deviation, return_intensity=True)
print('The chi-squared is %f and the loglikelihood is %f' % (chi_squared, probability) )
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
==============================================
Plot the model spectra vs the observed spectra
==============================================
"""

fig = plt.figure()

###### The observed spectra
ax1  = fig.add_subplot(121)
cax1 = ax1.imshow(spectra_observed_array, extent=((spectra_wavelengths[0] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, (spectra_wavelengths[-1] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, 1, 0),\
                aspect='normal', cmap=plt.cm.seismic,  vmin=0.2, vmax=1.8)

title = " Period = " + str(parameters['BINARY']['period'])
cb1   = fig.colorbar(cax1)

cb1.set_label('Flux', fontsize=18)

for ticks in cb1.ax.get_yticklabels():

     ticks.set_fontsize(14)

fig.autofmt_xdate()
plt.xlabel('RV (km/s)', fontsize=18)
plt.ylabel('Phase', fontsize=18)

###### The model spectra


ax2  = fig.add_subplot(122)
cax2 = ax2.imshow(spectra_model_array, extent=((spectra_wavelengths[0] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, (spectra_wavelengths[-1] - parameters['OTHER']['wave_center'])/\
                parameters['OTHER']['wave_center']*3.e5, 1, 0),
                aspect='normal', cmap=plt.cm.seismic,  vmin=0.2, vmax=1.8)
# ax2.set_xlim((data_wavelength_mcmc[0] - central_wavelength)/central_wavelength*3.e5,\
#                     (data_wavelength_mcmc[-1] - central_wavelength)/central_wavelength*3.e5)
cb2 = fig.colorbar(cax2)

cb2.set_label('Flux', fontsize=18)

for ticks in cb2.ax.get_yticklabels():

     ticks.set_fontsize(14)

fig.autofmt_xdate()
plt.xlabel ('RV (km/s)', fontsize=18)
plt.show()

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

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   We first initialise the input data (spectra),
"   orbital parameters, and jet parameters that we need to calculate the
"   absorption by the jet.
"   Next, create the post-AGB star (as a Fibonacci-grid), the binary system and
"   the jet configuration.
"   We then calculate the amount of absorption by the jet in the spectral line.
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
InputDir   = '../input_data/'+object_id+'/'
InputFile = datafile

###### Create the parameter dictionary with all jet, binary, and model parameters
parameters = parameters_DICT.create_parameters(InputDir+InputFile)
parameters['BINARY']['T_inf'] = geometry_binary.T0_to_IC(parameters['BINARY']['omega'],
                                                         parameters['BINARY']['ecc'],
                                                         parameters['BINARY']['period'],
                                                         parameters['BINARY']['T0'])

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
==================
Main MCMC sampling
==================
"""
###### Set the initial positions of the walkers ################################
parameters, parameters_init = \
            MCMC.mcmc_initial_positions(parameters,
                                  prev_chain=parameters['OTHER']['previous_walkers'],
                                  dir_previous_chain=parameters['OTHER']['dir_previous_chain'])

###### Check if the priors are in the accepted range ###########################
for walker in range(parameters['OTHER']['n_walkers']):

    parameters_add = MCMC.calc_additional_par(parameters, parameters_init[walker,:])

    while MCMC.ln_prior(parameters_init[walker,:], parameters, parameters_add) == -np.inf:

        pars, parameters_init[walker,:] = MCMC.mcmc_initial_positions(parameters,
                                      prev_chain=parameters['OTHER']['previous_walkers'],
                                      dir_previous_chain=parameters['OTHER']['dir_previous_chain'],
                                      single_walker=True)
        parameters_add = MCMC.calc_additional_par(parameters, parameters_init[walker,:])

###### Create the pool #########################################################
pool = MPIPool()

if not pool.is_master():
    pool.wait()
    syst.exit(0)
# spectra_observed = {1: {'2': np.array([1,1,2,1])}}
# spectra_background = {1: {'2': np.array([2,2,2,2])}}
# spectra_wavelengths = np.array([1,2,3,4])
# standard_deviation = {1: {'2': np.array([0.1,0.1,0.2,0.1])}}

sampler = emcee.EnsembleSampler(parameters['OTHER']['n_walkers'],
                                len(parameters['MODEL'].keys()),
                                MCMC.ln_probab,
                                args=(parameters,spectra_observed, spectra_background, spectra_wavelengths, standard_deviation),
                                pool=pool)

###### create the output file for the chain ####################################

OutputDir = '../MCMC_output/'+parameters['OTHER']['object_id']+'/results_'+parameters['OTHER']['jet_type']+'/output'

NewDirNumber  = 0
NewDirCreated = False

while NewDirCreated==False:

    if not os.path.exists(OutputDir+'_'+str(NewDirNumber)):

        os.makedirs(OutputDir+'_'+str(NewDirNumber))
        NewDirCreated = True

    else:

        NewDirNumber += 1

OutputDir = OutputDir+'_'+str(NewDirNumber)

shutil.copyfile(InputDir+InputFile, OutputDir+'/'+InputFile)

f = open(OutputDir+'/MCMC_chain_output.dat', 'ab')
f.close()

###### Run the mcmc chain ######################################################

for result in sampler.sample(parameters_init, iterations=parameters['OTHER']['n_iter'], storechain=False):
    position = result[0]
    f = open(OutputDir+'/MCMC_chain_output.dat', 'ab')
    np.savetxt(f, np.array(position))
    f.close()

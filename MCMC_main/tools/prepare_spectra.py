import sys
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main/tools')
sys.path.append('/lhome/dylanb/astronomy/MCMC_main/MCMC_main')
import os,glob
import pylab
import pyfits as pf
import numpy as np
import scipy
import matplotlib.pylab as plt
import scipy.constants as c
from PyAstronomy import pyasl
from scipy import interpolate
import csv
import pickle
import pandas as pd
import argparse
import parameters_DICT
import uncertainty_DICT
import geometry_binary

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ============================================================================
"   Prepare the input data, i.e. wavelength range, observed spectra, background
"   spectra, and standard deviation of the background spectra.
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

parser.add_argument('-range', dest='velocity_range',
                    help='The blue- and redshifted velocity range')

parser.add_argument('-line', dest='line',
                    help='The Balmer line')

parser.add_argument('-what', dest='what',
                    help='What file do you want to create')


args          = parser.parse_args()
object_id     = str(args.object_id)
datafile      = object_id+'.dat'
listfile      = object_id+'.list'
line          = str(args.line)
v_range       = float(args.velocity_range)
what          = str(args.what)

"""
================================
Binary system and jet properties
================================
"""

central_wavelengths = {'halpha': 6562.8, 'hbeta': 4861.35, 'hgamma': 4340.47, 'hdelta': 4101.73, 'random':5169}
central_wavelength  = central_wavelengths[line]
wave_min            = central_wavelength - (1000*v_range/c.speed_of_light) * central_wavelength
wave_max            = central_wavelength + (1000*v_range/c.speed_of_light) * central_wavelength

###### Read in the object specific and model parameters ########################
parameters = {}
InputDir   = '/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'
InputFile  = datafile

###### Create the parameter dictionary with all jet, binary, and model parameters
parameters = parameters_DICT.read_parameters(InputDir+InputFile)
parameters['BINARY']['T_inf'] = geometry_binary.T0_to_IC(parameters['BINARY']['omega'],
                                                         parameters['BINARY']['ecc'],
                                                         parameters['BINARY']['period'],
                                                         parameters['BINARY']['T0'])

parameters['OTHER']['central_wavelength'] = central_wavelength

def scale_intensity(wave_0,
                    wave_template,
                    intensity_template,
                    wave_spec,
                    intensity_spec,
                    norm=False,
                    wave_region=100e-10,
                    wave_0_d=100e-10):
    '''
    Returns the rescaled spectrum.
    norm=True if the spectrum is normalised.
    '''
    if norm==False:

        wave_min_a     = wave_0 - (wave_0_d + wave_region)
        wave_min_b     = wave_0 - (wave_0_d)
        wave_max_a     = wave_0 + (wave_0_d)
        wave_max_b     = wave_0 + (wave_0_d + wave_region)

        index_low  = np.argmax(intensity_template[np.where((wave_template > wave_min_a) & (wave_template < wave_min_b))])
        index_high = np.argmax(intensity_template[np.where((wave_template > wave_max_a) & (wave_template < wave_max_b))])
        I_low      = (intensity_template[np.where((wave_template > wave_min_a) & (wave_template < wave_min_b))])[index_low]
        I_high     = (intensity_template[np.where((wave_template > wave_max_a) & (wave_template < wave_max_b))])[index_high]
        wave_low   = (wave_template[np.where((wave_template > wave_min_a) & (wave_template < wave_min_b))])[index_low]
        wave_high  = (wave_template[np.where((wave_template > wave_max_a) & (wave_template < wave_max_b))])[index_high]
        gradient       = (I_high - I_low) / ( wave_high - wave_low)
        intercept      = I_high - gradient * wave_high
        scaling_interp = np.poly1d(np.array([gradient, intercept]))
        intensity_spec = intensity_spec * scaling_interp(wave_spec)

        # plt.plot(wave_template, intensity_template)
        # plt.plot(wave_spec, intensity_spec, label='after')
        # plt.legend()
        # plt.show()

    if norm==True:

        wave_min_a     = wave_0 - (wave_0_d + wave_region)
        wave_min_b     = wave_0 - (wave_0_d)
        wave_max_a     = wave_0 + (wave_0_d)
        wave_max_b     = wave_0 + (wave_0_d + wave_region)
        wave_low       = 0.5 * ( wave_min_a + wave_min_b )
        wave_high      = 0.5 * ( wave_max_a + wave_max_b )

        median_I_low   = np.percentile(intensity_spec[np.where((wave_template > wave_min_a) & (wave_template < wave_min_b))], 50)
        median_I_high  = np.percentile(intensity_spec[np.where((wave_template > wave_max_a) & (wave_template < wave_max_b))], 50)
        gradient       = (median_I_high - median_I_low) / (wave_high - wave_low)
        intercept      = median_I_high - gradient *  wave_high
        scaling_interp = np.poly1d(np.array([gradient, intercept]))

        intensity_spec = intensity_spec / scaling_interp(wave_spec)

        # plt.plot(wave_template, intensity_template)
        # # plt.plot(wave_spec, scaling_interp(wave_spec), label='interpolated')
        # plt.plot(wave_spec, intensity_spec, label='after')
        # plt.legend()
        # plt.xlim(6500, 6650)
        # plt.ylim(0,2)
        # plt.show()
    return intensity_spec, scaling_interp

def get_spectrum(filename, phase, parameters, v_range, bjd):


    datalist = glob.glob(filename)

    if len(datalist) > 0:

        # Read the data and the header is resp. 'spectrum' and 'header'
        spectrum = pf.getdata(datalist[0])
        header   = pf.getheader(datalist[0])

        crpix           = header.get('CRPIX1')-1
        crval           = header.get('CRVAL1')
        cdelt           = header.get('CDELT1')
        bvcor           = header.get('BVCOR')
        numberpoints    = len(spectrum)
        wavelengthbegin = (crval - crpix*cdelt)
        wavelengthend   = crval + (numberpoints-1)*cdelt
        wavelengths_exp = np.linspace(wavelengthbegin,wavelengthend,numberpoints)
        # Change from logarithmic to linear wavelength range
        wavelengths_lin = np.exp(wavelengths_exp)
        delta_wavelength = (1e3 * v_range / c.speed_of_light) * parameters['OTHER']['central_wavelength']
        indices         = np.where((parameters['OTHER']['central_wavelength'] - delta_wavelength < wavelengths_lin) & (wavelengths_lin < parameters['OTHER']['central_wavelength'] + delta_wavelength))
        # spectrum_wave_min = min(range(len(wavelengthslin)), key = lambda j: abs(wavelengthslin[j]- (parameters['OTHER']['central_wavelength'] - 48)))
        # spectrum_wave_max = min(range(len(wavelengthslin)), key = lambda j: abs(wavelengthslin[j]- (parameters['OTHER']['central_wavelength'] + 48)))

        wavelengths     = wavelengths_lin[indices]

        spectrum, interpol = scale_intensity(parameters['OTHER']['central_wavelength'],
                                            wavelengths_lin,
                                            np.ones(numberpoints),
                                            wavelengths_lin,
                                            spectrum,
                                            norm=True,
                                            wave_region=50,
                                            wave_0_d=50)

        spectrum = spectrum[indices]

        wavelengths = wavelengths * ( 1 - 1e3 * parameters['BINARY']['gamma'] / c.speed_of_light)
        return wavelengths, spectrum


def create_observed_spectra_file(object_id, line, parameters, InputDir):

    data_list = pd.read_csv(InputDir+object_id+'.list', delimiter=' ', names=['ID', 'full_date', 'object_id', 'bjd', 'bvcor', 'program_id', 'exposure_time', 'something', 'something_else', 'date'])

    phases    = ( (data_list.bjd - parameters['BINARY']['T_inf']) / parameters['BINARY']['period'] ) % 1

    filenames = [ '/lhome/dylanb/astronomy/objects/survey2020/'+object_id+'/*'+str(filenumber)+'_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits' for filenumber in data_list['ID']]

    data_spectra = {}

    for (spec, filename) in enumerate(filenames):

        wavelength_spectrum, spectrum = get_spectrum(filename, phases[spec], parameters, v_range, data_list.bjd[spec])

        # we do minus 0.5 such that 52.9 -> 52.4 -> 52
        place = int(round((phases[spec]/0.01) - 0.5,0))

        if (place in data_spectra) == False:

            data_spectra[place]                = {}
            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = spectrum

        else:

            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = spectrum

    for phase in data_spectra.keys():
        for spec in data_spectra[phase].keys():
            plt.plot(wavelength_spectrum, data_spectra[phase][spec])
            plt.show()

    f = open('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_observed_'+line+'.txt', 'wb')
    pickle.dump(data_spectra,f)
    f.close()

    f_wav = open('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_wavelength_'+line+'.txt', 'wb')
    pickle.dump(wavelength_spectrum, f_wav)
    f_wav.close()

    return 'complete'

def create_background_spectra_file(object_id, line, parameters, InputDir, plot=True):

    ##### Upload the file containing the list with all spectra
    data_list = pd.read_csv(InputDir+object_id+'.list', delimiter=' ', names=['ID', 'full_date', 'object_id', 'bjd', 'bvcor', 'program_id', 'exposure_time', 'something', 'something_else', 'date'])
    ##### Determine at which phase each spectra is taken
    phases    = ( (data_list.bjd - parameters['BINARY']['T_inf']) / parameters['BINARY']['period'] ) % 1
    ##### The filename of each spectra
    filenames = [ '/lhome/dylanb/astronomy/objects/survey2020/'+object_id+'/*'+str(filenumber)+'_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits' for filenumber in data_list['ID']]

    if line=='halpha':
        ##### Get the template spectrum with emission
        wave_template, spectrum_template = np.loadtxt('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_template_'+line+'.txt')

    ##### The filename of the photospheric spectrum
    file_photosphere = parameters['OTHER']['synthetic']
    ##### The wavelength and flux of the photospheric spectrum
    wave_photo, spectrum_photo = np.loadtxt('/lhome/dylanb/astronomy/MCMC_main/input_data/synthetic/'+file_photosphere)

    ##### Normalise the photospheric spectrum
    prim_spec_fake, scaling_inter_synth = scale_intensity(parameters['OTHER']['central_wavelength'],wave_photo,
                                        spectrum_photo,
                                        wave_photo,
                                        spectrum_photo,
                                        norm=True,
                                        wave_region=100,
                                        wave_0_d=100)

    spec_photo = spectrum_photo/scaling_inter_synth(wave_photo)

    interp_photo = interpolate.interp1d(wave_photo, spec_photo)

    data_spectra = {}

    for (spec, filename) in enumerate(filenames):

        wavelength_spectrum, spectrum = get_spectrum(filename, phases[spec], parameters, v_range, data_list.bjd[spec])

        prim_pos, sec_pos, prim_vel, sec_vel = geometry_binary.pos_vel_primary_secondary(
                                           phases[spec]*100,
                                           parameters['BINARY']['period'],
                                           parameters['BINARY']['omega'],
                                           parameters['BINARY']['ecc'],
                                           parameters['BINARY']['primary_asini'],
                                           parameters['BINARY']['primary_asini'],
                                           parameters['BINARY']['T_inf'],
                                           parameters['BINARY']['T0'])

        wave_photo_shifted = wavelength_spectrum * (1 - 1000. * -1. * prim_vel[1] / c.speed_of_light)

        interp_spectrum_photo = interp_photo(wave_photo_shifted)

        if line=='halpha':

            background_spectrum = spectrum_template + interp_spectrum_photo

        else:

            background_spectrum = interp_spectrum_photo

        # background_spectrum = interp_spectrum_photo
        # we do minus 0.5 such that 52.9 -> 52.4 -> 52
        place = int(round((phases[spec]/0.01) - 0.5,0))
        print(place)
        plt.plot(wavelength_spectrum, spectrum, label='obs')
        plt.plot(wavelength_spectrum, interp_spectrum_photo, label='synth')
        plt.plot(wavelength_spectrum, background_spectrum, label='background')
        plt.legend()
        plt.show()
        if (place in data_spectra) == False:

            data_spectra[place]                     = {}
            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = background_spectrum

        else:

            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = background_spectrum


    if plot==True:
        plot_dynamic_spectra(wavelength_spectrum, data_spectra, parameters)

    f = open('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_init_'+line+'.txt', 'wb')
    pickle.dump(data_spectra,f)
    f.close()


    return 'complete'

def create_template_spectra_file(object_id, line, parameters, InputDir, plot=True):

    ##### Upload the file containing the list with all spectra
    data_list = pd.read_csv(InputDir+object_id+'.list', delimiter=' ', names=['ID', 'full_date', 'object_id', 'bjd', 'bvcor', 'program_id', 'exposure_time', 'something', 'something_else', 'date'])
    ##### Determine at which phase each spectra is taken
    phases    = ( (data_list.bjd - parameters['BINARY']['T_inf']) / parameters['BINARY']['period'] ) % 1
    ##### The filename of each spectra
    filenames = [ '/lhome/dylanb/astronomy/objects/survey2020/'+object_id+'/*'+str(filenumber)+'_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits' for filenumber in data_list['ID']]
    ##### The filename of the photospheric spectrum
    file_photosphere = parameters['OTHER']['synthetic']
    ##### The wavelength and flux of the photospheric spectrum
    wave_photo, spectrum_photo = np.loadtxt('/lhome/dylanb/astronomy/MCMC_main/input_data/synthetic/'+file_photosphere)

    ##### Normalise the photospheric spectrum
    prim_spec_fake, scaling_inter_synth = scale_intensity(parameters['OTHER']['central_wavelength'],wave_photo,
                                        spectrum_photo,
                                        wave_photo,
                                        spectrum_photo,
                                        norm=True,
                                        wave_region=100,
                                        wave_0_d=100)

    spec_photo = spectrum_photo/scaling_inter_synth(wave_photo)

    interp_photo = interpolate.interp1d(wave_photo, spec_photo)

    ##### Get the template spectrum with emission
    template_ID = [ parameters['OTHER']['template'] ]

    if parameters['OTHER']['template_2'] != None:

        template_ID.append(parameters['OTHER']['template_2'])

        if parameters['OTHER']['template_3'] != None:

            template_ID.append(parameters['OTHER']['template_3'])

    data_spectra = {}

    for (spec, filename) in enumerate(filenames):

        wavelength_spectrum, spectrum = get_spectrum(filename, phases[spec], parameters, v_range, data_list.bjd[spec])

        prim_pos, sec_pos, prim_vel, sec_vel = geometry_binary.pos_vel_primary_secondary(
                                           phases[spec]*100,
                                           parameters['BINARY']['period'],
                                           parameters['BINARY']['omega'],
                                           parameters['BINARY']['ecc'],
                                           parameters['BINARY']['primary_asini'],
                                           parameters['BINARY']['primary_asini'],
                                           parameters['BINARY']['T_inf'],
                                           parameters['BINARY']['T0'])

        wave_photo_shifted = wavelength_spectrum * (1 - 1000. * -1. * prim_vel[1] / c.speed_of_light)

        interp_spectrum_photo = interp_photo(wave_photo_shifted)

        background_spectrum = spectrum - interp_spectrum_photo + 1
        # we do minus 0.5 such that 52.9 -> 52.4 -> 52
        place = int(round((phases[spec]/0.01) - 0.5,0))
        # print(place)
        # print(filename)
        # plt.plot(wavelength_spectrum, spectrum, label='obs')
        # plt.plot(wavelength_spectrum, interp_spectrum_photo, label='synth')
        # plt.plot(wavelength_spectrum, background_spectrum, label='background')
        # plt.legend()
        # plt.show()
        if (place in data_spectra) == False:

            data_spectra[place]                     = {}
            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = background_spectrum

        else:

            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = background_spectrum


    if len(template_ID)==1:
        ##### Only use one spectrum as template
        for phase in data_spectra.keys():

            for spec in data_spectra[phase].keys():

                if spec in template_ID:

                    template_spectrum = data_spectra[phase][spec] - 1

    else:
        ##### Use the mean of multiple spectra as template

        template_spectrum = np.zeros(len(wavelength_spectrum))
        for phase in data_spectra.keys():

            for spec in data_spectra[phase].keys():

                if spec in template_ID:

                    template_spectrum += data_spectra[phase][spec] - 1

        template_spectrum /= len(template_ID)

    std_spectra = {}

    for phase in data_spectra.keys():

        std_spectra[phase] = {}

        for spec in data_spectra[phase].keys():

            std_spectra[phase][spec] = abs(template_spectrum)

    if plot==True:
        plt.plot(wavelength_spectrum, template_spectrum)
        plt.show()
        plot_dynamic_spectra(wavelength_spectrum, data_spectra, parameters)

    np.savetxt('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_template_'+line+'.txt', np.array((wavelength_spectrum, template_spectrum)))

    f = open('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_stdev_init_'+line+'.txt', 'wb')
    pickle.dump(std_spectra,f)
    f.close()

    return 'complete'

def create_standard_deviation_spectra_file(object_id, line, parameters, InputDir):

    ##### Upload the file containing the list with all spectra
    data_list = pd.read_csv(InputDir+object_id+'.list', delimiter=' ', names=['ID', 'full_date', 'object_id', 'bjd', 'bvcor', 'program_id', 'exposure_time', 'something', 'something_else', 'date'])
    ##### Determine at which phase each spectra is taken
    phases    = ( (data_list.bjd - parameters['BINARY']['T_inf']) / parameters['BINARY']['period'] ) % 1
    ##### The filename of each spectra
    filenames = [ '/lhome/dylanb/astronomy/objects/survey2020/'+object_id+'/*'+str(filenumber)+'_HRF_OBJ_ext_CosmicsRemoved_log_merged_c.fits' for filenumber in data_list['ID']]

    if line=='halpha':
        ##### Get the template spectrum with emission
        wave_template, spectrum_template = np.loadtxt('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_template_'+line+'.txt')

    ##### The filename of the photospheric spectrum
    file_photosphere = parameters['OTHER']['synthetic']
    ##### The wavelength and flux of the photospheric spectrum
    wave_photo, spectrum_photo = np.loadtxt('/lhome/dylanb/astronomy/MCMC_main/input_data/synthetic/'+file_photosphere)

    ##### Normalise the photospheric spectrum
    prim_spec_fake, scaling_inter_synth = scale_intensity(parameters['OTHER']['central_wavelength'],wave_photo,
                                        spectrum_photo,
                                        wave_photo,
                                        spectrum_photo,
                                        norm=True,
                                        wave_region=100,
                                        wave_0_d=100)

    spec_photo = spectrum_photo/scaling_inter_synth(wave_photo)

    interp_photo = interpolate.interp1d(wave_photo, spec_photo)

    data_spectra = {}

    for (spec, filename) in enumerate(filenames):

        wavelength_spectrum, spectrum = get_spectrum(filename, phases[spec], parameters, v_range, data_list.bjd[spec])

        prim_pos, sec_pos, prim_vel, sec_vel = geometry_binary.pos_vel_primary_secondary(
                                           phases[spec]*100,
                                           parameters['BINARY']['period'],
                                           parameters['BINARY']['omega'],
                                           parameters['BINARY']['ecc'],
                                           parameters['BINARY']['primary_asini'],
                                           parameters['BINARY']['primary_asini'],
                                           parameters['BINARY']['T_inf'],
                                           parameters['BINARY']['T0'])

        wave_photo_shifted = wavelength_spectrum * (1 - 1000. * -1. * prim_vel[1] / c.speed_of_light)

        interp_spectrum_photo = interp_photo(wave_photo_shifted)

        if line=='halpha':

            background_spectrum = spectrum_template + interp_spectrum_photo

        else:

            background_spectrum = interp_spectrum_photo

        # background_spectrum = interp_spectrum_photo
        # we do minus 0.5 such that 52.9 -> 52.4 -> 52
        place = int(round((phases[spec]/0.01) - 0.5,0))
        print(place)
        plt.plot(wavelength_spectrum, spectrum, label='obs')
        plt.plot(wavelength_spectrum, interp_spectrum_photo, label='synth')
        plt.plot(wavelength_spectrum, background_spectrum, label='background')
        plt.legend()
        plt.show()
        if (place in data_spectra) == False:

            data_spectra[place]                     = {}
            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = background_spectrum

        else:

            data_spectra[place][data_list.ID[spec]] = {}
            data_spectra[place][data_list.ID[spec]] = background_spectrum


    if plot==True:
        plot_dynamic_spectra(wavelength_spectrum, data_spectra, parameters)

    f = open('/lhome/dylanb/astronomy/MCMC_main/input_data/'+object_id+'/'+line+'/'+object_id+'_stdev_init_'+line+'.txt', 'wb')
    pickle.dump(data_spectra,f)
    f.close()


    return 'complete'


"""
=============================================================
Plot the model spectra vs the observed spectra (interpolated)
=============================================================
"""
###### Interpolation ###########################################################

def plot_dynamic_spectra(spectra_wavelengths, spectra, parameters):
    """
    Plot the model spectra vs the observed spectra (interpolated)
    """

    ##### prepare a spectrum array
    spectra_array = np.zeros([100, len(spectra_wavelengths)])

    for phase in spectra.keys():

        for spectrum in spectra[phase].keys():

            spectra_array[phase,:] = spectra[phase][spectrum]

    spectra_dict = spectra
    n_phases = len(spectra_dict.keys())

    list_phases = []

    for count,ph in enumerate(spectra_dict.keys()):
        list_phases.append(ph)

    list_phases.sort()
    array_phases = 0.01 * np.array(list_phases)
    n_phases = len(array_phases)
    spectra_temp = np.zeros((len(spectra_wavelengths), n_phases))
    triple_spectra = np.zeros((len(spectra_wavelengths), 3*n_phases))

    triple_phases = np.zeros(3*n_phases)
    for i in range(3):
        triple_phases[i*n_phases:(i+1)*n_phases] = array_phases + (i-1)

    for count, key in enumerate(list_phases):
        mean_data = np.zeros(len(spectra_wavelengths))
        for spectr in spectra_dict[key]:

            mean_data += spectra_dict[key][spectr]

        mean_data /= len(spectra_dict[key])
        spectra_temp[:,count] = mean_data
        triple_spectra[:,count] = mean_data
        triple_spectra[:,count+n_phases] = mean_data
        triple_spectra[:,count+2*n_phases] = mean_data


    interpolated_data = np.zeros((len(spectra_wavelengths), 100))
    phases = np.arange(0, 1.0, 0.01)

    from scipy.interpolate import interp1d
    from scipy.interpolate import interp2d
    from scipy.interpolate import spline
    import matplotlib.gridspec as gridspec

    ###### observed spectra ########################################################

    colormap = plt.cm.seismic

    gs1 = gridspec.GridSpec(1, 1)
    my_dpi = 100
    fig = plt.figure(figsize=(1200/my_dpi, 600/my_dpi), dpi=my_dpi)
    axes = fig.add_subplot(gs1[0])
    for w, wave in enumerate(spectra_wavelengths):
        interp_wave = spline(triple_phases, triple_spectra[w,:], phases, order=1, kind='smoothest')
        interpolated_data[w,:] = interp_wave
        central_wavelength = parameters['OTHER']['central_wavelength']


    axes.imshow(interpolated_data.T, cmap=colormap, \
            extent=((spectra_wavelengths[0] - central_wavelength)/\
                            central_wavelength*3.e5, (spectra_wavelengths[-1] - central_wavelength)/\
                            central_wavelength*3.e5, 1, 0),\
                            origin='upper', vmin=0.2, vmax=1.8, aspect='auto')
    # plt.suptitle("Observations vs model")
    axes.set_xlabel('RV (km/s)', fontsize=18)
    axes.set_ylabel('Phase', fontsize=18)

    plt.show()


if what=='observed':
    print(create_observed_spectra_file(object_id, line, parameters,InputDir))

if what=='background':
    print(create_background_spectra_file(object_id, line, parameters,InputDir))

if what=='template':
    print(create_template_spectra_file(object_id, line, parameters,InputDir))

if what=='std':
    print(create_standard_deviation_spectra_file(object_id, line, parameters,InputDir))

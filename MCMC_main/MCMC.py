import numpy as np
import sys
import os, glob
import pickle
from scipy import constants
import emcee
from emcee.utils import MPIPool
import geometry_binary
import Star
import Cone
import create_jet
from scipy import constants
AU              = 1.496e+11     # 1AU in m


def ln_probab(pars_walker,
              pars,
              data_spectra,
              background_spectra,
              wavelengths_spectra,
              standard_deviation):
    """
        Calculates the probability of the model.

    Parameters
    ==========
    pars : dictionary
        All parameters
    pars_walker : np.array
        The model parameters of the walker
    data_spectra : dictionary
        The observed spectra ordered per phase
    Returns
    =======
    probability : float
        The probability of the model
    """
    pars_add = calc_additional_par(pars, pars_walker)

    lp = ln_prior(pars_walker, pars, pars_add)

    if np.isfinite(lp):

        chi_squared, probability = ln_likelihood(pars, pars_walker, pars_add, data_spectra, background_spectra, wavelengths_spectra, standard_deviation)
        probability += lp

        return probability

    return -np.inf

def ln_likelihood(pars,
                  pars_walker,
                  pars_add,
                  data_spectra,
                  background_spectra,
                  wavelengths_spectra,
                  standard_deviation,
                  return_intensity=False):
    """
        Calculates the likelihood function of the model.

    Parameters
    ==========
    pars: dictionary
        All parameters
    pars_walker : np.array
        The model parameters of the walker
    pars_add : dictionary
        The additional parameters that have to be calculated
    data_spectra : dictionary
        The observed spectra ordered per phase
    background_spectra : dictionary
        The background spectra ordered per phase
    wavelengths_spectra : np.array
        The wavelength array
    standard_deviation : dictionary
        The standard deviation ordered per phase
    Returns
    =======
    chi_squared : float
        The chi-squared of the model
    Likelihood : float
        The log likelihood of the model
    model_spectra : dictionary
        The model spectra
    """
    ###### Create the post-AGB star with a Fibonacci grid
    postAGB = Star.Star(pars_walker[pars['MODEL']['primary_radius']['id']],
                        np.array([0,0,0]),
                        pars_walker[pars['MODEL']['inclination']['id']],
                        pars['OTHER']['gridpoints_primary'])

    ###### Create the jet
    jet = create_jet.create_jet(pars['OTHER']['jet_type'], pars, pars_walker)

    ###### create the binary orbit
    primary_orbit   = {}
    secondary_orbit = {}

    chi_squared   = 0
    ln_likelihood = 0

    if return_intensity==True:

        model_spectra = {}

    for phase in data_spectra.keys():
    ###### iterate over each orbital phase

        prim_pos, sec_pos, prim_vel, sec_vel = geometry_binary.pos_vel_primary_secondary(
                                           phase,
                                           pars['BINARY']['period'],
                                           pars['BINARY']['omega'],
                                           pars['BINARY']['ecc'],
                                           pars_add['primary_sma_AU'],
                                           pars_add['secondary_sma_AU'],
                                           pars['BINARY']['T_inf'],
                                           pars['BINARY']['T0'])


        postAGB.centre = prim_pos
        jet.jet_centre = sec_pos
        if pars['OTHER']['tilt']==True:

            jet._set_orientation(np.array([sec_vel]))

        postAGB._set_grid()
        postAGB._set_grid_location()

        intensity = model(pars, pars_walker, pars_add, wavelengths_spectra, jet, postAGB)

        if return_intensity==True:

            model_spectra[phase] = {}

        for spectrum in data_spectra[phase].keys():
            ###### Iterate over all spectra with this phase

            sigma_squared                = standard_deviation[phase][spectrum]**2
            model_spectrum               = background_spectra[phase][spectrum] * intensity
            chi_squared_spectrum_array   = (data_spectra[phase][spectrum] - model_spectrum)**2 / sigma_squared
            ln_likelihood_spectrum_array = -0.5 * (chi_squared_spectrum_array + np.log(2. * np.pi * sigma_squared))
            chi_squared_spectrum         = np.sum(chi_squared_spectrum_array)
            ln_likelihood_spectrum       = np.sum(ln_likelihood_spectrum_array)

            chi_squared   += chi_squared_spectrum
            ln_likelihood += ln_likelihood_spectrum

            if return_intensity==True:

                model_spectra[phase][spectrum] = model_spectrum

    if return_intensity==True:

        return chi_squared, ln_likelihood, model_spectra

    else:

        return chi_squared, ln_likelihood



def model(pars, pars_walker, pars_add, wavelengths_spectra, jet, postAGB):
    """
        Calculates the amount of absorption of the light that travels through
        the jet

    Parameters
    ==========
    pars: dictionary
        All parameters
    pars_walker : np.array
        The model parameters of the walker
    pars_add : dictionary
        The additional parameters that have to be calculated
    data_spectra : dictionary
        The observed spectra ordered per phase
    wavelengths_spectra : np.array
        The wavelength array of the spectra
    jet : object
        The jet object
    postAGB : object
        The star object representing the post-AGB star

    Returns
    =======
    Intensity : np.array
        The intensity at each wavelength (assuming an initial normalised
        intensity of 1)
    """
    intensity = np.zeros(len(wavelengths_spectra))

    for pointAGB, coordAGB in enumerate(postAGB.grid_location):
        ###### For each ray of light from a grid point on the post-AGB star,
        ###### we calculate the absorption by the jet

        jet._set_gridpoints(coordAGB, pars['OTHER']['gridpoints_LOS'])

        if jet.gridpoints is None:
            ###### The ray does not pass through the jet

            intensity_pointAGB = pars['OTHER']['gridpoints_primary']**-1 * np.ones(len(wavelengths_spectra))

        else:
            ###### the ray passes through the jet

            jet._set_gridpoints_unit_vector()
            jet._set_gridpoints_polar_angle()
            ###### The jet velocity and density
            scaling_par_od          = -10**pars_walker[pars['MODEL']['c_optical_depth']['id']] # The scaling parameter
            optical_depth_ray       = np.zeros(len(wavelengths_spectra)) # The scaled optical depth at each wavelength bin
            jet_density_scaled      = jet.density(pars['OTHER']['gridpoints_LOS']) # The scaled number density of the jet
            jet_velocity            = jet.poloidal_velocity(pars['OTHER']['gridpoints_LOS'], pars['OTHER']['power_velocity']) # The velocity of the jet at each gridpoint (km/s)
            jet_radvel_km_per_s     = jet.radial_velocity(jet_velocity, pars_add['secondary_rad_vel']) # Radial velocity of each gridpoint (km/s)
            jet_radvel_m_per_s      = jet_radvel_km_per_s * 1000 # Radial velocity of each gridpoint (m/s)
            jet_delta_gridpoints_AU = np.linalg.norm(jet.gridpoints[0,:] - jet.gridpoints[1,:]) # The length of each gridpoint (AU)
            jet_delta_gridpoints_m  = jet_delta_gridpoints_AU * AU  # The length of each gridpoint (m)
            # The shifted central wavelength of the line
            jet_wavelength_0_rv     = pars['OTHER']['wave_center'] * ( 1. + jet_radvel_m_per_s / constants.c )

            indices_wavelengths   = [ np.abs(wave - wavelengths_spectra).argmin() for wave in jet_wavelength_0_rv]
            optical_depth_ray_LOS = jet_density_scaled * jet_delta_gridpoints_AU # The scaled optical depth for each gridpoint along the LOS
            np.add.at(optical_depth_ray, indices_wavelengths, optical_depth_ray_LOS)

            intensity_pointAGB = pars['OTHER']['gridpoints_primary']**-1 * np.exp(scaling_par_od * optical_depth_ray)

        intensity += intensity_pointAGB

    return intensity

def ln_prior(pars_walker, pars, pars_add):
    """
        Flat priors. The probability is set to -infty if one of the conditions
        is not met.

    Parameters
    ==========
    pars : dictionary
        All parameters
    pars_walker : np.array
        The model parameters of the walker
    pars_add : dictionary
        The additional parameters for this jet type
    Returns
    =======
    probability : float
        -np.inf or 0

    """

    jet_angle_id = pars['MODEL']['jet_angle']['id']
    incl_id      = pars['MODEL']['inclination']['id']
    radius_id    = pars['MODEL']['primary_radius']['id']
    v_max_id     = pars['MODEL']['velocity_max']['id']
    v_edge_id    = pars['MODEL']['velocity_edge']['id']

    probability = 0.0
    jet_height_above_star = pars_add['primary_sma_AU'] * ( 1 + pars_add['mass_ratio']) / np.tan(pars_walker[jet_angle_id])

    ###### Primary radius and jet velocity
    if not (pars_walker[radius_id] < jet_height_above_star
            and pars['MODEL']['primary_radius']['min'] < pars_walker[radius_id] < 0.725 * pars_add['roche_radius_primary_AU']
            and pars_walker[v_edge_id] < pars_walker[v_max_id]
    ):

        probability = -np.inf

    ###### All model parameters are within the correct range
    for param in pars['MODEL'].keys():

        param_id = pars['MODEL'][param]['id']

        if not pars['MODEL'][param]['min'] < pars_walker[param_id] < pars['MODEL'][param]['max']:

            probability = -np.inf

    if not probability==-np.inf:

        ###### The jet cavity angle is smaller than the inner jet angle or jet angle
        if 'jet_cavity_angle' in pars['MODEL'].keys():

            cavity_id = pars['MODEL']['jet_cavity_angle']['id']

            if (pars['OTHER']['jet_type']=='stellar_jet_simple'
                or pars['OTHER']['jet_type']=='stellar_jet'
                or pars['OTHER']['jet_type']=='X_wind_strict'
                or pars['OTHER']['jet_type']=='sdisk_wind_strict'
            ):

                if not pars_walker[cavity_id] < pars_walker[jet_angle_id]:

                    probability = -np.inf

            elif (pars['OTHER']['jet_type']=='x_wind'
                  or pars['OTHER']['jet_type']=='sdisk_wind'
            ):

                if not pars_walker[cavity_id] < pars_walker[jet_angle_id]:

                    probability = -np.inf

        ###### The jet inner angle is smaller than the jet outer angle
        if pars['OTHER']['jet_type']=='sdisk_wind' or pars['OTHER']['jet_type']=='x_wind':

                jet_angle_inner_id = pars['MODEL']['jet_angle_inner']['id']

                if not pars_walker[jet_angle_inner_id] < pars_walker[jet_angle_id]:

                    probability = -np.inf


        ###### inner radius of the disk is larger than twice the stellar radius and
        ###### outer radius of the disk is smaller than the roche radius of the companion
        if pars['OTHER']['jet_type']=='sdisk_wind' or pars['OTHER']['jet_type']=='sdisk_wind_strict':

            if not (2.*pars_add['secondary_radius'] < pars_add['disk_radius_in'] < pars_add['disk_radius_out'] < pars_add['roche_radius_s_AU']
            ):

                probability = -np.inf

    return probability

def calc_additional_par(pars, pars_walker):
    """
    calculate any parameter in the system from the model parameters, depending
    on the jet type.

    Parameters
    ==========
    pars : dictionary
        All parameters
    pars_walker : np.array
        The model parameters of the walker

    Returns
    =======
    pars_add : dictionary
        The additional parameters that have to be calculated

    """
    pars_add = {}

    incl_id                 = pars['MODEL']['inclination']['id']

    secondary_mass          = geometry_binary.calc_mass_sec(pars['BINARY']['primary_mass'],
                                                pars_walker[incl_id],
                                                pars['BINARY']['mass_function'])

    mass_ratio              = pars['BINARY']['primary_mass'] / secondary_mass

    primary_sma_AU          = pars['BINARY']['primary_asini'] / pars_walker[incl_id]
    secondary_sma_AU        = mass_ratio * primary_sma_AU
    roche_radius_primary    = geometry_binary.calc_roche_radius(mass_ratio, 'primary')
    roche_radius_primary_AU = roche_radius_primary * primary_sma_AU
    primary_orb_vel         = pars['BINARY']['primary_rad_vel'] / pars_walker[incl_id]
    secondary_orb_vel       = primary_orb_vel * mass_ratio
    secondary_rad_vel       = pars['BINARY']['primary_rad_vel'] * mass_ratio

    pars_add = {'primary_sma_AU': primary_sma_AU,
                      'secondary_sma_AU' : secondary_sma_AU,
                      'secondary_mass': secondary_mass,
                      'mass_ratio':mass_ratio,
                      'roche_radius_primary_AU':roche_radius_primary_AU,
                      'primary_orb_vel':primary_orb_vel,
                      'secondary_orb_vel':secondary_orb_vel,
                      'secondary_rad_vel':secondary_rad_vel}

    if pars['OTHER']['jet_type']=='sdisk_wind' or pars['OTHER']['jet_type']=='sdisk_wind_strict':
        G                  = constants.gravitational_constant
        M_sun              = 1.98847e30
        AU                 = 1.496e11
        R_sun              = 6.9551e8
        velocity_edge_id   = pars['MODEL']['velocity_edge']['id']
        velocity_max_id    = pars['MODEL']['velocity_max']['id']
        jet_angle_id       = pars['MODEL']['jet_angle']['id']
        jet_angle_inner_id = pars['MODEL']['jet_angle_inner']['id']
        v_M                = pars_walker[velocity_edge_id] * np.tan(pars_walker[jet_angle_id])**.5
        v_in               = v_M * np.tan(pars_walker[jet_angle_inner_id])
        secondary_radius   = 1.01*mass_sec**0.724 * R_sun / AU
        roche_radius_s     = geometry_binary.calc_roche_radius(mass_ratio, 'secondary')
        roche_radius_s_AU  = roche_radius_s * primary_sma_AU
        disk_radius_out    = G * secondary_mass * M_sun \
                          * (1000*pars_walker[velocity_edge_id])**-2 * AU**-1
        disk_radius_in     = G * secondary_mass * M_sun \
                          * (1000*v_in)**-2 * AU**-1
        parameters_extra = {'v_M': v_M,
                            'v_in': v_in,
                            'secondary_radius': secondary_radius,
                            'roche_radius_s_AU': roche_radius_s_AU,
                            'disk_radius_out': disk_radius_out,
                            'disk_radius_in': disk_radius_in
                            }
        pars_add.update(parameters_extra)

    return pars_add


def mcmc_initial_positions(pars, prev_chain=False, dir_previous_chain=None, single_walker=False):
    """
    Set the initial positions for the mcmc sampling

    Parameters
    ==========
    pars : dictionary
        The model parameters
    prev_chain : boolean
        True if there is a previous chain of walkers
    dir_previous_chain : string
        the directory to the previous chain
    single_walker : boolean
        True if we calculate the position for just one walker

    Returns
    =======
    pars : dictionary
        The model parameters including the parameter id
    pars_init : np.array
        The initial positions of the parameters
    """

    n_par     = len(pars['MODEL'].keys())
    n_walkers = int(pars['OTHER']['n_walkers'])

    if prev_chain==True:

        OutputDirPrevious = '../MCMC_output/'+pars['OTHER']['object_id']+'/results_'+pars['OTHER']['jet_type']+'/'+dir_previous_chain

        ###### Load the parameter id
        parameter_id = {}
        with open(OutputDirPrevious+'/parameter_id.dat','r') as f:
            lines = f.readlines()

        for line in lines:

            split_line         = line.split()
            par                = split_line[0]
            par_id             = int(split_line[1])
            parameters_id[par] = par_id

        ###### Load the previous walkers
        with open(OutputDirPrevious+'/MCMC_chain_'+pars['OTHER']['jet_type']+'.dat', 'r') as chain_file:
            walkers_lines = chain_file.readlines()[-n_walkers:]

        ###### create the chain array and add the parameter id
        walkers_list = [np.array(walker_line.split()) for walker_line in walkers_lines]
        pars_init = np.random.rand(n_walkers,n_par)

        for param in parameters_id.keys():

            pars['MODEL'][param]['id'] = parameters_id[param]

            for walker in n_walkers:

                pars_init[walker,parameters_id[param]] = walkers_list[walker][parameters_id[param]]

    elif single_walker==False:

        pars_init = np.random.rand(n_walkers,n_par)

        for n,param in enumerate(pars['MODEL'].keys()):

            pars_init[:,n] = pars['MODEL'][param]['min'] + \
                                   ( pars['MODEL'][param]['max'] - pars['MODEL'][param]['min'] )\
                                   * pars_init[:,n]
            pars['MODEL'][param]['id'] = n

    else:

        pars_init = np.random.rand(n_par)

        for param in pars['MODEL'].keys():

            n = pars['MODEL'][param]['id']
            pars_init[n] = pars['MODEL'][param]['min'] + \
                                   ( pars['MODEL'][param]['max'] - pars['MODEL'][param]['min'] )\
                                   * pars_init[n]

    return pars, pars_init

import numpy
import sys
import os, glob
import pickle



def lnlike(model_parameters, mass_prim, mass_sec, data_spectra, r_out, r_in, z_h, v_in):
    # Calculates the loglikelyhood of the model
    #inclination, alpha_out, alpha_in, constant_optdepth, v_axis, v_edge, v_disk_par, r_p = Theta
    #intensity, model_spectra, binsize = model(inclination, alpha_out, alpha_in, mass_prim, mass_sec, constant_optdepth, v_axis, v_in, v_edge, v_disk_par, r_p, data_spectra, r_out, r_in, z_h)

    intensity, model_spectra, binsize = model(model_parameters, )
    v_range = np.arange(-v_reach, v_reach + 0.1*binsize, binsize)
    wavelength_range = central_wavelength*(1. + v_range / 299792.458)
    
    chi2         = 0
    lnlikelyhood = 0
    imagenumber  = 0

    for key in model_spectra:

        interp_model = interpolate.interp1d(wavelength_range, model_spectra[key])
        for spectrum in data_spectra_mcmc[key]:
            sigma2 = (data_add_stdev_mcmc[key][spectrum] + standard_deviation[spectrum])**2
            if emission == True:
                chi2_spectrum = (interp_model(data_wavelength_mcmc)\
                                    *init_spec_mcmc[key][spectrum] \
                                    - data_spectra_mcmc[key][spectrum])**2\
                                    /sigma2
            else:
                chi2_spectrum = (interp_model(data_wavelength_mcmc) \
                                - data_spectra_mcmc[key][spectrum])**2\
                                /sigma2
            chi2 += np.sum(chi2_spectrum)

            lnlikelyhood_spectrum = -0.5*(chi2_spectrum +\
                        np.log(2. * np.pi * sigma2))
            lnlikelyhood += np.sum(lnlikelyhood_spectrum)


    BIC = np.log(12584) * 8 - 2 * (lnlikelyhood)
    chi2_red = chi2 / (12584 - 8)
    print 'chi2', chi2, 'lnlikelyhood', lnlikelyhood, 'BIC', BIC, 'red chi2', chi2_red
    return lnlikelyhood, intensity

def probab(Theta, mass_prim, data_spectra):
    # print 'theta:', Theta
    mass_sec = calc_mass_sec(mass_prim, Theta[0])
    v_M = Theta[5]*np.tan(Theta[1])**.5
    v_in = v_M*np.tan(Theta[2])**-.5
    r_out = 6.674e-11*mass_sec*1.989e30/(Theta[5]*1000.)**2/1.496e11/parameters['asini']*np.sin(Theta[0])
    r_in = 6.674e-11*mass_sec*1.989e30/(v_in*1000.)**2/1.496e11/parameters['asini']*np.sin(Theta[0])
    z_h = r_out/np.tan(Theta[1])
    lp = lnprior(Theta, mass_prim, mass_sec, r_out, r_in, v_in)
    if not np.isfinite(lp):
        return -np.inf
    lnlikel, intensity = lnlike(Theta, mass_prim, mass_sec, data_spectra, r_out, r_in, z_h, v_in)
    return lp + lnlikel

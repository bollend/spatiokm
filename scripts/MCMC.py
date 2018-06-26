# Fitting of velocities in and out the jet

usage = \
"""
Create a synthetic dynamic spectrum for the scattering
of continuum photons of the primary by the jet.
"""

import time
# import pyfits as pf
import numpy as np
import matplotlib.pylab as pl
import matplotlib.cm as cm
import matplotlib
import pylab
import sys
import os, glob
import pickle
import csv
# import imageio
import argparse
import disk_grid_fibonacci
from PyAstronomy import pyasl
from scipy.optimize import curve_fit
from scipy import interpolate
from astropy.stats import sigma_clip
from mpl_toolkits.mplot3d import Axes3D
font = {'size' : 22}
matplotlib.rc('font', **font)
# ##########
# ### Input data
# ##########
#

args = sys.argv


parser = argparse.ArgumentParser(description=usage)
# # positional
# parser.add_argument('-dat', dest = 'datfile', help='name of the parameterfile')
# # positional
parser.add_argument('-object', dest = 'nameobject', help='name of the object')
# # positional
parser.add_argument('-line', dest = 'line', help='minimum inclined jet velocity ')

args = parser.parse_args()
nameobject = args.nameobject
line = args.line

#################
"""
### date & time
"""
#################

now = time.strftime("%c")
with open('../date_run.txt', "a") as datefile:
    datefile.write('%s \t %10s \t nestedcos \t %10s \n' % (nameobject, line, time.strftime("%c")))

#################
"""
### parameters
"""
#################

parameters = {}
infile = open('../MCMC_objects/'+str(nameobject)+'/'+str(nameobject)+'.dat', 'r')
lines = infile.readlines()[2:]
for l in lines:
    title = l[:20].strip()
    value = eval(l[20:].split()[0])
    parameters[title] = value
infile.close()

"""
Fixed parameters
"""

emission = parameters['emission']
central_wavelength = parameters['w_c_'+str(line)]
w_begin = parameters['w_begin_'+str(line)]
w_end = parameters['w_end_'+str(line)]
omega = parameters['omega']
ecc = parameters['ecc']
T0 = parameters['T0']
P   = parameters['period']	#period (days)
period = P
K1 = parameters['K_p']
m_p = parameters['m_p']	#mass primary
v_prim = parameters['K_p']
fm = parameters['fm']
w   = 2.*np.pi/P
#t = np.linspace(0.,P,200)
Npoints_pathlength = parameters['points_pathlength']    #number of points along the path length trough the jet

R_p = parameters['R_p']	#orbital radius primary


####################
'''
synthetic photospheric spectrum, with (emission == True)
or without (emission == False) initial emission componenent)
'''
####################

if emission == True:
    f = open('../MCMC_objects/'+nameobject+'/'+str(nameobject) + '_init_'+parameters['template']+'.txt', 'r')
    init_spec = pickle.load(f)
    f.close()
else:
    wave_range, init_spec = np.loadtxt(\
                        '../MCMC_objects/'+nameobject+'/synthetic/'\
                        +parameters['photospheric']+'.ms_vsini14.txt',\
                        unpack = True, usecols = [0,1])
    interp = interpolate.interp1d(wave_range, init_spec)

###################
'''
 T0 set during inferior conjunction
'''
###################

nu_inf = (90.+ 1.*omega)*np.pi / 180.
E_inf = np.arccos((ecc + np.cos(nu_inf)) / (1 + ecc*np.cos(nu_inf)))
M_inf = E_inf - ecc * np.sin(E_inf)
n_inf = 2.*np.pi/P
if nu_inf < np.pi:
    T_inf = T0 - M_inf / n_inf
else:
    T_inf = T0 - P + M_inf / n_inf

###################
'''
Load the spectral data file + S/N
'''
###################

f = open('../MCMC_objects/'+nameobject+'/'+str(nameobject) + '_data_'+str(line)+'.txt', 'r')
data_spectra = pickle.load(f)
f.close()

# print data_spectra
f_wav = open('../MCMC_objects/'+nameobject+'/'+str(nameobject) + '_wavelength_'+str(line)+'.txt', 'r')
data_wavelength = pickle.load(f_wav)
f_wav.close()
# print data_wavelength
signal_to_noise = {}
standard_deviation = {}
f_snr = open('../MCMC_objects/'+nameobject+'/'+str(nameobject) + '_signal_to_noise_'+str(line)+'.txt', 'r')
lines = f_snr.readlines()[:]
for l in lines:
    title = l[:7].strip()
    value = eval(l[7:].split()[0])
    signal_to_noise[title] = value
    standard_deviation[title] = 1./value
    # print title, value, signal_to_noise[title]
f_snr.close()

'''
Cut the wavelength range
'''
wavmin = min(range(len(data_wavelength)), key = lambda j: abs(data_wavelength[j]- w_begin))
wavmax = min(range(len(data_wavelength)), key = lambda j: abs(data_wavelength[j]- w_end))
data_wavelength_mcmc =  data_wavelength[wavmin:wavmax]

data_spectra_mcmc = {}
init_spec_mcmc = {}
for key in data_spectra:
    data_spectra_mcmc[key] = {}
    init_spec_mcmc[key] = {}
    for spectrum in data_spectra[key]:
        if parameters['sigma_clip'] == True and str(line) == 'halpha':
            data_spectra_mcmc[key][spectrum] = sigma_clip(data_spectra[key][spectrum][wavmin:wavmax],
                                            sigma_lower = 50, sigma_upper = parameters['sigma_upper'])
            if emission == True:
                init_spec_mcmc[key][spectrum] = init_spec[key][spectrum][wavmin:wavmax]
        else:
            data_spectra_mcmc[key][spectrum] = data_spectra[key][spectrum][wavmin:wavmax]
            if emission == True:
                init_spec_mcmc[key][spectrum] = init_spec[key][spectrum][wavmin:wavmax]
        # pl.plot(data_wavelength_mcmc, data_spectra_mcmc[key][spectrum])
        # pl.show()
# pl.plot(data_wavelength, data_spectra[40]['580871'])
# pl.show()
# for row in data_spectra:
#     print row

# for wavel in data_wavelength:
#     print 'wavlength %3.5f' % wavel
#
# for titl in signal_to_noise:
#     print '%s \t %3.5f' % (titl, signal_to_noise[titl])
# pl.plot(1,1)
# pl.show()

#################
"""
##### defining the gridpoints on the primary
##### primary treated as a uniform disk, with points relative to the
##### radius of the primary
"""
#################

def disk_grid(r_p, Incl):
    Npoints_primary = parameters['points_primary']
    Point_prim = np.zeros((Npoints_primary,3))

    #### Fibonacci disk grid
    disk_grid = disk_grid_fibonacci.disk_grid_fibonacci(Npoints_primary, r_p, [0,0] )
    Point_prim = np.array([np.array(-disk_grid[:,0]), np.array(-disk_grid[:,1]*np.cos(Incl)), np.array(disk_grid[:,1]*np.sin(Incl))]).T

    #### Polar disk grid
    # point_prim = np.zeros((Npoints_primary,3))
    # for slicespoint_prim in range(prim_Nslices - 1):
    #     rdisk = (2.*(slices + 1.) + 1.)/(2.*prim_Nslices)
    #     point_prim[(1 + slices*prim_Nphi):(1 + (1 + slices)*prim_Nphi)] = np.array([np.array(rdisk*r_p*np.cos(n*2*np.pi/prim_Nphi)), np.array(-rdisk*r_p*np.sin(n*2*np.pi/prim_Nphi)*np.cos(incl)),np.array(rdisk*r_p*np.sin(n*2*np.pi/prim_Nphi)*np.sin(incl))]).T
    return Point_prim, Npoints_primary
#################
"""
### path length and positions in jet
"""
#################

def positions(t, pos_prim, pos_sec, p_prim, alpha, xs, ys, inclination):
    """
    Coordinates of the evolved star (xp, yp, zp), gridpoints along the line of sight through the jet (s), and their coordinates (position, r), and angle in the jet (theta_pos)
    """
    xp = pos_prim[0] + p_prim[0]
    yp = pos_prim[1] + p_prim[1]
    zp = p_prim[2]
    s = np.zeros(Npoints_pathlength)             #path lenght to primary of each point in jet
    position = np.zeros((Npoints_pathlength,3))    #position of each point in jet in 3D
    r = np.zeros((Npoints_pathlength,3))           # position of each point with secondary as origin
    r_unit = np.zeros((Npoints_pathlength,3))
    theta_pos = np.zeros(Npoints_pathlength)
    # sqrt = (2*(ys-yp)*np.sin(incl))**2 - 4*(np.cos(incl)**2*np.tan(alp)**2 - np.sin(incl)**2)*(- (xp-xs)**2 - (yp - ys)**2)
    sqrt = (2*zp*np.cos(inclination)*np.tan(alpha)**2 + 2*(ys - yp)*np.sin(inclination))**2 - 4*(np.cos(inclination)**2*np.tan(alpha)**2 - np.sin(inclination)**2)*(zp**2*np.tan(alpha)**2-(xp-xs)**2-(yp-ys)**2)

    if sqrt < 0:
        s1 = 0
        s2 = 0
        ds = 0

        s = np.zeros(Npoints_pathlength)
    else:

        # s1 = (-1*( 2*(ys - yp)*np.sin(incl)) + (sqrt)**.5)/(2*(np.cos(incl)**2*np.tan(alp)**2 - np.sin(incl)**2))
        # s2 = (-1*( 2*(ys - yp)*np.sin(incl)) - (sqrt)**.5)/(2*(np.cos(incl)**2*np.tan(alp)**2 - np.sin(incl)**2))
        s1 = (-1*( 2*zp*np.cos(inclination)*np.tan(alpha)**2 + 2*(ys - yp)*np.sin(inclination)) + (sqrt)**.5)/(2*(np.cos(inclination)**2*np.tan(alpha)**2 - np.sin(inclination)**2))
        s2 = (-1*( 2*zp*np.cos(inclination)*np.tan(alpha)**2 + 2*(ys - yp)*np.sin(inclination)) - (sqrt)**.5)/(2*(np.cos(inclination)**2*np.tan(alpha)**2 - np.sin(inclination)**2))

        if s1 < -0.000005 and s2 < 0:
            s1 = 0
            s2 = 0
            ds = 0
        elif inclination > alpha:

            s_min = s1
            s_max = s2
            ds = (s_max - s_min)/(Npoints_pathlength-1)
            s = np.arange(s_min, s_max + 0.1*ds, ds)
            position[:,0] += xp
            position[:,1] += yp + s*np.sin(inclination)
            position[:,2] += zp + s*np.cos(inclination)
            r[:,:] = position[:,:] - pos_sec
            r_tot = (r[:,0]**2+r[:,1]**2+r[:,2]**2)**.5
            r_unit[:,0]= r[:,0]/r_tot
            r_unit[:,1]= r[:,1]/r_tot
            r_unit[:,2]= r[:,2]/r_tot
            theta_pos[:] = np.arctan((r[:,0]**2 + r[:,1]**2)**.5/r[:,2])
            # pl.plot(xp,yp, 'o', color = 'k')
            # pl.plot(xs, ys,'o')
            # pl.plot(position[:,0], position[:,1])
            # pl.xlim(-2.,2.)
            # pl.ylim(-2.,2.)
            # pl.show()

        elif inclination < alpha:

            s_min = np.max([s1,s2])
            s_max = 5.*s_min
            ds = (s_max - s_min)/(Npoints_pathlength-1)
            s = np.arange(s_min, s_max + 0.1*ds, ds)
            position[:,0] += xp
            position[:,1] += yp + s*np.sin(inclination)
            position[:,2] += zp + s*np.cos(inclination)
            r[:,:] = position[:,:] - pos_sec
            r_tot = (r[:,0]**2+r[:,1]**2+r[:,2]**2)**.5
            r_unit[:,0]= r[:,0]/r_tot
            r_unit[:,1]= r[:,1]/r_tot
            r_unit[:,2]= r[:,2]/r_tot
            theta_pos[:] = np.arctan((r[:,0]**2 + r[:,1]**2)**.5/r[:,2])
            # pl.plot(xp,yp, 'o', color = 'k')
            # pl.plot(xs, ys,'o')
            # pl.plot(position[:,0], position[:,1])
            # pl.xlim(-2.,2.)
            # pl.ylim(-2.,2.)
            # pl.show()

    return s1, s2, s, ds, position, r, r_unit, theta_pos

def velocity_density(I, POINT, t, theta, pos, R_unit, n_unit, alpha_out, alpha_in, \
                    inclination, v_axis, v_in, v_edge, Orbit_sec):
    """
    Calculate the radial velocity and density at each gridpoint in the jet
    """
    if pos[0,1] > pos[-1,1]:

        # print pos[0,1],pos[-1,1], 'here'
        v_s = np.zeros(Npoints_pathlength)
        rho_s = np.zeros(Npoints_pathlength)

    else:
        v_theta = np.zeros(Npoints_pathlength)
        v_s = np.zeros(Npoints_pathlength)
        rho_s = np.zeros(Npoints_pathlength)

        v_M = (v_in - v_edge)/np.cos(np.pi/2.*alpha_in/alpha_out) + v_edge
        v_theta[np.abs(theta) > alpha_in] = v_edge + (v_M - v_edge)*np.cos(np.pi/2.*theta[np.abs(theta) > alpha_in]/alpha_out)
        v_s[np.abs(theta) > alpha_in] = -v_theta[np.abs(theta) > alpha_in]*np.sum(R_unit[(np.abs(theta) > alpha_in),:]*n_unit, axis = 1)\
         - Orbit_sec*v_prim*np.sin(w*t)
        rho_s[np.abs(theta) > alpha_in] = (v_theta[np.abs(theta) > alpha_in]**.5 * pos[(np.abs(theta) > alpha_in),2]**-2)

        v_theta[np.abs(theta) < alpha_in] = v_edge + (v_M - v_edge)*np.cos(np.pi/2.*theta[np.abs(theta) < alpha_in]/alpha_out)\
        + (v_axis - v_M)*np.cos(np.pi/2.*theta[np.abs(theta) < alpha_in]/alpha_in)
        v_s[np.abs(theta) < alpha_in] = -v_theta[np.abs(theta) < alpha_in]*np.sum(R_unit[(np.abs(theta) < alpha_in),:]*n_unit, axis = 1)\
         - Orbit_sec*v_prim*np.sin(w*t)
        rho_s[np.abs(theta) < alpha_in] = (v_in*.5 * (v_theta[np.abs(theta) < alpha_in]/v_in)**-.5 * pos[(np.abs(theta) < alpha_in),2]**-2)
        if np.isnan(rho_s).any():
            rho_s[np.abs(theta) < alpha_in] = (v_in**.5 * pos[(np.abs(theta) < alpha_in),2]**-2)

    #############

    return v_s, rho_s


#################
"""
### create synthetic dynamic spectra
"""
#################

number_of_bins = parameters['wavelength_bins']        # for image plot
v_reach = parameters['velocity_range']



def model(inclination, alpha_out, alpha_in, mass_prim, mass_sec,
            constant_optdepth, v_axis, v_in, v_edge, r_p, data_spec):
    model_spec = {}
    for key in data_spec:
        model_spec[key] = {}
    t = 0.01*np.array(model_spec.keys())*P
    R_s = 1*mass_prim/mass_sec
    phasenumber = np.array(model_spec.keys())
    point_prim, Npoints_primary = disk_grid(r_p, inclination)    # points on the primary component (assumed to be a uniform disk)
    n_unit = np.array([0,np.sin(inclination),np.cos(inclination)]) # normal vector along the line of sight
    # binsize = v_axis/(0.5*(number_of_bins - 1))  # for image plot
    binsize = v_reach/(0.5*(number_of_bins - 1))
    Orbit_sec = R_p*mass_prim/mass_sec
    # intensity = np.zeros((len(model_spectra.keys()), number_of_bins))
    intensity = np.zeros((100, number_of_bins))
    # density_summed = np.zeros((len(model_spectra.keys()), number_of_bins))
    density_summed = np.zeros((100, number_of_bins))
    for i in range(len(t)):
        ke = pyasl.KeplerEllipse(R_p, P, e = parameters['ecc'], Omega=0.,
                                    i = 0.0, w = omega)
        pos_prim = -ke.xyzPos((t[i] + (T_inf - T0))%P)
        ke_sec = pyasl.KeplerEllipse(R_s, P, e=parameters['ecc'], Omega=0., i = 0.0,
                                    w = omega + 180)
        pos_sec = -ke_sec.xyzPos((t[i] + (T_inf - T0))%P)
        ### RV of phostospheric spectra
        ke_syn = pyasl.KeplerEllipse(R_p, P, e=parameters['ecc'], Omega=0., i = 90, w = omega)
        phaseplot_syn = np.arange(0., 1.001*P, 0.01*P)
        vel_syn = ke_syn.xyzVel((phaseplot_syn + (T_inf - T0))% P)
        velplot_min_syn = (np.max(vel_syn[::,2])*0.5 - np.min(vel_syn[::,2])*0.5)
        vel_syn[::,2] = vel_syn[::,2]/velplot_min_syn
        vel_value_syn = ke_syn.xyzVel((t[i] + (T_inf - T0))%P)/velplot_min_syn
        rvprim_syn = K1*vel_value_syn[2]

        # Coordinates of the companion star
        xs = pos_sec[0]
        ys = pos_sec[1]
        S1 = np.zeros((Npoints_primary, 3))
        S2 = np.zeros((Npoints_primary, 3))
        S = np.zeros((Npoints_primary, Npoints_pathlength))
        dS = np.zeros((Npoints_primary))
        Position = np.zeros((Npoints_primary, Npoints_pathlength, 3))
        R = np.zeros((Npoints_primary, Npoints_pathlength, 3))
        R_unit = np.zeros((Npoints_primary, Npoints_pathlength, 3))
        Theta_pos = np.zeros((Npoints_primary, Npoints_pathlength))
        density = np.zeros((Npoints_primary, Npoints_pathlength))
        radial_velocity = np.zeros((Npoints_primary, Npoints_pathlength))
        v_gradient = np.zeros((Npoints_primary, Npoints_pathlength))
        v_gradient_inv = np.zeros((Npoints_primary, Npoints_pathlength))

        ### 3d plot
        # fig = pl.figure()
        # ax = fig.gca(projection='3d')
        ###
        # v_range = np.arange(-v_axis, v_axis + 0.1*binsize, binsize)
        v_range = np.arange(-v_reach, v_reach + 0.1*binsize, binsize)
        tau_los = np.zeros((len(v_range), Npoints_primary))
        '''
         create photosperic spectra (with or without emission component)
        '''
        if emission == False:
            wavelength_range = central_wavelength*(1. + v_range / 299792.458)
            wavelength_range_synth = wavelength_range*(1. - rvprim_syn / 299792.458)
            init_spec = np.full((Npoints_primary, len(v_range)), interp(wavelength_range_synth)).T

        for point in range(Npoints_primary):
            S1[point,:], S2[point,:], S[point,:], dS[point], Position[point,:,:],\
                R[point,:,:], R_unit[point,:,:], Theta_pos[point,:] =\
                 positions(t[i], pos_prim, pos_sec, point_prim[point,:],\
                                                alpha_out, xs, ys, inclination)
            radial_velocity[point,:], density[point,:] = \
                    velocity_density(i, point, t[i], Theta_pos[point,:], \
                    Position[point,:,:], R_unit[point,:,:], n_unit, alpha_out, alpha_in, \
                    inclination, v_axis, v_in, v_edge, Orbit_sec)
            opacity = np.std(radial_velocity[point,:])**-1
            density[density > 1.e30] = 0     # remove -inf densities
            if opacity > 1.e30: opacity = 0
            v_gradient[point, 1:-1] = np.abs((radial_velocity[point, 2:] - radial_velocity[point,0:-2]) / (2*dS[point]))
            v_gradient[point, 0] = np.abs((radial_velocity[point,1] - radial_velocity[point, 0]) / dS[point])
            v_gradient[point, -1] = np.abs((radial_velocity[point,-1] - radial_velocity[point, -2]) / dS[point])

            v_gradient[np.isnan(v_gradient)] = 0
            v_gradient_inv[point,:] = v_gradient[point,:]**-1
            v_gradient_inv[v_gradient_inv > 1.e30] = 0
            # pl.subplot(211)
            # pl.plot(S[point,:], radial_velocity[point,:],'k')
            # pl.subplot(212)
            # pl.plot(S[point,:], v_gradient[point,:], 'r')
            # print v_gradient[point,:],v_gradient_inv[point,:]
            # pl.show()
            # pl.plot(radial_velocity[i,:], -density[i,:]*dS)
            # print 'rv', 'density',radial_velocity[i,:], -density[i,:]*dS
            # pl.xlim((-600,600))
            # pl.show()

            for vr in range(len(radial_velocity[point,:])):
                index = int(round((radial_velocity[point,vr] + v_reach)/(2. * v_reach) * number_of_bins))
                tau_los[index, point] += v_gradient_inv[point, vr] * density[point, vr] * dS[point]
                density_summed[phasenumber[i], index] -= density[point,vr]
                # pl.plot(v_range, density_summed[i,:])
                # pl.show()
                ### 3d plot
                # ax.plot(Position[point, :, 0], Position[point, :, 1], Position[point, :, 2],'r+')

            ### starting with a normalised spectrum (non-photospheric)
            # intensity_los = Npoints_primary**-1.*np.exp(-10**constant_optdepth*tau_los)
            ### starting with the photosperic component:
            '''
            emission == True:
            Calculating the absorption for each LOS, starting with an intensity
            normalised to one (without photosperic absorption component).
            emission == False:
            Calculating the absorption for each LOS, starting with the photosperic component.
            The photosperic component is eventually subtracted (So we are only left with
            the absorption component):
            '''
            if emission == True:
                intensity_los = Npoints_primary**-1.*np.exp(-10**constant_optdepth*tau_los)
            else:
                intensity_los = init_spec*Npoints_primary**-1.*np.exp(-10**constant_optdepth*tau_los)
                intensity_los = intensity_los + (1 - init_spec)*Npoints_primary**-1

            ###
            # pl.plot(np.arange(0,len(intensity_los[:,point]),1),intensity_los[:,point])
            # pl.show()
            intensity[phasenumber[i],:] = np.sum(intensity_los, axis = 1)
            model_spec[phasenumber[i]] = intensity[phasenumber[i],:]
            # pl.plot(radial_velocity[0,:], radial_velocity[0,:])
            # pl.show()
        ### 3d plot
        # ax.plot(pos_prim[0] + point_prim[:,0], pos_prim[1] + point_prim[:,1], pos_prim[2] + point_prim[:,2], '+')
        # cone_z = np.arange(0, 3, 0.2)
        # cone_theta = np.arange(0, 2 * np.pi + np.pi / 50, np.pi / 50)
        # ax.scatter(xs,ys,0,'g+')
        # for zval in cone_z:
        #     cone_x = xs + np.tan(alp) * zval * np.array([np.cos(cone_q) for cone_q in cone_theta])
        #     cone_y = ys + np.tan(alp) * zval * np.array([np.sin(cone_q) for cone_q in cone_theta])
        #     ax.plot(cone_x, cone_y, zval, 'b+')
        # # X, Y = np.meshgrid(cone_x, cone_y)
        # # Z = ((X - pos_sec[0])**2 + (Y - pos_sec[2])**2)**.5/np.tan(alp)
        # # ax.plot_surface(X, Y, Z)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # ax.axis('equal')
        # # print r_p, np.shape(point_prim)
        # pl.show()
        #
        # for inten in range(int(number_time_steps)):
        #     pl.plot(v_range, intensity[inten,:])
        #     pl.show()
    return intensity, model_spec, binsize

def lnprior(Theta, mass_prim, mass_sec):
    Incl, Alph_out, Alph_in, Constant_optdepth, V_axis, V_in, V_edge, r_Prim = Theta
    # print Theta, 'heeeere'
    q = mass_prim/mass_sec
    ## inner jet starts at a radius larger than twice the approximated companion radius (using Demircan 1991)
    # R_companion = 1.01*mass_sec**0.724*0.00465/parameters['asini']*np.sin(Incl)
    ## Eggleton 1983: Roche radius of the primary component in terms of a1
    roche_radius_p = 0.49*q**(2./3) / (0.6*q**(2./3) + np.log(1 + q**(1./3)))*(1 + q)
    # roche_radius_s = 0.49*q**(-2./3) / (0.6*q**(-2./3) + np.log(1 + q**(-1./3)))*(1 + q)

    ## test the binary configuration
    # circ = np.arange(0,2*np.pi+0.1,0.1)
    # fig, ax = pl.subplots()
    # ax.plot(roche_radius_p*np.cos(circ)-1,roche_radius_p*np.sin(circ),'k')
    # ax.plot(roche_radius_s*np.cos(circ)+q, roche_radius_s*np.sin(circ),'k')
    # circle1= pl.Circle((-1,0), r_Prim, color='b')
    #
    # ax.add_artist(circle1)
    # ax.plot(-1,0,'o',color = 'r')
    # ax.plot(q,0,'o',color = 'r')
    # circle2= pl.Circle((q,0), R_companion, color='b')
    # ax.add_artist(circle2)
    # ax.set_xlim((-2,2))
    # ax.set_ylim((-2,2))
    # pl.show()

    if parameters['inc_min'] < Incl < parameters['inc_max']\
    and Alph_in < Alph_out < parameters['alp_max'] \
    and parameters['alp_min'] < Alph_in\
    and V_edge < V_in < V_axis\
    and parameters['v_axis_min'] < V_axis < parameters['v_axis_max']\
    and parameters['v_edge_min'] < V_edge < parameters['v_edge_max']\
    and V_axis*np.cos(Incl) < 2000.\
    and parameters['r_Prim_min'] < r_Prim < roche_radius_p\
    and parameters['constant_min'] < Constant_optdepth < parameters['constant_max']:
        return 0.0
    return -np.inf

# def chi_squared(Theta, inclination, alpha, mass_prim, mass_sec, power, constant_optdepth, v_axis, v_edge, data_spectra):
def lnlike(Theta, mass_prim, mass_sec, data_spectra):
    # inclination, alpha, mass_sec, power, constant_optdepth, v_axis, v_edge, r_p = Theta
    inclination, alpha_out, alpha_in, constant_optdepth, v_axis, v_in, v_edge, r_p = Theta
    intensity, model_spectra, binsize = model(inclination, alpha_out, alpha_in, mass_prim, mass_sec, constant_optdepth, v_axis, v_in, v_edge, r_p, data_spectra)
    v_range = np.arange(-v_reach, v_reach + 0.1*binsize, binsize)
    wavelength_range = central_wavelength*(1. + v_range / 299792.458)
    # print wavelength_range, 'wavelength_range'
    # print data_wavelength, 'data_wavelength'
    chi2 = 0
    lnlikelyhood = 0
    imagenumber = 0
    images = []
    for key in model_spectra:
        interp_model = interpolate.interp1d(wavelength_range, model_spectra[key])
        for spectrum in data_spectra_mcmc[key]:
            if emission == True:
                chi2_spectrum = (interp_model(data_wavelength_mcmc)\
                                    *init_spec_mcmc[key][spectrum] \
                                    - data_spectra_mcmc[key][spectrum])**2\
                                    /(standard_deviation[spectrum]**2)\
                                    + 2.*n.pi*np.log(standard_deviation[spectrum])
            else:
                chi2_spectrum = (interp_model(data_wavelength_mcmc) \
                                - data_spectra_mcmc[key][spectrum])**2/\
                                (standard_deviation[spectrum]**2)\
                                + 2.*n.pi*np.log(standard_deviation[spectrum])
            chi2 += np.sum(chi2_spectrum)
            # lnlikelyhood_spectrum = -0.5*(chi2_spectrum + np.log(2.*np.pi*standard_deviation[spectrum]**2))
            lnlikelyhood_spectrum = -0.5*(chi2_spectrum)
            lnlikelyhood += np.sum(lnlikelyhood_spectrum)

    ############
    ####movie
    ############
    #         import imageio
    #         pl.plot(data_wavelength_mcmc, interp_model(data_wavelength_mcmc), 'b')
    #         pl.plot(data_wavelength_mcmc, data_spectra_mcmc[key][spectrum])
    #         # pl.plot(data_wavelength_mcmc, interp_model(data_wavelength_mcmc) - data_spectra_mcmc[key][spectrum], 'k--')
    #         pl.ylabel('spectrum')
    #         pl.xlabel('wavelength (angstrom)')
    #         pl.ylim([0.,2.])
    #         im_phase = float(key)
    #         im_phase /= 100.
    #         pl.title('phase = %2.2f' % im_phase)
    #         pl.tight_layout()
    #         fname = '_tmp%05d.png'%imagenumber
    #
    #         pl.savefig('../MCMC_objects/'+str(nameobject)+'/movie/'+nameobject+fname)
    #         images.append(imageio.imread('../MCMC_objects/'+str(nameobject)+'/movie/'+nameobject+fname))
    #
    #         pl.clf()
    #         imagenumber += 1
	# # os.system('../MCMC_objects/'+str(nameobject)+'/movie/movie.gif')
    # imageio.mimsave('../MCMC_objects/'+str(nameobject)+'/movie/movie.gif', images)
    ############
    #### end movie
    ############

    print 'chi2', chi2, 'lnlikelyhood', lnlikelyhood
    return lnlikelyhood, intensity

def probab(Theta, mass_prim, data_spectra):
    # print 'theta:', Theta
    mass_sec = calc_mass_sec(mass_prim, Theta[0])
    lp = lnprior(Theta, mass_prim, mass_sec)
    if not np.isfinite(lp):
        return -np.inf
    lnlikel, intensity = lnlike(Theta, mass_prim, mass_sec, data_spectra)
    return lp + lnlikel

def calc_mass_sec(mp, inc):
    a0 = -fm*mp**2/(np.sin(inc)**3)
    a1 = -2*fm*mp/(np.sin(inc)**3)
    a2 = -fm/(np.sin(inc)**3)
    Q = (3.*a1 - a2**2)/9.
    R = (9.*a2*a1 - 27*a0 - 2.*a2**3)/54.
    D = Q**3 + R**2
    S = (R + D**.5)**(1./3.)
    T = (R - D**.5)**(1./3.)
    ms = -1./3.*a2 + (S + T)
    return ms

#################
"""
### main
"""
#################

import emcee
from emcee.utils import MPIPool

ndim = 8
nwalkers = parameters['nwalkers']

niter = parameters['niter']
#pos_min = np.array([55.*np.pi/180, 35.*np.pi/180, 10.*np.pi/180,\
#        0, 500, 350, 1, 0.05])
#pos_max = np.array([80.*np.pi/180, 78.*np.pi/180, 70.*np.pi/180.,\
#        9, 1400, 750, 150, 1.]) 
pos_min = np.array([parameters['inc_min'], parameters['alp_min'], parameters['alp_min'],\
             parameters['constant_min'],
             parameters['v_axis_min'], parameters['v_edge_min'], parameters['v_edge_min'], parameters['r_Prim_min']])
pos_max = np.array([parameters['inc_max'], parameters['alp_max'], parameters['alp_max'],\
             parameters['constant_max'],\
             parameters['v_axis_max'], parameters['v_axis_max'], parameters['v_edge_max'], parameters['r_Prim_max']])
pos_size = pos_max - pos_min


if parameters['previous_walkers_nc'] == True:
    '''
    Start from previous chain if available
    '''
    chain_file = open('../MCMC_objects/'+str(nameobject)+'/results_nested_cos/MCMC_chain_'+str(nameobject)+'_'+str(line)+'_'\
                +str(w_begin)+'_'+str(w_end)+'_walk'+str(nwalkers)+'_iter'+str(niter)+'.dat', "r")
    walkers_lines = chain_file.readlines()[-nwalkers:]
    pos_list = [np.array(walker_line.split()) for walker_line in walkers_lines]
    pos0 = np.zeros((nwalkers,ndim))
    for posit_walk in range(nwalkers):
        for posit_dim in range(ndim):
            pos0[posit_walk,posit_dim] = float(pos_list[posit_walk][posit_dim])
else:
    '''
    Otherwise start a new chain
    '''
    poss0 = [pos_min + pos_size*np.random.rand(ndim) for i in range(nwalkers)]
    pos0 = np.zeros((nwalkers,ndim))
    for posit_walk in range(nwalkers):
        for posit_dim in range(ndim):
            pos0[posit_walk,posit_dim] = float(poss0[posit_walk][posit_dim])


for row in range(nwalkers):
    '''
    Check if all priors are in the accepted range
    '''
    mass_sec = calc_mass_sec(m_p, pos0[row,0])
    while lnprior(pos0[row,:], m_p, mass_sec) == -np.inf:
        '''
        Change those priors with lnprob = -inf
        '''
        pos0[row,:] = pos_min + pos_size*np.random.rand(ndim)
        mass_sec = calc_mass_sec(m_p, pos0[row,0])

pool = MPIPool()
if not pool.is_master():
    pool.wait()
    sys.exit(0)

sampler = emcee.EnsembleSampler(nwalkers, ndim, probab, args = (m_p, data_spectra), pool=pool)

#time0 = time.time()

######### begin save chain
f = open('../MCMC_objects/'+str(nameobject)+'/results_nested_cos/MCMC_chain_'+str(nameobject)+'_'+str(line)+'_'\
                +str(w_begin)+'_'+str(w_end)+'_walk'+str(nwalkers)+'_iter'+str(niter)+'.dat', "a")
f.close()

for result in sampler.sample(pos0, iterations=niter, storechain=False):
    position = result[0]
    f = open('../MCMC_objects/'+str(nameobject)+'/results_nested_cos/MCMC_chain_'+str(nameobject)+'_'+str(line)+'_'\
                +str(w_begin)+'_'+str(w_end)+'_walk'+str(nwalkers)+'_iter'+str(niter)+'.dat', "a")
    np.savetxt(f, np.array(position))
    f.close()
######### end save chain

# pos, prob, state = sampler.run_mcmc(pos0, niter)
time1 = time.time()
print 'this took %4f seconds' % (time1 - time0)

# samples = sampler.flatchain
# samples.shape
# f = open('../MCMC_objects/'+str(nameobject)+'/results_nested_cos/MCMC_'+str(nameobject)+'_'+str(line)+'_'+str(w_begin)+'_'+str(w_end)+\
#             '_walk'+str(nwalkers)+'_iter'+str(niter)+'_abs.txt', 'w')
# pickle.dump(samples,f)
# f.close()
# import corner




# Chi_squared = chi_squared([9.60864951e-01,4.64895561e-01,4.40932747e+00,7.81017189e+00,
#                             7.14033317e+02,1.93325863e+02,2.53937592e+00], m_p, data_spectra)
# chi2 = chi_squared([incl, alp, m_s, p, constant, v_0, v_alph, r_p], m_p, data_spectra)
# chi2 = chi_squared([incl, alp, m_s, constant, v_0, v_alph, r_p], m_p, data_spectra)
# incl, alp, constant, v_0, v_alph, r_p = [1.047,0.82, 9., 800., 100., 0.2]
# incl, alp_out, alp_in, constant, v_0, v_alph, r_p = \
#         [60*np.pi/180.,47*np.pi/180., 40*np.pi/180., 11., 850., 85., 0.5]
# incl, alp_out, alp_in, constant, v_0, v_in, v_alph, r_p = [1.07,0.9,0.7, 6.9, 900., 150, 55, 0.84] # good!!
# incl, alp, constant, v_0, v_alph, r_p = [1.162,1.0, 7.6, 900., 55, 0.95] #
# incl, alp, constant, v_0, v_alph, r_p = [1.07,0.97, 7.2, 1050., 55, 0.84] #good
# incl, alp_out, alp_in, constant, v_0, v_alph, r_p = [  1.28, 1.2, 0.21, 12, 750, 135, 0.6]\
#
# loglikelyhood, intensity = \
#         lnlike([incl, alp_out, alp_in, constant, v_0, v_in, v_alph, r_p], m_p, calc_mass_sec(m_p, incl), data_spectra)


#################
"""
### date & time
"""
#################

now = time.strftime("%c")
## date and time representation
with open('../date_run.txt', "a") as datefile:
    datefile.write('%10s \t %10s \t nestedcos ended \t %10s \n' % (nameobject, line, time.strftime("%c")))

#################
"""
### remove NaN
"""
#################

# where_are_NaNs = np.isnan(intensity)
# intensity[where_are_NaNs] = 1.



#################
"""
### create synthetic dynamic spectra
"""
#################
# fig, ax = pl.subplots()
# # ax.imshow(density_summed, extent=(-v_0, v_0, 1, 0),
#                                     # aspect='normal', cmap=pl.cm.gnuplot,  vmin=0.04*np.min(density_summed), vmax=0)
# # print v_0
# pl.imshow(intensity, extent=(-v_reach, v_reach, 1, 0),
#                                     aspect='normal', cmap=pl.cm.gnuplot,  vmin=0.0, vmax=1)
# title = " Period = "+str(P)
# cb=pl.colorbar()
# cb.set_label('Flux', fontsize=18)
# for ticks in cb.ax.get_yticklabels():
#      ticks.set_fontsize(14)
# fig.autofmt_xdate()
#
# pl.show()
########################
"""
plots the RV motion of the primary and secondary component
"""
########################

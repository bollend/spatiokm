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
"   Plots the geometry of the binary system and the structure of the jet
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


"""
===========================
Calculating binary geometry
===========================
"""

# Constants in cgs
Rsol_cgs = 6.957e10
G_cgs    = 6.674e-8
Msol_cgs = 2e33
R_cgs    = 8.314e7
k_b_cgs  = 1.38e-16
mol_cgs  = 6.022e23
mu_cgs   = 0.8
rho_cgs  = 1e-6

# constants in si
Rsol        = 6.957e8
Msol        = 2e30
Lsol        = 3.827e+26
R           = 8.314
G           = 6.674e-11
k_b         = 1.380e-23
sigma       = 5.67e-8
mass_proton = 1.67e-27
mol         = 6.022e23
mu          = mu_cgs * 1e-3
rho         = 1e-6

sec_in_yr = 60*60*24*365
Rsol_in_au = 215.032
e = np.exp(1)

##### binary #####

def roche_r(q):
    return 0.49*q**(2/3) / (0.6 * q**(2/3) + np.log( 1 + q**(1/3)))


a_p               = parameters['BINARY']['primary_asini'] / np.sin(pars_model['inclination']) # in AU
M_p               = 0.6
M_s               = geometry_binary.calc_mass_sec(M_p, pars_model['inclination'], parameters['BINARY']['mass_function']) # in AU
mass_ratio        = M_p/M_s
a                 = a_p * ( 1 + mass_ratio )
a_s               = a - a_p
roche_radius_p_a  = roche_r(mass_ratio)
roche_radius_s_a  = roche_r(mass_ratio**-1)
roche_radius_p_AU = roche_radius_p_a * a
roche_radius_s_AU = roche_radius_s_a * a
inclination       = pars_model['inclination']

print(mass_ratio, M_s)
print(roche_radius_p_a, roche_radius_s_a)
print(roche_radius_p_AU, roche_radius_s_AU, a)
##### primary #####

radius_p = pars_model['primary_radius']


##### jet #####
jet_angle                 = pars_model['jet_angle']# radians
velocity_centre           = pars_model['velocity_max']         # km/s
velocity_edge             = pars_model['velocity_edge']           # km/s
jet_cavity_angle          = pars_model['jet_cavity_angle']
jet_tilt                  = pars_model['jet_tilt']
exp_velocity              = pars_model['exp_velocity']

jet_type                  = parameters['OTHER']['jet_type']
jet_velocity_max          = 1000 * velocity_centre                      # m/s
jet_velocity_min          = 1000 * velocity_edge                        # m/s
jet_density_max           = 10**(parameters['OTHER']['bestfit_p'])                                      # m^-3
jet_density_max_kg_per_m3 = jet_density_max                             # kg/m^3
power_velocity            = parameters['OTHER']['power_velocity']

if jet_type!='sdisk_wind' and jet_type!='x_wind':
    power_density             = pars_model['power_density']

if jet_type=='sdisk_wind':
    jet_angle_inner   = pars_model['jet_angle_inner']
    scaling_par       = pars_model['scaling_par']
    power_density_in  = pars_model['power_density_in']
    power_density_out = pars_model['power_density_out']

if jet_type=='x_wind':
    jet_angle_inner   = pars_model['jet_angle_inner']
    power_density_in  = pars_model['power_density_in']
    power_density_out = pars_model['power_density_out']





"""
========
Plotting
========
"""


import matplotlib as mpl


ratio = 1


## Set limits
tsmin = 4.028
tsmax = 184.78
y_min  = 0
y_max  = 110

## Fontsize
fts  = ratio*20
lbsz = ratio*10
lbs  = ratio*16
lgds = ratio*12

## Plots here
plt.rc('font', weight='bold', size=fts)
plt.rc('xtick', labelsize=lbs)
mpl.rc('ytick', labelsize=lbs)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}\boldmath'] #for \text command
plt.rcParams['axes.linewidth']      = 1.
plt.rc('text', usetex=True)

## Figure
# fig, axes = plt.subplots(1, 4, figsize=(ratio*210/30,ratio*0.8*297/30) )

## Ticks
# minor_yticks = phases[1::2]
# major_yticks = phases[0::2]

# Color
# num_plots = len(spectra)
colormap = plt.cm.Blues


fig, axes = plt.subplots(2,1, figsize=(12,12))




# ##### jet angle #####
# axes[0].plot([-5*a*np.sin(jet_angle),5*a*np.sin(jet_angle)]+a_s,[-5*a*np.cos(jet_angle),5*a*np.cos(jet_angle)], color='gray')
# axes[0].plot([5*a*np.sin(jet_angle),-5*a*np.sin(jet_angle)]+a_s,[-5*a*np.cos(jet_angle),5*a*np.cos(jet_angle)], color='gray')

x_length = 5*a*np.sin(jet_angle)
##### jet inner angle #####
if jet_angle_inner:
    axes[0].plot([-5*a*np.sin(jet_angle_inner),5*a*np.sin(jet_angle_inner)]+a_s,[-5*a*np.cos(jet_angle_inner),5*a*np.cos(jet_angle_inner)], color='white', ls=':', lw=1.5)
    axes[0].plot([5*a*np.sin(jet_angle_inner),-5*a*np.sin(jet_angle_inner)]+a_s,[-5*a*np.cos(jet_angle_inner),5*a*np.cos(jet_angle_inner)], color='white', ls=':', lw=1.5)

# ##### jet cavity angle #####
# axes[0].plot([-5*a*np.sin(jet_cavity_angle),5*a*np.sin(jet_cavity_angle)]+a_s,[-5*a*np.cos(jet_cavity_angle),5*a*np.cos(jet_cavity_angle)], color='lightgray')
# axes[0].plot([5*a*np.sin(jet_cavity_angle),-5*a*np.sin(jet_cavity_angle)]+a_s,[-5*a*np.cos(jet_cavity_angle),5*a*np.cos(jet_cavity_angle)], color='lightgray')

# ##### filling between the jet regions #####
#
# axes[0].fill_between([-5*a*np.sin(jet_angle),5*a*np.sin(jet_angle)]+a_s, [-5*a*np.cos(jet_angle_inner),5*a*np.cos(jet_angle_inner)], [-x_length/np.tan(jet_cavity_angle),x_length/np.tan(jet_cavity_angle)], color='lightgray')
# axes[0].fill_between([-5*a*np.sin(jet_angle),5*a*np.sin(jet_angle)]+a_s, [x_length/np.tan(jet_cavity_angle),-x_length/np.tan(jet_cavity_angle)], [5*a*np.cos(jet_angle_inner),-5*a*np.cos(jet_angle_inner)], color='lightgray')
#
# axes[0].fill_between([-5*a*np.sin(jet_angle),5*a*np.sin(jet_angle)]+a_s, [-5*a*np.cos(jet_angle),5*a*np.cos(jet_angle)], [-x_length/np.tan(jet_angle_inner),x_length/np.tan(jet_angle_inner)], color='gray')
# axes[0].fill_between([-5*a*np.sin(jet_angle),5*a*np.sin(jet_angle)]+a_s, [x_length/np.tan(jet_angle_inner),-x_length/np.tan(jet_angle_inner)], [5*a*np.cos(jet_angle),-5*a*np.cos(jet_angle)], color='gray')





axes[0].arrow(-a_p, 0, a*np.sin(inclination), a*np.cos(inclination), color='k', head_width=0.03, head_length=0.06, zorder=40)
primary = plt.Circle((-a_p, 0), radius_p, color='darkorange', zorder=49)
# primary_roche = plt.Circle((-a_p, 0), roche_radius_p_AU, color='grey', ls='--', fill=False)
axes[0].add_artist(primary)
# axes[0].add_artist(primary_roche)
axes[0].scatter([a_s],[0], marker="*", s=400, color='darkorange', zorder=50)
axes[0].scatter([0],[0], marker="x", s=50, color='red')



axes[0].set_xlim(-1,1)
axes[0].set_ylim(-0.501,0.501)
# axes[0].set_xlim(-1.5*a, 1.5*a)
# axes[0].set_ylim(-0.5*a,a)
axes[0].set_xlabel('X(AU)')
axes[0].set_ylabel('Z(AU)')
axes[0].minorticks_on()
axes[0].xaxis.set_ticks_position('both')
axes[0].yaxis.set_ticks_position('both')
axes[0].set_aspect('equal')


##### BEGIN PLOT JET #####
import matplotlib.colors as colors
p_v = pars_model['exp_velocity']
power = 2
p_rho_out = pars_model['power_density_out']
p_rho_in = pars_model['power_density_in']

theta = np.arange(0.,jet_angle + .5, .01)
Npoints_pathlength = len(theta)
jet_tilt = 0

def density(x, y):
    rho = np.zeros(np.shape(x))

    for row in range(len(x[:,0])):
        for column in range(len(x[0,:])):
            theta_element = np.arctan(x[row,column]/y[row,column])
            theta_element_tilt = np.arctan(x[row,column]/y[row,column]) + jet_tilt
            if x[row,column] == 0:
                rho[row,column] = 0
            elif np.abs(theta_element_tilt) < jet_cavity_angle:
                rho[row,column] = 0
            elif np.abs(theta_element_tilt) <  jet_angle_inner:
                rho[row,column] = (np.abs(theta_element_tilt)/(jet_angle_inner))**p_rho_in \
                * np.dot([x[row,column],y[row,column]], [-np.sin(jet_tilt), np.cos(jet_tilt)])**-2
            elif np.abs(theta_element_tilt) <  jet_angle:
                rho[row,column] = (np.abs(theta_element_tilt)/(jet_angle_inner))**p_rho_out \
                * np.dot([x[row,column],y[row,column]], [-np.sin(jet_tilt), np.cos(jet_tilt)])**-2
            else:
                rho[row,column] = 0
    return np.log10(rho)

def velocity(x, y):
    velocity_x = np.zeros(np.shape(x))
    velocity_y = np.zeros(np.shape(x))

    e_exponent = np.exp(p_v)


#     print(np.shape(x))
    for row in range(len(x[:,0])):
        for column in range(len(x[0,:])):
            theta_element = np.arctan(x[row,column]/y[row,column])
            theta_element_tilt = np.arctan(x[row,column]/y[row,column]) + jet_tilt
            if theta_element_tilt == 0:
                velocity_x[row,column] = 0
                velocity_y[row,column] = 0
            elif np.abs(theta_element_tilt) < jet_cavity_angle:
                v_theta = 0
                velocity_x[row,column] = 0
                velocity_y[row,column] = 0
            elif np.abs(theta_element_tilt) < jet_angle:
                exp_angles = ( ( np.abs(theta_element_tilt) - jet_cavity_angle ) / (jet_angle - jet_cavity_angle) )**power
                factor = ( e_exponent**-(exp_angles) - e_exponent**-1 ) / ( 1 - e_exponent**-1 )
                v_theta = velocity_edge + (velocity_centre - velocity_edge)**factor
                velocity_x[row,column] = v_theta * np.sin(theta_element)
                velocity_y[row,column] = v_theta * np.cos(theta_element)
#                 print(v_theta)
            else:
                velocity_x[row,column] = 0
                velocity_y[row,column] = 0

    return velocity_x, velocity_y


dimx = (-2,1)
dimy = (-0.5,0.5)
x = np.linspace(dimx[0], dimx[1], 1000)
y = np.linspace(dimy[0], dimy[1], 1000)

X, Y = np.meshgrid(x + a_s, y)
Z = density(X, Y)


axes[0].pcolormesh(X+a_s, Y, Z, vmin=-5, vmax = 1, cmap='Blues')

axes[0].plot([a_s+3*np.tan(jet_tilt), a_s-3*np.tan(jet_tilt)], [-3, 3], color='k', ls='--')

# plt.colorbar(pad=0.01, aspect=40, label='Scaled density')
# plt.clim(-4, 0);
# plt.show()


Xv = np.linspace(dimx[0], dimx[1], 40)
Yv = np.linspace(dimy[0], dimy[1], 40)
U, V = np.meshgrid(Xv, Yv)

Uv, Vv = velocity(U, V)

mask = np.logical_or(Uv != 0,Vv !=0)
U, V = U[mask], V[mask]
Uv, Vv = Uv[mask], Vv[mask]

##### END PLOT JET #####


##### BEGIN IMBEDDED JET PLOT #####


# These are in unitless percentages of the figure size. (0,0 is bottom left)

left, bottom, width, height = [0.23, 0.65, 0.20, 0.20]

# left, bottom, width, height = [0.685, 0.65, 0.24, 0.24] #BD
ax2 = fig.add_axes([left, bottom, width, height])
ax2.axes.get_xaxis().set_ticks([])
ax2.axes.get_yaxis().set_ticks([])
ax2.axis('off')
import matplotlib.image as mpimg
img=mpimg.imread('iras_densityvelocity_log.png')
ax2.imshow(img)



# jet_tilt = pars_model['jet_tilt']
# dimx = (-2,2)
# dimy = (0,3)
# x = np.linspace(dimx[0], dimx[1], 1000)
# y = np.linspace(dimy[0], dimy[1], 1000)
#
# X, Y = np.meshgrid(x, y)
# Z = density(X, Y)
#
#
# ax2.plot([0,-3*np.tan(jet_tilt)], [0, 3], color='k', ls='--')
#
# jetmesh = ax2.pcolormesh(X, Y, Z, vmin=-5, vmax = 1, cmap='Blues')
#
# cb = plt.colorbar(jetmesh, pad=0.01, aspect=40, label='Log scaled density')
# cb.set_clim(-5, 1);
# ax2.axis('equal')
#
# ax2.set_xlabel('Y(AU)')
# ax2.set_ylabel('Z(AU)')
# ax2.minorticks_on()


##### END IMBEDDED JET PLOT #####

##### XY plot ######
primary = plt.Circle((-a_p, 0), radius_p, color='darkorange')
primary_roche = plt.Circle((-a_p, 0), roche_radius_p_AU, color='grey', ls='--', fill=False)
secondary_roche = plt.Circle((a_s, 0), roche_radius_s_AU, color='grey', ls='--', fill=False)
axes[1].add_artist(primary)
axes[1].add_artist(primary_roche)
axes[1].add_artist(secondary_roche)
axes[1].scatter([a_s],[0], marker="*", s=400, color='darkorange', zorder=50)
axes[1].scatter([0],[0], marker="x", s=50, color='red')

axes[1].set_xlim(-1,1)
axes[1].set_ylim(-0.501,0.501)
# axes[0].set_xlim(-1.5*a, 1.5*a)
# axes[0].set_ylim(-0.5*a,a)

axes[1].set_xlabel('X(AU)')
axes[1].set_ylabel('Y(AU)')
axes[1].minorticks_on()
axes[1].xaxis.set_ticks_position('both')
axes[1].yaxis.set_ticks_position('both')

axes[1].set_aspect('equal')

##### Roche potential

def roche_potential(x,y):
    return 2 /( 1 + mass_ratio**-1) * (x**2+y**2)**-0.5\
     + 2 * mass_ratio**-1 / ( 1 + mass_ratio**-1) * ( (x-1)**2+y**2 )**-0.5\
      + ( x - mass_ratio**-1/(1+mass_ratio**-1) )**2 + y**2


x = np.linspace(-0.75,1.5,200)
y = np.linspace(-1,1,200)

X,Y = np.meshgrid(x,y)
Z = roche_potential(X,Y)
print(Z)
##### BD: 3.962  IRAS:
# axes[0].contour(X*a-a_p, Y*a, Z,[-2,3.962], cmap='RdGy')
axes[1].contour(X*a-a_p, Y*a, Z,[-2,3.99], cmap='RdGy')

# axes[1].contour(X, Y, Z,[-2,3.69], cmap='RdGy')

plt.subplots_adjust(left=0.11, bottom=0.09, right=0.96, top=0.96, wspace=0.2, hspace=0.215)

plt.savefig('geometry.png', dpi=500)
plt.show()

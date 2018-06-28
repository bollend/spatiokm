import numpy as np

def T0_to_IC(omega, ecc, P, T0):
    '''
    Set T0 to time of inferior conjunction by shifting from
    time of periastron passage.

    Parameters
    ----------
    omega : float
        Longitude of the ascending node [rad]
    ecc : float
        Orbital eccentricity (0-1)
    P : float
        Orbital period of the binary system
    T0 : float
        Time of periastron passage

    Returns
    -------
    Time of inferior conjunction : float
        The time when the primary component is closest to the observer
    '''
def positions(position_primary, position_secondary,\
             point_primary, jet_angle, inclination):
    """
    Determines the coordinates of the gridpoints along
    the line-of-sight through the jet and their angle in the jet

    Parameters
    ----------
    position_primary : numpy array
        The (x,y,z)-coordinates of the primary in AU
    position_secondary : numpy array
        The (x,y,z)-coordinates of the secondary in AU
    point_primary : numpy array
        The (x,y,z)-coordinates of the point on the primary in AU
        relative to the centre of the primary
    jet_angle : float
        The half opening angle of the jet in radians
    inclination : float
        The inclination angle of the binary system in radians

    Returns
    -------
    jet_entry : float
        value of line-of-sight parameter 's' for which the line-of-sight
        enters the jet
    jet_exit : float
        value of line-of-sight parameter 's' for which the line-of-sight
        leaves the jet
    jet_los_gridpoints : numpy array
        values of line-of-sight parameter 's' for which the line-of-sight
        goes through the jet
    delta_s : float
        difference between two consecutive values of 's'
        (delta_s = s_n+1 - s_n)

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

        if s1 < 0 and s2 < 0:
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

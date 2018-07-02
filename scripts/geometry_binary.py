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
    nu_inf = (90. + 1.*omega) * np.pi / 180.
    E_inf = np.arccos((ecc + np.cos(nu_inf)) / (1 + ecc * np.cos(nu_inf)))
    M_inf = E_inf - ecc * np.sin(E_inf)
    n_inf = 2.*np.pi / P

    if nu_inf < np.pi:
        return T0 - M_inf / n_inf
    else:
        return T0 - P + M_inf / n_inf

def disk_grid(radius_primary, inclination, number_of_gridpoints):
    import disk_grid_fibonacci
    '''
    Create a Fibonacci grid of N gridpoints to represent the
    evolved star with a radius R, for a certain inclination.

    Parameters
    ----------
    radius_primary : float
        Stellar radius of the evolved star in units of the semi-major axis
    inclination : float
        Inclination of the orbital plane
    number_of_gridpoints : integer
        The number of gridpoints on the Fibonacci grid.

    Returns
    -------
    (x,y,z) coordinates: numpy.ndarray
        The coordinates of the gridpoints relative to the centre of the
        evolved component in an array with dimensions (Npoints, 3)
    '''
    grid = disk_grid_fibonacci.disk_grid_fibonacci(number_of_gridpoints,\
            radius_primary, [0,0])
    grid_primary = np.array( [np.array(-grid[:,0]),\
            np.array(-grid[:,1] * np.cos(inclination)), \
            np.array(grid[:,1] * np.sin(inclination))] ).T

    return grid_primary

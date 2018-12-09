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
    grid_primary : numpy array
        The coordinates of the gridpoints relative to the centre of the
        evolved component in an array with dimensions (Npoints, 3)
    '''
    grid = disk_grid_fibonacci.disk_grid_fibonacci(number_of_gridpoints,\
            radius_primary, [0,0])
    grid_primary = np.array( [np.array(-grid[:,0]),\
            np.array(-grid[:,1] * np.cos(inclination)), \
            np.array(grid[:,1] * np.sin(inclination))] ).T

    return grid_primary

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

    def calc_launch_radius_velocity(mass_secondary, sma):
        """
        Calculates the launch radius of the X-wind (at the X-region) with the
        secondary component (companion star) as origin.

        Parameters
        ----------
        mass_secondary : float
            Stellar mass of the main sequence star (companion) in units of
            solar mass
        sma : float
            Semi-major axis of the evolved star (primary component)

        Returns
        -------
        launch_radius : float
            The launch radius of the X-region in units of the semi-major axis
        keplerian_velocity : float
            The Keplerian velocity at the radius of the X-region in the inner
            disk
        '''
        """
        # Determine the radius of the main-sequence star, using the empirical
        # mass-stellar radius relation of Demircan, 1991 in units of AU
        radius_secondary_AU = 1.01 * mass_secondary**0.724 * 0.00465
        radius_secondary_sma = radius_secondary_AU / sma
        # The radius of the launch point at the X-region in the disk
        launch_radius_AU = 2. * radius_secondary_AU
        launch_radius_sma = 2. * radius_secondary_sma
        keplerian_velocity= 30.* (mass_secondary / launch_radius_AU)**.5
        #(gravitational_constant*mass_secondary/radius_secondary_AU)

        return launch_radius_sma, keplerian_velocity

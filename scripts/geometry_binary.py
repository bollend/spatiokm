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

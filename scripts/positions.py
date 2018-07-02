import numpy as np

def positions(position_primary, position_secondary,\
             point_on_primary, jet_angle, inclination, N_points_pathlength):
    """
    Determines the coordinates of the gridpoints along
    the line-of-sight through the jet and their angle in the jet

    Parameters
    ----------
    position_primary : numpy array
        The (x,y,z)-coordinates of the primary in AU
    position_secondary : numpy array
        The (x,y,z)-coordinates of the secondary in AU
    point_on_primary : numpy array
        The (x,y,z)-coordinates of the point on the primary in AU
        relative to the centre of the primary
    jet_angle : float
        The half opening angle of the jet in radians
    inclination : float
        The inclination angle of the binary system in radians
    N_points_pathlength : integer
        The number of points through the jet along the line-of-sight

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

    xp = position_primary[0] + point_on_primary[0]
    yp = position_primary[1] + point_on_primary[1]
    zp = point_on_primary[2]

    # Path length to primary of each point in jet
    pathlength_parameter_points = np.zeros(N_points_pathlength)
    # Position of each point in the jet along the LOS in (x,y,z)-coordinates
    position_points_LOS = np.zeros((N_points_pathlength,3))
    # Position of each point with secondary as origin
    position_points_jet_origin = np.zeros((N_points_pathlength,3))
    # Unit vector of each line from the secondary component to the point in
    # the jet
    r_unit_positions = np.zeros((N_points_pathlength,3))
    # Angle between the line from the secondary component to the point in
    # the jet and the jet axis
    angle_positions_jet = np.zeros(N_points_pathlength)

    sqrt = (2 * zp * np.cos(inclination) * np.tan(jet_angle)**2 + 2 * (ys - yp)\
            * np.sin(inclination))**2 - 4 * (np.cos(inclination)**2\
            * np.tan(jet_angle)**2 - np.sin(inclination)**2)\
            * (zp**2 * np.tan(jet_angle)**2 - (xp - xs)**2 - (yp - ys)**2)

    if sqrt < 0:
        jet_entry = 0
        jet_exit = 0
        delta_s = 0

        s = np.zeros(N_points_pathlength)

    else:

        jet_entry = (-1 * (2 * zp * np.cos(inclination) * np.tan(jet_angle)**2 \
            + 2*(ys - yp) * np.sin(inclination)) \
            + (sqrt)**.5) / (2 * (np.cos(inclination)**2 * np.tan(jet_angle)**2\
            - np.sin(inclination)**2))
        jet_exit = (-1 * ( 2 * zp * np.cos(inclination) * np.tan(jet_angle)**2 \
            + 2 * (ys - yp) * np.sin(inclination)) \
            - (sqrt)**.5) / (2 * (np.cos(inclination)**2 * np.tan(jet_angle)**2 \
            - np.sin(inclination)**2))

        if jet_entry < 0 and jet_exit < 0:
            jet_entry = 0
            jet_exit = 0
            delta_s = 0

        elif inclination > jet_angle:

            delta_s = (jet_exit - jet_entry) / (N_points_pathlength - 1)
            s = np.arange(jet_entry, jet_exit + 0.1 * delta_s, delta_s)
            position[:,0] += xp
            position[:,1] += yp + s * np.sin(inclination)
            position[:,2] += zp + s * np.cos(inclination)
            r[:,:] = position[:,:] - pos_sec
            r_tot = (r[:,0]**2 + r[:,1]**2 + r[:,2]**2)**.5
            r_unit[:,0]= r[:,0] / r_tot
            r_unit[:,1]= r[:,1] / r_tot
            r_unit[:,2]= r[:,2] / r_tot
            theta_pos[:] = np.arctan((r[:,0]**2 + r[:,1]**2)**.5 / r[:,2])
            # pl.plot(xp,yp, 'o', color = 'k')
            # pl.plot(xs, ys,'o')
            # pl.plot(position[:,0], position[:,1])
            # pl.xlim(-2.,2.)
            # pl.ylim(-2.,2.)
            # pl.show()

        elif inclination < jet_angle:

            jet_entry = np.max([jet_entry,jet_exit])
            jet_exit = 5. * jet_entry
            delta_s = (jet_exit - jet_entry) / (N_points_pathlength-1)
            s = np.arange(jet_entry, jet_exit + 0.1*delta_s, delta_s)
            position[:,0] += xp
            position[:,1] += yp + s * np.sin(inclination)
            position[:,2] += zp + s * np.cos(inclination)
            r[:,:] = position[:,:] - pos_sec
            r_tot = (r[:,0]**2 + r[:,1]**2 + r[:,2]**2)**.5
            r_unit[:,0]= r[:,0] / r_tot
            r_unit[:,1]= r[:,1] / r_tot
            r_unit[:,2]= r[:,2] / r_tot
            theta_pos[:] = np.arctan((r[:,0]**2 + r[:,1]**2)**.5 / r[:,2])
            # pl.plot(xp,yp, 'o', color = 'k')
            # pl.plot(xs, ys,'o')
            # pl.plot(position[:,0], position[:,1])
            # pl.xlim(-2.,2.)
            # pl.ylim(-2.,2.)
            # pl.show()

    return jet_entry, jet_exit, s, delta_s, position, r, r_unit, theta_pos

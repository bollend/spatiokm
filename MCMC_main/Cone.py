"""
This class implements a cone geometry and associated functions
"""

import numpy as np

class Jet_model(object):
    """ A 3d model representing a stellar jet in a binary system

    The jet cone can be initialised with a half-opening angle.
    Origin and orientation are optional.

    Parameters
    ==========

    inclination : float
        Inclination angle of the orbital plane of the binary system.
    jet_angle : float
        The half opening angle of the cone in radians.
    velocity_max : float
        The velocity along the jet axis
    velocity_edge : float
        The velocity along the jet edge
    jet_type : string
        The jet configuration (disk wind, nested cosine or x-wind, stellar jet)
    jet_centre : numpy array, optional
        Default value is np.array([0, 0, 0])
    jet_angle_inner : float, optional
        The inner angle of the boundary between two jet regions. Default value is None.
    velocity_inner : float
        The velocity at the inner jet angle boundary. Default value is None.
    jet_orientation : numpy array
        Default value is np.array([0, 0, 1])

    Attributes
    ==========

    inclination
    jet_angle
    jet_centre
    jet_orientation

    Raises
    ======

    TypeError
        When 'jet_centre' is not a numpy array.

    Examples
    ========
    tbc

    """

    def __init__(self,
                inclination,
                jet_angle,
                jet_type,
                jet_centre=np.array([0, 0, 0]),\
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False):

        self.inclination             = inclination
        self.jet_angle               = jet_angle
        self.jet_centre              = jet_centre
        self.jet_orientation         = jet_orientation
        self.ray                     = np.array([0, np.sin(self.inclination), np.cos(self.inclination)])
        self.jet_type                = jet_type
        self.jet_tilt                = jet_tilt
        self.double_tilt             = double_tilt
        self.south_tilt_intersection = False

    @property
    def jet_angle(self):
        return self._jet_angle

    @jet_angle.setter
    def jet_angle(self, value):
        if value < 0 or value > 0.5*np.pi:
            raise ValueError('The half-opening angle of the cone should be '
                            'between 0 and pi/2 radians.')
        self._jet_angle = value

    @property
    def inclination(self):
        return self._inclination

    @inclination.setter
    def inclination(self, value):
        if value < 0 or value > 0.5*np.pi:
            raise ValueError(''' The inclination angle of the binary system should be
                            between 0 and pi/2 radians.
                            ''')
        self._inclination = value

    def discriminant(a, b, c):
        """
        Returns the discriminant of a second order equation:
        a * t**2 + b * t + c = 0
        """
        return b**2 - 4 * a * c

    def unit_vector(self, vector):
        """
        Returns the unit vector of the vector.
        """
        vector_norm = np.linalg.norm(vector, axis=1)
        vector_norm_T = np.array([vector_norm]).T
        return vector / vector_norm_T

    def angle_between_vectors(self, v1, v2, unit=False):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'.
        The dot-product of the two vectors is clipped since the result might
        be slightly higher than 1 or lower than -1 due to numerical errors.
        """

        if unit==False:
            v1_u = self.unit_vector(v1)
            v2_u = self.unit_vector(v2)
        else:
            v1_u = v1
            v2_u = v2
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def entry_exit_ray_cone(self, origin_ray, angle_jet, jet_centre, orientation=np.array([0,0,1])):
        """(origin_ray, angle_jet, jet_centre, self.jet_orientation)
        Calculate the discriminant of the second order equation for the intersection
        of a ray and a cone

        Parameters
        ==========
        origin_ray : array
            The point on the surface of the primary which the line-of-sight
            intersects

        Returns
        =======
        entry_parameter : float
            The first solution of the equation (at the jet entry).
        exit_parameter : float
            The second solution of the equation (at the jet exit).

        """

        CO = origin_ray - jet_centre
        a  = np.dot(self.ray, self.jet_orientation)**2 - np.cos(angle_jet)**2
        b  = 2 * (np.dot(self.ray, self.jet_orientation) * np.dot(CO, self.jet_orientation)\
            - np.dot(self.ray, CO) * np.cos(angle_jet)**2)
        c  = np.dot(CO, self.jet_orientation)**2 - \
            np.dot(CO,CO) * np.cos(angle_jet)**2

        Discriminant = b**2 - 4 * a * c

        if Discriminant <= 0:
            # The ray does not intersect the cone
            entry_parameter, exit_parameter = None, None
        else:
            # The ray intersects the cone at an entry and exit point
            parameter_1     = (-b - Discriminant**.5) / (2 * a)
            parameter_2     = (-b + Discriminant**.5) / (2 * a)
            entry_parameter = min(parameter_1, parameter_2)
            exit_parameter  = max(parameter_1, parameter_2)

        return entry_parameter, exit_parameter

    def intersection(self, origin_ray, angle_jet, jet_centre, number_of_gridpoints):
        """
        Determines the coordinates of the intersection between the jet cone and
        the line-of-sight.

        Parameters
        ==========
        origin_ray : array
            The point on the surface of the primary which the line-of-sight
            intersects
        number_of_gridpoints : integer
            The number of gridpoints along the line-of-sight through the jet
        Returns
        =======
        jet_entry_par : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            enters the jet.
        jet_exit_par : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            leaves the jet.
        positions_along_los : array
            The positions of the gridpoints along the line-of-sight that go
            through the jet.

        """
        # Calculate the discriminant of the second order equation for the
        # intersection of the ray and the cone
        jet_entry_par, jet_exit_par = self.entry_exit_ray_cone(origin_ray, angle_jet, jet_centre, self.jet_orientation)

        check_south                  = False
        self.south_tilt_intersection = False

        if jet_entry_par is None:
            # The ray does not intersect the cone
            check_south = True
            jet_entry, jet_exit, self.gridpoints = None, None, None

        elif jet_exit_par < 0:
            # The ray intersects the cone in the wrong direction (away from the observer)
            jet_entry_par, jet_exit_par, self.gridpoints = None, None, None
            check_south = True

        elif jet_entry_par > 0:
            # the ray intersects the cone in the correct direction in two places
            jet_pos_entry = origin_ray + jet_entry_par * self.ray
            jet_pos_exit = origin_ray + jet_exit_par * self.ray

            if jet_pos_exit[2] < 0:
                # The jet intersects the south cone

                if self.double_tilt==True:

                    check_south = True

                else:

                    jet_pos_parameters = np.linspace(jet_entry_par, \
                                                jet_exit_par, number_of_gridpoints)
                    self.gridpoints    = origin_ray + np.outer(jet_pos_parameters,
                                                                self.ray)

            elif jet_pos_entry[2] > 0:
                # The jet intersects the north cone
                jet_pos_parameters = np.linspace(jet_entry_par, \
                                            jet_exit_par, number_of_gridpoints)
                self.gridpoints    = origin_ray + np.outer(jet_pos_parameters,
                                                            self.ray)

            elif jet_pos_entry[2] < 0:
                # The jet intersects both the north and south cone, but the star
                # is located in the jet.
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

        else:
            # The jet intersects both the north and south cone.
            jet_pos_entry = origin_ray + jet_entry_par * self.ray
            jet_pos_exit = origin_ray + jet_exit_par * self.ray

            if jet_pos_entry[2] > 0 or jet_pos_exit[2] < 0:
                # The star is in the north cone of the jet or in the south cone
                # of the jet
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

            elif jet_pos_entry[2] < 0 and jet_pos_exit[2] > 0:
                # The jet intersects the north cone in one point.
                # (inclination angle < jet angle)
                jet_entry_par = np.copy(jet_exit_par)
                jet_exit_par  = None
                jet_pos_parameters  = np.linspace(jet_entry_par,\
                                    jet_entry_par + 5., number_of_gridpoints)
                self.gridpoints     = origin_ray + np.outer(jet_pos_parameters,
                                                            self.ray)
            else:

                check_south = True
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None


        if check_south==True and self.double_tilt==True:
            # We check for intersection with the south lobe
            jet_entry_par, jet_exit_par = self.entry_exit_ray_cone(origin_ray, angle_jet, jet_centre, self.jet_orientation*np.array([1.,1.,-1.]))

            if jet_entry_par is None:
                # The ray does not intersect the cone
                jet_entry, jet_exit, self.gridpoints = None, None, None

            elif jet_exit_par < 0:
                # The ray intersects the cone in the wrong direction (away from the observer)
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

            elif jet_entry_par > 0:
                # the ray intersects the cone in the correct direction in two places
                jet_pos_entry = origin_ray + jet_entry_par * self.ray
                jet_pos_exit = origin_ray + jet_exit_par * self.ray

                if jet_pos_exit[2] < 0:
                    # The jet intersects the south cone
                    jet_pos_parameters = np.linspace(jet_entry_par, \
                                                jet_exit_par, number_of_gridpoints)
                    self.gridpoints    = origin_ray + np.outer(jet_pos_parameters,
                                                                self.ray)
                    self.south_tilt_intersection = True

                elif jet_pos_entry[2] > 0:
                    # The jet intersects the north cone
                    jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

                elif jet_pos_entry[2] < 0:
                    # The jet intersects both the north and south cone, but the star
                    # is located in the jet.
                    jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

            else:
                # The jet intersects both the north and south cone, so the ray
                # doesn't go through the south cone in the right direction
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

        return jet_entry_par, jet_exit_par, self.gridpoints

    #
    # def jet_azimuthal_velocity(self, positions_LOS, vel_keplerian, sma_primary,\
    #                             number_of_gridpoints):
    #     """
    #     Determines the azimuthal velocity component at each grid point along the
    #     line-of-sight through the jet. The angular momentum along the
    #     streamlines in the jet is conserved.
    #
    #     Parameters
    #     ==========
    #     positions_LOS : array
    #         The positions of the gridpoints along the line-of-sight that go
    #         through the jet.
    #     vel_keplerian : array
    #         The Keplerian rotational velocity at the emerging point of the
    #         stream line corresponding to the grid points
    #     sma_primary : float
    #         The semi-major axis of the primary component (evolved star) in
    #         units of AU
    #     number_of_gridpoints : integer
    #         The number of gridpoints
    #     Returns
    #     =======
    #     azimuthal velocity : array
    #         The azimuthal velocity for each grid point along the line-of-sight
    #     """
    #     # Determines the coordinates of the grid point relative to
    #     # the jet centre.
    #     positions_relto_jet        = positions_LOS - self.jet_centre
    #     factor                     = positions_relto_jet[:,2] / self.gridpoints_unit_vector[2]
    #     disk_launch_point          = positions_relto_jet - factor * self.gridpoints_unit_vector
    #     rad_distance_launch_point  = (disk_launch_point[:,0]**2 + disk_launch_point[:,1]**2)**.5
    #     rad_distance_positions     = (positions_LOS[:,0]**2 + positions_LOS[:,1]**2)**.5
    #     # Differentiate between disk wind and X-wind (The launch point in the
    #     # X-wind model is a source point)
    #
    #     if self.type=="disk wind":
    #         azimuthal_vel_magnitude    = vel_keplerian\
    #                                     * (rad_distance_launch_point / rad_distance_positions)
    #         azimuthal_velocity         = azimuthal_vel_magnitude \
    #                                     * np.array([-1. * positions_relto_jet[:,1], \
    #                                     positions_relto_jet[:,0], np.zeros(number_of_gridpoints)]).T \
    #                                     / rad_distance_positions
    #
    #
    #     elif self.type=="stellar jet" or self.type=="simple stellar jet" or self.type=="x-wind":
    #         radius_launch_point, kepl_vel_launch_point = calc_launch_radius(self.mass_sec, sma_primary)
    #         # We add the radius of the X-region relative to the secondary component (companion star)
    #         # to the radial distance of the positions in the jet relative to the jet axis
    #         # in order to avoid infinite velocities
    #         rad_distance_positions_corrected = radius_launch_point + rad_distance_positions
    #         factor                           = radius_launch_point / rad_distance_positions
    #         azimuthal_vel_magnitude          = kepl_vel_launch_point * factor**.5
    #         azimuthal_velocity               = azimuthal_vel_magnitude \
    #                                             * np.array([-1. * positions_relto_jet[:,1], \
    #                                             positions_relto_jet[:,0], np.zeros(number_of_gridpoints)]).T \
    #                                             / rad_distance_positions
    #     return azimuthal_velocity

class Jet(Jet_model):
    """
    A jet with a velocity and density

    Parameters
    ==========
    Attributes
    ==========
    """
    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False):

        self.velocity_max     = velocity_max
        self.velocity_edge    = velocity_edge
        super().__init__(inclination, jet_angle, jet_type, jet_centre,\
                        jet_orientation, jet_tilt, double_tilt)

    def _set_gridpoints(self, origin_ray, number_of_gridpoints):

        self.jet_entry_par, self.jet_exit_par, self.gridpoints = \
                    self.intersection(origin_ray, self.jet_angle, self.jet_centre, number_of_gridpoints)

    def _set_gridpoints_unit_vector(self):
        """
        Sets the unit vector for the vector from the jet centre to the gridpoints.
        """
        if self.gridpoints is not None:

            self.gridpoints_unit_vector = self.unit_vector(self.gridpoints - self.jet_centre)

        else:

            self.gridpoints_unit_vector = None

    def _set_gridpoints_polar_angle(self):
        """
        Sets the polar angle of the gridpoints in the jet relative to the
        jet axis. The centre shift is applied when the centre of the cone is not located
        at the jet centre, i.e., for a disk wind.
        """
        if self.gridpoints_unit_vector is not None:

            if self.double_tilt==False or self.south_tilt_intersection==False:

                self.polar_angle_gridpoints = self.angle_between_vectors(self.gridpoints_unit_vector,\
                                            self.jet_orientation, unit=True)
                self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > 0.5 * np.pi)]\
                        = np.pi - self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > 0.5 * np.pi)]

            else:

                self.polar_angle_gridpoints = self.angle_between_vectors(self.gridpoints_unit_vector,\
                                            self.jet_orientation*np.array([1.,1.,-1.]), unit=True)
                self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > 0.5 * np.pi)]\
                        = np.pi - self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > 0.5 * np.pi)]

        else:

            self.polar_angle_gridpoints = None

    def _set_orientation(self, travel_direction):
        """
        Sets the orientation of the jet axis
        """
        a = np.array([0,0,1])
        v = self.unit_vector(travel_direction)
        r = np.cross(a,v)

        self.jet_orientation = a * np.cos(-1*self.jet_tilt) + np.cross(r,a) * np.sin(-1*self.jet_tilt) + r * np.dot(r,a) * (1 - np.cos(-1*self.jet_tilt))
        self.jet_orientation = self.jet_orientation.flatten()

    def radial_velocity(self, velocities, radvel_secondary):
        """
        Calculates the radial velocity of the jet velocities along the grid points
        """
        radvel = - velocities * np.sum(self.gridpoints_unit_vector * self.ray, axis=1)\
                 - radvel_secondary
        return radvel

    def radial_velocity_gradient(self, radial_velocity, dS,
                                km_to_m=False, AU_to_m=False):
        if km_to_m==True:
            radial_velocity *= 0.001
        if AU_to_m==True:
            dS *= 1.496e11
        radvel_gradient  = np.zeros(len(radial_velocity))
        radvel_gradient[1:-1] = np.abs((radial_velocity[2:] - radial_velocity[0:-2]) / (2*dS))
        radvel_gradient[0]    = np.abs((radial_velocity[1] - radial_velocity[ 0]) / dS)
        radvel_gradient[-1]   = np.abs((radial_velocity[-1] - radial_velocity[ -2]) / dS)

        return radvel_gradient

class Stellar_jet_simple(Jet):
    """
    A stellar jet with a single velocity law and density law. The jet has a
    certain jet angle, inner and outer velocity, and jet centre and orientation.
    """
    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                power_density,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt)

        self.jet_cavity_angle = jet_cavity_angle
        self.power_density    = power_density

    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has a single velocity law
        The velocity for the gridpoints in the jet which have a polar angle
        smaller than the cavity angle is not calculated
        """
        pol_velocity = np.zeros(number_of_gridpoints)
        indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
        pol_velocity[indices] = self.velocity_max + (self.velocity_edge - self.velocity_max)\
                * (np.abs(self.polar_angle_gridpoints[indices] - self.jet_cavity_angle)\
                / (self.jet_angle - self.jet_cavity_angle) )**power

        return pol_velocity

    def density(self, number_of_gridpoints):
        """
        The density in the jet for each gridpoint. The density is a function
        of the polar angle and the height of the jet.
        """
        if self.south_tilt_intersection==False:

            density = np.zeros(number_of_gridpoints)
            indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
            density[indices] = \
                        (self.polar_angle_gridpoints[indices]\
                        / self.jet_angle)**self.power_density \
                        * np.dot(self.gridpoints[indices,:] - self.jet_centre, self.jet_orientation)**-2
        else:

            density = np.zeros(number_of_gridpoints)
            indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
            density[indices] = \
                        (self.polar_angle_gridpoints[indices]\
                        / self.jet_angle)**self.power_density \
                        * np.dot(self.gridpoints[indices,:]*np.array([1.,1.,-1.]) - self.jet_centre, self.jet_orientation)**-2

        return density

class Stellar_jet(Jet):
    """
    A stellar jet with a single velocity law and density law. The velocity
    and density law are more flexible that the simple stellar jet. The
    jet has a certain jet angle, inner and outer velocity, and jet centre
    and orientation.
    """

    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                exp_velocity,
                power_density,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt)

        self.exp_velocity     = exp_velocity
        self.power_density    = power_density
        self.jet_cavity_angle = jet_cavity_angle

    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has a single velocity law, with a varying velocity profile
        The velocity is not calculated for the gridpoints in the jet with
        a polar angle smaller than the cavity angle.
        """
        pol_velocity = np.zeros(number_of_gridpoints)
        indices           = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
        e_exponent          = np.exp(self.exp_velocity)

        exp_angles = np.abs(self.polar_angle_gridpoints[indices] / self.jet_angle)**power

        factor = ( e_exponent**-(exp_angles) - e_exponent**-1 )\
                 / ( 1 - e_exponent**-1 )

        pol_velocity[indices] \
                = self.velocity_edge + (self.velocity_max - self.velocity_edge) * factor

        return pol_velocity

    def density(self, number_of_gridpoints):
        """
        The density in the jet for each gridpoint. The density is a function
        of the polar angle and the height of the jet.
        """
        if self.south_tilt_intersection==False:

            density = np.zeros(number_of_gridpoints)
            indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
            density[indices] = \
                        (self.polar_angle_gridpoints[indices]/self.jet_angle)**self.power_density \
                        * np.dot(self.gridpoints[indices,:] - self.jet_centre, self.jet_orientation)**-2

        else:

            density = np.zeros(number_of_gridpoints)
            indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
            density[indices] = (self.polar_angle_gridpoints[indices] / self.jet_angle)**self.power_density \
                                * np.dot(self.gridpoints[indices,:]*np.array([1.,1.,-1.]) \
                                - self.jet_centre, self.jet_orientation)**-2

        return density

class X_wind(Jet):
    """
    A jet with a single velocity law and double density law. The velocity
    and density law are similar to that of the stellar jet. The
    jet has a certain jet angle, inner, middle, and outer velocity, and jet centre
    and orientation.
    """
    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                exp_velocity,
                power_density_in,
                power_density_out,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0,
                jet_angle_inner=0):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt)

        self.exp_velocity       = exp_velocity
        self.power_density_in   = power_density_in
        self.power_density_out  = power_density_out
        self.jet_cavity_angle   = jet_cavity_angle
        self.jet_angle_inner    = jet_angle_inner

    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has a single velocity law, with a varying velocity profile
        The velocity is not calculated for the gridpoints in the jet with
        a polar angle smaller than the cavity angle.
        """
        pol_velocity = np.zeros(number_of_gridpoints)
        indices      = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
        e_exponent   = np.exp(self.exp_velocity)
        exp_angles   = np.abs((self.polar_angle_gridpoints[indices] - self.jet_cavity_angle)\
                / (self.jet_angle - self.jet_cavity_angle) )**power

        factor = ( e_exponent**-(exp_angles) - e_exponent**-1 )\
                 / ( 1 - e_exponent**-1 )

        pol_velocity[indices] \
                = self.velocity_edge + (self.velocity_max - self.velocity_edge) * factor

        return pol_velocity

    def density(self, number_of_gridpoints):
        """
        The density in the jet for each gridpoint. The density is a function
        of the polar angle and the height of the jet.
        """
        density     = np.zeros(number_of_gridpoints)
        indices_in  = np.where( (self.polar_angle_gridpoints > self.jet_cavity_angle) & (self.polar_angle_gridpoints < self.jet_angle_inner) )
        indices_out = np.where(self.polar_angle_gridpoints > self.jet_angle_inner)

        if self.south_tilt_intersection==False:

            density[indices_in] = (self.polar_angle_gridpoints[indices_in] / self.jet_angle_inner)**self.power_density_in \
                                * np.dot(self.gridpoints[indices_in,:] - self.jet_centre, self.jet_orientation)**-2

            density[indices_out] = (self.polar_angle_gridpoints[indices_out] / self.jet_angle_inner)**self.power_density_out \
                                * np.dot(self.gridpoints[indices_out,:] - self.jet_centre, self.jet_orientation)**-2

        else:

            density[indices_in] = (self.polar_angle_gridpoints[indices_in] / self.jet_angle_inner)**self.power_density_in \
                                * np.dot(self.gridpoints[indices_in,:]*np.array([1.,1.,-1.]) - self.jet_centre, self.jet_orientation)**-2
            density[indices_out] = (self.polar_angle_gridpoints[indices_out] / self.jet_angle_inner)**self.power_density_out \
                                * np.dot(self.gridpoints[indices_out,:]*np.array([1.,1.,-1.]) - self.jet_centre, self.jet_orientation)**-2

        return density

class X_wind_strict(Jet):
    """
    A jet with a single velocity law and density law. The velocity
    and density law are similar to that of the stellar jet. The
    jet has a certain jet angle, inner and outer velocity, and jet centre
    and orientation.
    """

    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                exp_velocity,
                power_density,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt)

        self.jet_cavity_angle = jet_cavity_angle
        self.exp_velocity     = exp_velocity
        self.power_density    = power_density

    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has a single velocity law, with a varying velocity profile
        The velocity is not calculated for the gridpoints in the jet with
        a polar angle smaller than the cavity angle.
        """
        pol_velocity = np.zeros(number_of_gridpoints)
        indices           = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
        e_exponent        = np.exp(self.exp_velocity)
        exp_angles        = np.abs((self.polar_angle_gridpoints[indices] - self.jet_cavity_angle)\
                / (self.jet_angle - self.jet_cavity_angle))**power

        factor = ( e_exponent**-(exp_angles) - e_exponent**-1 )\
                 / ( 1 - e_exponent**-1 )

        pol_velocity[indices] \
                = self.velocity_edge + (self.velocity_max - self.velocity_edge) * factor

        return pol_velocity

    def density(self, number_of_gridpoints):
        """
        The density in the jet for each gridpoint. The density is a function
        of the polar angle and the height of the jet.
        """

        density = np.zeros(number_of_gridpoints)
        indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)

        if self.south_tilt_intersection==False:

            density[indices] = (self.polar_angle_gridpoints[indices] / self.jet_angle)**self.power_density \
                                * np.dot(self.gridpoints[indices,:] - self.jet_centre, self.jet_orientation)**-2

        else:

            density[indices] = (self.polar_angle_gridpoints[indices] / self.jet_angle)**self.power_density \
                                * np.dot(self.gridpoints[indices,:]*np.array([1.,1.,-1.]) - self.jet_centre, self.jet_orientation)**-2
        return density

class Disk_wind(Jet):
    """
    A disk wind jet with a velocity and density

    Parameters
    ==========
    Attributes
    ==========
    """
    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt)

        self.jet_cavity_angle         = jet_cavity_angle

    def _set_orientation(self, travel_direction):
        """
        Sets the orientation of the jet axis
        """
        a = np.array([0,0,1])
        v = self.unit_vector(travel_direction)
        r = np.cross(a,v)

        self.jet_orientation = a * np.cos(-1*self.jet_tilt) + np.cross(r,a) * np.sin(-1*self.jet_tilt) + r * np.dot(r,a) * (1 - np.cos(-1*self.jet_tilt))
        self.jet_orientation = self.jet_orientation.flatten()

    def _set_centre_shift(self, disk_radius):
        """
        Sets the shifted centre of the disk wind
        """
        self.centre_shift = disk_radius/np.tan(self.jet_angle)

        if self.jet_tilt==False:

            self.centre_shift = disk_radius/np.tan(self.jet_angle)
            self.jet_centre_outflow_north = self.jet_centre[2] - self.centre_shift
            self.jet_centre_outflow_south = self.jet_centre[2] + self.centre_shift

        elif self.double_tilt==False:

            self.jet_centre_outflow_north = self.jet_centre - self.centre_shift * self.jet_orientation
            self.jet_centre_outflow_south = self.jet_centre + self.centre_shift * self.jet_orientation

        elif self.double_tilt==True:

            self.jet_centre_outflow_north = self.jet_centre - self.centre_shift * self.jet_orientation
            self.jet_centre_outflow_south = self.jet_centre - np.array([1,1,-1]) * self.centre_shift * self.jet_orientation

    def _set_gridpoints(self, origin_ray, number_of_gridpoints):

        self.jet_entry_par, self.jet_exit_par, self.gridpoints = \
                    self.intersection(origin_ray,
                                    self.jet_angle,
                                    self.jet_centre,
                                    number_of_gridpoints)

    def _set_gridpoints_unit_vector(self, number_of_gridpoints):
        """
        Sets the unit vector for the vector from the jet centre to the gridpoints.
        The centre shift is applied when the centre of the cone is not located
        at the jet centre, i.e., for a disk wind.
        """
        if self.gridpoints is not None:

            indices_north            = np.where(self.gridpoints[:,2] > 0)
            indices_south            = np.where(self.gridpoints[:,2] < 0)
            self.gridpoints_unit_vector = np.zeros((number_of_gridpoints, 3))

            self.gridpoints_unit_vector[indices_north[0],:] = \
                self.unit_vector(self.gridpoints[indices_north[0],:] - (self.jet_centre_outflow_north))
            self.gridpoints_unit_vector[indices_south[0],:] = \
                self.unit_vector(self.gridpoints[indices_south[0],:] - (self.jet_centre_outflow_south))

        else:

            self.gridpoints_unit_vector = None

    def intersection(self, origin_ray, angle_jet, jet_centre, number_of_gridpoints):
        """
        Determines the coordinates of the intersection between the jet cone and
        the line-of-sight for a disk wind

        Parameters
        ==========
        origin_ray : array
            The point on the surface of the primary which the line-of-sight
            intersects
        number_of_gridpoints : integer
            The number of gridpoints along the line-of-sight through the jet
        Returns
        =======
        jet_entry_par : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            enters the jet.
        jet_exit_par : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            leaves the jet.
        positions_along_los : array
            The positions of the gridpoints along the line-of-sight that go
            through the jet.

        """
        # Calculate the discriminant of the second order equation for the
        # intersection of the ray and the cone

        jet_entry_par_north, jet_exit_par_north = \
                        self.entry_exit_ray_cone(origin_ray, self.jet_angle, self.jet_centre_outflow_north)
        jet_entry_par_south, jet_exit_par_south = \
                        self.entry_exit_ray_cone(origin_ray, self.jet_angle, self.jet_centre_outflow_south)

        if (jet_entry_par_north is None and jet_entry_par_south is None)\
              or\
           (jet_entry_par_north < 0 and jet_entry_par_south < 0):
            # The ray does not intersect the north or south lobe
            jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

        elif (jet_entry_par_north is not None and jet_entry_par_south is None)\
                or\
             (jet_entry_par_north is not None and jet_entry_par_south < 0):
            # The ray only intersects the north lobe or the ray intersects both
            # lobes, but the south lobe in the wrong direction

            if jet_entry_par_north < 0:
                # The ray intersects the North cone in the wrong direction
                # (away from the observer) or the entry point is behind the star
                # (the star is located in the jet)
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

            elif self.jet_angle < self.inclination:
                # The jet half-opening angle is smaller than the inclination
                # angle of the system. The line-of-sight will have a jet entry
                # and jet exit point.
                jet_pos_parameters = np.linspace(jet_entry_par_north, \
                                     jet_exit_par_north, number_of_gridpoints)
                self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)
                jet_entry_par      = jet_entry_par_north
                jet_exit_par       = jet_exit_par_north

            elif self.jet_angle > self.inclination:
                # The jet half-opening angle is larger than the inclination angle
                # of the system. The line-of-sight will only have a jet entry point
                jet_entry_par      = np.copy(jet_exit_par_north)
                jet_exit_par       = None
                jet_pos_parameters = np.linspace(jet_entry_par, \
                                        jet_entry_par + 5., number_of_gridpoints)
                self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)

        elif (jet_entry_par_north is None and jet_entry_par_south is not None)\
               or\
             (jet_entry_par_north < 0 and jet_entry_par_south is not None):
            # The ray only intersects the south lobe or the ray intersects both
            # lobes, but the north lobe in the wrong direction

            if jet_entry_par_south < 0:
                # The ray intersects the south cone in the wrong direction
                # (away from the observer) or the entry point is behind the star
                # (the star is located in the jet)
                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

            elif self.jet_angle < self.inclination:
                # The jet half-opening angle is smaller than the inclination
                # angle of the system. The line-of-sight will have a jet entry
                # and jet exit point.
                jet_entry_par      = jet_entry_par_south
                jet_exit_par       = jet_exit_par_south
                jet_pos_parameters = np.linspace(jet_entry_par, \
                                     jet_exit_par, number_of_gridpoints)
                self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)

            elif self.jet_angle > self.inclination:
                # The jet half-opening angle is larger than the inclination angle
                # of the system. The line-of-sight will only have a jet entry point
                jet_entry_par      = np.copy(jet_exit_par_south)
                jet_exit_par       = None
                jet_pos_parameters = np.linspace(jet_entry_par, \
                                        jet_entry_par + 5., number_of_gridpoints)
                self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)

        elif jet_entry_par_north > 0 and jet_exit_par_south > 0:
            # The ray intersects both lobes in the right direction

            if jet_entry_par_north < jet_entry_par_south:
                # The ray intersects the north lobe

                if self.jet_angle < self.inclination:
                    # The jet half-opening angle is smaller than the inclination
                    # angle of the system. The line-of-sight will have a jet entry
                    # and jet exit point.
                    jet_pos_parameters = np.linspace(jet_entry_par_north, \
                                         jet_exit_par_north, number_of_gridpoints)
                    self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)
                    jet_entry_par      = jet_entry_par_north
                    jet_exit_par       = jet_exit_par_north

                elif self.jet_angle > self.inclination:
                    # The jet half-opening angle is larger than the inclination angle
                    # of the system. The line-of-sight will only have a jet entry point
                    jet_entry_par      = np.copy(jet_exit_par_north)
                    jet_exit_par       = None
                    jet_pos_parameters = np.linspace(jet_entry_par, \
                                            jet_entry_par + 5., number_of_gridpoints)
                    self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)

            elif jet_exit_par_south > jet_exit_par_north:
                # The ray intersects the south lobe

                if self.jet_angle < self.inclination:
                    # The jet half-opening angle is smaller than the inclination
                    # angle of the system. The line-of-sight will have a jet entry
                    # and jet exit point.
                    jet_entry_par      = jet_entry_par_south
                    jet_exit_par       = jet_exit_par_south
                    jet_pos_parameters = np.linspace(jet_entry_par, \
                                         jet_exit_par, number_of_gridpoints)
                    self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)

                elif self.jet_angle > self.inclination:
                    # The jet half-opening angle is larger than the inclination angle
                    # of the system. The line-of-sight will only have a jet entry point
                    jet_entry_par      = np.copy(jet_exit_par_south)
                    jet_exit_par       = None
                    jet_pos_parameters = np.linspace(jet_entry_par, \
                                            jet_entry_par + 5., number_of_gridpoints)
                    self.gridpoints    = origin_ray + np.outer(jet_pos_parameters, self.ray)

            else:
                # The ray intersects both lobes in the right direction, which means
                # the ray intersects the disk. Hence, the ray will be blocked by
                # the disk.

                jet_entry_par, jet_exit_par, self.gridpoints = None, None, None

        return jet_entry_par, jet_exit_par, self.gridpoints

class Sdisk_wind(Disk_wind):
    """
    A jet with a double velocity law and density law. The jet is launched from
    the region above and below the accretion disk
    """

    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                scaling_par,
                power_density_in,
                power_density_out,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0,
                jet_angle_inner=None):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt,
                        jet_cavity_angle)

        self.jet_centre_outflow = self.jet_centre - self.centre_shift
        self.jet_angle_inner    = jet_angle_inner
        self.scaling_par        = scaling_par
        self.power_density_in      = power_density_in
        self.power_density_out      = power_density_out

    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has two velocity laws, with an outer velocity profile that
        is scaled to the Keplerian velocity of the position in the disk
        from where the jet is launched, and an inner velocity law similar to that
        of the strict stellar wind.
        The velocity is not calculated for the gridpoints in the jet with
        a polar angle smaller than the cavity angle.
        """

        pol_velocity = np.zeros(number_of_gridpoints)
        indices_in        = np.where( (self.polar_angle_gridpoints > self.jet_cavity_angle) & (self.polar_angle_gridpoints < self.jet_angle_inner) )
        indices_out       = np.where(self.polar_angle_gridpoints > self.jet_angle_inner)
        v_edge_scaled     = self.scaling_par * self.velocity_edge
        v_M               = v_edge_scaled * np.tan(self.jet_angle)**.5
        v_in_scaled       = v_M * np.tan(self.jet_angle_inner)**-.5

        ###### The innner velocities
        pol_velocity[indices_in] = self.velocity_max + (v_in_scaled - self.velocity_max)\
                    * ( np.abs(self.polar_angle_gridpoints[indices_in] - self.jet_cavity_angle)\
                    / (self.jet_angle_inner - self.jet_cavity_angle) )**power

        ###### The outer velocities following a scaled Keplerian velocity
        pol_velocity[indices_out] = v_M * np.tan(self.polar_angle_gridpoints[indices_out])**-.5

        return pol_velocity

    def density(self, number_of_gridpoints):
        """
        The density in the jet for each gridpoint. The density is a function
        of the polar angle and the height of the jet.
        """
        density     = np.zeros(number_of_gridpoints)
        indices_in  = np.where( (self.polar_angle_gridpoints > self.jet_cavity_angle) & (self.polar_angle_gridpoints < self.jet_angle_inner) )
        indices_out = np.where(self.polar_angle_gridpoints > self.jet_angle_inner)

        density[indices_in] = (self.polar_angle_gridpoints[indices_in] / self.jet_angle_inner)**self.power_density_in \
                            * np.dot(self.gridpoints[indices,:] - self.jet_centre, self.jet_orientation)**-2
        density[indices_out] = (self.polar_angle_gridpoints[indices_out] / self.jet_angle_inner)**self.power_density_out \
                            * np.dot(self.gridpoints[indices,:] - self.jet_centre, self.jet_orientation)**-2

    density     = np.zeros(number_of_gridpoints)
    indices_in  = np.where( (self.polar_angle_gridpoints > self.jet_cavity_angle) & (self.polar_angle_gridpoints < self.jet_angle_inner) )
    indices_out = np.where(self.polar_angle_gridpoints > self.jet_angle_inner)

    if self.south_tilt_intersection==False:

        density[indices_in] = (self.polar_angle_gridpoints[indices_in] / self.jet_angle_inner)**self.power_density_in \
                            * np.dot(self.gridpoints[indices_in,:] - self.jet_centre, self.jet_orientation)**-2

        density[indices_out] = (self.polar_angle_gridpoints[indices_out] / self.jet_angle_inner)**self.power_density_out \
                            * np.dot(self.gridpoints[indices_out,:] - self.jet_centre, self.jet_orientation)**-2

    else:

        density[indices_in] = (self.polar_angle_gridpoints[indices_in] / self.jet_angle_inner)**self.power_density_in \
                            * np.dot(self.gridpoints[indices_in,:]*np.array([1.,1.,-1.]) - self.jet_centre, self.jet_orientation)**-2
        density[indices_out] = (self.polar_angle_gridpoints[indices_out] / self.jet_angle_inner)**self.power_density_out \
                            * np.dot(self.gridpoints[indices_out,:]*np.array([1.,1.,-1.]) - self.jet_centre, self.jet_orientation)**-2

    return density
        return density

class Sdisk_wind_strict(Disk_wind):
    """
    A jet with a single Keplerian velocity law and density law. The jet is
    launched from the region above and below the accretion disk.
    """

    def __init__(self,
                inclination,
                jet_angle,
                velocity_max,
                velocity_edge,
                scaling_par,
                power_density,
                jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_tilt=0,
                double_tilt=False,
                jet_cavity_angle=0,
                centre_shift=None):

        super().__init__(inclination,
                        jet_angle,
                        velocity_max,
                        velocity_edge,
                        jet_type,
                        jet_centre,
                        jet_orientation,
                        jet_tilt,
                        double_tilt,
                        jet_cavity_angle,
                        centre_shift)

        self.jet_centre_outflow = jet_centre - centre_shift
        self.scaling_par        = scaling_par
        self.power_density      = power_density

    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has a single velocity law, with a velocity profile that
        is scaled to the Keplerian velocity of the position in the disk
        from where the jet is launched.
        The velocity is not calculated for the gridpoints in the jet with
        a polar angle smaller than the cavity angle.
        """
        pol_velocity = np.zeros(number_of_gridpoints)
        indices           = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)
        v_edge_scaled     = self.scaling_par * self.velocity_edge
        v_M               = v_edge_scaled * np.tan(self.jet_angle)**.5
        pol_velocity[indices] = v_M * np.tan(self.polar_angle_gridpoints[indices])**-.5

        return pol_velocity

    def density(self, number_of_gridpoints):
        """
        The density in the jet for each gridpoint. The density is a function
        of the polar angle and the height of the jet.
        """
        density = np.zeros(number_of_gridpoints)
        indices = np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)

        density[indices] = (self.polar_angle_gridpoints[indices] / self.jet_cavity_angle)**self.power_density \
                            * np.dot(self.gridpoints[indices,:] - self.jet_centre, self.jet_orientation)**-2

        return density

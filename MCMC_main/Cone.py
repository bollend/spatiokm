"""
This class implements a cone geometry and associated functions
"""

from sympy.geometry.point import Point3D
from sympy.geometry import Ray3D
from sympy.geometry import Line
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
    jet_inner_angle : float, optional
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

    def __init__(self, inclination, jet_angle, jet_type, jet_centre=np.array([0, 0, 0]),\
                jet_orientation=np.array([0, 0, 1])):

        self.inclination     = inclination
        self.jet_angle       = jet_angle
        self.jet_centre      = jet_centre
        self.jet_orientation = jet_orientation
        self.ray             = np.array([0, np.sin(self.inclination), np.cos(self.inclination)])
        self.jet_type        = jet_type

        if self.jet_type == 'disk_wind':
            self.jet_centre == jet_centre - z_h
        # else:
        #     self.jet_source_point == self.jet_source_point

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
        return vector / np.linalg.norm(vector)

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

    def entry_exit_ray_cone(self, origin_ray, angle_jet):
        """
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
        CO = origin_ray - self.jet_centre
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

    def intersection(self, origin_ray, angle_jet, number_of_gridpoints):
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
        jet_entry_par, jet_exit_par = self.entry_exit_ray_cone(origin_ray, angle_jet)

        if jet_entry_par == None:
            # The ray does not intersect the cone
            jet_entry, jet_exit, self.gridpoints = None, None, None
        else:
            # The ray intersects the cone at s1 and s2
            if jet_entry_par < 0:
                # The ray intersects the cone in the wrong direction (away from the observer)
                # or the entry point is behind the star (the star is located in the jet)
                jet_entry_par, jet_exit_par, self.gridpoints \
                                        = None, None, None

            elif self.jet_angle < self.inclination:
                # The jet half-opening angle is smaller than
                # the inclination angle of the system. The line-of-sight
                # will have a jet entry and exit point.
                jet_pos_parameters = np.linspace(jet_entry_par, \
                                    jet_exit_par, number_of_gridpoints)
                self.gridpoints    = origin_ray + np.outer(jet_pos_parameters,
                                                       self.ray)

            elif self.jet_angle > self.inclination:
                # The jet half-opening angle is larger than
                # the inclination angle of the system. The line-of-sight
                # will only have a jet entry point.
                jet_entry_par = np.copy(jet_exit_par)
                jet_exit_par  = None
                jet_pos_parameters  = np.linspace(jet_entry_par,\
                                    jet_entry_par + 5., number_of_gridpoints)
                self.gridpoints     = origin_ray + np.outer(jet_pos_parameters,
                                                            self.ray)


        return jet_entry_par, jet_exit_par, self.gridpoints


    # def jet_poloidal_velocity(self, positions_LOS, number_of_gridpoints, power=2):
    #     """
    #     Determines the poloidal velocity component at each grid point along the
    #     line-of-sight through the jet
    #
    #     Parameters
    #     ==========
    #     positions_LOS : array
    #         The positions of the gridpoints along the line-of-sight that go
    #         through the jet.
    #     number_of_gridpoints : integer
    #         The number of gridpoints
    #     power : float (optional)
    #         The velocity law power factor (default value = 2)
    #     Returns
    #     =======
    #     poloidal velocity : array
    #         The poloidal velocity for each grid point along the line-of-sight
    #     """
    #     poloidal_velocity = np.zeros(number_of_gridpoints)
    #
    #     if self.jet_type == "simple stellar jet":
    #         # The jet has a single velocity law
    #         # The velocity for the gridpoints in the jet which have a polar angle
    #         # smaller than the cavity angle is not calculated
    #         vel_gridpoints[np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)] \
    #                 = self.velocity_max + (self.velocity_edge - self.velocity_max)\
    #                 * ( (self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)] - self.jet_cavity_angle)\
    #                 / (self.jet_angle - self.jet_cavity_angle) )**power
    #
    #     if self.jet_type == "stellar jet":
    #         # The jet has a single velocity law
    #         # The velocity for the gridpoints in the jet which have a polar angle
    #         # smaller than the cavity angle is not calculated
    #         vel_gridpoints[np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)] \
    #                 = self.velocity_edge + ((self.velocity_max - self.velocity_edge)\
    #                 * (self.sigma_stellar**-( (self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)] - self.jet_cavity_angle)\
    #                 / (self.jet_angle - self.jet_cavity_angle) )**2 - self.sigma_stellar**-1) / (1. - self.sigma_stellar**-1))
    #
    #     elif self.jet_type == "x-wind":
    #         # The jet has two velocity laws, one representing an inner stellar jet
    #         # and the other representing the x-wind in the outer region of the jet
    #         index_inner  = self.polar_angle_gridpoints < self.jet_inner_angle
    #         index_outer  = self.polar_angle_gridpoints > self.jet_inner_angle
    #         cos_boundary = np.cos(0.5 * np.pi * self.jet_inner_angle / self.jet_angle)
    #         cos_inner    = np.cos(0.5 * np.pi * self.polar_angle_gridpoints[index_inner] / self.jet_inner_angle)
    #         cos_outer    = np.cos(0.5 * np.pi * self.polar_angle_gridpoints / self.jet_angle)
    #         vel_diff_be  = self.velocity_inner - self.velocity_edge
    #         v_M          = vel_diff_be / cos_boundary + self.velocity_edge
    #         poloidal_velocity[index_inner] = self.velocity_edge \
    #                   + (v_M - self.velocity_edge) * cos_outer[index_inner] \
    #                   + (self.velocity_max - v_M) * cos_inner[index_inner]
    #         poloidal_velocity[index_outer] = self.velocity_edge \
    #                   + (v_M - self.velocity_edge) * cos_outer[index_outer]
    #
    #     elif self.jet_type == "strict x-wind":
    #         # The jet has one velocity law, representing the x-wind
    #         # in the outer region of the jet. The inner region (smaller than 30degr)
    #         # is a cavity
    #         index  = self.polar_angle_gridpoints > self.jet_inner_angle
    #         cos_boundary = np.cos(0.5 * np.pi * self.jet_inner_angle / self.jet_angle)
    #         cos_outer    = np.cos(0.5 * np.pi * self.polar_angle_gridpoints / self.jet_angle)
    #         vel_diff_be  = self.velocity_inner - self.velocity_edge
    #         v_M          = vel_diff_be / cos_boundary + self.velocity_edge
    #         poloidal_velocity[index_outer] = self.velocity_edge \
    #                   + (v_M - self.velocity_edge) * cos_outer[index]
    #
    #     elif self.jet_type == "disk wind":
    #         # The jet has two velocity laws, one representing an inner stellar jet
    #         # and the other representing the disk wind
    #         index_inner = self.polar_angle_gridpoints < self.jet_inner_angle
    #         index_outer = self.polar_angle_gridpoints > self.jet_inner_angle
    #         v_M         = self.velocity_edge * np.tan(self.jet_angle)**.5
    #         poloidal_velocity[index_inner] = self.velocity_max + (self.velocity_inner - self.velocity_max) \
    #                   * (self.polar_angle_gridpoints[index_inner] / self.jet_inner_angle)**power
    #         poloidal_velocity[index_outer] = v_M * np.tan(self.polar_angle_gridpoints[index_outer])**.5
    #
    #     return poloidal_velocity
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
    #
    #
    # def radial_velocity(self, velocities, radvel_secondary):
    #     """ Calculates the radial velocity of the jet velocities along the grid points """
    #     radvel = - velocities * np.sum(self.gridpoints_unit_vector * self.ray, axis=1)\
    #              - radvel_secondary
    #     return radvel
    #
    #
    # def jet_density(self, jet_height, angles, value_inner=0, power=2, \
    #                 power_inner=2, power_outer=2):
    #     """
    #     Determines the density in the jet at each grid point along the line-of-sight
    #
    #     Parameters
    #     ==========
    #     jet_height : array
    #         The height in the jet of each gridpoint along the line-of-sight
    #     angles : array
    #         The angles in the jet of the ray along the line-of-sight
    #     power : float
    #         The density law power factor (default value = 2)
    #     power_inner : float
    #         The inner density law power factor (default value = 2)
    #     power_outer : float
    #         The outer density law power factor (default value = 2)
    #     Returns
    #     =======
    #     density : array
    #         The density in the jet at each grid point along the line-of-sight
    #     """
    #     density = np.zeros(number_of_gridpoints)
    #     if self.jet_type == "simple stellar jet" or self.jet_type == "stellar jet":
    #         density[np.where(angles > self.jet_cavity_angle)] = \
    #                 (angles[np.where(angles > self.jet_cavity_angle)]/self.jet_angle)**power * jet_height**-2
    #
    #     # elif self.jet_type = "stellar jet":
    #     #     density[np.where(angles > self.jet_cavity_angle)] = \
    #
    #     elif self.jet_type == "x-wind" or self.jet_type == "disk wind":
    #         density[np.abs(values) <= value_inner] =\
    #                 (values[np.abs(theta) < alpha_in]/value_inner)**power_inner\
    #                 * jet_height[(np.abs(theta) < alpha_in),2]**-2
    #         density[np.abs(values) >= value_outer] =\
    #                 (values[np.abs(theta) > alpha_in]/value_inner)**power_outer\
    #                 * jet_height[(np.abs(theta) < alpha_in),2]**-2
    #
    #     return density

class Jet(Jet_model):
    """
    A jet with a velocity and density

    Parameters
    ==========
    Attributes
    ==========
    """
    def __init__(self, inclination,
                jet_angle, velocity_max,
                velocity_edge,
                jet_type, jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None):
        self.jet_cavity_angle = jet_cavity_angle
        self.velocity_max     = velocity_max
        self.velocity_edge    = velocity_edge
        super().__init__(inclination, jet_angle, jet_type, jet_centre,\
                        jet_orientation)

    def _set_gridpoints(self, origin_ray, number_of_gridpoints):
        self.jet_entry_par, self.jet_exit_par, self.gridpoints = \
                    self.intersection(origin_ray, self.jet_angle, number_of_gridpoints)

    def _set_gridpoints_unit_vector(self):
        """
        Sets the unit vector for the vector from the jet centre to the gridpoints
        """
        self.gridpoints_unit_vector = self.unit_vector(self.gridpoints - self.jet_centre)

    def _set_gridpoints_polar_angle(self):
        """
        Sets the polar angle of the gridpoints in the jet relative to the
        jet axis.
        """
        self.polar_angle_gridpoints = self.angle_between_vectors(self.gridpoints_unit_vector,\
                                      self.jet_orientation, unit=True)
        self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > 0.5 * np.pi)]\
                = np.pi - self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > 0.5 * np.pi)]


    def radial_velocity(self, velocities, radvel_secondary):
        """
        Calculates the radial velocity of the jet velocities along the grid points
        """
        radvel = - velocities * np.sum(self.gridpoints_unit_vector * self.ray, axis=1)\
                 - radvel_secondary
        return radvel

class Stellar_jet_simple(Jet):
    """
    """
    def __init__(self, inclination,
                jet_angle, velocity_max,
                velocity_edge, jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None):

        super().__init__(inclination, jet_angle,
                        velocity_max,
                        velocity_edge, jet_type,
                        jet_centre, jet_orientation,
                        jet_cavity_angle)

        self.jet_type = "simple stellar jet"


    def poloidal_velocity(self, number_of_gridpoints, power):
        """
        The jet has a single velocity law
        The velocity for the gridpoints in the jet which have a polar angle
        smaller than the cavity angle is not calculated
        """
        poloidal_velocity = np.zeros(number_of_gridpoints)

        if self.jet_type == "simple stellar jet":
            # The jet has a single velocity law
            # The velocity for the gridpoints in the jet which have a polar angle
            # smaller than the cavity angle is not calculated
            vel_gridpoints[np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)] \
                    = self.velocity_max + (self.velocity_edge - self.velocity_max)\
                    * ( (self.polar_angle_gridpoints[np.where(self.polar_angle_gridpoints > self.jet_cavity_angle)] - self.jet_cavity_angle)\
                    / (self.jet_angle - self.jet_cavity_angle) )**power

        return poloidal_velocity



    def density(self):
            """

            """
            density = np.zeros(number_of_gridpoints)
            density[np.where(angles > self.jet_cavity_angle)] = \
                        (angles[np.where(angles > self.jet_cavity_angle)]\
                        /self.jet_angle)**power \
                        * self.gridpoints[:,2]**-2

            return density

class Stellar_jet(Jet):
    """
    """

    def __init__(self, inclination, jet_angle,
                velocity_max,
                velocity_edge, jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None):

        Jet.__init__(self, inclination, jet_angle,
                    velocity_max, velocity_edge,
                    jet_centre=np.array([0, 0, 0]),
                    jet_orientation=np.array([0, 0, 1]),
                    jet_cavity_angle=None)
    def poloidal_velocity(self):
        """
        """
        poloidal_velocity = np.zeros(number_of_gridpoints)

    def azimuthal_velocity(self):
        """
        """

    def radial_velocity(self):
        """
        """

    def density(self):
        """
        """


class X_wind(Jet):
    """
    """

    def __init__(self, inclination, jet_angle,
                velocity_max,
                velocity_edge, jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None,
                velocity_inner=None, jet_inner_angle=None, z_h=None):

        Jet.__init__(self, inclination, jet_angle,
                    velocity_max, velocity_edge,
                    jet_centre=np.array([0, 0, 0]),
                    jet_orientation=np.array([0, 0, 1]),
                    jet_cavity_angle=None)

    def poloidal_velocity(self):
        """
        """
        poloidal_velocity = np.zeros(number_of_gridpoints)

    def azimuthal_velocity(self):
        """
        """

    def radial_velocity(self):
        """
        """

    def density(self):
        """
        """
class X_wind_strict(Jet):
    """
    """

    def __init__(self, inclination, jet_angle,
                velocity_max, velocity_edge,
                jet_type, jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None,
                velocity_inner=None,
                jet_inner_angle=None, z_h=None):

        super().__init__(self, inclination, jet_angle,
                    velocity_max, velocity_edge,
                    jet_centre=np.array([0, 0, 0]),
                    jet_orientation=np.array([0, 0, 1]),
                    jet_cavity_angle=None)

    def poloidal_velocity(self):
        """
        """
        poloidal_velocity = np.zeros(number_of_gridpoints)

    def azimuthal_velocity(self):
        """
        """

    def radial_velocity(self):
        """
        """

    def density(self):
        """
        """
class Disk_wind(Jet):
    """
    """

    def __init__(self, inclination, jet_angle,
                velocity_max,
                velocity_edge, jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None,
                velocity_inner=None,
                jet_inner_angle=None, z_h=None):

        if self.jet_type == 'disk_wind':
            self.jet_centre == jet_centre - z_h
        super().__init__(self, inclination, jet_angle,
                    velocity_max,
                    velocity_edge,
                    jet_centre=np.array([0, 0, 0]),
                    jet_orientation=np.array([0, 0, 1]),
                    jet_cavity_angle=None)

    def poloidal_velocity(self):
        """
        """
        poloidal_velocity = np.zeros(number_of_gridpoints)

    def azimuthal_velocity(self):
        """
        """

    def radial_velocity(self):
        """
        """

    def density(self):
        """
        """
class Disk_wind_strict(Jet):
    """
    """

    def __init__(self, inclination, jet_angle,
                velocity_max,
                velocity_edge,jet_type,
                jet_centre=np.array([0, 0, 0]),
                jet_orientation=np.array([0, 0, 1]),
                jet_cavity_angle=None,
                velocity_inner=None,
                jet_inner_angle=None,
                z_h=None):

        if self.jet_type == 'disk_wind':
            self.jet_centre == jet_centre - z_h
        super().__init__(self, inclination, jet_angle,
                    velocity_max,
                    velocity_edge,
                    jet_centre=np.array([0, 0, 0]),
                    jet_orientation=np.array([0, 0, 1]),
                    jet_cavity_angle=None)

    def poloidal_velocity(self):
        """
        """
        poloidal_velocity = np.zeros(number_of_gridpoints)

    def azimuthal_velocity(self):
        """
        """

    def radial_velocity(self):
        """
        """

    def density(self):
        """
        """

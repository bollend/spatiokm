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
    velocity_axis : float
        The velocity along the jet axis
    velocity_edge : float
        The velocity along the jet edge
    jet_type : string
        The jet configuration (disk wind, nested cosine or x-wind, stellar jet)
    jet_centre : array, optional
        Default value is [0, 0, 0]
    jet_inner_angle : float, optional
        The inner angle of the boundary between two jet regions. Default value is None.
    velocity_inner : float
        The velocity at the inner jet angle boundary. Default value is None.
    orientation : array
        Default value is [0, 0, 1]

    Attributes
    ==========

    inclination
    jet_angle
    velocity_axis
    velocity_edge
    jet_type
    jet_centre
    jet_inner_angle
    velocity_boundary
    orientation


    Raises
    ======

    ???TypeError
        When 'jet_centre' is not a numpy array.

    Examples
    ========
    tbc

    """

    def __init__(self, inclination, jet_angle, velocity_axis, velocity_edge,\
                jet_type, jet_centre=np.array([0, 0, 0]), jet_inner_angle=None,\
                velocity_boundary=None, orientation=np.array([0, 0, 1]), z_h):

        self.jet_angle = jet_angle
        self.jet_inner_angle = jet_inner_angle
        self.velocity_axis = velocity_axis
        self.velocity_edge = velocity_edge
        self.origin_type = origin_type
        if jet_type == 'disk_wind':
            jet_centre[2] += z_h
        self.jet_centre = jet_centre
        self.jet_inner_angle = jet_inner_angle
        self.velocity_boundary = velocity_boundary
        self.orientation = orientation
        self.inclination = inclination

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
        return self._inclination

    def determinant(a, b, c):
        """
        Returns the determinant of a second order equation:
        a * t**2 + b * t + c = 0
        """
        return b**2 - 4 * a * c

    def unit_vector(vector):
        """
        Returns the unit vector of the vector.
        """
        return vector / np.linalg.norm(vector)

    def angle_between(v1, v2):
        """
        Returns the angle in radians between vectors 'v1' and 'v2'
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def entry_exit_ray_cone(self, origin_ray, ray):
        """
        Calculate the determinant of the second order equation for the intersection
        of a ray and a cone

        Parameters
        ==========
        origin_ray : array
            The point on the surface of the primary which the line-of-sight
            intersects
        ray : array
            The ray along the line-of-sight starting from the surface of the
            primary.

        Returns
        =======
        entry_parameter : float
            The first solution of the equation (at the jet entry).
        exit_parameter : float
            The second solution of the equation (at the jet exit).

        """
        CO = origin_ray - self.jet_centre
        a = np.dot(ray, self.orientation)**2 - np.cos(self.jet_angle)**2
        b = 2 * (np.dot(ray, self.orientation) * np.dot(CO, self.orientation)\
            - np.dot(ray, CO) * np.cos(self.jet_angle)**2)
        c = np.dot(CO, self.orientation)**2 - \
            np.dot(CO,CO) * np.cos(self.jet_angle)**2

        Delta = b**2 - 4 * a * c

        if Delta <= 0:
            # The ray does not intersect the cone
            entry_parameter, exit_parameter = None, None
        else:
            # The ray intersects the cone at an entry and exit point
            parameter_1 = (-b - Delta**.5) / (2 * a)
            parameter_2 = (-b + Delta**.5) / (2 * a)
            entry_parameter = min(parameter_1, parameter_2)
            exit_parameter = max(parameter_1, parameter_2)

        return entry_parameter, exit_parameter


    def intersection(self, origin_ray, ray, number_of_gridpoints):
        """
        Determines the coordinates of the intersection between the jet cone and
        the line-of-sight.

        Parameters
        ==========
        origin_ray : array
            The point on the surface of the primary which the line-of-sight
            intersects
        ray : array
            The unit vector of the ray along the line-of-sight starting from the surface of the
            primary.
        number_of_gridpoints : integer
            The number of gridpoints along the line-of-sight through the jet
        Returns
        =======
        jet_entry_parameter : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            enters the jet.
        jet_exit_parameter : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            leaves the jet.
        positions_along_los : array
            The positions of the gridpoints along the line-of-sight that go
            through the jet.

        """
        # Calculate the determinant Delta of the second order equation for the
        # intersection of the ray and the cone
        jet_entry_parameter, jet_exit_parameter = self.entry_exit_ray_cone(origin_ray, ray)

        if jet_entry_parameter == None:
            # The ray does not intersect the cone
            jet_entry, jet_exit = None, None
        else:
            # The ray intersects the cone at s1 and s2
            if (jet_entry_parameter < 0 and jet_exit_parameter < 0):
                # The ray intersects the cone in the wrong direction (away from the observer)
                jet_entry_parameter, jet_exit_parameter, jet_entry, jet_exit \
                                        = [None for _ in range(4)]

            elif self.jet_angle < self.inclination:
                # The jet half-opening angle is smaller than
                # the inclination angle of the system. The line-of-sight
                # will have a jet entry and exit point.
                jet_positions_parameters = np.linspace(jet_entry_parameter, \
                                    jet_exit_parameter, number_of_gridpoints)
                positions = origin_ray + np.outer(jet_positions_parameters, ray)

            elif self.jet_angle > self.inclination:
                # The jet half-opening angle is larger than
                # the inclination angle of the system. The line-of-sight
                # will only have a jet entry point.
                jet_entry_parameter = np.copy(jet_exit_parameter)
                jet_exit_parameter = None
                jet_positions_parameters = np.linspace(jet_entry_parameter,\
                                    jet_entry_parameter + 5., number_of_gridpoints)
                positions = origin_ray + np.outer(jet_positions_parameters, ray)


        return jet_entry_parameter, jet_exit_parameter, positions

        def velocity(self, phase, positions_LOS, rv_secondary):
            """
            Determines the velocity at each grid point along the line-of-sight
            through the jet

            Parameters
            ==========
            phase : float
                Orbital phase
            positions_LOS : array
                The positions of the gridpoints along the line-of-sight that go
                through the jet.
            rv_secondary : float
                The radial velocity of the secondary component at that orbital phase
            Returns
            =======
            rad_velocity : array
                The radial velocity for each grid point along the line-of-sight
            """

            if self.jet_type == "stellar jet":
                vel_gridpoints = self.velocity_axis + (self.velocity_edge - self.velocity_axis)\
                            * (polar_angle_gridpoints / self.jet_angle)**power
                radvel_gridpoints = - vel_gridpoints * np.sum(R_unit*n_unit, axis = 1) - Orbit_sec*v_prim*np.sin(w*t)

            elif self.jet_type == "x-wind":
                rad_velocity = self.velocity_xwind(phase, positions_LOS, rv_secondary)

            elif self.jet_type == "disk wind":
                rad_velocity = self.velocity_disk_wind(phase, positions_LOS, rv_secondary)
            return rad_velocity

        def velocity_stellar_jet(self, phase, position_LOS, rv_secondary):
            """
            Determines the velocity at each grid point along the line-of-sight
            through the jet for the stellar jet model.

            Parameters
            ==========
            phase : float
                Orbital phase
            positions_LOS : array
                The positions of the gridpoints along the line-of-sight that go
                through the jet.
            rv_secondary : float
                The radial velocity of the secondary component at that orbital phase
            Returns
            =======
            radvel_gridpoints : array
                The radial velocity for each grid point along the line-of-sight
            """
            vel_gridpoints = self.velocity_axis + (self.velocity_edge - self.velocity_axis)\
                        * (polar_angle_gridpoints / self.jet_angle)**power
            radvel_gridpoints = - vel_gridpoints * np.sum(R_unit*n_unit, axis = 1) - Orbit_sec*v_prim*np.sin(w*t)


            return radvel_gridpoints

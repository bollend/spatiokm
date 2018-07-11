"""
This class implements a cone geometry and associated functions
"""

from sympy.geometry.point import Point3D
from sympy.geometry import Ray3D
from sympy.geometry import Line
import numpy as np

class Cone3D(object):
    """ A 3d cone object representing a stellar jet

    The cone can be initialised with a half-opening angle.
    Origin and orientation are optional.

    Parameters
    ==========

    jet_centre : array, optional
        Default value is [0, 0, 0]
    jet_angle : float
        The half opening angle of the cone in radians.
    inclination : float
        Inclination angle of the orbital plane of the binary system.
    orientation : array
        Default value is [0, 0, 1]

    Attributes
    ==========

    jet_centre
    jet_angle
    inclination
    orientation

    Raises
    ======

    TypeError
        When 'jet_centre' is not a numpy array.

    Examples
    ========
    tbc

    """

    def __init__(self, jet_angle, inclination, jet_centre=np.array([0, 0, 0]),\
                orientation=np.array([0, 0, 1])):

        self.jet_angle = jet_angle
        self.jet_centre = jet_centre
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
            # The ray intersects the cone at an entry point and exit point
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
            #The ray does not intersect the cone
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
                positions_along_los = origin_ray + np.outer(jet_positions_parameters, ray)

            elif self.jet_angle > self.inclination:
                # The jet half-opening angle is larger than
                # the inclination angle of the system. The line-of-sight
                # will only have a jet entry point.
                jet_entry_parameter = np.copy(jet_exit_parameter)
                jet_exit_parameter = None
                jet_positions_parameters = np.linspace(jet_entry_parameter,\
                                    jet_entry_parameter + 5., number_of_gridpoints)
                positions_along_los = origin_ray + np.outer(jet_positions_parameters, ray)


        return jet_entry_parameter, jet_exit_parameter, positions_along_los

        def velocity_density():

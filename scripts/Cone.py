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

    centre : array, optional
        Default value is [0, 0, 0]
    jet_angle : float
        The half opening angle of the cone in radians.
    inclination : float
        Inclination angle of the orbital plane of the binary system.
    orientation : array
        Default value is [0, 0, 1]

    Attributes
    ==========

    centre
    jet_angle
    inclination
    orientation

    Raises
    ======

    TypeError
        When 'centre' is not a numpy array.

    Examples
    ========
    tbc

    """

    def __init__(self, jet_angle, inclination, centre=np.array([0, 0, 0]),\
                orientation=np.array([0, 0, 1])):

        self._jet_angle = jet_angle
        self.centre = centre
        self.orientation = orientation
        self._inclination = inclination

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

    def determinant_intersect_ray_cone(self, origin_ray, ray):
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
        Determinant : float
            The determinant of the second order equation

        """
        CO = origin_ray - self.centre
        a = np.dot(ray, self.orientation)**2 - np.cos(self._jet_angle)**2
        b = 2 * (np.dot(ray, self.orientation) * np.dot(CO, self.orientation)\
            - np.dot(ray, CO) * np.cos(self._jet_angle)**2)
        c = np.dot(CO, self.orientation)**2 - \
            np.dot(CO,CO) * np.cos(self._jet_angle)**2
        return b**2 - 4 * a * c


    def intersection(self, origin_ray, ray):
        """
        Determines the coordinates of the intersection between the jet cone and
        the line-of-sight.

        Parameters
        ==========
        origin_ray : array
            The point on the surface of the primary which the line-of-sight
            intersects
        ray : array
            The ray along the line-of-sight starting from the surface of the
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
        jet_entry : array
            Point where the line-of-sight enters the jet.
        jet_exit : array
            Point where the line-of-sight leaves the jet

        """
        # Calculate the determinant Delta of the second order equation for the
        # intersection of the ray and the cone
        Delta = self.determinant_intersect_ray_cone(origin_ray, ray)

        if Delta <= 0:
            #The ray does not intersect the cone
            jet_entry_parameter = None
            jet_exit_parameter = None
            jet_entry = None
            jet_exit = None

        else:
            # The ray intersects the cone at s1 and s2
            jet_entry_parameter = (-b - Delta**.5) / (2 * a)
            jet_exit_parameter = (-b + Delta**.5) / (2 * a)

            if (jet_entry_parameter < 0 and jet_exit_parameter < 0):
                # The ray intersects the cone in the wrong direction (away from the observer)
                jet_entry_parameter, jet_exit_parameter, jet_entry, jet_exit = None

            elif jet_angle < inclination

        return jet_entry_parameter, jet_exit_parameter, jet_entry, jet_exit




        if not isintance(ray, np.ndarray):
            raise TypeError('The ray should be an numpy array')


    # def equation(self, x='x', y='y', z='z'):
    #     """The equation of the cone.
    #     Parameters
    #     ==========
    #     x : str, optional
    #         Label for the x-axis. Default value is 'x'.
    #     y : str, optional
    #         Label for the y-axis. Default value is 'y'.
    #     z : str, optional
    #         Label for the y-axis. Default value is 'y'.
    #     Returns
    #     =======
    #     equation : sympy expression
    #
    #     Examples
    #     ========
    #     pass
    #     """
    #     x = _symbol(x, real=True)
    #     y = _symbol(y, real=True)
    #     t1 = ((x - self.center.x) / self.hradius)**2
    #     t2 = ((y - self.center.y) / self.vradius)**2
    #     return t1 + t2 - 1

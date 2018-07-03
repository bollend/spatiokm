"""
This class implements a cone geometry and associated functions
"""

from sympy.geometry.point import Point3D
from sympy.geometry import Ray3D
import numpy as np

class Cone3D(object):
    """ A 3d cone object

    The cone can be initialised with a half-opening angle.
    Origin and orientation are optional.

    Parameters
    ==========

    centre : Point, optional
        Default value is Point3D(0, 0, 0)
    angle : float
        The half opening angle of the cone in radians
    orientation : Point or a direction vector
        Default value is Point3D(0, 0, 1)

    Attributes
    ==========

    centre
    angle
    orientation

    Raises
    ======

    TypeError
        When 'centre' is not a Point.

    Examples
    ========
    tbc

    """

    def __init__(self, angle, centre=Point3D(0, 0, 0),\
                orientation=Point3D(0, 0, 1)):

        self._angle = angle
        self.centre = centre
        self.orientation = orientation

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        if value < 0 or value > 0.5*np.pi:
            raise ValueError('The half-opening angle of the cone should be'
                             ' between 0 and pi/2')
        self._angle = value

    def intersection(self, ray):
        """
        Determines the coordinates of the intersection between the jet cone and
        the line-of-sight.

        Parameters
        ==========
        ray : Ray3D (sympy)
            The ray along the line-of-sight starting from the surface of the
            primary.

        Returns
        =======
        jet_entry_parameter : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            enters the jet.
        jet_exit_parameter : float
            Value of line-of-sight parameter 's' for which the line-of-sight
            leaves the jet.
        jet_entry : Point3D
            Point where the line-of-sight enters the jet.
        jet_exit : Point3D
            Point where the line-of-sight leaves the jet

        """
        if not isintance(ray, Ray3D):
            raise TypeError('The ray should be of type Ray3D')

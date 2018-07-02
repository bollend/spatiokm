"""
This class implements a cone geometry and associated functions
"""

from sympy.geometry.point import Point3D
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

    def intersection():

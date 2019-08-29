"""
A star object
"""

import numpy as np
import geometry_binary

class Star(object):
    """
    A star object with a location in 3D space.
    """
    def __init__(self, radius, centre, inclination, gridpoints):
        self.radius = radius
        self.centre = centre
        self.inclination = inclination
        self.gridpoints = gridpoints

    def get_radius(self):
        return self.radius

    def _set_grid(self):
        self.grid = geometry_binary.disk_grid(self.radius, self.inclination, self.gridpoints)

    def _set_grid_location(self):
        self.grid_location = self.centre + self.grid

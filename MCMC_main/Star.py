"""
A star object
"""

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
        self.grid = None
        self.grid_location = None

    def get_radius(self):
        """
        returns the radius of the Star object
        """
        return self.radius

    def _set_grid(self):
        self.grid = geometry_binary.disk_grid(self.radius, self.inclination, self.gridpoints)

    def _set_grid_location(self):
        self.grid_location = self.centre + self.grid

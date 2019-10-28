import numpy as np

def line_plane_intersection(plane_normal, plane_point, ray_direction, ray_point, epsilon=1e-6):

	ndotu = plane_normal.dot(ray_direction)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = ray_point - plane_point
	si = -plane_normal.dot(w) / ndotu
	Psi = w + si * ray_direction + ray_point
	return Psi

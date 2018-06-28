import numpy as np

def positions(t, pos_prim, pos_sec, p_prim, alpha, xs, ys, inclination):
    """
    Coordinates of the evolved star (xp, yp, zp), gridpoints along the line of sight through the jet (s), and their coordinates (position, r), and angle in the jet (theta_pos)
    """
    xp = pos_prim[0] + p_prim[0]
    yp = pos_prim[1] + p_prim[1]
    zp = p_prim[2]
    s = np.zeros(Npoints_pathlength)             #path lenght to primary of each point in jet
    position = np.zeros((Npoints_pathlength,3))    #position of each point in jet in 3D
    r = np.zeros((Npoints_pathlength,3))           # position of each point with secondary as origin
    r_unit = np.zeros((Npoints_pathlength,3))
    theta_pos = np.zeros(Npoints_pathlength)
    # sqrt = (2*(ys-yp)*np.sin(incl))**2 - 4*(np.cos(incl)**2*np.tan(alp)**2 - np.sin(incl)**2)*(- (xp-xs)**2 - (yp - ys)**2)
    sqrt = (2*zp*np.cos(inclination)*np.tan(alpha)**2 + 2*(ys - yp)*np.sin(inclination))**2 - 4*(np.cos(inclination)**2*np.tan(alpha)**2 - np.sin(inclination)**2)*(zp**2*np.tan(alpha)**2-(xp-xs)**2-(yp-ys)**2)

    if sqrt < 0:
        s1 = 0
        s2 = 0
        ds = 0

        s = np.zeros(Npoints_pathlength)
    else:

        # s1 = (-1*( 2*(ys - yp)*np.sin(incl)) + (sqrt)**.5)/(2*(np.cos(incl)**2*np.tan(alp)**2 - np.sin(incl)**2))
        # s2 = (-1*( 2*(ys - yp)*np.sin(incl)) - (sqrt)**.5)/(2*(np.cos(incl)**2*np.tan(alp)**2 - np.sin(incl)**2))
        s1 = (-1*( 2*zp*np.cos(inclination)*np.tan(alpha)**2 + 2*(ys - yp)*np.sin(inclination)) + (sqrt)**.5)/(2*(np.cos(inclination)**2*np.tan(alpha)**2 - np.sin(inclination)**2))
        s2 = (-1*( 2*zp*np.cos(inclination)*np.tan(alpha)**2 + 2*(ys - yp)*np.sin(inclination)) - (sqrt)**.5)/(2*(np.cos(inclination)**2*np.tan(alpha)**2 - np.sin(inclination)**2))

        if s1 < 0 and s2 < 0:
            s1 = 0
            s2 = 0
            ds = 0
        elif inclination > alpha:

            s_min = s1
            s_max = s2
            ds = (s_max - s_min)/(Npoints_pathlength-1)
            s = np.arange(s_min, s_max + 0.1*ds, ds)
            position[:,0] += xp
            position[:,1] += yp + s*np.sin(inclination)
            position[:,2] += zp + s*np.cos(inclination)
            r[:,:] = position[:,:] - pos_sec
            r_tot = (r[:,0]**2+r[:,1]**2+r[:,2]**2)**.5
            r_unit[:,0]= r[:,0]/r_tot
            r_unit[:,1]= r[:,1]/r_tot
            r_unit[:,2]= r[:,2]/r_tot
            theta_pos[:] = np.arctan((r[:,0]**2 + r[:,1]**2)**.5/r[:,2])
            # pl.plot(xp,yp, 'o', color = 'k')
            # pl.plot(xs, ys,'o')
            # pl.plot(position[:,0], position[:,1])
            # pl.xlim(-2.,2.)
            # pl.ylim(-2.,2.)
            # pl.show()

        elif inclination < alpha:

            s_min = np.max([s1,s2])
            s_max = 5.*s_min
            ds = (s_max - s_min)/(Npoints_pathlength-1)
            s = np.arange(s_min, s_max + 0.1*ds, ds)
            position[:,0] += xp
            position[:,1] += yp + s*np.sin(inclination)
            position[:,2] += zp + s*np.cos(inclination)
            r[:,:] = position[:,:] - pos_sec
            r_tot = (r[:,0]**2+r[:,1]**2+r[:,2]**2)**.5
            r_unit[:,0]= r[:,0]/r_tot
            r_unit[:,1]= r[:,1]/r_tot
            r_unit[:,2]= r[:,2]/r_tot
            theta_pos[:] = np.arctan((r[:,0]**2 + r[:,1]**2)**.5/r[:,2])
            # pl.plot(xp,yp, 'o', color = 'k')
            # pl.plot(xs, ys,'o')
            # pl.plot(position[:,0], position[:,1])
            # pl.xlim(-2.,2.)
            # pl.ylim(-2.,2.)
            # pl.show()

    return s1, s2, s, ds, position, r, r_unit, theta_pos

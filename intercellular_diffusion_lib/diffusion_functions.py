import pandas as pd
import numpy as np


def stokes_einstein(x):
    return ((1.38e-23 * 298.15) / (6 * np.pi * 8.9e-4 * x))


def D_eff(D, q, cell_um):
    return (D*q*cell_um)/(D+q*cell_um)


def diffuse(u, dx2, D, b, dt):
    un = u.copy()
    Y, X = un.shape
    # TODO: conn = connectivity(un, pd_rate)
    # TODO: beta = production(un, b)
    for y in range(0, Y):
        for x in range(0, X):
            # Flux in the diffusion equation is positional dependant
            # therefore, the order of the variables is of the uptmost importance
            # Moving from left to right, up to down this becomes:
            Cs = []
            if y > 0:
                # Check if upper is available
                Cs.append(u[y-1, x])
            if y < Y-1:
                # Check if lower is available
                Cs.append(u[y+1, x])
            if x > 0:
                # Check if left is available
                Cs.append(u[y, x-1])
            if x < X-1:
                # Check if right is available
                Cs.append(u[y, x+1])
            if len(Cs) < 2:
                un[y, x] = diffuse_1point(u[y, x], D, *Cs, dx2, b, dt)
            elif len(Cs) < 3:
                un[y, x] = diffuse_2point(u[y, x], D, *Cs, dx2, b, dt)
            elif len(Cs) < 4:
                un[y, x] = diffuse_3point(u[y, x], D, *Cs, dx2, b, dt)
            elif len(Cs) == 4:
                un[y, x] = diffuse_4point(u[y, x], D, *Cs, dx2, b, dt)
            else:
                # No connectivity
                continue
    return un


def production(c, b):
    return np.zeros(c.shape)  # + (b * np.where)


def diffuse_1point(c0, D, x1, dx2, b, dt):
    c = c0 + dt*D * ((x1 - c0)/dx2) + b
    return c


def diffuse_2point(c0, D, x1, x2, dx2, b, dt):
    c = c0 + dt*D * ((x1 - 2*c0 + x2)/dx2) + b
    return c


def diffuse_3point(c0, D, x1, x2, y1, dx2, b, dt):
    c = c0 + dt*D * ((x1 - 3*c0 + x2 + y1)/dx2) + b
    return c


def diffuse_4point(c0, D, x1, x2, y1, y2, dx2, b, dt):
    c = c0 + dt*D * ((x1 - 4*c0 + x2 + y1 + y2)/dx2) + b
    return c


# These funcs are currently dead due to deprecation of pd_rate
# def pd_perm(x, pd_rate=0):
#     return x*pd_rate


# def connectivity(c, pd_rate):
#     """
#     Returns a connectivity mask for the current cell array
#     """
#     return pd_perm(np.ones(c.shape), pd_rate=pd_rate)

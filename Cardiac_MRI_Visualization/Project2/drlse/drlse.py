import math
import scipy
import numpy as np


def dirac(phi, epsilon):
    """ assume that phi is 2d  """
    row_num, col_num = phi.shape
    ret = np.zeros(phi.shape)
    for i in range(row_num):
        for j in range(col_num):
            if -epsilon <= phi[i, j] <= epsilon:
                ret[i, j] = (1/2/epsilon) * (1 + math.cos(math.pi * phi[i, j] / epsilon))
    return ret


def drlse_edge_detect(phi_0, g ,lambd, mu, alfa, epsilon, timestep, iter):
    phi = phi_0
    temp = np.dstack(np.gradient(g))
    vx, vy = temp[:, :, 0], temp[:, :, 1]
    for k in range(1, iter+1):
        phi = neumann_bound_cond(phi)
        temp = np.dstack(np.gradient(phi))
        phi_x, phi_y = temp[:, :, 0], temp[:, :, 1]
        s = np.sqrt(phi_x**2 + phi_y**2)
        smallNumber = 1e-10
        Nx = phi_x / (s+smallNumber)
        Ny = phi_y / (s+smallNumber)
        curvature = div(np.dstack([Nx, Ny]))
        dist_reg_term = distReg_p2(phi)
        dirac_phi = dirac(phi, epsilon)
        area_term = dirac_phi * g
        edge_term = dirac_phi * (vx*Nx + vy*Ny) + dirac_phi*g*curvature
        phi += timestep * (mu * dist_reg_term + lambd*edge_term + alfa*area_term)

    return phi


def distReg_p2(phi):
    temp = np.dstack(np.gradient(phi))
    phi_x, phi_y = temp[:, :, 0], temp[:, :, 1]
    s = np.sqrt(phi_x ** 2 + phi_y ** 2)
    a = (s >= 0) & (s <= 1)
    a = a.astype(float)
    b = s > 1
    b = b.astype(float)
    ps = a*np.sin(2*np.pi*s)/(2*np.pi) + b*(s-1)
    dps = ((ps != 0)*ps+(ps == 0))/((s != 0)*s+(s == 0))
    f = div(np.dstack([dps*phi_x - phi_x, dps*phi_y - phi_y])) + scipy.ndimage.filters.laplace(phi)
    return f


def div(f):
    """ assume that f is 2d """
    fx = f[:, :, 0]
    fy = f[:, :, 1]

    dfxdx = np.gradient(fx, axis=0)
    dfydy = np.gradient(fy, axis=1)
    divF = np.add(dfxdx, dfydy)

    return divF


def neumann_bound_cond(f):
    """" assume that f is 2d """
    nrow, ncol = f.shape
    g = np.array(f)

    g[0, 0] = g[2, 2]
    g[0, ncol-1] = g[2, ncol-3]
    g[nrow-1, 0] = g[nrow-3, 2]
    g[nrow-1, ncol-1] = g[nrow-3, ncol-3]

    g[0, 1:ncol-1] = g[2, 1:ncol-1]
    g[nrow-1, 1:ncol-1] = g[nrow-3, 1:ncol-1]

    g[1:nrow-1, 0] = g[1:nrow-1, 2]
    g[1:nrow-1, ncol-1] = g[1:nrow-1, ncol-3]
    return g

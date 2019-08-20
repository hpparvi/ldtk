from numpy import zeros, sqrt
from numba import njit, generated_jit


@njit
def m_trianqular_1d(mu, pv):
    a, b = sqrt(pv[0]), 2*pv[1]
    u, v = a * b, a * (1. - b)
    return 1. - u*(1.-mu) - v*(1.-mu)**2


@njit
def m_triangular_2d(mu, pv):
    nsets = pv.shape[0]
    model = zeros((nsets, mu.size))
    for i in range(nsets):
        a, b = sqrt(pv[i, 0]), 2 * pv[i, 1]
        u, v = a * b, a * (1. - b)
        model[i, :] = 1. - u * (1. - mu) - v * (1. - mu) ** 2
    return model


@njit
def m_triangular_3d(mu, pv):
    n_pvs = pv.shape[0]
    n_filters = pv.shape[1]
    model = zeros((n_pvs, n_filters, mu.size))
    for ipv in range(n_pvs):
        for ifl in range(n_filters):
            a, b = sqrt(pv[ipv, ifl, 0]), 2 * pv[ipv, ifl, 1]
            u, v = a * b, a * (1. - b)
            model[ipv, ifl, :] = 1. - u * (1. - mu) - v * (1. - mu) ** 2
    return model


@generated_jit(nopython=True)
def m_triangular(mu, pv):
    if pv.ndim == 1:
        return m_trianqular_1d
    elif pv.ndim == 2:
        return m_triangular_2d
    elif pv.ndim == 3:
        return m_triangular_3d

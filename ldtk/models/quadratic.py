from numpy import zeros
from numba import njit, generated_jit


@njit
def m_quadratic_1d(mu, pv):
    return 1. - pv[0]*(1.-mu) - pv[1]*(1.-mu)**2


@njit
def m_quadratic_2d(mu, pv):
    nsets = pv.shape[0]
    model = zeros((nsets, mu.size))
    for i in range(nsets):
        model[i, :] = 1. - pv[i, 0]*(1.-mu) - pv[i, 1]*(1.-mu)**2
    return model


@njit
def m_quadratic_3d(mu, pv):
    n_pvs = pv.shape[0]
    n_filters = pv.shape[1]
    model = zeros((n_pvs, n_filters, mu.size))
    for ipv in range(n_pvs):
        for ifl in range(n_filters):
            model[ipv, ifl, :] = 1. - pv[ipv, ifl, 0]*(1.-mu) - pv[ipv, ifl, 1]*(1.-mu)**2
    return model


@generated_jit(nopython=True)
def m_quadratic(mu, pv):
    if pv.ndim == 1:
        return m_quadratic_1d
    elif pv.ndim == 2:
        return m_quadratic_2d
    elif pv.ndim == 3:
        return m_quadratic_3d

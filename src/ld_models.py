import numpy as np

class LDModel(object):
    npar = None

    def __init__(self):
        raise NotImplementedError

    def __call__(self, mu, pv):
        raise NotImplementedError

    @classmethod
    def eval(cl,mu,pv):
        raise NotImplementedError


class LinearModel(LDModel):
    """Linear limb darkening model (Mandel & Agol, 2001)."""
    npar = 1

    @classmethod
    def evaluate(cl,mu,pv):
        return 1. - pv[0]*(1.-mu)


class QuadraticModel(LDModel):
    """Quadratic limb darkening model (Mandel & Agol, 2001)."""
    npar = 2

    @classmethod
    def evaluate(cl,mu,pv):
        return 1. - pv[0]*(1.-mu) - pv[1]*(1.-mu)**2


class NonlinearModel(LDModel):
    """Nonlinear limb darkening model (Mandel & Agol, 2001)."""
    npar = 4

    @classmethod
    def evaluate(cl,mu,pv):
        return (1. - (pv[0]*(1.-mu**0.5) + pv[1]*(1.-mu**0.5) +
                      pv[2]*(1.-mu**1.5) + pv[3]*(1.-mu**2  )))


class GeneralModel(LDModel):
    """General limb darkening model (Gimenez, 2006)"""
    npar = None

    @classmethod
    def evaluate(cl,mu,pv):
        return 1. - np.sum([c*(1.-mu**i) for i,c in enumerate(pv)])


models = {'linear':    LinearModel,    'ln': LinearModel,
          'quadratic': QuadraticModel, 'qd': QuadraticModel,
          'nonlinear': NonlinearModel, 'nl': NonlinearModel,
          'general':   GeneralModel,   'ge': GeneralModel}

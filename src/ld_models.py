class LDModel(object):
    npar = None

    def __init__(self):
        raise NotImplementedError

    def __call__(self, mu, pv):
        raise NotImplementedError

    @classmethod
    def eval(cl,mu,pv):
        raise NotImplementedError


class QuadraticModel(LDModel):
    """Quadratic limb-darkening law as  described in (Mandel & Agol, 2001).
    """

    npar = 2

    @classmethod
    def evaluate(cl,mu,pv):
        return 1. - pv[0]*(1.-mu) - pv[1]*(1.-mu)**2


models = {'quadratic':QuadraticModel, 'qd':QuadraticModel}

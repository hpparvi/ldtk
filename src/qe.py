from numpy import asarray, zeros_like, interp, loadtxt

class QE(object):
    def __init__(self, instrument):
        self.instrument = instrument

    def __call__(self, wl):
        raise NotImplementedError


class TabulatedQE(QE):
    """Quantum efficiency tabulated as a function of wavelength.
    """
    def __init__(self, instrument, wl_or_fname, tm=None):
        """
        Parameters

        name        : str           filter name
        wl_or_fname : array or str  a list of wavelength or filename
        tm          : array         a list of transmission values
        """
        super(TabulatedQE,self).__init__(instrument)
        if isinstance(wl_or_fname,str):
            self.wl, self.tm = loadtxt(wl_or_fname).T
        else:
            self.wl = asarray(wl)
            self.tm = asarray(tm)
        self.tm *= tmf

    def __call__(self, wl):
        return interp(wl, self.wl, self.tm, 0., 0.)

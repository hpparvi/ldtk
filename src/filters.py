from numpy import asarray, zeros_like, interp, loadtxt

class Filter(object):
    def __init__(self, name):
        self.name = name

    def __call__(self, wl):
        raise NotImplementedError


class TabulatedFilter(Filter):
    """Tabulated filter where the transmission is
       listed as a function of wavelength.
    """
    def __init__(self, name, wl_or_fname, tm=None, tmf=1.):
        """
        Parameters

        name        : str           filter name
        wl_or_fname : array or str  a list of wavelength or filename
        tm          : array         a list of transmission values
        tmf         : float         should be set to 1e-2 if the transmission is given in percents
        """
        super(TabulatedFilter,self).__init__(name)
        if isinstance(wl_or_fname,str):
            self.wl, self.tm = loadtxt(wl_or_fname).T
        else:
            self.wl = asarray(wl)
            self.tm = asarray(tm)
        self.tm *= tmf

    def __call__(self, wl):
        return interp(wl, self.wl, self.tm, 0., 0.)


class BoxcarFilter(Filter):
    """A simple boxcar filter."""
    def __init__(self, name, wl_min, wl_max):
        super(BoxcarFilter,self).__init__(name)
        self.wl_min = wl_min
        self.wl_max = wl_max
        
    def __call__(self, wl):
        w = zeros_like(wl)
        w[(wl>self.wl_min) & (wl<self.wl_max)] = 1.
        return w

"""
Limb darkening toolkit
Copyright (C) 2015  Hannu Parviainen <hpparvi@gmail.com>

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

import os
import warnings
import astropy.io.fits as pf

from ftplib import FTP
from itertools import product
from os.path import join, exists
from pickle import dump, load

from astropy.utils.exceptions import AstropyWarning
from tqdm.auto import tqdm

from .core import ldtk_root, FN_TEMPLATE, TEFF_POINTS, LOGG_POINTS, Z_POINTS, is_inside, SpecIntFile, message

warnings.filterwarnings('ignore')

edir_medres = 'SpecIntFITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011'
edir_lowres = 'SpecInt50FITS/PHOENIX-ACES-AGSS-COND-SPECINT-2011'


class Client(object):
    def __init__(self, limits=None, verbosity: int = 1, offline_mode: bool = False,
                 update_server_file_list: bool = False, cache: str = None, lowres: bool = True):
        """LDTk client

        Args:
            limits: Teff, log g, and metallicity limits.
            verbosity: verbosity level.
            offline_mode: tries to use cached files only if set to ``True``.
            update_server_file_list: updates the server file list if set to ``True``.
            cache: path to the cache directory where the downloaded files should be stored.
            lowres: uses spectra binned to 5 nm if set to ``True``, otherwise uses the original files.
        """
        self.eftp = 'phoenix.astro.physik.uni-goettingen.de'
        self.use_lowres = lowres
        self.files = None
        self.verbosity = verbosity
        self.offline_mode = offline_mode

        if lowres:
            self.edir = edir_lowres
            self._cache = cache or join(ldtk_root, 'cache_lowres')
        else:
            self.edir = edir_medres
            self._cache = cache or join(ldtk_root, 'cache')

        self._server_file_list = join(ldtk_root, 'server_file_list.pkl')

        if not exists(self._cache):
            os.mkdir(self._cache)

        if exists(self._server_file_list) and not update_server_file_list:
            with open(self._server_file_list, 'rb') as fin:
                self.files_in_server = load(fin)
        else:
            self.files_in_server = self.get_server_file_list()
            with open(self._server_file_list, 'wb') as fout:
                dump(self.files_in_server, fout)

        if limits:
            self.set_limits(*limits)

    def _local_path(self, teff_or_fn, logg=None, z=None):
        """Creates the path to the local version of the file."""
        fn = teff_or_fn if isinstance(teff_or_fn, str) else self.create_name(teff_or_fn, logg, z)
        return join(self._cache, 'Z' + fn[13:17], fn)

    def _local_exists(self, teff_or_fn, logg=None, z=None):
        """Tests if a file exists in the local cache. """
        return exists(self._local_path(teff_or_fn, logg, z))

    def create_name(self, teff, logg, z):
        """Creates a SPECINT filename given teff, logg, and z."""
        return FN_TEMPLATE.format(teff=int(teff), logg=logg, z=z)

    def set_limits(self, teff_lims, logg_lims, z_lims):
        self.teffl = teff_lims
        self.teffs = is_inside(TEFF_POINTS, teff_lims)
        self.nteff = len(self.teffs)
        self.loggl = logg_lims
        self.loggs = is_inside(LOGG_POINTS, logg_lims)
        self.nlogg = len(self.loggs)
        self.zl = z_lims
        self.zs = is_inside(Z_POINTS, z_lims)
        self.nz = len(self.zs)
        self.pars = [p for p in product(self.teffs, self.loggs, self.zs)]
        self.files = [SpecIntFile(*p, cache=self._cache) for p in product(self.teffs, self.loggs, self.zs)]
        self.clean_file_list()
        self.check_file_corruption([f.local_path for f in self.files])

        self.not_cached = len(self.files) - sum([f.local_exists for f in self.files])
        fsize = 0.334 if self.use_lowres else 16
        if self.not_cached > 0:
            message("Need to download {:d} files, approximately {:.2f} MB".format(self.not_cached, fsize * self.not_cached))

    def get_server_file_list(self):
        ftp = FTP(self.eftp)
        ftp.login()
        ftp.cwd(self.edir)
        files_in_server = {}
        zdirs = sorted(ftp.nlst())
        for zdir in tqdm(zdirs, desc='Updating server file list'):
            ftp.cwd(zdir)
            files_in_server[zdir] = sorted(ftp.nlst())
            ftp.cwd('..')
        ftp.close()
        return files_in_server

    def files_exist(self, files=None):
        """Tests if a file exists in the FTP server."""
        return [f.name in self.files_in_server[f._zstr] for f in self.files]

    def clean_file_list(self):
        """Removes files not in the FTP server."""
        self.files = [f for f, e in zip(self.files, self.files_exist()) if e]

    def download_uncached_files(self, force=False):
        """Downloads the uncached files to a local cache."""

        if self.not_cached > 0 or force:
            ftp = FTP(self.eftp)
            ftp.login()
            ftp.cwd(self.edir)
            file_paths = []

            with tqdm(desc='LDTk downloading uncached files', total=self.not_cached) as pb:
                for fid, f in enumerate(self.files):
                    if not exists(join(self._cache, f._zstr)):
                        os.mkdir(join(self._cache, f._zstr))
                    if not f.local_exists or force:
                        ftp.cwd(f._zstr)
                        localfile = open(f.local_path, 'wb')
                        ftp.retrbinary('RETR ' + f.name, localfile.write)
                        localfile.close()
                        ftp.cwd('..')
                        self.not_cached -= 1
                        pb.update(1)
                        file_paths.append(f.local_path)
                    else:
                        if self.verbosity > 1:
                            print('Skipping an existing file: {:s}'.format(f.name))
            ftp.close()
            return self.check_file_corruption(file_paths)
        return False

    def check_file_corruption(self, files):
        """Checks local cache for corrupted files."""

        with warnings.catch_warnings():
            warnings.simplefilter('error', category=AstropyWarning)
            corrupt = False
            for file in files:
                if exists(file) and file.endswith('.fits'):
                    try:
                        with pf.open(file, checksum=True) as hdul:
                            pass
                    except (AstropyWarning, OSError):
                        corrupt = True
                        del hdul[0].data
                        os.remove(file)
            return corrupt

    @property
    def local_filenames(self):
        return [f.local_path for f in self.files]

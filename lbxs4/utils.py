import os
import numpy as np
from functools import wraps
import time
import hashlib
import healpy as hp

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        t = time.strftime("%H:%M:%S", time.gmtime(te-ts))
        print(f"func:{f.__name__} took: {t}")
        return result
    return wrap

def hash_array(arr):
    return hashlib.sha224(arr).hexdigest()

def camb_clfile(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls, tensCls or ScalCls types) returned as a dict of numpy arrays.
    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.
    """
    with open(fname) as f:
        firstline = next(f)
    keys = [i.lower() for i in firstline.split(' ') if i.isalpha()][1:]
    cols = np.loadtxt(fname).transpose()

    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)

    cls = {k : np.zeros(lmax + 1, dtype=float) for k in keys}

    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int) # type: ignore
    w = lambda ell :ell * (ell + 1) / (2. * np.pi)
    wpp = lambda ell : ell ** 2 * (ell + 1) ** 2 / (2. * np.pi)
    wptpe = lambda ell :np.sqrt(ell.astype(float) ** 3 * (ell + 1.) ** 3) / (2. * np.pi) 
    for i, k in enumerate(keys):
        if k == 'pp':
            we = wpp(ell)
        elif 'p' in k and ('e' in k or 't' in k):
            we = wptpe(ell)
        else:
            we = w(ell)
        cls[k][ell[idc]] = cols[i + 1][idc] / we[idc]
    return cls

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret


def arc2cl(arc):
    return np.radians(arc/60)**2
def cl2arc(cl):
    return np.rad2deg(np.sqrt(cl))*60

def noise(arr):
    return cl2arc(1/sum(1/arc2cl(arr)))

def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map

    Parameters
    ----------
    m : map or array of maps
      map(s) to be rotated
    coord : sequence of two character
      First character is the coordinate system of m, second character
      is the coordinate system of the output map. As in HEALPIX, allowed
      coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)

    Example
    -------
    The following rotate m from galactic to equatorial coordinates.
    Notice that m can contain both temperature and polarization.
    >>>> change_coord(m, ['G', 'C'])
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))

    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))

    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]

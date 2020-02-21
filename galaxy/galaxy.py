
import numpy as np
from astropy.io import fits

from spextra import Spextrum

from scopesim.source.source_templates import Source

from .models import GalaxyBase


def galaxysource(sed,           # The SED of the galaxy
                 z,             # redshift
                 mag,           # magnitude
                 filter,        # passband
                 plate_scale,   # the plate scale "/pix
                 spec_scale,    # the spectral scale A/pix (or another unit)
                 r_eff,         # effective radius
                 n,             # sersic index
                 ellip,         # ellipticity
                 theta,         # position angle
                 extend,        # extend in units of r_eff
                 vmax=0,        # maximum velocity
                 sigma=0):      # velocity dispersion
    """
    Galaxy is created always at (x,y)=(0,0) so we don't need to create a huge image containing
    the whole FoV

    Image size depends on parameter extend (in units of r_eff) and
    wavelength is limited by the passband with some padding so
    don't create huge blocks


    Parameters
    ----------
    sed
    z
    r_eff
    n

    Returns
    -------

    """
    image_size = (r_eff * extend / plate_scale)   # TODO: Needs unit check
    x_0 = int(image_size/2)
    y_0 = int(image_size/2)
    x, y = np.grid(np.arange(int(image_size), int(image_size)))
    galaxy = GalaxyBase(x_0=x_0,
                        y_0=y_0,
                        r_eff=r_eff,
                        n=n,
                        ellip=ellip,
                        theta=theta,
                        vmax=vmax,
                        sigma=sigma)

    sp = Spextrum(sed).redshift(z)
    src = Source()

    if (vmax == 0) and (sigma == 0):  # imaging case
        img = galaxy.flux(x, y)
        scaled_sp = sp.scale_to_magnitude(amplitude=mag, filter_name=filter)

        src.fields  = img
        src.spectra = scaled_sp

    return src

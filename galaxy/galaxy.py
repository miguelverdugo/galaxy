
import numpy as np
from astropy.io import fits
import astropy.units as u

from spextra import Spextrum

from scopesim.source.source_templates import Source

from .models import GalaxyBase


def galaxysource(sed,           # The SED of the galaxy
                 z=0,             # redshift
                 mag=15,           # magnitude
                 filter_name="g",        # passband
                 plate_scale=0.1,   # the plate scale "/pix
                 r_eff=25,         # effective radius
                 n=4,             # sersic index
                 ellip=0.1,         # ellipticity
                 theta=0,         # position angle
                 extend=2):        # extend in units of r_eff

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
    if isinstance(mag, u.Quantity) is False:
        mag = mag * u.ABmag
    if isinstance(plate_scale, u.Quantity) is False:
        plate_scale = plate_scale * u.arcsec
    if isinstance(r_eff, u.Quantity) is False:
        r_eff = r_eff * u.arcsec

    sp = Spextrum(sed).redshift(z=z)
    scaled_sp = sp.scale_to_magnitude(amplitude=mag, filter_name=filter_name)

    r_eff = r_eff.to(u.arcsec)
    plate_scale = plate_scale.to(u.arcsec)

    image_size = 2 * (r_eff.value * extend / plate_scale.value)  # TODO: Needs unit check
    x_0 = image_size // 2
    y_0 = image_size // 2

    x, y = np.meshgrid(np.arange(image_size),
                       np.arange(image_size))

    galaxy = GalaxyBase(x=x, y=y, x_0=x_0, y_0=y_0,
                        r_eff=r_eff.value, amplitude=1,  n=n,
                        ellip=ellip, theta=theta)

    img = galaxy.intensity

    w, h = img.shape
    header = fits.Header({"CRPIX1": w // 2,
                          "CRPIX2": h // 2,
                          "CRVAL1": 0,
                          "CRVAL2": 0,
                          "CDELT1": plate_scale.to(u.deg).value,
                          "CDELT2": plate_scale.to(u.deg).value,
                          "CUNIT1": "DEG",
                          "CUNIT2": "DEG"})

    hdu = fits.PrimaryHDU(data=img, header=header)

    src = Source()
    src.spectra = [scaled_sp]
    src.fields = [hdu]

    return src


def galaxysource3d(sed,           # The SED of the galaxy
                   z=0,             # redshift
                   mag=15,           # magnitude
                   filter_name="g",        # passband
                   plate_scale=0.1,   # the plate scale "/pix
                   r_eff=25,         # effective radius
                   n=4,             # sersic index
                   ellip=0.1,         # ellipticity
                   theta=0,         # position angle
                   vmax=100,
                   sigma=100,
                   extend=2,        # extend in units of r_eff
                   ngrid=10):       # griding parameter

    if isinstance(mag, u.Quantity) is False:
        mag = mag * u.ABmag
    if isinstance(plate_scale, u.Quantity) is False:
        plate_scale = plate_scale * u.arcsec
    if isinstance(r_eff, u.Quantity) is False:
        r_eff = r_eff * u.arcsec
    if isinstance(vmax, u.Quantity) is False:
        vmax = vmax*u.km/u.s
    if isinstance(vmax, u.Quantity) is False:
        sigma = sigma*u.km/u.s

    sp = Spextrum(sed).redshift(z=z)
    scaled_sp = sp.scale_to_magnitude(amplitude=mag, filter_name=filter_name)

    r_eff = r_eff.to(u.arcsec)
    plate_scale = plate_scale.to(u.arcsec)
    vmax = vmax.to(u.km/u.s)
    sigma = sigma.to(u.km/u.s)

    image_size = 2 * (r_eff.value * extend / plate_scale.value)  # TODO: Needs unit check
    x_0 = image_size // 2
    y_0 = image_size // 2

    x, y = np.meshgrid(np.arange(image_size),
                       np.arange(image_size))

    galaxy = GalaxyBase(x=x, y=y, x_0=x_0, y_0=y_0,
                        r_eff=r_eff.value, amplitude=1, n=n,
                        ellip=ellip, theta=theta)

    img = galaxy.intensity
    velfield = galaxy.velfield
    dispmap = galaxy.dispmap
    masks = galaxy.get_masks(ngrid=ngrid)

    w, h = img.shape
    header = fits.Header({"CRPIX1": w // 2,
                          "CRPIX2": h // 2,
                          "CRVAL1": 0,
                          "CRVAL2": 0,
                          "CDELT1": plate_scale.to(u.deg).value,
                          "CDELT2": plate_scale.to(u.deg).value,
                          "CUNIT1": "DEG",
                          "CUNIT2": "DEG"})

    src = Source()
    src.fields = []
    src.spectra = []

    hdu = fits.PrimaryHDU()
    hdulist = fits.HDUList(hdu)
    for m in masks:

        data = m*img
        hdulist.append(fits.PrimaryHDU(data=data, header=header))

        vel = np.median(m*velfield)
        sigma = np.median(m*dispmap)
        spectra = scaled_sp.redshift(vel=vel)  # TODO: check if broadening is working as expected


    src.fields = hdulist

    return src
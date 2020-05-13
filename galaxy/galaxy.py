
import numpy as np
from astropy.io import fits
import astropy.units as u

from spextra import Spextrum

from scopesim.source.source_templates import Source

from .models import GalaxyBase


def galaxy(sed,           # The SED of the galaxy
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
    Creates a source object of a galaxy described by its Sersic index and other
    parameters.

    This function is ideal for imaging

    Parameters
    ----------
    sed
    z
    r_eff
    n

    Returns
    -------
    src : scopesim.Source
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

    img = galaxy.intensity.value

    w, h = img.shape
    header = fits.Header({"NAXIS": 2,
                          "NAXIS1": 2*x_0 + 1,
                          "NAXIS2": 2*y_0 + 1,
                          "CRPIX1": w // 2,
                          "CRPIX2": h // 2,
                          "CRVAL1": 0,
                          "CRVAL2": 0,
                          "CDELT1": -1*plate_scale.to(u.deg).value,
                          "CDELT2": plate_scale.to(u.deg).value,
                          "CUNIT1": "DEG",
                          "CUNIT2": "DEG",
                          "CTYPE1": 'RA---TAN',
                          "CTYPE2": 'DEC--TAN',
                          "SPEC_REF": 0})

    hdu = fits.ImageHDU(data=img, header=header)
    hdu.writeto("deleteme.fits", overwrite=True)
    src = Source()
    src.spectra = [scaled_sp]
    src.fields = [hdu]

    return src


def galaxy3d(sed,           # The SED of the galaxy
             z=0,             # redshift
             mag=15,           # magnitude
             filter_name="g",        # passband
             plate_scale=1,   # the plate scale "/pix
             r_eff=25,         # effective radius
             n=4,             # sersic index
             ellip=0.1,         # ellipticity
             theta=0,         # position angle
             vmax=100,
             sigma=100,
             extend=2,        # extend in units of r_eff
             ngrid=10):       # griding parameter

    """
    Creates a source object of a galaxy described by its Sersic index and other
    parameters. It also generates a velocity field (set by vmax) and
    a velocity dispersion map (set by sigma).

    The maps are binned according to the ngrid parameter, higher ngrid will create
    finer binned fields but it may increase the computation time.

    The ngrid parameter does not specify the number of bins. A ngrid=10 will create
    around 40 independent regions whilst a ngrid of 100 will create around 2300 regions

    This function is ideal for spectroscopy

    Parameters
    ----------
    sed
    z
    mag
    filter_name
    plate_scale
    r_eff
    n
    ellip
    theta
    vmax
    sigma
    extend
    ngrid

    Returns
    -------

    src : scopesim.Source
    """

    if isinstance(mag, u.Quantity) is False:
        mag = mag * u.ABmag
    if isinstance(plate_scale, u.Quantity) is False:
        plate_scale = plate_scale * u.arcsec
    if isinstance(r_eff, u.Quantity) is False:
        r_eff = r_eff * u.arcsec
    if isinstance(vmax, u.Quantity) is False:
        vmax = vmax*u.km/u.s
    if isinstance(sigma, u.Quantity) is False:
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
                        ellip=ellip, theta=theta, vmax=vmax, sigma=sigma)

    intensity = galaxy.intensity
    velocity = galaxy.velocity
    dispersion = galaxy.dispersion
    masks = galaxy.get_masks(ngrid=ngrid)

    w, h = intensity.shape
    header = fits.Header({"NAXIS": 2,
                          "NAXIS1": 2 * x_0 + 1,
                          "NAXIS2": 2 * y_0 + 1,
                          "CRPIX1": w // 2,
                          "CRPIX2": h // 2,
                          "CRVAL1": 0,
                          "CRVAL2": 0,
                          "CDELT1": -1*plate_scale.to(u.deg).value,
                          "CDELT2": plate_scale.to(u.deg).value,
                          "CUNIT1": "DEG",
                          "CUNIT2": "DEG",
                          "CTYPE1": 'RA---TAN',
                          "CTYPE2": 'DEC--TAN',
                          "SPEC_REF": 0})

    src = Source()
    src.fields = []
    src.spectra = []
    total_flux = np.sum(intensity.value)
    #hdu = fits.PrimaryHDU()
    hdulist = []
    for i, m in enumerate(masks):

        data = m * intensity.value
        factor = np.sum(data) / total_flux
        header["SPEC_REF"] = i

        med_vel = np.median(m*velocity)
        med_sig = np.median(m*dispersion)
        # TODO: check in speXtra if broadening is working as expected
        spec = scaled_sp.redshift(vel=med_vel) * factor
        hdu = fits.ImageHDU(data=data, header=header)
        hdulist.append(hdu)
        src.spectra.append(spec)

    src.fields = fits.HDUList(hdulist)

    return src



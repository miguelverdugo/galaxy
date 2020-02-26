
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
    print(image_size, "*"*15)
    x, y = np.meshgrid(np.arange(image_size),
                       np.arange(image_size))

    galaxy = GalaxyBase(x_0=x_0, y_0=y_0,
                        r_eff=r_eff.value, amplitude=1,  n=n,
                        ellip=ellip, theta=theta)

    img = galaxy.flux(x, y)

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



def split_moments(ngrid=10,
                  plate_scale=0.1,  # the plate scale "/pix
                  r_eff=25,  # effective radius
                  n=4,  # sersic index
                  ellip=0.1,  # ellipticity
                  theta=0,  # position angle
                  extend=2,
                  vmax=100,
                  sigma=100):  # extend in units of r_eff
    """
    Returns

    Parameters
    ----------
    ngrid

    Returns
    -------

    A list of ndarrays and velocities and sigmas

    """
    image_size = 2 * (r_eff * extend / plate_scale)  # TODO: Needs unit check
    x_0 = image_size // 2
    y_0 = image_size // 2
    print(image_size, "*" * 15)
    x, y = np.meshgrid(np.arange(image_size),
                       np.arange(image_size))

    galaxy = GalaxyBase(x_0=x_0, y_0=y_0,
                        r_eff=r_eff/plate_scale, amplitude=1, n=n,
                        ellip=ellip, theta=theta,
                        vmax=vmax, sigma=sigma)

    img = galaxy.flux(x, y)
    img = img/np.sum(img)
    vel_field = galaxy.velfield(x, y)
    sigma_field = galaxy.dispfield(x, y)

    total_field = sigma_field

    total_split = np.linspace(np.min(total_field), np.max(total_field), ngrid+1)
    sigma_split = np.linspace(np.min(sigma_field), np.max(sigma_field), ngrid//2)

    vel_split = np.linspace(np.min(vel_field), np.max(vel_field), ngrid + 1)

    vel_grid = np.round((ngrid/2)*vel_field/np.max(vel_field)) * np.max(vel_field)
    sigma_grid = np.round((ngrid/2)*sigma_field/np.max(sigma_field)) * np.max(sigma_field)
    total_field = np.round(vel_grid + sigma_grid)
    uniques = np.unique(total_field)

#    img = np.round((ngrid)*np.log10(img)/np.max(np.log10(img))) * np.max(img)
    subfields = []
    i = 0
    for value in uniques:

        mask = np.ma.masked_where(total_field ==value, total_field).mask

#        subfield = np.ma.masked_array(total_field, np.logical_not(mask))
        vel_median = np.ma.mean(np.ma.masked_array(vel_field, np.logical_not(mask)))
        sig_median = np.ma.median(np.ma.masked_array(sigma_field, np.logical_not(mask)))
        print(vel_median, sig_median)
        subfields.append(i + 0*np.ma.masked_array(img, np.logical_not(mask)))
        i = i + 1


    return subfields




def galaxysource3d():
    pass


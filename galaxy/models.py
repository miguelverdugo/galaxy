# -*- coding: utf-8 -*-

import numpy as np

from astropy import units as u
from astropy.units import Quantity, UnitsError
from astropy.utils.decorators import deprecated
from astropy.modeling.core import (Fittable1DModel, Fittable2DModel,
                                   ModelDefinitionError)

from astropy.modeling.parameters import Parameter, InputParameterError
from astropy.modeling.utils import ellipse_extent


TWOPI = 2 * np.pi
FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
GAUSSIAN_SIGMA_TO_FWHM = 2.0 * np.sqrt(2.0 * np.log(2.0))

# ----- Models copied from astropy.modeling.functional_models  -------------
# ----- They are here for completeness and templating     ------------------
class Sersic2D(Fittable2DModel):
    r"""
    Two dimensional Sersic surface brightness profile.

    Parameters
    ----------
    amplitude : float
        Surface brightness at r_eff.
    r_eff : float
        Effective (half-light) radius
    n : float
        Sersic Index.
    x_0 : float, optional
        x position of the center.
    y_0 : float, optional
        y position of the center.
    ellip : float, optional
        Ellipticity.
    theta : float, optional
        Rotation angle in radians, counterclockwise from
        the positive x-axis.

    See Also
    --------
    Gaussian2D, Moffat2D

    Notes
    -----
    Model formula:

    .. math::

        I(x,y) = I(r) = I_e\exp\left\{-b_n\left[\left(\frac{r}{r_{e}}\right)^{(1/n)}-1\right]\right\}

    The constant :math:`b_n` is defined such that :math:`r_e` contains half the total
    luminosity, and can be solved for numerically.

    .. math::

        \Gamma(2n) = 2\gamma (b_n,2n)

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        from astropy.modeling.models import Sersic2D
        import matplotlib.pyplot as plt

        x,y = np.meshgrid(np.arange(100), np.arange(100))

        mod = Sersic2D(amplitude = 1, r_eff = 25, n=4, x_0=50, y_0=50,
                       ellip=.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)


        plt.figure()
        plt.imshow(log_img, origin='lower', interpolation='nearest',
                   vmin=-1, vmax=2)
        plt.xlabel('x')
        plt.ylabel('y')
        cbar = plt.colorbar()
        cbar.set_label('Log Brightness', rotation=270, labelpad=25)
        cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
        plt.show()

    References
    ----------
    .. [1] http://ned.ipac.caltech.edu/level5/March05/Graham/Graham2.html
    """

    amplitude = Parameter(default=1)
    r_eff = Parameter(default=1)
    n = Parameter(default=4)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    ellip = Parameter(default=0)
    theta = Parameter(default=0)
    _gammaincinv = None

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta):
        """Two dimensional Sersic profile function."""

        if isinstance(theta, u.Quantity) is False:
            theta = theta * u.deg

        theta = theta.to(u.rad)

        if cls._gammaincinv is None:
            try:
                from scipy.special import gammaincinv
                cls._gammaincinv = gammaincinv
            except ValueError:
                raise ImportError('Sersic2D model requires scipy > 0.11.')

        bn = cls._gammaincinv(2. * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * sin_theta + (y - y_0) * cos_theta
        x_min = -(x - x_0) * cos_theta + (y - y_0) * sin_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-bn * (z ** (1 / n) - 1))


    @property
    def input_units(self):
        if self.x_0.unit is None:
            return None
        else:
            return {'x': self.x_0.unit,
                    'y': self.y_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x_0': inputs_unit['x'],
                'y_0': inputs_unit['x'],
                'r_eff': inputs_unit['x'],
                'theta': u.rad,
                'amplitude': outputs_unit['z']}

# ------------------ END --------------------------------------------------------

# ----- Copied from astromodels -----


class VelField(Fittable2DModel):
    r"""
        Two dimensional Velocity Field following arctan approximation

        Parameters
        ----------

        ellip : float, u.Quantity
            Ellipticity on the sky

        theta : float, u.Quantity
            Position angle of the major axis wrt to north (=up) measured counterclockwise,
        vmax : float, u.Quantity
            Constant rotation velocity for R>>rd,

        r_eff : float
            scale length of galaxy (assumed to be turnover radius)

        x0 : float, optional
            x position of the center.
        y0 : float, optional
            y position of the center.

        q : float, optional
            Disk thickness
        """
    vmax = Parameter(default=100)
    r_eff = Parameter(default=1)

    ellip = Parameter(default=0)    # maximum ellipticity 1 - q,  make tests
    theta = Parameter(default=0)

    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)

    q = Parameter(default=0.2)

    @staticmethod
    def evaluate(x, y, vmax, r_eff,  ellip, theta, x_0, y_0, q):
        """
        Two dimensional velocity field, arctan approximation
        TODO: Be consistent with Sersic2D

        """
        if isinstance(theta, u.Quantity) is False:
            theta = theta * u.deg

        r_d = r_eff  # For now,  for n=1  r_eff = 1.678 * r_d
        theta = (-theta).to(u.rad)
        # get inclination from ellipticity
        incl = np.arccos(np.sqrt(((1 - ellip) ** 2 - q ** 2) / (1 - q ** 2)))

        r = ((x - x_0) ** 2 + (y - y_0) ** 2) ** 0.5

        #   azimuthal angle in the plane of the galaxy = cos(theta) = cost
        cost = (-(x - x_0) * np.sin(theta) + (y - y_0) * np.cos(theta)) / (r + 0.00001)
        vrot = vmax*2 / np.pi*np.arctan(r/r_d)         #arctan model

        return vrot * np.sin(incl) * cost

    @property
    def input_units(self):
        if self.x_0.unit is None:
            return None
        else:
            return {'x': self.x_0.unit,
                    'y': self.y_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x_0': inputs_unit['x'],
                'y_0': inputs_unit['x'],
                'r_eff': inputs_unit['x'],
                'phi': u.deg,
                'vrot': outputs_unit['z']}

#------  End -----

class DispersionField(Fittable2DModel):

    r"""
        Two dimensional Velocity Dispersion
        At the moment just a gaussian distribution
        TODO: Investigate the possible real distributions

        Parameters
        ----------

        incl : float, u.Quantity
            Inclination inclination between the normal to the galaxy plane and the line-of-sight,

        phi : float, u.Quantity
            Position angle of the major axis wrt to north (=up) measured counterclockwise,
        sigma : float, u.Quantity
            velocity dispersion

        r_d : float
            scale length of galaxy (assumed to be turnover radius) it will be used as sigma

        x0 : float, optional
            x position of the center.
        y0 : float, optional
            y position of the center.
        """
   # incl = Parameter(default=45)
    ellip = Parameter(default=0)
    theta = Parameter(default=0)
    sigma = Parameter(default=100)
    r_eff = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)


    @staticmethod
    def evaluate(x, y, ellip, theta, sigma, r_eff, x_0, y_0):
        """
        TODO: Be consistent with Sersic2D

        """
     #   if isinstance(incl, u.Quantity) is False:
     #       incl = incl * u.deg
        if isinstance(theta, u.Quantity) is False:
            theta = theta * u.deg

        """Two dimensional Gaussian function"""
        theta = theta.to(u.rad)
    #    incl = incl.to(u.rad)

        # get ellipticity from inclination
       # ellip = 1 - np.sqrt((1 - q ** 2) * np.cos(incl) ** 2 + q ** 2)

        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * sin_theta + (y - y_0) * cos_theta
        x_min = -(x - x_0) * cos_theta + (y - y_0) * sin_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
        result = sigma * np.exp(-z**2)
        return result

        #  x_stddev = x_maj
        #  y_stddev = x_min
        #  cost2 = np.cos(theta) ** 2
        #  sint2 = np.sin(theta) ** 2
        #  sin2t = np.sin(2. * theta)
        #  xstd2 = x_stddev ** 2
        #  ystd2 = y_stddev ** 2
        #   xdiff = x - x_0
        #   ydiff = y - y_0
        #   a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
        #   b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
        #   c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))

        #  return sigma * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) + (c * ydiff ** 2)))


    @property
    def input_units(self):
        if self.x_0.unit is None:
            return None
        else:
            return {'x': self.x_0.unit,
                    'y': self.y_0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x_0': inputs_unit['x'],
                'y_0': inputs_unit['x'],
                'r_eff': inputs_unit['x'],
                'phi': u.deg,
                'sigma': outputs_unit['z']}


class GalaxyBase:

    def __init__(self, x, y, x_0, y_0, amplitude, r_eff, ellip, theta, n=4, vmax=0, sigma=0, q=0.2):

        self.x = x
        self.y = y
        self.amplitude = amplitude
        self.r_eff = r_eff
        self.x_0 = x_0
        self.y_0 = y_0
        self.ellip = ellip
        self.theta = theta
        self.n = n
        self.vmax = vmax
        self.sigma = sigma
        self.q = q

    @property
    def intensity(self):
        mod = Sersic2D(x_0=self.x_0,
                       y_0=self.y_0,
                       amplitude=self.amplitude,
                       r_eff=self.r_eff,
                       n=self.n,
                       ellip=self.ellip,
                       theta=self.theta)
        return mod(self.x, self.y)

    @property
    def velfield(self):
        """
        Velocity field according to the supplied parameters

        Returns
        -------

        """


        if self.vmax > 0:
            mod = VelField(x_0=self.x_0,
                           y_0=self.y_0,
                           r_eff=self.r_eff,
                           ellip=self.ellip,
                           theta=self.theta,
                           vmax=self.vmax,
                           q=self.q)
            result = mod(self.x, self.y)
        else:
            result = np.ones(shape=self.x.shape)

        return result


    @property
    def dispmap(self):
        """
        Velocity dispersion map according to the supplied parameters

        Returns
        -------

        """

        if self.sigma > 0:
            mod = DispersionField(x_0=self.x_0,
                                  y_0=self.y_0,
                                  r_eff=self.r_eff,
                                  ellip=self.ellip,
                                  theta=self.theta,
                                  sigma=self.sigma)
            result = mod(self.x, self.y)
        else:
            result = np.ones(shape=self.x.shape)

        return result


    def regrid(self, ngrid=10):
        """
        Regrid the smooth velocity field to regions with similar velocity and  velocity dispersion

        Parameters
        ----------
        ngrid: integer


        Returns
        -------
        A numpy array with sectors numbered
        """
        velfield = self.velfield
        dispfield = self.dispmap

        vel_grid = np.round((ngrid // 2) * velfield / np.max(velfield)) * np.max(velfield)
        sigma_grid = np.round((ngrid // 2) * dispfield / np.max(dispfield)) * np.max(dispfield)
        total_field = vel_grid + sigma_grid
        uniques = np.unique(total_field)
        idx = np.arange(uniques.size)

        for v, i in zip(uniques, idx):
            total_field[total_field == v] = i+1

        return total_field

    def get_masks(self, ngrid=10):

        grid = self.regrid(ngrid=ngrid)
        uniques = np.unique(grid)
        masklist = []
        for value in uniques:
            mask = np.ma.masked_where(grid == value, grid, copy=True).mask.astype(int)
            masklist.append(mask)

        return masklist



    @classmethod
    def from_file_moments(cls, filename, flux_ext=1, vel_ext=None, disp_ext=None):
        """
        Read the moments from a fits file and creates a Galaxy object.

        TODO: implement!

        Sometimes the moments are in different extensions, sometimes in different files

        Also some cases the moments are binned so we need to be careful with binning again




        Parameters
        ----------
        filename
        flux_ext
        vel_ext
        disp_ext

        Returns
        -------

        """

        pass









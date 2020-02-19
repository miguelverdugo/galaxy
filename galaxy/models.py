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

        if cls._gammaincinv is None:
            try:
                from scipy.special import gammaincinv
                cls._gammaincinv = gammaincinv
            except ValueError:
                raise ImportError('Sersic2D model requires scipy > 0.11.')

        bn = cls._gammaincinv(2. * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
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

        incl : float, u.Quantity
            Inclination inclination between the normal to the galaxy plane and the line-of-sight,

        phi : float, u.Quantity
            Position angle of the major axis wrt to north (=up) measured counterclockwise,
        vmax : float, u.Quantity
            Constant rotation velocity for R>>rd,

        r_d : float
            scale length of galaxy (assumed to be turnover radius)

        x0 : float, optional
            x position of the center.
        y0 : float, optional
            y position of the center.
        """
    incl = Parameter(default=45)
    phi = Parameter(default=0)
    vmax = Parameter(default=100)
    r_d = Parameter(default=1)
    x0 = Parameter(default=0)
    y0 = Parameter(default=0)
    v0 = Parameter(default=0)


    @classmethod
    def evaluate(cls, x, y, incl, phi, vmax, r_d, x0, y0, v0):
        """
        TODO: Be consistent with Sersic2D
        (x,y) kartesian sky coordinates,
        (x0,y0) kartesian sky coordiantes of rotation centre of galaxy,
        V0 velocity of centre wrt observer,
        incl inclination angle between the normal to the galaxy plane and the line-of-sight,
        phi position angle of the major axis wrt to north (=up) measured counterclockwise,
        Vmax constant rotation for R>>rd,
        rd scale length of galaxy (assumed to be turnover radius)
        """
        if isinstance(incl, u.Quantity) is False:
            incl = incl * u.deg
        if isinstance(phi, u.Quantity) is False:
            phi = phi * u.deg

        phi = phi.to(u.rad)
        incl = incl.to(u.rad)
        r = ((x - x0) ** 2 + (y - y0) ** 2) ** 0.5
        #   azimuthal angle in the plane of the galaxy = cos(theta) = cost
        cost = (-(x - x0) * np.sin(phi) + (y - y0) * np.cos(phi)) / (r + 0.00001)

        vrot = vmax*2/np.pi*np.arctan(r/r_d)         #arctan model

        return v0 + vrot * np.sin(incl) * cost

    @property
    def input_units(self):
        if self.x0.unit is None:
            return None
        else:
            return {'x': self.x0.unit,
                    'y': self.y0.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise UnitsError("Units of 'x' and 'y' inputs should match")
        return {'x0': inputs_unit['x'],
                'y0': inputs_unit['x'],
                'r_d': inputs_unit['x'],
                'phi': u.deg,
                'vrot': outputs_unit['z']}




#------ copied from astromodels
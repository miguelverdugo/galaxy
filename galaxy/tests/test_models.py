
from galaxy.models import Sersic2D, VelField, DispersionField
import numpy as np
import matplotlib.pyplot as plt
import pytest


PLOTS = True

class TestSersic:

    def test_simple(self, plot=PLOTS):
        """
        See if example runs
        """
        x, y = np.meshgrid(np.arange(100), np.arange(100))

        mod = Sersic2D(amplitude=1, r_eff=25, n=4, x_0=50, y_0=50,
                       ellip=.5, theta=-1)
        img = mod(x, y)
        log_img = np.log10(img)

        if plot is True:
            plt.figure()
            plt.imshow(log_img, origin='lower', interpolation='nearest',
                   vmin=-1, vmax=2)
            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar()
            cbar.set_label('Log Brightness', rotation=270, labelpad=25)
            cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
            plt.show()
            assert True


class TestVelField:

    def test_simple(self, plot=PLOTS):
        x, y = np.meshgrid(np.arange(100), np.arange(100))

        vmax = 100
        mod = VelField(vmax=vmax, r_eff=25, x_0=50, y_0=50,
                       ellip=.5, theta=0)
        img = mod(x, y)
        max = np.max(img)
        min = np.min(img)


        if plot is True:
            plt.figure()
            plt.imshow(img, origin='lower', interpolation='nearest')

            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar()
            cbar.set_label('Velocity', rotation=270, labelpad=25)
           # cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
            plt.show()



class TestDispField:

    def test_simple(self, plot=PLOTS):
        x, y = np.meshgrid(np.arange(100), np.arange(100))

        sigma = 100
        mod = DispersionField(sigma=sigma, r_eff=25, x_0=50, y_0=50,
                              ellip=.0, theta=0)
        img = mod(x, y)

        max = np.max(img)
        assert np.isclose(sigma, max)

        if plot is True:
            plt.figure()
            plt.imshow(img, origin='lower', interpolation='nearest')

            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar()
            cbar.set_label('Dispersion', rotation=270, labelpad=25)
           # cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
            plt.show()

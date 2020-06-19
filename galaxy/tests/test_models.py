
from galaxy.models import Sersic2D, VelField, DispersionField, GalaxyBase
from galaxy import galaxy, galaxy3d
import numpy as np
import matplotlib.pyplot as plt
import pytest
from scopesim.source.source_templates import Source


PLOTS = True

# Testing the basic models

class TestSersic:

    def test_simple(self, plot=PLOTS):
        """
        See if example runs
        """
        x, y = np.meshgrid(np.arange(100), np.arange(100))

        mod = Sersic2D(amplitude=1, r_eff=25, n=4, x_0=50, y_0=50,
                       ellip=.5, theta=30)
        img = mod(x, y)
        log_img = np.log10(img)

        if plot is True:
            plt.figure(figsize=(7, 7))
            plt.imshow(log_img, origin='lower', interpolation='nearest',
                       vmin=-1, vmax=2, cmap="inferno")
            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar(fraction=0.04)
            cbar.set_label('Log Brightness', rotation=270, labelpad=25)
            cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
            plt.show()
            assert True


class TestVelField:

    def test_simple(self, plot=PLOTS):
        x, y = np.meshgrid(np.arange(100), np.arange(100))

        vmax = 100
        mod = VelField(vmax=vmax, r_eff=15, x_0=50, y_0=50,
                       ellip=.5, theta=30)
        img = mod(x, y)
        max = np.max(img)
        min = np.min(img)


        if plot is True:
            plt.figure(figsize=(7, 7))
            plt.imshow(img, origin='lower', interpolation='nearest')
            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar(fraction=0.04, pad=0.02)
            cbar.set_label('Velocity [km/s]', rotation=270, labelpad=15)
           # cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
            plt.show()



class TestDispField:

    def test_simple(self, plot=PLOTS):
        x, y = np.meshgrid(np.arange(200), np.arange(200))

        sigma = 100
        mod = DispersionField(sigma=sigma, r_eff=45, x_0=100, y_0=100,
                              ellip=.5, theta=30)
        img = mod(x, y)

        max = np.max(img)
        assert np.isclose(sigma, max)

        if plot is True:
            plt.figure(figsize=(8, 8))
            plt.imshow(img, origin='lower', interpolation='nearest')

            plt.xlabel('x')
            plt.ylabel('y')
            cbar = plt.colorbar(fraction=0.04, pad=0.02)
            cbar.set_label('Dispersion', rotation=270, labelpad=15)
           # cbar.set_ticks([-1, 0, 1, 2], update_ticks=True)
            plt.show()


class TestGalaxy1D:

    def test_galaxy_creation(self):

        gal = galaxy(sed="kc96/s0", z=0)
        assert isinstance(gal, Source)

    def test_plot_galaxy(self, plot=PLOTS):
        gal = galaxy(sed="kc96/s0", z=0)
        data = gal.fields[0].data
        print(data)
        if plot is True:
            plt.imshow(np.log10(data), origin='lower', interpolation='nearest')
            plt.show()



class TestGalaxyBase:


    def test_regrid(self):

        r_eff = 37  # effective radius
        n = 1  # sersic index
        ellip = 0.5  # ellipticity
        theta = 30  # position angle

        vmax = 100
        sigma = 100  # extend in units of r_eff

        x, y = np.meshgrid(np.arange(200), np.arange(200))
        galaxy = GalaxyBase(x, y, x_0=100, y_0=100,
                            r_eff=r_eff, amplitude = 1, n=n,
                            ellip=ellip, theta=theta,
                            vmax=vmax, sigma=sigma)

        grid = galaxy.regrid(ngrid=10)

        print("LEVELS:", 15*"*", np.unique(grid))
        print("N_LEVELS:", 15*"*", np.unique(grid).shape)
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(grid, origin="lower")
       # cbar = plt.colorbar()
       # cbar.set_label('Dispersion', rotation=270, labelpad=25)
        plt.show()

    def test_get_masks(self):
        r_eff = 37  # effective radius
        n = 1  # sersic index
        ellip = 0.6  # ellipticity
        theta = 30  # position angle

        vmax = 36
        sigma = 71  # extend in units of r_eff

        x, y = np.meshgrid(np.arange(200), np.arange(200))
        galaxy = GalaxyBase(x, y, x_0=100, y_0=100,
                            r_eff=r_eff, amplitude=1, n=n,
                            ellip=ellip, theta=theta,
                            vmax=vmax, sigma=sigma)

        masks = galaxy.get_masks(ngrid=10)
        numbers = np.arange(len(masks))
        print(15*"*", len(masks))
        first = masks[10]

        for m, n in zip(masks, numbers):
            print(n)
            first = first + m * n
            plt.imshow(m, origin="lower") #, cmap="tab20")
           # plt.show()

        #plt.imshow(first, origin="lower") #, vmin=0, vmax=4)
        cbar = plt.colorbar()
        print(np.min(first), np.max(first))

        plt.show()


class TestGalaxy3D:


    def test_instance(self):
        gal = galaxy3d(sed="kc96/s0", z=0)
        assert isinstance(gal, Source)






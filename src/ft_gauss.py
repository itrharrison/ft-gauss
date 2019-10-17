import numpy as np
import operator
from numpy import fft as ft
from matplotlib import pyplot as plt
from matplotlib import rc
from scipy import ndimage

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

def cropND(img, bounding):
  start = tuple(map(lambda a, da: a//2-da//2, img.shape, bounding))
  end = tuple(map(operator.add, start, bounding))
  slices = tuple(map(slice, start, end))
  return img[slices]

def gaussian(x, loc, scale):

  retVar = np.exp(-0.5*((x - loc)/scale)**2.)/(scale*np.sqrt(2.*np.pi))

  return retVar

n_pix = 64
upscale = 0.5

x_grid = np.linspace(-5,5,n_pix)
f_grid = np.linspace(-5,5,n_pix*upscale)

g1 = gaussian(x_grid, 0, 1.)
g2 = gaussian(x_grid, 0, 2.)
g3 = gaussian(x_grid, 0, 0.5)

F_g1 = ft.fftshift(ft.fft(g1))
F_g2 = ft.fftshift(ft.fft(g2))
F_g3 = ft.fftshift(ft.fft(g3))

F_g2_interp = np.interp(f_grid, x_grid, np.abs(F_g2))
F_g3_interp = np.interp(f_grid, x_grid, np.abs(F_g3))

plt.figure(1, figsize=(3*4.5, 3*3.75))
plt.subplot(331)
plt.plot(x_grid, g1, '-')
plt.subplot(332)
plt.plot(x_grid, g2, '-')
plt.subplot(333)
plt.plot(x_grid, g3, '-')

plt.subplot(334)
plt.plot(x_grid, np.real(F_g1), '-', label='Real')
plt.plot(x_grid, np.imag(F_g1), '-', label='Imaginary')
plt.plot(x_grid, np.abs(F_g1), '-', label='Amplitude')
plt.legend()
plt.subplot(335)
plt.plot(x_grid, np.real(F_g2), '-')
plt.plot(x_grid, np.imag(F_g2), '-')
plt.plot(x_grid, np.abs(F_g2), '-')
plt.plot(f_grid, F_g2_interp, '+')
plt.subplot(336)
plt.plot(x_grid, np.real(F_g3), '-')
plt.plot(x_grid, np.imag(F_g3), '-')
plt.plot(x_grid, np.abs(F_g3), '-')
plt.plot(f_grid, F_g3_interp, '+')

plt.subplot(338)
plt.plot(x_grid, cropND(F_g2_interp,(n_pix,)), '+')

plt.subplot(339)
plt.plot(x_grid, cropND(F_g3_interp,(n_pix,)), '+')

plt.savefig('plots/three_gaussians.png', dpi=300, bbox_inches='tight')
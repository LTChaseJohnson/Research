import numpy as np
import matplotlib.pyplot as plt
from math import *
from matplotlib.patches import Polygon
from scipy import integrate
import scipy.special as sp

a = np.linspace(-1,1,1500)
s = np.linspace(-1,1,1500)

F = (np.sin(a*4*pi/2))**2/(a*4*pi/2)**2

beta = 3.67
f = 8.
lamb = 527.
M = 0.108

ds = 8*pi

G = e**((-4*beta**2*(s*4*pi)**2)/(ds**2))

plt.figure(1)
L1, = plt.plot(a,F)
L2, = plt.plot(s,G)
plt.ylabel('Normalized intensity')
plt.xlabel('Scaled image diameter: $x/ds$')
plt.legend([L1,L2],["Fraunhofer Diffraction","Gaussian Approximation"], loc='best')
plt.show()
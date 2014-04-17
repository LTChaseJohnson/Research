# The purpose of this program is to allow user inputs based on
# experimental constraints on optics and deliver exposure
# times and capture frequency.

# Exposure length is calculated so particle will go from leading
# edge of the pixel to trailing edge of the pixel

import numpy as np
import matplotlib.pyplot as plt
from math import *
from matplotlib.patches import Polygon
from scipy import integrate
import scipy.special as sp

#Os = input('Enter particle size (microns): ')
Os = 40.
#pix = input('Enter pixel size (microns): ')
pix = 7.
#Widthpix,Heightpix = input('Enter # pixels width,height: ')
Widthpix,Heightpix = 2336.,1720.
#FOV = input('Enter sensor field of view (mm): ')
FOV = 152.
#fstop = input('Enter f#: ')
fstop = 8.
#lam = input('Enter laser wavelength (nm): ')
lam = 527.
lamb = lam/(10**9)
#do = input('Enter object distance (mm): ')
do = 210.
U = input('Enter flow speed (m/s): ')
alphad = input('Enter flow angle: ')
alpha = alphad*pi/180
beta = np.sqrt(3.67)

# Sensor Size in microns
Szw = Widthpix * pix
Szh = Heightpix * pix

# Magnification and required focal length
M = (Szw/1000.)/FOV
print ('Optical magnification: '),M
Ui = M*U
print ('Image speed (m/s): '),Ui
f = (do*M)/(M+1)
print ('Required focal length for equipment setup: '),f

# Diffraction limited airy disc size in microns
ds = 2.44 * (1+M) * fstop * (lam/1000)
print ('Diffraction limited airy disc size (microns): '),ds

# Image size (must be at least 2.5 x pixel size)
Is = ((M**2 * Os**2) + ds**2)**0.5
dtau = np.linspace(-Is/2,Is/2,200)
print ('Image size (microns): '),Is
print ('Recommended images size (microns): '),pix*2.5

# Recommended exposure time will allow for 1/3 of image to 
# pass pixel.
dte = Is/(3.*Ui)
print ('Recommended exposure time (us): '),dte
Es = input('Enter exposure time (us): ')
 
# Sampling time based on streak time across 3 pixels
tsample = (pix + ((1/3.)*Is))/Ui
fps = (1/tsample)*10**6
print ('Recommended sampling time (us): '),tsample
print ('Recommended sampling frequency (fps): '),fps

def I(x,beta,ds,a,b):
    def func(s):
        return np.exp(-4*beta**2*x**2/ds**2)
    return integrate.quad(lambda s:func(s),a,b)[0]

xp = -50*np.random.random(5)
yp = -5*np.random.random(5)
plt.figure(1)
plt.scatter(xp,yp)

N = 100                                  #Number of frames to expose
rp = 0.5*Is
t = np.linspace(0,N,N+1)                 #Time step for each frame
E = np.empty_like(t)                     #Empty exposure array

for i in range(5):
    for j in range(N):
        x = xp[i]+cos(alpha)*Ui*t[j]
        if (x+rp <0) or (x-rp >0):
            E[j]=0
        else: E[j]=I(x,beta,ds,x,x+Ui*Es)

plt.figure(2)
plt.bar(t,E)
plt.xlabel('Time Steps')
plt.ylabel('Exposure')
plt.xlim(min(t),max(t))

"""# Assuming the leading edge of a particle is aligned with the trailing
# edge of the pixel, the pixel exposure will be:

def Intensity(x):
    return np.exp(-4*beta**2*x**2/ds**2)

I = Intensity(dtau)



plt.figure(1)
#plt.subplot(2,1,1)
plt.plot(dtau,I)
plt.title('Exposures of Three Sequential Pixels')
plt.ylabel('Normalized Intensities',fontsize=18)

Is1 = Is/2
section1 = np.linspace(Is1-(Ui*Es),Is1,100)
plt.fill_between(section1,Intensity(section1),facecolor='#FF6666')
#print section1
Is2 = Is1-(Ui*(tsample))+pix
section2 = np.linspace(Is2-(Ui*Es),Is2,100)
#print section2
plt.fill_between(section2,Intensity(section2),facecolor='#66FF66')
Is3 = Is1-(Ui*(2.*tsample))+(2.*pix)
section3 = np.linspace(Is3-(Ui*Es),Is3,100)
plt.fill_between(section3,Intensity(section3),facecolor='#6666FF')
#print section3

Ep1, err = quad(Intensity, np.min(section1), np.max(section1))
print ('Exposure 1 value: '),Ep1
Ep2, err = quad(Intensity, np.min(section2), np.max(section2))
print ('Exposure 2 value: '),Ep2
Ep3, err = quad(Intensity, np.min(section3), np.max(section3))
print ('Exposure 3 value: '),Ep3

plt.figure(2)
plt.title('Exposure Distribution of Sequentially Fired Pixels')
plt.ylabel('Exposure Based on Normalized Intensity Profile')
cell = np.linspace(1,3,3)
Ep = [Ep1,Ep2,Ep3]
plt.bar(-cell,Ep)"""

plt.show()
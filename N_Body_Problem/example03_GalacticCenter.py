"""
===============================================================================
LYNX project
Few-Bodies Gravitational System Sumilation
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""

from numpy import loadtxt, pi, array
import time
from common.gravSystem import System
from common.plots import *
from integrators.RK4 import RK4
from integrators.velocityVerlet import velVerlet
from integrators.logHreg import logHreg
from integrators.BS import bulStoer, integrate



# Relevant Constants in the units
# (years, AU, Solar_masses)
# Newtonian Gravitational Constant
G = 4*pi**2

# Conversion factors
arcsec_in_au = 8000 # 1 arcsec in au



# Read the initial data
initial_data_file = "N_Body_Problem/data/GalacticCenter.dat"
(x1,y1,z1,vx1,vy1,vz1,mass) = loadtxt(initial_data_file, unpack = True)

# Convert from SI units to (years, AU, Solar_Mass) units
x = x1*arcsec_in_au
y = y1*arcsec_in_au
z = z1*arcsec_in_au
vx = vx1*arcsec_in_au
vy = vy1*arcsec_in_au
vz = vz1*arcsec_in_au


# Number of particles
N = len(mass)
print('\nNumber of particles: ', N)

names = [['SgrA*', 'black'],
         ['Star01', 'crimson'], 
         ['Star02', 'cornflowerblue'],
         ['Star03', 'darkgreen'],
         ['Star04', 'darkorange'],
         ['Star05', 'darkviolet'],
         ['Star06', 'darkturquoise'],
         ['Star07', 'deeppink'],
         ['Star08', 'gold'],
         ['Star09', 'indigo'],
         ['Star10', 'lime'],
         ['Star11', 'maroon'],
         ['Star12', 'navy'],
         ['Star13', 'olive']]

# Creation of the system and the initial conditions
S = System(mass, G)
q0 = array([x,y,z,vx,vy,vz]).T

# Creation of the time grid (in years)
t_0 = 0.
t_f = 100.

# Number of steps in the grid
n = 700000

# Constant stepsize defined by the number of steps in the grid
dt = (t_f - t_0)/n


# --------------------------------------------------------------------------- #
# -------------------------CHOOSE THE INTEGRATOR----------------------------- #
# --------------------------------------------------------------------------- #
# 1. RK4 method
# 2. Velocity Verlet method
# 3. Bulirsch-Stoer method
# 4. Adam's method
# 5. Logarithm Hamiltonian regularization method (Not yet)
# --------------------------------------------------------------------------- #

intgrtr = 1

start = time.time()

match intgrtr:
    case 1:
        # Integration of the equations of motion using the RK4 method
        q = RK4(S.EoM, q0, t_0, t_f, dt)
        integrator = 'RK4'
    case 2:
        # Integration of the equations of motion using the velocity Verlet method
        q = velVerlet(S.EoM, q0, t_0, t_f, dt)
        integrator = 'Velocity Verlet'
    case 3:
        # Integration of the equations of motion using the Bulirsch-Stoer method
        _, q = bulStoer(S.EoM, t_0, q0, t_f, H=0.01, tol=1e-11)
        integrator = 'BS method'
    case 4:
        # Integration of the equations of motion using the Adam's method
        q = adams(S.EoM, q0, t_0, t_f, dt)
        integrator = 'Adam\'s method'


end = time.time()
print('\nEl tiempo de computo con el uso de ', integrator,' fue:', end - start, '\n\n')

# Energy of the system
T = zeros(n)
U = zeros(n)
for i in range(n):
    T[i] = S.KineticEnergy(q[i,:,:])
    U[i] = S.PotentialEnergy(q[i,:,:])

print('\nEl cambio en la energía total a lo largo de la integración fue de:')
print((T[0]+U[0]) - (T[-1] + U[-1]), '\n\n')

# Plot the orbits
plot3D(q[::50], names, integrator = integrator)

# Plot the energy
energyPlot(T, U, integrator = integrator)

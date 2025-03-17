"""
===============================================================================
LYNX project
Few-Bodies Gravitational System Sumilation
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""

from numpy import loadtxt, pi, array, any
import time
from common.gravSystem import System
from common.plots import *
from integrators.RK4 import RK4
from integrators.velocityVerlet import velVerlet
from integrators.logHreg import logHreg


# Relevant Constants in the units
# (years, AU, Solar_masses)
# Newtonian Gravitational Constant
G = 4.*pi**2

# Conversion factors
m_Sun = 1.98855e30 # Solar mass in kg
au_in_meters = 1.49598261e11 # 1 au in meters
year_in_seconds = 3600*24*365 # 1 year in seconds

# Read the initial data
initial_data_file = "data/sun_earth.dat"
(x,y,z,vx,vy,vz,mass) = loadtxt(initial_data_file, unpack = True)

# Convert from SI units to (years, AU, Solar_Mass) units
x = x/au_in_meters
y = y/au_in_meters
z = z/au_in_meters
vx = vx*year_in_seconds/au_in_meters
vy = vy*year_in_seconds/au_in_meters
vz = vz*year_in_seconds/au_in_meters
mass = mass/m_Sun

# Number of particles
N = len(mass)
print('\nNumber of particles: ', N)

names = [['Sun', 'crimson'], 
         ['Earth', 'cornflowerblue']]

# Creation of the system and the initial conditions
S = System(mass, G)
q0 = array([x,y,z,vx,vy,vz]).T

# Creation of the time grid (in years)
t_0 = 0.
t_f = 5.

# Number of steps in the grid
n = 40000

# Constant stepsize defined by the number of steps in the grid
dt = (t_f - t_0)/n




# --------------------------------------------------------------------------- #
# -------------------------CHOOSE THE INTEGRATOR----------------------------- #
# --------------------------------------------------------------------------- #
# 1. RK4 method
# 2. Velocity Verlet method
# 3. Logarithm Hamiltonian regularization method
# --------------------------------------------------------------------------- #

intgrtr = 2

start = time.time()

match intgrtr:
    case 1:
        # Integration of the equations of motion using the RK4 method
        q = RK4(S.EoM, q0, t_0, t_f, dt)
        integrator = 'RK4'
    case 2:
        # Integration of the equations of motion using the velocity Verlet method
        q = velVerlet(S.EoM, q0, t_0, t_f, dt)
        integrator = 'Velocity verlet'
    case 'logHreg':
        # Integration of the equations of motion using the logarithm Hamiltonian method
        q = logHreg(S.EoM, S.PotentialEnergy, q0, t_0, t_f, dt)
        integrator = 'logHreg'


end = time.time()
print('\nEl tiempo de computo con el uso de ', integrator,' fue:', end - start, '\n\n')

# Energy of the system
T = zeros(n)
U = zeros(n)
for i in range(n):
    T[i] = S.KineticEnergy(q[i,:,:])
    U[i] = S.PotentialEnergy(q[i,:,:])


# Plot the orbits
plot3D(q, names, integrator = integrator)

# Plot the energy
energyPlot(T, U, integrator = integrator)

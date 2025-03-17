"""
===============================================================================
4th Order Runge-Kutta algorithm for the LYNX project.
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""
from numpy import zeros, linspace

def RK4(ODE, q0, t0, tf, dt=1e-6):
    '''
    ------------------------------------------
    RK4(ODE, q0, t0, tf, dt)
    ------------------------------------------
    4th Order Runge-Kutta method for solving 
    a system of ODEs.
    Arguments:
    ODE: function defining the system of ODEs
    q0: numpy array with the initial values of
        the functions in the ODEs system
    t0: independent parameter initial value
    tf: independent parameter final value
    dt: stepsize for the iteration
    ------------------------------------------
    '''

    n = int((tf - t0)/dt)
    tt = linspace(t0, tf, n)
    N = len(q0)
    q = zeros([n, N, 6]) # array [steps, masses, 6 variables]
    q[0,:,:] = q0

    for i in range(n-1):
        k1 = dt*ODE(tt[i], q[i,:,:])
        k2 = dt*ODE(tt[i] + dt/2., q[i,:,:] + k1/2.)
        k3 = dt*ODE(tt[i] + dt/2., q[i,:,:] + k2/2.)
        k4 = dt*ODE(tt[i] + dt, q[i,:,:] + k3)
        q[i+1,:,:] = q[i,:,:] + (k1 + 2.*k2 + 2.*k3 + k4)/6.

    return q


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE CODE LYNX.')
    print('YOU NEED TO RUN THE main.py FILE TO EXECUTE THE CODE.')
    print('')
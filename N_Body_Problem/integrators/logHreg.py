"""
===============================================================================
LogH regularization algorithm for the LYNX project.
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""

from numpy import zeros, linspace

def logHreg(ODE, potential, q0, t0, tf, dt=1e-6):
    '''
    ------------------------------------------
    logHreg(ODE, q0, t0, tf, dt)
    logH regularization method for solving 
    a system of ODEs.
    ------------------------------------------
    Arguments:
    ODE: function defining the system of ODEs
    q0: numpy array with the initial values of
        the functions in the ODEs system
    t0: independent parameter initial value
    tf: independent parameter final value
    dt: stepsize for the iteration
    ------------------------------------------
    Dependences: NumPy
    ------------------------------------------
    '''
    n = int((tf - t0)/dt) # Number of steps
    tt = linspace(t0, tf, n)
    N = len(q0) # Number of particles

    # Initial condition
    q = zeros([n, N, 6]) # array [steps, masses, 6 variables]
    #q[0,0] = t0
    q[0,:,:] = q0
    
    for i in range(1,n):
        # K(dt/2)
        U = -1e-4
        #print(potential(q[i-1,:,:]))
        h = dt/(-U)
        v_half = q[i-1,:,3:] + ODE(tt[i-1], q[i-1,:,:])[:,3:]*h/2
        # D(h)
        q[i,:,0:3] = q[i-1,:,0:3] + v_half*h
        # K(dt/2)
        U = -1e-4#potential(q[i,:,:])
        h = dt/(-U)
        q[i,:,3:] = v_half + ODE(tt[i], q[i,:,:])[:,3:]*h/2

    return q



###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE CODE LYNX.')
    print('YOU NEED TO RUN THE main.py FILE TO EXECUTE THE CODE.')
    print('')
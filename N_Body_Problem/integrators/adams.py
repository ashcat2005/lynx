"""
===============================================================================
4th Order Runge-Kutta algorithm for the LYNX project.
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""
from numpy import zeros, linspace, ones, sin
from math import factorial



def adams(ODE, q0, t0, tf, dt=1e-6):
    '''
    ------------------------------------------
    adams(ODE, q0, t0, tf, dt)
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
    N = len(q0)
    q = zeros([n, N, 6]) # array [steps, masses, 6 variables]
    q[0,:,:] = q0

    h = dt*ones([N, 6])

    def Delta(x, j, f = ODE):
        S = zeros([N, 6])
        for i in range(j+1):
            C = ((-1)**i)*factorial(j)//(factorial(i)*factorial(j-i))
            S += C*f(0,x - i*h)
        return S
    
    for i in range(n-1):
        q[i+1,:,:] = q[i,:,:] + dt*(ODE(0,q[i,:,:]) + (1/2)*Delta(q[i,:,:],1) \
                                    + (5/12)*Delta(q[i,:,:], 2)\
                                    + (3/8)*Delta(q[i,:,:], 3)\
                                    + (251/720)*Delta(q[i,:,:], 4)\
                                    + (95/288)*Delta(q[i,:,:], 5)\
                                    + (19087/60480)*Delta(q[i,:,:], 6)\
                                    + (5257/17280)*Delta(q[i,:,:], 7))
    return q


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE CODE LYNX.')
    print('YOU NEED TO RUN THE main.py FILE TO EXECUTE THE CODE.')
    print('')

"""
===============================================================================
Newtonian Gravitational Equations of  Motion for the LYNX project.
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""
from numpy import zeros, sqrt, sum, pi, any
import sys

class System():
    def __init__(self, mass, G = 4.*pi**2):
        if any(mass<=0):
            sys.exit('All the masses must be positive')
        self.N = len(mass)
        self.mass = mass
        self.G = G

    def EoM(self, t0, q0):
        '''
        ------------------------------------------
        EoM(t0,q0) 
        ------------------------------------------
        Equations of Motion for N-particles 
        
        Arguments:
        t0: time parameter (not necessary for 
            the Newtonian problem)
        q0: numpy array with the initial condition
            data:
            q0[0] = particle 1
            q0[1] = particle 2
            etc.
            q0[0] = [x0, y0, z0, vx0, vy0, vz0]
        mass: masses of the particles
        ------------------------------------------
        '''
        q1 = zeros(q0.shape)
        q1[:,0:3] = q0[:,3:] # Components of the velocity
        
        for i in range(self.N):
            Deltaxyz = q0[i,0:3] - q0[:,0:3]
            # Distance between particles
            r = sqrt(sum(Deltaxyz*Deltaxyz, axis=1))
            # To avoid divivision by zero in the self-force terms
            # The terms vanish due to the Delta term in the numerator! 
            r[i] = 1 
            q1[i,3] = -self.G * sum(Deltaxyz[:,0] * self.mass/(r**3))
            q1[i,4] = -self.G * sum(Deltaxyz[:,1] * self.mass/(r**3))
            q1[i,5] = -self.G * sum(Deltaxyz[:,2] * self.mass/(r**3))
        return q1
    
    def KineticEnergy(self, q):
        '''
        Kinetic Energy 
        '''
        (_,_,_,vx,vy,vz) = q.transpose()
        v2 = vx**2 + vy**2 + vz**2
        T = 0.5*sum(self.mass*v2)
        return T
    
    def PotentialEnergy(self, q):
        '''
        Potential Energy
        '''
        (x,y,z,_,_,_) = q.transpose()
        U = 0.
        for i in range(0,self.N):
            deltax = x[i] - x
            deltay = y[i] - y
            deltaz = z[i] - z
            r = sqrt(deltax**2 + deltay**2 + deltaz**2)
            # To avoid divivision by zero and put the value of the term to zero!
            r[i] = 1e300 
        U += - 0.5*self.G * self.mass[i] * sum(self.mass/r)
        return U
    
    def TotalEnergy(self, q):
        '''
        Total Energy calculation
        '''
        return self.KineticEnergy(q) + self.PotentialEnergy(q)



###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE CODE LYNX.')
    print('YOU NEED TO RUN THE main.py FILE TO EXECUTE THE CODE.')
    print('')
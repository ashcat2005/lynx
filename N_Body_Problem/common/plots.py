"""
===============================================================================
Plotting algorithm of the orbits for the LYNX project.
===============================================================================
@author: Eduard Larrañaga - 2024
===============================================================================
"""

import matplotlib.pyplot as plt
from numpy import amax, amin, zeros

def plot3D(q, names, integrator='', savefig=False, filename='orbit.png'):
    # Limits for the plot
    boundary = max(abs(amax(q[:,:,0:3])), abs(amin(q[:,:,0:3])))*(1+0.1)

    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(projection='3d')
    for i in range(len(names)):
        ax.plot(q[:,i,0],q[:,i,1],q[:,i,2], label=names[i][0], color=names[i][1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim3d(-boundary, boundary)
    ax.set_ylim3d(-boundary, boundary)
    ax.set_zlim3d(-boundary, boundary)
    ax.legend()
    ax.title.set_text('Orbits calculated using '+integrator)
    if savefig:
        plt.savefig(filename)
    plt.show()

def energyPlot(T, U, integrator='', savefig=False, filename='energy.png'):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot()
    ax.plot(T, color='cornflowerblue', label='Kinetic Energy')
    ax.plot(U, color='crimson', label='Potential Energy')
    ax.plot(T+U, color='black', label='Total Energy')
    ax.set_ylabel('Total Energy')
    ax.set_xlabel('t')
    ax.title.set_text('Conserved Quantities using '+integrator)
    plt.grid()
    plt.legend()
    if savefig:
        plt.savefig(filename)
    plt.show()


###############################################################################

if __name__ == '__main__':
    print('')
    print('THIS IS A MODULE DEFINING ONLY A PART OF THE CODE LYNX.')
    print('YOU NEED TO RUN THE main.py FILE TO EXECUTE THE CODE.')
    print('')

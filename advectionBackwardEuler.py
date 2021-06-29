# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:31:10 2021

@author: nikol
"""

from fenics import * 
from ufl.algorithms.ad import expand_derivatives
import numpy as np


msh = IntervalMesh(10, 0, 1)                        #create mesh
V = FunctionSpace(msh, "CG", 1)                     #create 1D function space

T = 2.0                                             # final time
num_steps = 5                                     # number of time steps
dt = T / num_steps                                  # time step size

c= Constant(2.0)                                    #Constant for advection equation  
u_D = Expression('x[0]-c*t', degree=1, c=c,t=0)     #analytical solution for boundaries
        
def boundary(x, on_boundary):
            return on_boundary
        
bc = DirichletBC(V, u_D, boundary)                  #boundary conditions


u_n = interpolate(u_D, V)                           #u at time 0
u = TrialFunction(V)                                #Trialfunction
k0= Function(V)                                         
v0=TestFunction(V)                                  #Testfunction
u0= u_n + dt*Constant(1.0)*k0
F = u*v0*dx+u0*v0*dx- dt*c*Dx(u0,0)*v0*k0*dx        


# Residual
r = u - u_n + dt * c * Dx((u+u_n)/2,0)
# Add SUPG stabilisation terms (from https://fenicsproject.org/qa/13458/how-implement-supg-properly-advection-dominated-equation/)
h = 0.1  # should be interval size. Don't know how to extract this from msh right now.
tau = h/(2.0*c) # tau from SUPG fenics example
F += tau * c * Dx(v0,0) * r * dx
a, L = lhs(F), rhs(F)      



u = Function(V)
t = 0
arrayY=[]
arrayX=[]
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

    u_e = interpolate(u_D, V)
    error = np.abs(u_e.vector().get_local() - u.vector().get_local()).max()
    print('t = %.2f: error = %.3g' % (t, error))
    #assert(error < 10**-13)
    arrayY.append(u.vector().get_local().max())
    arrayX.append(t)

    # Update previous solution
    u_n.assign(u)

with open('advectionBackwardEuler2.txt', 'w') as file:    
    for i in range(num_steps):
        file.write('{},{}\n'.format(arrayX[i],arrayY[i]))

# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 09:11:12 2021

@author: nikol
"""

from fenics import * 

from irksome import GaussLegendre, RadauIIA, Dt, TimeStepper


  
from ufl.algorithms.ad import expand_derivatives

import numpy as np


N = 100
x0 = 0.0
x1 = 10.0
y0 = 0.0
y1 = 10.0
nx = ny = 8
msh = IntervalMesh(10, 0, 1)
V = FunctionSpace(msh, "CG", 1)


  
T = 2.0            # final time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size

 

u_D = Expression('x[0]-c*t', degree=1, c=2,t=0)
        
def boundary(x, on_boundary):
            return on_boundary
        
bc = DirichletBC(V, u_D, boundary)


u_n = interpolate(u_D, V)

u = TrialFunction(V)
v = TestFunction(V)
c=Constant(2.0)
F = u*v*dx + u_n*v*dx - dt*c*Dx(u,0)*v*dx

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
    
    arrayY.append(u.vector().get_local().max())
    arrayX.append(t)
    print(u.vector().get_local())
    # Update previous solution
    u_n.assign(u)
    #for vec in vertices(msh):
        #print(vec.point().x())

with open('advectionBackwardEuler.txt', 'w') as file:    
    for i in range(num_steps):
        file.write('{},{}\n'.format(arrayX[i],arrayY[i]))

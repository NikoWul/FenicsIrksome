# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 13:22:31 2021

@author: nikol
"""

from fenics import * 

from irksome import GaussLegendre, RadauIIA, Dt, TimeStepper


  
from ufl.algorithms.ad import expand_derivatives

import numpy as np




nx = ny = 8

msh = IntervalMesh(10, 0, 1)
V = FunctionSpace(msh, "CG", 1)

E=V.ufl_element()*V.ufl_element()
Vbig= FunctionSpace(V.mesh(),E)
  
T = 2.0            # final time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size

 
E=V.ufl_element()*V.ufl_element()
Vbig= FunctionSpace(V.mesh(),E)
c=Constant(2.0)
u_D = Expression('x[0]-c*t', degree=1, c=c,t=0)
        
def boundary(x, on_boundary):
            return on_boundary
        
bc = []
for i in range(2):
    bc.append(DirichletBC(Vbig.sub(i), u_D, boundary))
u_n = interpolate(u_D, V)


(sigma, u) = TrialFunctions(Vbig)

k=Function(Vbig)
k0,k1=split(k)
v0, v1 = TestFunctions(Vbig)
u0= u_n + dt*Constant(0.5)*k0+dt*Constant(0.5)*k1
u1 = u0 + dt*Constant(0.5)*k0+dt*Constant(0.5)*k1     
F = sigma*v0*dx + u*v1*dx + u0*dx + u1*dx - dt*Dx(u0,0)*v0*k0*dx - dt*Dx(u1,0)*v1*k1*dx
a, L = lhs(F), rhs(F)
w=Function(Vbig)



u = Function(V)
t = 0
arrayY=[]
arrayX=[]
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, w, bc)

    # Plot solution
    #plot(u)

    # Compute error at vertices
    u_e = interpolate(u_D, V)
    print(w.vector().get_local())
    arrayY.append(w.vector().get_local().max())
    arrayX.append(t)

    # Update previous solution
    u_n.assign(u)

with open('advectionHeun.txt', 'w') as file:    
    for i in range(num_steps):
        file.write('{},{}\n'.format(arrayX[i],arrayY[i]))
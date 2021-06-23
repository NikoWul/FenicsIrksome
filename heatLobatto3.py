# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:44:16 2021

@author: nikol
"""
from fenics import * 

from irksome import GaussLegendre, RadauIIA, Dt, TimeStepper, LobattoIIIC

from ufl.log import error
  
from ufl.algorithms.ad import expand_derivatives

import numpy as np

T = 2.0            # final time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

nx = ny = 8
msh = UnitSquareMesh(nx, ny)
V = FunctionSpace(msh, "P", 1)

bt =  LobattoIIIC(2)

ns = bt.num_stages  

A=bt.A
t = 0
print(A)
print(dt)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)



u_n = interpolate(Constant(0.0), V)

E=V.ufl_element()*V.ufl_element()
Vbig= FunctionSpace(V.mesh(),E)
k = Function(Vbig)
k0, k1 = split(k)
v0, v1 = TestFunctions(Vbig)
(u_0, u_1) = TrialFunctions(Vbig)
v = TestFunction(V)
u0 = u_0 + A[0][0] * dt * k0 + A[0][1]  * dt * k1
u1 = u_1 +  A[1][0] * dt * k0 + A[1][1] * dt * k1
F = (inner(k0 , v0) * dx + inner(k1, v1) * dx +inner(grad(u0), grad(v0)) * dx + inner(grad(u1), grad(v1)) * dx)
a, L = lhs(F), rhs(F)
print(F)

def boundary(x, on_boundary):
    return on_boundary
bc = []
for i in range(2):
    bc.append(DirichletBC(Vbig.sub(i), u_D, boundary))
  
vtkfile = File("heat_gaussian/solution.pvd")    
u = Function(Vbig)

arrayY=[]
arrayX=[]
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)
    u_ref = interpolate(u_D, V)
    error_normalized = (u_ref - u.sub(0)) / u_ref
    # project onto function space
    error_pointwise = project(abs(error_normalized), V)
    # determine L2 norm to estimate total error
    error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))
    error_pointwise.rename("error", " ")
    print('t = %.2f: error = %.3g' % (t, error_total))
    # Compute error at vertices
    arrayY.append(error_total)
    arrayX.append(t)
    # Update previous solution
    u_n.assign(u)
    
#print(arrayX)
#print(arrayY)







# -*- coding: utf-8 -*-
"""
Created on Wed May 26 16:44:16 2021

@author: nikol
"""
from fenics import * 

from irksome import GaussLegendre, RadauIIA, LobattoIIIC

from ufl.log import error
  
from ufl.algorithms.ad import expand_derivatives

import numpy as np

T = 1.0            # final time
num_steps = 1      # number of time steps
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

# Define boundary conditions
s = 2
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t', degree=2, alpha=alpha, beta=beta, t=0)
du_Ddt = s * [None]
for i in range(s):
    du_Ddt[i] = Expression('beta', degree=2, alpha=alpha, beta=beta, t=0)
    du_Ddt[i].t = t + bt.c[i] * dt

u_ini = interpolate(u_D, V)

E = V.ufl_element() * V.ufl_element()
Vbig = FunctionSpace(V.mesh(), E)
k0, k1 = TrialFunctions(Vbig)
v0, v1 = TestFunctions(Vbig)
f = Expression('beta - 2 - 2*alpha', degree=2, alpha=alpha, beta=beta, t=0)
u0 = u_ini + A[0][0] * dt * k0 + A[0][1] * dt * k1
u1 = u_ini + A[1][0] * dt * k0 + A[1][1] * dt * k1
F = (inner(k0 , v0) * dx + inner(grad(u0), grad(v0)) * dx) + (inner(k1, v1) * dx + inner(grad(u1), grad(v1)) * dx) - f * v1 * dx - f * v0 * dx
a, L = lhs(F), rhs(F)
print(F)

def boundary(x, on_boundary):
    return on_boundary
bc = []
for i in range(2):
    bc.append(DirichletBC(Vbig.sub(i), du_Ddt[i], boundary))
  
vtkfile = File("heat_gaussian/solution.pvd")    

arrayY=[]
arrayX=[]
k = Function(Vbig)

for n in range(num_steps):

    # Update current time
    for i in range(s):
        du_Ddt[i].t = t + bt.c[i] * dt
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, k, bc)
    u_ref = interpolate(u_D, V)
    u_sol = project(u_ini + bt.b[0] * k.sub(0) + bt.b[1] * k.sub(1), V)
    error_normalized = (u_ref - u_sol) / u_ref
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
    u_ini.assign(u_sol)
    
#print(arrayX)
#print(arrayY)







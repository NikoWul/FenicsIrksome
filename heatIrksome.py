# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 17:18:20 2021

@author: nikol
"""
from fenics import * 
from irksome import Dt, LobattoIIIC, getForm

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

A=bt.A
t = 0

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)

E=V.ufl_element()*V.ufl_element()
Vbig= FunctionSpace(V.mesh(),E)
u=TrialFunction(V)
v = TestFunction(V)
u_n = interpolate(Constant(0.0), V)
F_n = F = inner (Dt(u), v) * dx + inner (grad(u), grad(v)) * dx

def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

F , bcnew = getForm(F_n, bt, V, dt, bc)
a, L = lhs(F), rhs(F)

u = Function(V)

arrayY=[]
arrayX=[]
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bcnew)
    u_ref = interpolate(u_D, V)
    error_normalized = (u_ref - u) / u_ref
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
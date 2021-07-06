from fenics import * 
from irksome import Dt, LobattoIIIC, getForm

import numpy as np

from ufl.algorithms.ad import expand_derivatives    

T = 1.0            # final time
num_steps = 1     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

nx = ny = 8
msh = UnitSquareMesh(nx, ny)
V = FunctionSpace(msh, "P", 1)

bt =  LobattoIIIC(2)

A=bt.A
t = 0
ns=  bt.num_stages

E=V.ufl_element()*V.ufl_element()
Vbig= FunctionSpace(V.mesh(),E)

def boundary(x, on_boundary):
    return on_boundary

# Define boundary condition
du_Ddt = ns * [None]
bc = []
for i in range(ns):
    du_Ddt[i] = Expression('2*beta*t', degree=2, alpha=alpha, beta=beta, t=0)
    du_Ddt[i].t = t + bt.c[i] * dt
    bc.append(DirichletBC(Vbig.sub(i), du_Ddt[i], boundary))

u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)





f = ns * [None]
for i in range(ns):
    f[i] = Expression('2*beta*t - 2 - 2*alpha', degree=2, alpha=alpha, beta=beta, t=0)
    f[i].t = t + bt.c[i] * dt

#rhs= 2*beta*t - 2 - 2*alpha
#x,y = SpatialCoordinate(msh)
#ueaxt= 2*beta*t - 2 - 2*alpha
#rhs=expand_derivatives(ueaxt)

u=TrialFunction(V)
v = TestFunction(V)
u_n = interpolate(Constant(0.0), V)
F_n = inner (Dt(u), v) * dx + inner (grad(u), grad(v)) * dx - inner(rhs,v)*dx

def boundary(x, on_boundary):
    return on_boundary
bc = DirichletBC(V, u_D, boundary)

F , bcnew, vbits = getForm(F_n, bt, V, dt, bc, f)

#F=F - f[1] * vbits[1] * dx - f[0] * vbits[0] * dx
a, L = lhs(F), rhs(F)

u = Function(V)
k=Function(Vbig)

arrayY=[]
arrayX=[]
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, k, bc)
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
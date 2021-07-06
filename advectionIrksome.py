
from fenics import * 

from irksome import GaussLegendre, RadauIIA, Dt, getForm

import numpy as np



bt=  RadauIIA(1)
print(bt.A)
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
u_n = interpolate(u_D, V)
        

u=TrialFunction(V)
v = TestFunction(V)
F_n = F = u*v*dx + u_n*v*dx-c*dt*Dt(u)*v*dx

def boundary(x, on_boundary):
            return on_boundary
        
bc = DirichletBC(V, u_D, boundary)
F, bcs = getForm(F_n, bt, V, dt, bc)
print(F)

# Residual
r = u - u_n + dt * c * Dx((u+u_n)/2,0)
# Add SUPG stabilisation terms (from https://fenicsproject.org/qa/13458/how-implement-supg-properly-advection-dominated-equation/)
h = 0.1  # should be interval size. Don't know how to extract this from msh right now.
tau = h/(2.0*c) # tau from SUPG fenics example
F += tau * c * Dx(v,0) * r * dx
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

    # Plot solution
    #plot(u)
    u_ref = interpolate(u_D, V)
    error_normalized = (u_ref - u) / u_ref
    # project onto function space
    error_pointwise = project(abs(error_normalized), V)
    # determine L2 norm to estimate total error
    error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))
    error_pointwise.rename("error", " ")
    print('t = %.2f: error = %.3g' % (t, error_total))
    # Compute error at vertices
    #u_e = interpolate(u_D, V)
    #print(u.vector().get_local())
    arrayY.append(u.vector().get_local().max())
    arrayX.append(t)

    # Update previous solution
    u_n.assign(u)

with open('advectionHeun.txt', 'w') as file:    
    for i in range(num_steps):
        file.write('{},{}\n'.format(arrayX[i],arrayY[i]))
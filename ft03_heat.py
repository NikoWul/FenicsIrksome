"""
FEniCS tutorial demo program: Heat equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  u'= Laplace(u) + f  in the unit square
  u = u_D             on the boundary
  u = u_0             at t = 0

  u = 1 + x^2 + alpha*y^2 + \beta*t
  f = beta - 2 - 2*alpha
"""

from __future__ import print_function
from fenics import *
import numpy as np
from irksome import GaussLegendre, RadauIIA, Dt, TimeStepper

butcher_tableau = RadauIIA(1)
ns = butcher_tableau.num_stages 

T = 10.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
t=Constant(0.0)


# Create mesh and define function space
nx = ny = 10
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'CG', 1)


# Define boundary condition
#vel = Expression(('x[0]', 'x[1]', 'x[2]'), element=V.ufl_element())
u_D = Expression('2*t',
                 degree=1, t=0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)  #in Irksome demos u
#u_n = project(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(2)

F = inner(v,Dt(u))*dx-inner(f*u,Dx(v))*dx
#a, L = lhs(F), rhs(F)
luparams = {"mat_type": "aij",
              "ksp_type": "preonly",
              "pc_type": "lu"}

stepper = TimeStepper(F, butcher_tableau, t, dt, u_n, bcs=bc,
                        solver_parameters=luparams)

while (float(t) < 1.0):
      if (float(t) + float(dt) > 1.0):
          dt.assign(1.0 - float(t))
      stepper.advance()
      print(float(t))
      t.assign(float(t) + float(dt))








#a, L = lhs(F), rhs(F)

# Time-stepping
#u = Function(V)

#for n in range(num_steps):

    # Update current time
 #   t += dt
  #  u_D.t = t

    # Compute solution
   # solve(a == L, u, bc)

    # Plot solution
    #plot(u)

    # Compute error at vertices
    #u_e = interpolate(u_D, V)
    #error = np.abs(u_e.vector().array() - u.vector().array()).max()
    #print('t = %.2f: error = %.3g' % (t, error))

    # Update previous solution
    #u_n.assign(u)

# Hold plot
#interactive()

from irksome import GaussLegendre, RadauIIA, Dt, getForm
from fenics import * 

from irksome import GaussLegendre, RadauIIA, LobattoIIIC

from ufl.log import error
  
from ufl.algorithms.ad import expand_derivatives

import numpy as np

from fenics import * 

from irksome import GaussLegendre, RadauIIA, LobattoIIIC

from ufl.log import error
  
from ufl.algorithms.ad import expand_derivatives

import numpy as np

def heatreplace( v, u, k):
    L= (inner(k , v) * dx + inner(grad(u), grad(v)) * dx)
    return L

def mul(one, other):
    return MixedFunctionSpace((one, other))


T = 2.0            # final time
t = 0              # current time
num_steps = 20     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# define geometry and spatial discretization
nx = ny = 8
msh = UnitSquareMesh(nx, ny)
def boundary(x, on_boundary):
    return on_boundary

# get RK scheme
bt =  LobattoIIIC(3)
nus = bt.num_stages  
A = bt.A

print(bt.c)

# Create mixed function space depending on number of stages
V = FunctionSpace(msh, "P", 1)

E = V.ufl_element()
TH = MixedElement([E, E,E])

for j in range(nus-1):
    E = E*V.ufl_element()
Vbig = FunctionSpace(V.mesh(), TH)



# Define boundary conditions
# Important: Use derivative of BC, since our unknowns are the stages! See [1] eq. (18)
du_Ddt = nus * [None]
bc = []
for i in range(nus):
    du_Ddt[i] = Expression('2*beta*t', degree=3, alpha=alpha, beta=beta, t=0)
    du_Ddt[i].t = t
    for j in range (i-1):
        du_Ddt[i].t = du_Ddt[i].t + bt.c[j] * dt
    if(nus==1):
        bc.append(DirichletBC(Vbig, du_Ddt[i], boundary))
    else:
        bc.append(DirichletBC(Vbig.sub(i), du_Ddt[i], boundary))

# Define initial condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t*t', degree=2, alpha=alpha, beta=beta, t=0)
u_ini = interpolate(u_D, V)

# Define problems rhs. Important: If f is time dependent, we need the same procedure like for the boundary conditions.
f = nus * [None]
for i in range(nus):
    f[i] = Expression('2*beta*t - 2 - 2*alpha', degree=2, alpha=alpha, beta=beta, t=0)
    f[i].t = t + bt.c[i] * dt

k = TrialFunction(Vbig)
v= TestFunction(Vbig)

ks=split(k)
vs=split(v)


# Define solutions per stage. Todo: Should be generalized via a for-loop
u = nus * [None]
for i in range(nus):
    uhelp=u_ini
    for j in range (nus):
        uhelp= uhelp+ A[i][j] *dt * ks[j]
    u[i]=uhelp

rh = 0
for i in range(nus):
    rh= rh + f[i]* vs[i]*dx



# Assemble weak form. Todo: Should be generalized via for-loop
F=0
for i in range(nus):
    F= F+ heatreplace(vs[i],u[i],ks[i])
F=F-rh
a, L = lhs(F), rhs(F)
print(F)

vtkfile = File("heat_gaussian/solution.pvd")    

# Unknown: stages k
k = Function(Vbig)


arrayY = []
arrayX = []

for n in range(num_steps):

    # Update BCs and rhs wrt current time.
    for i in range(nus):
        du_Ddt[i].t = t + bt.c[i] * dt
        f[i].t = t + bt.c[i] * dt

    # Compute solution for stages
    solve(a == L, k, bc)   
    # Assemble solution from stages
    if(nus==1):
         u_sol = project(u_ini + dt * (bt.b[0] * k ), V)
    elif(nus==2):
        u_sol = project(u_ini + dt * (bt.b[0] * k.sub(0) + bt.b[1] * k.sub(1)), V)
    else:
        u_sol = project(u_ini + dt * (bt.b[0] * k.sub(0) + bt.b[1] * k.sub(1)+ bt.b[2] * k.sub(2)), V)
    # Update initial condition with solution
    u_ini.assign(u_sol)
    # Update time and compute reference solution
    t += dt
    u_D.t = t
    u_ref = interpolate(u_D, V)
    # Compute error
    error_normalized = (u_ref - u_sol) / u_ref
    error_pointwise = project(abs(error_normalized), V)
    error_total = sqrt(assemble(inner(error_pointwise, error_pointwise) * dx))  # determine L2 norm to estimate total error
    error_pointwise.rename("error", " ")
    print('t = %.2f: error = %.3g' % (t, error_total))
    # Compute error at vertices
    arrayY.append(error_total)
    arrayX.append(t)
    

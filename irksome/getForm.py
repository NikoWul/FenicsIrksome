import numpy
from fenics import *
from ufl.classes import Zero
from .deriv import TimeDerivative
from ufl.constantvalue import as_ufl
from ufl.corealg.multifunction import MultiFunction
from ufl.algorithms.analysis import has_exact_type
from ufl.classes import CoefficientDerivative
from ufl.algorithms.map_integrands import map_integrand_dags
from functools import partial


class MyReplacer(MultiFunction):
    def __init__(self, mapping):
        super().__init__()
        self.replacements = mapping

    def expr(self, o):
        if o in self.replacements:
            return self.replacements[o]
        else:
            return self.reuse_if_untouched(o, *map(self, o.ufl_operands))


def replace(e, mapping):
    """Replace subexpressions in expression.
    @param e:
        An Expr or Form.
    @param mapping:
        A dict with from:to replacements to perform.
    """
    mapping2 = dict((k, as_ufl(v)) for (k, v) in mapping.items())

    # Workaround for problem with delayed derivative evaluation
    # The problem is that J = derivative(f(g, h), g) does not evaluate immediately
    # So if we subsequently do replace(J, {g: h}) we end up with an expression:
    # derivative(f(h, h), h)
    # rather than what were were probably thinking of:
    # replace(derivative(f(g, h), g), {g: h})
    #
    # To fix this would require one to expand derivatives early (which
    # is not attractive), or make replace lazy too.
    if has_exact_type(e, CoefficientDerivative):
        # Hack to avoid circular dependencies
        from ufl.algorithms.ad import expand_derivatives
        e = expand_derivatives(e)

    return map_integrand_dags(MyReplacer(mapping2), e)


def getForm(F, butch, V, dt, bc, f):
    """Given a time-dependent variational form and a
    :class:`ButcherTableau`, produce UFL for the s-stage RK method.
    :arg F: UFL form for the semidiscrete ODE/DAE
    :arg butch: the :class:`ButcherTableau` for the RK method being used to
         advance in time.
    :arg dt: a :class:`Constant` referring to the size of the current
         time step.
    :arg bc:  a :class:`DirichletBC` object
         containing (possible time-dependent) boundary conditions imposed
         on the system.
    On output, we return a tuple consisting of four parts:
       - Fnew, the :class:`Form`
       - `bcnew`, a list of :class:`firedrake.DirichletBC` objects to be posed
         on the stages
"""

    # extract parameters from the butcher tableau butch
    A = butch.A
    num_stages = butch.num_stages
    c=butch.c

    # create expanded Function Space
    E = V.ufl_element()
    for i in range(num_stages-1):
        E = E*V.ufl_element()
    Vbig = FunctionSpace(V.mesh(), E)
    t=0

    # create Testfunctions and Trialfunctions from function space V, create functions k_i from expanded Functionspace
    vnew = TestFunction(Vbig)
    k = Function(Vbig)
    v = TestFunction(V)
    u = TrialFunction(V)
    u0bits = split(u)
    vbits = split(v)
    vbigbits = split(vnew)
    kbits = split(k)

    # make objects Ak
    kbits_np = numpy.zeros((num_stages, num_stages), dtype="object")
    for i in range(num_stages):
        for j in range(num_stages):
            kbits_np[i, j] = kbits[i]
    Ak = A @ kbits_np

    # transformation to Runge-Kutta variational form
    Fnew = Zero()
    for i in range(num_stages):
        repl = {t: t + c[i] * dt}
        for j, (ubit, vbit, kbit) in enumerate(zip(u0bits, vbits, kbits)):
            repl[ubit] = ubit + dt * Ak[i, j]
            repl[vbit] = vbigbits[ i + j]
            repl[TimeDerivative(ubit)] = kbits_np[i, j]
            if (len(ubit.ufl_shape) == 1):
                for kk, kbitbit in enumerate(kbits_np[i, j]):
                    repl[TimeDerivative(ubit[kk])] = kbitbit
                    repl[ubit[kk]] = repl[ubit][kk]
                    repl[vbit[kk]] = repl[vbit][kk]
        Fnew += replace(F, repl)

    # create list of boundary conditions
    #Fnew=Fnew- f[1] * vbigbits[1] * dx - f[0] * vbigbits[0] * dx
    bcnew = []
    for i in range(num_stages):
        bcnew.append(bc)

    return Fnew, bcnew, vbigbits
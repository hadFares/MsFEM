from .problem import A, f, A_HOM, ExactSolution, HomogSolution, exact_solution
from .mesh import Mesh1D
from .solution import DiscreteSolution, P1Interpolant
from .fem_p1 import solve as fem_p1_solve
from .msfem import solve as msfem_solve, msfem_basis
from . import errors, plots, timing

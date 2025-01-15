import sys

import dolfin
from fenics import *

############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################


def LinearSolver(A, x, b, param, P=False):
    if param["Linear Solver"]["Type of Solver"] == "Default":
        solve(A, x.vector(), b)

    elif param["Linear Solver"]["Type of Solver"] == "Iterative Solver":
        soltype = param["Linear Solver"]["Iterative Solver"]
        precon = param["Linear Solver"]["Preconditioner"]

        solver = PETScKrylovSolver(soltype, precon)

        if P == False:
            solver.set_operator(A)

        else:
            solver.set_operators(A, P)

        # TODO - generate those parameters also for all the apps but Heterodimer
        solver.parameters["relative_tolerance"] = param["Linear Solver"][
            "Iterative Solver Options"
        ]["Relative tolerance"]
        solver.parameters["absolute_tolerance"] = param["Linear Solver"][
            "Iterative Solver Options"
        ]["Absolute tolerance"]
        solver.parameters["nonzero_initial_guess"] = param["Linear Solver"][
            "Iterative Solver Options"
        ]["Non-zero initial guess"]
        solver.parameters["monitor_convergence"] = param["Linear Solver"][
            "Iterative Solver Options"
        ]["Monitor convergence"]
        solver.parameters["report"] = param["Linear Solver"][
            "Iterative Solver Options"
        ]["Report"]
        solver.parameters["maximum_iterations"] = int(
            param["Linear Solver"]["Iterative Solver Options"][
                "Maximum iterations"
            ]
        )

        solver.solve(x.vector(), b)

    elif param["Linear Solver"]["Type of Solver"] == "MUMPS":
        solver = PETScLUSolver("mumps")

        solver.solve(A, x.vector(), b)

    else:
        print("Choosed solver not available!")
        sys.exit(0)

    return x

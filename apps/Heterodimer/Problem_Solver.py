#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Main file for the heterodimer model problem solution.
#
# ------------------------------------------------------------------------------
# Author: Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

# sys and os are needed to manage paths for finding files.
import os
import sys
from ast import Constant
from pyclbr import Function
from dolfin import MPI

from fenics import *

# Retrieve the directory from where this script is executed.
cwd = os.getcwd()
# Allows search for modules both in the current directory and in "utilities".
sys.path.append(cwd)
sys.path.append(cwd + "/../../utilities")

# import NeuroFEM utilites.
import Common_main
import Formulation_Elements as FE
import LinearSolver_handler
import ParameterFile_handler
import XDMFHandler
import BCs_handler
import ErrorComputationHandler as ErrComp
import Mesh_handler


################################################################################
# Parallel execution.
# @TODO rethink and merge with utilities/IOHandler.py to set levels of message
# priority and manage the output in parallel execution.
#
# Ok di Mattia. Dopo consegna tesi.
################################################################################
def parallel_initialisation():
    parallel = False
    rank = 0

    if MPI.comm_world.Get_size() == 1:
        print("Serial execution.")
    else:
        rank = MPI.comm_world.Get_rank()
        parallel = True
        if rank == 0:
            print(f"Parallel execution on {MPI.comm_world.Get_size()} ranks.")

    return parallel, rank


def printp(s, p, r):
    if (p == False) or (p == True and r == 0):
        print(s)


################################################################################
# Solver.
################################################################################
def problemconvergence(filename, conv):
    """problemconvergence function used to test the convergence of the solver.

    This function solves the very same problem on the same domain with different
    mesh refinements. This way the convergence of the scheme can be tested,
    provided an exact solution.

    Args:
        filename (str): name of the file `filename.prm` containing the data
        necessary to solve the problem.
        conv (int): number of iterations desired for the convergence test.
        A minimum of three iterations are suggested to better estimate the slope
        of convergence rate, and test it against the expected one.
    """
    errors = ErrComp.ComputeErrorsHeterodimer()

    for it in range(0, conv):
        # Convergence iteration solver.
        problemsolver(filename, it, True, errors)
        if MPI.comm_world.Get_rank() == 0:
            print("Iteration completed.")

    # Save the errors' table to a .csv file.
    errors.save_to_csv("errors_convergence_test-" + filename)
    # Print the errors.
    if MPI.comm_world.Get_rank() == 0:
        print(errors.erros_L2_DG_final)
        print(errors.errors_energy_final)

    # Print them in a .png file
    # errors.plot_errors_in_space("Energy", normalised=True)
    # errors.plot_errors_in_space("L2DG", normalised=True)


def problemsolver(filename, iteration=0, conv=False, errors=False):
    """problemsolver function that manages the solution routine of the
    Heterodimer model. It a system of partial differential (reaction diffusion)
    equations. Its complete description can be found in Antonietti, Bonizzoni,
    Corti, Dall'Olio (2023).

    Args:
        filename (str): name of the file from which parameters are read.
        iteration (int, optional): number of iterations for convergence test.
        Defaults to 0.
        conv (bool, optional): a converge test is carried out if conv is set to
        True. Defaults to False.
        errors (bool, optional): dataframe containing the errors of the
        numerical solutions with resect to the exact solution. Defaults to
        False.
    """
    # Set the parallel environment.
    # --------------------------------------------------------------------------
    is_parallel, this_rank = parallel_initialisation()

    # Parameters.
    # --------------------------------------------------------------------------
    # Import the parameters given the filename.
    param = ParameterFile_handler.readprmfile(filename)

    # How ghost entities are managed: Ghost entities are copies of mesh facets
    # or cells that reside on multiple processes in parallel simulations to
    # enable communication and computation across process boundaries.
    # "Shared facets" are used for communication between processes: multiple
    # processes share copies of certain facets. Improves parallel efficiency.
    parameters["ghost_mode"] = "shared_facet"

    # Output rules.
    # --------------------------------------------------------------------------
    if conv:
        OutputFN = (
            param["Output"]["Output XDMF File Name"] + "_Ref_" + str(iteration)
        )
    else:
        OutputFN = param["Output"]["Output XDMF File Name"]

    dont_print_to_video = param["Output"]["Limit print messages"]

    # Handling of the mesh and of the measures.
    # --------------------------------------------------------------------------
    mesh = Mesh_handler.MeshHandler(param, iteration)
    # Scale the mesh by the characteristic length. The mesh is now adimensional.
    mesh.scale(1 / param["Scaling Parameters"]["Characteristic Length"])
    # Normal vector of the mesh elements' faces.
    n = FacetNormal(mesh)
    # Cell diameter.
    h = CellDiameter(mesh)
    # Importing the Boundary Identifier.
    BoundaryID = BCs_handler.ImportFaceFunction(param, mesh)
    # Measures with boundary distinction definition.
    ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

    # Functional spaces and uknowns.
    # --------------------------------------------------------------------------
    P = FiniteElement(
        "DG",
        mesh.ufl_cell(),
        int(param["Spatial Discretization"]["Polynomial Degree"]),
    )
    # Set the vectorial space of the whole system.
    P_system = MixedElement([P, P])

    # Connecting the FEM element to the mesh discretization.
    X = FunctionSpace(mesh, P)
    X_system = FunctionSpace(mesh, P_system)

    # System unknown.
    x = Function(X_system)
    # Split system function to access components.
    x_c, x_q = x.split()

    # Time initialisation.
    # --------------------------------------------------------------------------
    T = param["Temporal Discretization"]["Final Time"]
    dt = param["Temporal Discretization"]["Time Step"]

    dt_out = param["Output"]["Timestep of File Save"]
    it_out = int(dt_out / dt)

    t = 0.0
    it = 0

    # Initial conditions.
    # --------------------------------------------------------------------------
    # Previous timestep functions' definition.
    c_old = Function(X)
    q_old = Function(X)

    # Previous - previous timestep functions' definition.
    c_oold = Function(X)
    q_oold = Function(X)

    # Construct the variational formulation.
    variational_form = FE.HeterodimerVariationalFormulation(
        X_system, mesh, ds_vent, ds_skull, h, n, param
    )
    # Initial conditions' construction.
    c_old, q_old = variational_form.initial_conditions_constructor(
        param, X, c_old, q_old, t, conv
    )
    c_oold, q_oold = variational_form.initial_conditions_constructor(
        param, X, c_oold, q_oold, t - dt, conv
    )

    # Assign the initial values component to the ukwnown.
    x_c.assign(c_old)
    x_q.assign(q_old)

    # Save the initial solution.
    # --------------------------------------------------------------------------
    xdmf_saver = XDMFHandler.SolutionSaver(
        OutputFN,
        mesh,
        [X, X],
        [
            "Concentration: healthy proteins",
            "Concentration: misfolded proteins",
        ],
        ["c", "q"],
        save_multiple_files=param["Output"]["Multiple output files"],
    )
    # Save the initial solution.
    xdmf_saver.save_solution([x_c, x_q], t)

    # Initialise error computation.
    if conv:
        errors.initialise(X, mesh, h, n, param)

    # Iterate on time.
    # --------------------------------------------------------------------------
    # Problem resolution cicle.
    it_tot = int(T / dt)

    while abs(it - it_tot) > 0.5:
        # Temporal advancement.
        t += dt
        it += 1

        # Variational formulation construction.
        a, L = variational_form.compute(t, c_old, q_old, c_oold, q_oold)

        # Construct the linear system.
        A = assemble(a)
        b = assemble(L)

        # Solve the linear system.
        x = LinearSolver_handler.LinearSolver(A, x, b, param)
        if dont_print_to_video == False:
            printp(
                "Problem solved at time {:.6f}".format(t),
                is_parallel,
                this_rank,
            )

        # Save the current solution.
        x_c, x_q = x.split(deepcopy=True)
        if (it % it_out) < DOLFIN_EPS:
            xdmf_saver.save_solution([x_c, x_q], t)

        # Time advancement of the solution
        c_oold.assign(c_old)
        q_oold.assign(q_old)

        c_old.assign(x_c)
        q_old.assign(x_q)

    # Error of approximation
    if conv:
        errors.compute_errors(x_c, x_q, iteration, t, it)


################################################################################
# Main.
################################################################################
# This line checks that the current script is run as main program.
if __name__ == "__main__":
    Common_main.main(sys.argv[1:], cwd, "/../../physics/Heterodimer")

    if MPI.comm_world.Get_rank() == 0:
        print("Problem Solved.")

    # Finalize MPI
    MPI.barrier(MPI.comm_world)  # Ensure all processes have reached this point

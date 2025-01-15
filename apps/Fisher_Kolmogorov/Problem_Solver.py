import mshr
import meshio
import math
import dolfin
from mpi4py import MPI
from fenics import *
import numpy as np
import matplotlib.pyplot as plt
import pyrameters as PRM
import os
import sys
import getopt
import pandas as pd

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/../../utilities")

import ParameterFile_handler as prmh
import BCs_handler
import Mesh_handler
import XDMF_handler
import LinearSolver_handler
import HDF5_handler
import Common_main
import Formulation_Elements as VFE
import Formulation_ElementsExp as VFEE
import ErrorComputation_handler


# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

    errors = pd.DataFrame(columns=["Error_L2_c", "Error_DG_c"])

    for it in range(0, conv):
        # Convergence iteration solver
        errors = problemsolver(filename, it, True, errors)
        errors.to_csv("ErrorsDGP4P5.csv")


def problemsolver(filename, iteration=0, conv=False, errors=False):

    # Import the parameters given the filename
    param = prmh.readprmfile(filename)

    parameters["ghost_mode"] = "shared_facet"

    # Handling of the mesh
    mesh = Mesh_handler.MeshHandler(param, iteration)

    mesh.scale(1 / param["Scaling Parameters"]["Characteristic Length"])

    # Importing the Boundary Identifier
    BoundaryID = BCs_handler.ImportFaceFunction(param, mesh)

    # Computing the mesh dimensionality
    D = mesh.topology().dim()

    # Functional Spaces
    if param["Spatial Discretization"]["Method"] == "DG-FEM":
        P = FiniteElement(
            "DG",
            mesh.ufl_cell(),
            int(param["Spatial Discretization"]["Polynomial Degree"]),
        )

    elif param["Spatial Discretization"]["Method"] == "CG-FEM":
        P = FiniteElement(
            "CG",
            mesh.ufl_cell(),
            int(param["Spatial Discretization"]["Polynomial Degree"]),
        )

    # Connecting the FEM element to the mesh discretization
    X = FunctionSpace(mesh, P)

    # Construction of tensorial space
    X9 = TensorFunctionSpace(mesh, "DG", 0)

    # Diffusion tensors definition
    if param["Model Parameters"]["Isotropic Diffusion"] == "No":
        K = Function(X9)
        filenameK = param["Model Parameters"][
            "Axonal Diffusion Tensor File Name"
        ]
        Kname = param["Model Parameters"][
            "Name of Axonal Diffusion Tensor in File"
        ]
        K = HDF5_handler.ImportPermeabilityTensor(filenameK, mesh, K, Kname)
    else:
        K = False

    # Time step and normal definitions
    dt = param["Temporal Discretization"]["Time Step"]
    T = param["Temporal Discretization"]["Final Time"]

    n = FacetNormal(mesh)

    x = Function(X)

    # Test function definition
    v = TestFunction(X)

    # Measure with boundary distinction definition
    ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

    # Time Initialization
    t = 0.0

    # Output file name definition
    if conv:
        OutputFN = (
            param["Output"]["Output XDMF File Name"]
            + "_Ref"
            + str(iteration)
            + "_"
        )
    else:
        OutputFN = param["Output"]["Output XDMF File Name"]

    # Choose the solver method
    if param["Spatial Discretization"]["Formulation"] == "Classical":

        # Solution functions definition
        c = TrialFunction(X)

        # Previous timestep functions definition
        c_old = Function(X)
        c_oold = Function(X)

        # Initial Condition Construction
        c_old = InitialConditionConstructor(param, mesh, X, c_old, conv, t)
        c_oold = InitialConditionConstructor(
            param, mesh, X, c_oold, conv, t - dt
        )
        x = c_old
    else:

        # Trial functions definition
        l = TrialFunction(X)

        # Previous timestep functions definition
        l_old = Function(X)

        # Initial Condition Construction
        l_old = InitialConditionConstructor(param, mesh, X, l_old, conv, t)
        l_old.vector()[:] = np.log(l_old.vector()[:])

        x.assign(l_old)

    # Save the time initial solution
    XDMF_handler.FKSolutionSave(
        OutputFN,
        x,
        t,
        dt,
        param["Spatial Discretization"]["Formulation"],
        mesh,
        X,
    )

    it = 0
    # Problem Resolution Cicle
    while t < T:

        # Temporal advancement
        t += dt
        it += 1

        # Variational Formulation Construction
        if param["Spatial Discretization"]["Formulation"] == "Classical":
            a, L = VariationalFormulation(
                param, c, v, dt, n, c_old, c_oold, K, t, mesh, ds_vent, ds_skull
            )

            A = assemble(a)
            b = assemble(L)

            if param["Spatial Discretization"]["Method"] == "CG-FEM":

                # Dirichlet Boundary Conditions vector construction
                bc = VFE.DirichletBoundary(X, param, BoundaryID, t, mesh)

                [bci.apply(A) for bci in bc]
                [bci.apply(A) for bci in bc]

            # Linear System Resolution
            x = LinearSolver_handler.LinearSolver(A, x, b, param)

        else:
            a, L = VariationalFormulationExp(
                param, x, v, dt, n, l_old, K, t, mesh, ds_vent, ds_skull
            )

            solve((a - L) == 0, x)

        # Save the time initial solution
        if it % param["Output"]["Timestep of File Save"] == 0:
            XDMF_handler.FKSolutionSave(
                OutputFN,
                x,
                t,
                dt,
                param["Spatial Discretization"]["Formulation"],
                mesh,
                X,
            )

        if MPI.comm_world.Get_rank() == 0:
            print("Problem at time {:.6f}".format(t), "solved")

        # Time advancement of the solution
        if param["Spatial Discretization"]["Formulation"] == "Classical":
            c_oold.assign(c_old)
            c_old.assign(x)

        else:
            l_old.assign(x)

    # Error of approximation
    if conv:
        errors = ErrorComputation_handler.FK_Errors(
            param, x, errors, mesh, iteration, t, n
        )

        if MPI.comm_world.Get_rank() == 0:
            print(errors)

    return errors


def VariationalFormulation(
    param, c, v, dt, n, c_old, c_oold, K, time, mesh, ds_vent, ds_skull
):

    time_prev = time - param["Temporal Discretization"]["Time Step"]
    period = param["Temporal Discretization"]["Problem Periodicity"]

    # Time-Discretization Parameter
    theta = param["Temporal Discretization"]["Theta-Method Parameter"]

    # Spatial Polynomial Order
    deg = param["Spatial Discretization"]["Polynomial Degree"]

    h = CellDiameter(mesh)

    # Scaling parameters
    U = param["Scaling Parameters"]["Characteristic Length"]
    tau = param["Scaling Parameters"]["Characteristic Time"]

    # Reaction Coefficient Extraction
    alpha = Constant(tau * param["Model Parameters"]["Reaction Coefficient"])

    # Diffusion Parameter Extraction
    d_ext = (tau / (U * U)) * param["Model Parameters"][
        "Extracellular diffusion"
    ]
    d_axn = (tau / (U * U)) * param["Model Parameters"]["Axonal diffusion"]

    # Bilinear forms Continuous Galerkin
    a = (
        VFE.dot_L2(1 / dt, c, v)
        + theta * VFE.ac(d_ext, d_axn, K, c, v)
        - theta * VFE.dot_L2(alpha, c, v)
    )

    # Semi-implicit Nonlinear Treatment
    if abs(theta - 1) < 0.001:
        a = a + theta * VFE.dot_L2(alpha * c_old, c, v)

    elif abs(theta - 0.5) < 0.001:
        a = a + theta * VFE.dot_L2(alpha * (1.5 * c_old - 0.5 * c_oold), c, v)

    # RHS Continuous Galerkin
    L = (
        VFE.dot_L2(1 / dt, c_old, v)
        - (1 - theta) * VFE.ac(d_ext, d_axn, K, c_old, v)
        + (1 - theta) * VFE.dot_L2(alpha, c_old, v)
        + theta * VFE.F(v, time, param)
        + (1 - theta) * VFE.F(v, time_prev, param)
        + theta * VFE.F_N(v, param, ds_vent, ds_skull, time, period)
        + theta * VFE.F_N(v, param, ds_vent, ds_skull, time_prev, period)
    )

    # Semi-implicit Nonlinear Treatment
    if abs(theta) < 0.001:
        L = L - (1 - theta) * VFE.dot_L2(alpha * c_old, c_old, v)

    elif abs(theta - 0.5) < 0.001:
        L = L - (1 - theta) * VFE.dot_L2(
            alpha * (1.5 * c_old - 0.5 * c_oold), c_old, v
        )

    # DISCONTINUOUS GALERKIN TERMS
    if param["Spatial Discretization"]["Method"] == "DG-FEM":

        # Definition of the stabilization parameters
        eta = Constant(
            param["Spatial Discretization"]["Discontinuous Galerkin"][
                "Penalty Parameter"
            ]
        )

        # Bilinear forms Disontinuous Galerkin
        a = (
            a
            + theta * VFE.ac_DG(d_ext, d_axn, K, c, v, eta, deg, h, n)
            + theta
            * VFE.ac_DG_D(
                c, v, param, ds_vent, ds_skull, d_ext, d_axn, K, eta, deg, h, n
            )
        )

        # RHS Discontinuous Galerkin
        L = (
            L
            - (1 - theta) * VFE.ac_DG(d_ext, d_axn, K, c_old, v, eta, deg, h, n)
            - (1 - theta)
            * VFE.ac_DG_D(
                c_old,
                v,
                param,
                ds_vent,
                ds_skull,
                d_ext,
                d_axn,
                K,
                eta,
                deg,
                h,
                n,
            )
            + theta
            * VFE.F_DG_D(
                v,
                param,
                ds_vent,
                ds_skull,
                d_ext,
                d_axn,
                K,
                eta,
                deg,
                h,
                n,
                time,
                period,
            )
            + (1 - theta)
            * VFE.F_DG_D(
                v,
                param,
                ds_vent,
                ds_skull,
                d_ext,
                d_axn,
                K,
                eta,
                deg,
                h,
                n,
                time_prev,
                period,
            )
        )

    return a, L


#########################################################################################################################
# 						Variational Formulation Definition					#
#########################################################################################################################


def VariationalFormulationExp(
    param, l, v, dt, n, l_old, K, time, mesh, ds_vent, ds_skull
):

    time_prev = time - param["Temporal Discretization"]["Time Step"]
    period = param["Temporal Discretization"]["Problem Periodicity"]

    # Time-Discretization Parameter
    theta = param["Temporal Discretization"]["Theta-Method Parameter"]

    # Spatial Polynomial Order
    deg = param["Spatial Discretization"]["Polynomial Degree"]

    h = CellDiameter(mesh)

    # Scaling parameters
    U = param["Scaling Parameters"]["Characteristic Length"]
    tau = param["Scaling Parameters"]["Characteristic Time"]

    # Reaction Coefficient Extraction
    alpha = Constant(tau * param["Model Parameters"]["Reaction Coefficient"])

    # Diffusion Parameter Extraction
    d_ext = (tau / (U * U)) * param["Model Parameters"][
        "Extracellular diffusion"
    ]
    d_axn = (tau / (U * U)) * param["Model Parameters"]["Axonal diffusion"]

    # Bilinear forms Continuous Galerkin
    a = (
        VFEE.dot_L2_exp(1 / dt, l, v)
        + theta * VFEE.ac_exp(d_ext, d_axn, K, l, v)
        - theta * VFEE.dot_L2_exp(alpha, l, v)
        + 2 * theta * (1 - theta) * VFEE.dot_L2_exp(alpha * exp(l_old), l, v)
        + theta * theta * VFEE.dot_L2_exp(alpha, 2 * l, v)
    )

    # RHS Continuous Galerkin
    L = (
        VFEE.dot_L2_exp(1 / dt, l_old, v)
        - (1 - theta) * VFEE.ac_exp(d_ext, d_axn, K, l_old, v)
        + theta * VFEE.F(v, time, param)
        + (1 - theta) * VFEE.F(v, time_prev, param)
        + (1 - theta) * VFEE.dot_L2_exp(alpha, l_old, v)
        - (1 - theta) * (1 - theta) * VFEE.dot_L2_exp(alpha, 2 * l_old, v)
        + theta * VFEE.F_N(v, param, ds_vent, ds_skull, time, period)
        + theta * VFEE.F_N(v, param, ds_vent, ds_skull, time_prev, period)
    )

    # DISCONTINUOUS GALERKIN TERMS
    if param["Spatial Discretization"]["Method"] == "DG-FEM":

        # Definition of the stabilization parameters
        eta = Constant(
            param["Spatial Discretization"]["Discontinuous Galerkin"][
                "Penalty Parameter"
            ]
        )

        # Bilinear forms Disontinuous Galerkin
        a = (
            a
            + theta * VFEE.ac_DG_exp(d_ext, d_axn, K, l, v, eta, deg, h, n)
            + theta
            * VFEE.ac_DG_D_exp(
                l, v, param, ds_vent, ds_skull, d_ext, d_axn, K, eta, deg, h, n
            )
        )

        # RHS Discontinuous Galerkin
        L = (
            L
            - (1 - theta)
            * VFEE.ac_DG_exp(d_ext, d_axn, K, l_old, v, eta, deg, h, n)
            - (1 - theta)
            * VFEE.ac_DG_D_exp(
                l, v, param, ds_vent, ds_skull, d_ext, d_axn, K, eta, deg, h, n
            )
            + theta
            * VFEE.F_DG_D(
                v,
                param,
                ds_vent,
                ds_skull,
                d_ext,
                d_axn,
                K,
                eta,
                deg,
                h,
                n,
                time,
                period,
            )
            + (1 - theta)
            * VFEE.F_DG_D(
                v,
                param,
                ds_vent,
                ds_skull,
                d_ext,
                d_axn,
                K,
                eta,
                deg,
                h,
                n,
                time_prev,
                period,
            )
        )

    return a, L


######################################################################################################
# 				Constructor of Initial Condition 				     #
######################################################################################################
def InitialConditionConstructor(param, mesh, X, c_old, conv, time):

    # Constant Initialization
    c0 = param["Model Parameters"]["Initial Condition"]
    x0 = Constant(c0)

    # Initial Condition Importing from Files
    if conv:
        c0 = param["Convergence Test"]["Exact Solution"]
        x0 = Expression(
            c0,
            degree=int(param["Spatial Discretization"]["Polynomial Degree"]),
            t=time,
        )

    c_old = interpolate(x0, X)

    if param["Model Parameters"]["Initial Condition from File"] == "Yes":

        c_old = HDF5_handler.ImportICfromFile(
            param["Model Parameters"]["Initial Condition File Name"],
            mesh,
            c_old,
            param["Model Parameters"]["Name of IC Function in File"],
        )

    return c_old


######################################################################
# 				Main 				     #
######################################################################

if __name__ == "__main__":

    Common_main.main(sys.argv[1:], cwd, "/../../physics/FisherKolmogorov")

    if MPI.comm_world.Get_rank() == 0:
        print("Problem Solved!")

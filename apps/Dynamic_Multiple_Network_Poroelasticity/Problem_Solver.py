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
import ErrorComputation_handler


# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

    errors = pd.DataFrame(
        columns=[
            "Error_L2_pC",
            "Error_L2_pA",
            "Error_L2_pV",
            "Error_L2_pE",
            "Error_L2_u",
            "Error_DG_pC",
            "Error_DG_pA",
            "Error_DG_pV",
            "Error_DG_pE",
            "Error_DG_u",
        ]
    )

    for it in range(0, conv):
        # Convergence iteration solver
        errors = problemsolver(filename, it, True, errors)
        errors.to_csv("solution/ErrorsDGP6P6.csv")


# PROBLEM SOLVER
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

    # Pressures and Displacement Functional Spaces
    if param["Spatial Discretization"]["Method"] == "DG-FEM":
        Pp = FiniteElement(
            "DG",
            mesh.ufl_cell(),
            int(
                param["Spatial Discretization"][
                    "Polynomial Degree for Pressure"
                ]
            ),
        )
        Pu = VectorElement(
            "DG",
            mesh.ufl_cell(),
            int(
                param["Spatial Discretization"][
                    "Polynomial Degree for Displacement"
                ]
            ),
        )

    elif param["Spatial Discretization"]["Method"] == "CG-FEM":
        Pp = FiniteElement(
            "CG",
            mesh.ufl_cell(),
            int(
                param["Spatial Discretization"][
                    "Polynomial Degree for Pressure"
                ]
            ),
        )
        Pu = VectorElement(
            "CG",
            mesh.ufl_cell(),
            int(
                param["Spatial Discretization"][
                    "Polynomial Degree for Displacement"
                ]
            ),
        )

    # Mixed FEM Spaces
    element = MixedElement([Pp, Pp, Pp, Pp, Pu])

    # Connecting the FEM element to the mesh discretization
    X = FunctionSpace(mesh, element)

    # Construction of tensorial space
    X9 = TensorFunctionSpace(mesh, "DG", 0)

    # Arterial permeability tensor
    if (
        param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Isotropic Permeability"
        ]
        == "No"
    ):
        K_At = Function(X9)
        filenameK = param["Model Parameters"]["Fluid Networks"][
            "Arterial Network"
        ]["Permeability Tensor File Name"]
        Kname = param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Name of Permeability Tensor in File"
        ]
        K_At = HDF5_handler.ImportPermeabilityTensor(
            filenameK, mesh, K_At, Kname
        )
    else:
        K_At = False

    # Capillary permeability tensor
    if (
        param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Isotropic Permeability"
        ]
        == "No"
    ):
        K_Ct = Function(X9)
        filenameK = param["Model Parameters"]["Fluid Networks"][
            "Capillary Network"
        ]["Permeability Tensor File Name"]
        Kname = param["Model Parameters"]["Fluid Networks"][
            "Capillary Network"
        ]["Name of Permeability Tensor in File"]
        K_Ct = HDF5_handler.ImportPermeabilityTensor(
            filenameK, mesh, K_Ct, Kname
        )
    else:
        K_Ct = False

    # Venous permeability tensor
    if (
        param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Isotropic Permeability"
        ]
        == "No"
    ):
        K_Vt = Function(X9)
        filenameK = param["Model Parameters"]["Fluid Networks"][
            "Venous Network"
        ]["Permeability Tensor File Name"]
        Kname = param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Name of Permeability Tensor in File"
        ]
        K_Vt = HDF5_handler.ImportPermeabilityTensor(
            filenameK, mesh, K_Vt, Kname
        )
    else:
        K_Vt = False

    # CSF-ISF permeability tensor
    if (
        param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Isotropic Permeability"
        ]
        == "No"
    ):
        K_Et = Function(X9)
        filenameK = param["Model Parameters"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Permeability Tensor File Name"]
        Kname = param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Name of Permeability Tensor in File"
        ]
        K_Et = HDF5_handler.ImportPermeabilityTensor(
            filenameK, mesh, K_Et, Kname
        )
    else:
        K_Et = False

    # Paramerer importation
    dt = param["Temporal Discretization"]["Time Step"]
    T = param["Temporal Discretization"]["Final Time"]
    beta = param["Temporal Discretization"]["Newmark Beta Parameter"]
    gamma = param["Temporal Discretization"]["Newmark Gamma Parameter"]

    n = FacetNormal(mesh)

    # Solution functions definition
    x = Function(X)
    pc, pa, pv, pe, u = TrialFunctions(X)

    # Test functions definition
    qc, qa, qv, qe, v = TestFunctions(X)

    # Previous timestep functions definition
    x_old = Function(X)
    to_a_old = Function(X)
    to_a_old_2 = Function(X)
    to_z_old = Function(X)

    pc_old, pa_old, pv_old, pe_old, u_old = x_old.split(deepcopy=True)
    pa_app, pc_app, pv_app, pe_app, a_old = to_a_old.split(deepcopy=True)
    pa_app, pc_app, pv_app, pe_app, z_old = to_z_old.split(deepcopy=True)

    # Measure with boundary distinction definition
    ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

    # Time Initialization
    t = 0.0

    # Initial Condition Construction
    x = InitialConditionConstructor(
        param, mesh, X, x, pa_old, pc_old, pv_old, pe_old, u_old, t, conv
    )
    to_z_old = VelocityInitialConditionConstructor(
        param, mesh, X, to_z_old, z_old, conv, t
    )
    pa_app, pc_app, pv_app, pe_app, z_old = to_z_old.split(deepcopy=True)

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

    # Save the time initial solution
    XDMF_handler.MPETSolutionSave(
        OutputFN,
        x,
        t,
        param["Temporal Discretization"]["Time Step"],
        param["Scaling Parameters"]["Characteristic Pressure"],
        param["Scaling Parameters"]["Characteristic Length"],
    )

    # Time advancement of the solution
    x_old.assign(x)
    pc_old, pa_old, pv_old, pe_old, u_old = split(x_old)

    # Problem Resolution Cicle
    while t < T:

        # Temporal advancement
        t += param["Temporal Discretization"]["Time Step"]

        # Variational Formulation Construction
        a, L = VariationalFormulation(
            param,
            pc,
            pa,
            pv,
            pe,
            u,
            qc,
            qa,
            qv,
            qe,
            v,
            dt,
            n,
            u_old,
            pc_old,
            pa_old,
            pv_old,
            pe_old,
            z_old,
            a_old,
            K_At,
            K_Vt,
            K_Ct,
            K_Et,
            t,
            mesh,
            ds_vent,
            ds_skull,
        )

        # Problem Solver Definition
        A = assemble(a)
        b = assemble(L)

        # Dirichlet Boundary Conditions vector construction and imposition
        if param["Spatial Discretization"]["Method"] == "CG-FEM":
            bc = VFE.DirichletBoundary(X, param, BoundaryID, t, mesh)

            [bci.apply(A) for bci in bc]
            [bci.apply(b) for bci in bc]

        # Linear System Resolution
        x = LinearSolver_handler.LinearSolver(A, x, b, param)

        # Save the solution at time t
        # if math.fmod(t, param['Output']['Timestep of File Save']) < (param['Output']['Timestep of File Save']/1000):
        XDMF_handler.MPETSolutionSave(
            OutputFN,
            x,
            t,
            param["Temporal Discretization"]["Time Step"],
            param["Scaling Parameters"]["Characteristic Pressure"],
            param["Scaling Parameters"]["Characteristic Length"],
        )

        # Newmark Advancement Step
        to_a_old_2.assign(to_a_old)
        to_a_old.assign(
            (x - x_old) / (beta * dt * dt)
            - to_z_old / (beta * dt)
            + (2 * beta - 1) / (2 * beta) * to_a_old_2
        )
        to_z_old.assign(
            to_z_old + dt * (gamma * to_a_old + (1 - gamma) * to_a_old_2)
        )
        x_old.assign(x)

        pa_app, pc_app, pv_app, pe_app, a_old = to_a_old.split()
        pa_app, pc_app, pv_app, pe_app, z_old = to_z_old.split()
        pc_old, pa_old, pv_old, pe_old, u_old = x_old.split(deepcopy=True)

        # End of the time iteration
        if MPI.comm_world.Get_rank() == 0:
            print("Problem at time {:.6f}".format(t), "solved")

    # Error of approximation
    if conv:
        errors = ErrorComputation_handler.MPET_Errors(
            param, x, errors, mesh, iteration, t, n
        )

        if MPI.comm_world.Get_rank() == 0:
            print(errors)

        return errors


#############################################################################
# 						Variational Formulation Definition					#
#############################################################################


def VariationalFormulation(
    param,
    pc,
    pa,
    pv,
    pe,
    u,
    qc,
    qa,
    qv,
    qe,
    v,
    dt,
    n,
    u_old,
    pc_old,
    pa_old,
    pv_old,
    pe_old,
    z_old,
    a_old,
    K_At,
    K_Vt,
    K_Ct,
    K_Et,
    time,
    mesh,
    ds_vent,
    ds_skull,
):

    period = param["Temporal Discretization"]["Problem Periodicity"]
    time_prev = time - param["Temporal Discretization"]["Time Step"]

    # Time-Discretization Parameters
    theta = Constant(param["Temporal Discretization"]["Theta-Method Parameter"])
    beta = Constant(param["Temporal Discretization"]["Newmark Beta Parameter"])
    gamma = Constant(
        param["Temporal Discretization"]["Newmark Gamma Parameter"]
    )

    # Spatial Polynomial Order
    degP = param["Spatial Discretization"]["Polynomial Degree for Pressure"]
    degU = param["Spatial Discretization"]["Polynomial Degree for Displacement"]

    h = CellDiameter(mesh)

    # Scaling Parameters
    U = param["Scaling Parameters"]["Characteristic Length"]
    P = param["Scaling Parameters"]["Characteristic Pressure"]
    tau = param["Scaling Parameters"]["Characteristic Time"]

    # Lamè Parameters Extraction
    G = Constant(
        (1.0 / P)
        * param["Model Parameters"]["Elastic Solid Tissue"][
            "First Lamé Parameter"
        ]
    )
    l = Constant(
        (1.0 / P)
        * param["Model Parameters"]["Elastic Solid Tissue"][
            "Second Lamé Parameter"
        ]
    )
    rho = Constant(
        (U * U / (tau * tau * P))
        * param["Model Parameters"]["Elastic Solid Tissue"]["Tissue Density"]
    )

    # Biot's Coefficients Extraction
    alphaA = Constant(
        param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Biot Coefficient"
        ]
    )
    alphaC = Constant(
        param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Biot Coefficient"
        ]
    )
    alphaV = Constant(
        param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Biot Coefficient"
        ]
    )
    alphaE = Constant(
        param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Biot Coefficient"
        ]
    )

    # Coupling Parameters Extraction
    wAC = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Arterial-Capillary Coupling Parameter"
        ]
    )
    wAV = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Arterial-Venous Coupling Parameter"
        ]
    )
    wEA = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "CSF-Arterial Coupling Parameter"
        ]
    )
    wEV = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "CSF-Venous Coupling Parameter"
        ]
    )
    wEC = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "CSF-Capillary Coupling Parameter"
        ]
    )
    wVC = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Venous-Capillary Coupling Parameter"
        ]
    )

    # External Coupling Parameters Extraction
    w_ext_C = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "External Coupling Parameter"
        ]
    )
    w_ext_A = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "External Coupling Parameter"
        ]
    )
    w_ext_V = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "External Coupling Parameter"
        ]
    )
    w_ext_E = Constant(
        (P * tau)
        * param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "External Coupling Parameter"
        ]
    )

    # Permeability Parameters Extraction
    KC = (tau * P / (U * U)) * param["Model Parameters"]["Fluid Networks"][
        "Capillary Network"
    ]["Permeability"]
    KA = (tau * P / (U * U)) * param["Model Parameters"]["Fluid Networks"][
        "Arterial Network"
    ]["Permeability"]
    KV = (tau * P / (U * U)) * param["Model Parameters"]["Fluid Networks"][
        "Venous Network"
    ]["Permeability"]
    KE = (tau * P / (U * U)) * param["Model Parameters"]["Fluid Networks"][
        "CSF-ISF Network"
    ]["Permeability"]

    # Time Derivative Terms Extraction
    cC = Constant(
        P
        * param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Time Derivative Coefficient"
        ]
    )
    cA = Constant(
        P
        * param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Time Derivative Coefficient"
        ]
    )
    cV = Constant(
        P
        * param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Time Derivative Coefficient"
        ]
    )
    cE = Constant(
        P
        * param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Time Derivative Coefficient"
        ]
    )

    # LINEAR POROELASTICITY MOMENTUM EQUAION
    aU = (
        VFE.dot_L2(rho / (beta * dt * dt), u, v)
        + VFE.a_el(G, l, u, v)
        - VFE.bp(alphaA, v, pa, 1)
        - VFE.bp(alphaC, v, pc, 1)
        - VFE.bp(alphaV, v, pv, 1)
        - VFE.bp(alphaE, v, pe, 1)
    )

    LU = (
        VFE.dot_L2(rho / (beta * dt * dt), u_old, v)
        + VFE.F_el(v, time, param, mesh)
        + VFE.F_N_el(v, param, ds_vent, ds_skull, mesh, time, period)
        + VFE.dot_L2(rho / (beta * dt), z_old, v)
        + (1 - 2 * beta) / (2 * beta) * VFE.dot_L2(rho, a_old, v)
    )

    # CAPILLARY PRESSURE BALANCE EQUATION
    aC = (
        VFE.tP_der(cC, pc, qc, dt)
        + theta * gamma / beta * VFE.bp(alphaC, u, qc, dt)
        + theta * VFE.ap(KC, K_Ct, pc, qc)
        + theta * VFE.C_coupl(wAC, pc, pa, qc)
        + theta * VFE.C_ext_coupl(w_ext_C, pc, qc)
        + theta * VFE.C_coupl(wVC, pc, pv, qc)
        + theta * VFE.C_coupl(wEC, pc, pe, qc)
    )

    LC = (
        VFE.tP_der(cC, pc_old, qc, dt)
        + theta * gamma / beta * VFE.bp(alphaC, u_old, qc, dt)
        - (1 - theta) * VFE.ap(KC, K_Ct, pc_old, qc)
        - (1 - theta) * VFE.C_coupl(wAC, pc_old, pa_old, qc)
        - (1 - theta) * VFE.C_ext_coupl(w_ext_C, pc_old, qc)
        - (1 - theta) * VFE.C_coupl(wVC, pc_old, pv_old, qc)
        - (1 - theta) * VFE.C_coupl(wEC, pc_old, pe_old, qc)
        + theta * VFE.F("Capillary", qc, time, param)
        + (1 - theta) * VFE.F("Capillary", qc, time_prev, param)
        + theta
        * VFE.F_N(
            qc, "Capillary Network", param, ds_vent, ds_skull, time, period
        )
        + (1 - theta)
        * VFE.F_N(
            qc, "Capillary Network", param, ds_vent, ds_skull, time_prev, period
        )
        + (theta * gamma / beta - 1) * VFE.bp(alphaC, z_old, qc, 1)
        + theta * (gamma / (2 * beta) - 1) * VFE.bp(alphaC, a_old, qc, 1 / dt)
    )

    # ARTERIAL PRESSURE BALANCE EQUATION
    aA = (
        VFE.tP_der(cA, pa, qa, dt)
        + theta * gamma / beta * VFE.bp(alphaA, u, qa, dt)
        + theta * VFE.ap(KA, K_At, pa, qa)
        + theta * VFE.C_coupl(wAC, pa, pc, qa)
        + theta * VFE.C_ext_coupl(w_ext_A, pa, qa)
        + theta * VFE.C_coupl(wAV, pa, pv, qa)
        + theta * VFE.C_coupl(wEA, pa, pe, qa)
    )

    LA = (
        VFE.tP_der(cA, pa_old, qa, dt)
        + theta * gamma / beta * VFE.bp(alphaA, u_old, qa, dt)
        - (1 - theta) * VFE.ap(KA, K_At, pa_old, qa)
        - (1 - theta) * VFE.C_coupl(wAC, pa_old, pc_old, qa)
        - (1 - theta) * VFE.C_ext_coupl(w_ext_A, pa_old, qa)
        - (1 - theta) * VFE.C_coupl(wAV, pa_old, pv_old, qa)
        - (1 - theta) * VFE.C_coupl(wEA, pa_old, pe_old, qa)
        + theta * VFE.F("Arterial", qa, time, param)
        + (1 - theta) * VFE.F("Arterial", qa, time_prev, param)
        + theta
        * VFE.F_N(
            qa, "Arterial Network", param, ds_vent, ds_skull, time, period
        )
        + (1 - theta)
        * VFE.F_N(
            qa, "Arterial Network", param, ds_vent, ds_skull, time_prev, period
        )
        + (theta * gamma / beta - 1) * VFE.bp(alphaA, z_old, qa, 1)
        + theta * (gamma / (2 * beta) - 1) * VFE.bp(alphaA, a_old, qa, 1 / dt)
    )

    # VENOUS PRESSURE BALANCE EQUATION
    aV = (
        VFE.tP_der(cV, pv, qv, dt)
        + theta * gamma / beta * VFE.bp(alphaV, u, qv, dt)
        + theta * VFE.ap(KV, K_Vt, pv, qv)
        + theta * VFE.C_coupl(wVC, pv, pc, qv)
        + theta * VFE.C_ext_coupl(w_ext_V, pv, qv)
        + theta * VFE.C_coupl(wAV, pv, pa, qv)
        + theta * VFE.C_coupl(wEV, pv, pe, qv)
    )

    LV = (
        VFE.tP_der(cV, pv_old, qv, dt)
        + theta * gamma / beta * VFE.bp(alphaV, u_old, qv, dt)
        - (1 - theta) * VFE.ap(KV, K_Vt, pv_old, qv)
        - (1 - theta) * VFE.C_coupl(wVC, pv_old, pc_old, qv)
        - (1 - theta) * VFE.C_ext_coupl(w_ext_V, pv_old, qv)
        - (1 - theta) * VFE.C_coupl(wAV, pv_old, pa_old, qv)
        - (1 - theta) * VFE.C_coupl(wEV, pv_old, pe_old, qv)
        + theta * VFE.F("Venous", qv, time, param)
        + (1 - theta) * VFE.F("Venous", qv, time_prev, param)
        + theta
        * VFE.F_N(qv, "Venous Network", param, ds_vent, ds_skull, time, period)
        + (1 - theta)
        * VFE.F_N(
            qv, "Venous Network", param, ds_vent, ds_skull, time_prev, period
        )
        + (theta * gamma / beta - 1) * VFE.bp(alphaV, z_old, qv, 1)
        + theta * (gamma / (2 * beta) - 1) * VFE.bp(alphaV, a_old, qv, 1 / dt)
    )

    # CSF-ISF PRESSURE BALANCE EQUATION
    aE = (
        VFE.tP_der(cE, pe, qe, dt)
        + theta * gamma / beta * VFE.bp(alphaE, u, qe, dt)
        + theta * VFE.ap(KE, K_Et, pe, qe)
        + theta * VFE.C_coupl(wEC, pe, pc, qe)
        + theta * VFE.C_coupl(wEA, pe, pa, qe)
        + theta * VFE.C_coupl(wEV, pe, pv, qe)
        + theta * VFE.C_ext_coupl(w_ext_E, pe, qe)
    )

    LE = (
        VFE.tP_der(cE, pe_old, qe, dt)
        + theta * gamma / beta * VFE.bp(alphaE, u_old, qe, dt)
        - (1 - theta) * VFE.ap(KE, K_Et, pe_old, qe)
        - (1 - theta) * VFE.C_coupl(wEC, pe_old, pc_old, qe)
        - (1 - theta) * VFE.C_ext_coupl(w_ext_E, pe_old, qe)
        - (1 - theta) * VFE.C_coupl(wEA, pe_old, pa_old, qe)
        - (1 - theta) * VFE.C_coupl(wEV, pe_old, pv_old, qe)
        + theta * VFE.F("CSF-ISF", qe, time, param)
        + (1 - theta) * VFE.F("CSF-ISF", qe, time_prev, param)
        + theta
        * VFE.F_N(qe, "CSF-ISF Network", param, ds_vent, ds_skull, time, period)
        + (1 - theta)
        * VFE.F_N(
            qe, "CSF-ISF Network", param, ds_vent, ds_skull, time_prev, period
        )
        + (theta * gamma / beta - 1) * VFE.bp(alphaE, z_old, qe, 1)
        + theta * (gamma / (2 * beta) - 1) * VFE.bp(alphaE, a_old, qe, 1 / dt)
    )

    # DISCONTINUOUS GALERKIN TERMS
    if param["Spatial Discretization"]["Method"] == "DG-FEM":

        # Definition of the stabilization parameters
        etaC = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Capillary Pressure"
        ]
        etaA = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Arterial Pressure"
        ]
        etaV = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Venous Pressure"
        ]
        etaE = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for CSF-ISF Pressure"
        ]
        etaU = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Displacement"
        ]

        # LINEAR POROELASTICITY MOMENTUM EQUAION
        aU = (
            aU
            - VFE.b_DG(alphaA, v, n, pa, 1)
            - VFE.b_DG_D(
                "Arterial Network",
                alphaA,
                pa,
                v,
                ds_vent,
                ds_skull,
                n,
                1,
                param,
            )
            - VFE.b_DG(alphaV, v, n, pv, 1)
            - VFE.b_DG_D(
                "Venous Network", alphaV, pv, v, ds_vent, ds_skull, n, 1, param
            )
            - VFE.b_DG(alphaC, v, n, pc, 1)
            - VFE.b_DG_D(
                "Capillary Network",
                alphaC,
                pc,
                v,
                ds_vent,
                ds_skull,
                n,
                1,
                param,
            )
            - VFE.b_DG(alphaE, v, n, pe, 1)
            - VFE.b_DG_D(
                "CSF-ISF Network", alphaE, pe, v, ds_vent, ds_skull, n, 1, param
            )
            + VFE.a_el_DG(G, l, u, v, etaU, degU, h, n)
            + VFE.a_el_DG_D(
                G, l, u, v, ds_skull, ds_vent, etaU, degU, h, n, param
            )
        )

        LU = LU + VFE.F_el_DG_D(
            G,
            l,
            u,
            v,
            ds_skull,
            ds_vent,
            etaU,
            degU,
            h,
            n,
            param,
            mesh,
            time,
            period,
        )

        # CAPILLARY PRESSURE BALANCE EQUATION
        aC = (
            aC
            + theta * VFE.ap_DG(KC, K_Ct, pc, qc, etaC, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pc,
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                K_Ct,
                etaC,
                degP,
                h,
                n,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaC, u, n, qc, dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "Capillary Network",
                alphaC,
                qc,
                u,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
        )

        LC = (
            LC
            - (1 - theta) * VFE.ap_DG(KC, K_Ct, pc_old, qc, etaC, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pc_old,
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                K_Ct,
                etaC,
                degP,
                h,
                n,
            )
            + theta
            * VFE.F_DG_D(
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                K_Ct,
                etaC,
                degP,
                h,
                n,
                time,
                period,
            )
            + (1 - theta)
            * VFE.F_DG_D(
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                K_Ct,
                etaC,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaC, u_old, n, qc, dt)
            + (theta * gamma / beta - 1) * VFE.b_DG(alphaC, z_old, n, qc, 1)
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG(alphaC, a_old, n, qc, 1 / dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "Capillary Network",
                alphaC,
                qc,
                u_old,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
            + (theta * gamma / beta - 1)
            * VFE.b_DG_D(
                "Capillary Network",
                alphaC,
                qc,
                z_old,
                ds_vent,
                ds_skull,
                n,
                1,
                param,
            )
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG_D(
                "Capillary Network",
                alphaC,
                qc,
                a_old,
                ds_vent,
                ds_skull,
                n,
                1 / dt,
                param,
            )
            + (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaC,
                n,
                time,
                period,
                mesh,
            )
            - (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaC,
                n,
                time_prev,
                period,
                mesh,
            )
        )

        # ARTERIAL PRESSURE BALANCE EQUATION
        aA = (
            aA
            + theta * VFE.ap_DG(KA, K_At, pa, qa, etaA, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pa,
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K_At,
                etaA,
                degP,
                h,
                n,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaA, u, n, qa, dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "Arterial Network",
                alphaA,
                qa,
                u,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
        )

        LA = (
            LA
            - (1 - theta) * VFE.ap_DG(KA, K_At, pa_old, qa, etaA, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pa_old,
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K_At,
                etaA,
                degP,
                h,
                n,
            )
            + theta
            * VFE.F_DG_D(
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K_At,
                etaA,
                degP,
                h,
                n,
                time,
                period,
            )
            + (1 - theta)
            * VFE.F_DG_D(
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K_At,
                etaA,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaA, u_old, n, qa, dt)
            + (theta * gamma / beta - 1) * VFE.b_DG(alphaA, z_old, n, qa, 1)
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG(alphaA, a_old, n, qa, 1 / dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "Arterial Network",
                alphaA,
                qa,
                u_old,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
            + (theta * gamma / beta - 1)
            * VFE.b_DG_D(
                "Arterial Network",
                alphaA,
                qa,
                z_old,
                ds_vent,
                ds_skull,
                n,
                1,
                param,
            )
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG_D(
                "Arterial Network",
                alphaA,
                qa,
                a_old,
                ds_vent,
                ds_skull,
                n,
                1 / dt,
                param,
            )
            + (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaA,
                n,
                time,
                period,
                mesh,
            )
            - (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaA,
                n,
                time_prev,
                period,
                mesh,
            )
        )

        # VENOUS PRESSURE BALANCE EQUATION
        aV = (
            aV
            + theta * VFE.ap_DG(KV, K_Vt, pv, qv, etaV, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pv,
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K_Vt,
                etaV,
                degP,
                h,
                n,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaV, u, n, qv, dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "Venous Network", alphaV, qv, u, ds_vent, ds_skull, n, dt, param
            )
        )

        LV = (
            LV
            - (1 - theta) * VFE.ap_DG(KV, K_Vt, pv_old, qv, etaV, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pv_old,
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K_Vt,
                etaV,
                degP,
                h,
                n,
            )
            + theta
            * VFE.F_DG_D(
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K_Vt,
                etaV,
                degP,
                h,
                n,
                time,
                period,
            )
            + (1 - theta)
            * VFE.F_DG_D(
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K_Vt,
                etaV,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaV, u_old, n, qv, dt)
            + (theta * gamma / beta - 1) * VFE.b_DG(alphaV, z_old, n, qv, 1)
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG(alphaV, a_old, n, qv, 1 / dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "Venous Network",
                alphaV,
                qv,
                u_old,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
            + (theta * gamma / beta - 1)
            * VFE.b_DG_D(
                "Venous Network",
                alphaV,
                qv,
                z_old,
                ds_vent,
                ds_skull,
                n,
                1,
                param,
            )
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG_D(
                "Venous Network",
                alphaV,
                qv,
                a_old,
                ds_vent,
                ds_skull,
                n,
                1 / dt,
                param,
            )
            + (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaV,
                n,
                time,
                period,
                mesh,
            )
            - (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaV,
                n,
                time_prev,
                period,
                mesh,
            )
        )

        # CSF-ISF PRESSURE BALANCE EQUATION
        aE = (
            aE
            + theta * VFE.ap_DG(KE, K_Et, pe, qe, etaE, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pe,
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                KE,
                K_Et,
                etaE,
                degP,
                h,
                n,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaE, u, n, qe, dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "CSF-ISF Network",
                alphaE,
                qe,
                u,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
        )

        LE = (
            LE
            - (1 - theta) * VFE.ap_DG(KE, K_Et, pe_old, qe, etaE, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pe_old,
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                KE,
                K_Et,
                etaE,
                degP,
                h,
                n,
            )
            + theta
            * VFE.F_DG_D(
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                KE,
                K_Et,
                etaE,
                degP,
                h,
                n,
                time,
                period,
            )
            + (1 - theta)
            * VFE.F_DG_D(
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                KE,
                K_Et,
                etaE,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + (theta * gamma) / beta * VFE.b_DG(alphaE, u_old, n, qe, dt)
            + (theta * gamma / beta - 1) * VFE.b_DG(alphaE, z_old, n, qe, 1)
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG(alphaE, a_old, n, qe, 1 / dt)
            + (theta * gamma)
            / beta
            * VFE.b_DG_D(
                "CSF-ISF Network",
                alphaE,
                qe,
                u_old,
                ds_vent,
                ds_skull,
                n,
                dt,
                param,
            )
            + (theta * gamma / beta - 1)
            * VFE.b_DG_D(
                "CSF-ISF Network",
                alphaE,
                qe,
                z_old,
                ds_vent,
                ds_skull,
                n,
                1,
                param,
            )
            - theta
            * (1 - gamma / (2 * beta))
            * VFE.b_DG_D(
                "CSF-ISF Network",
                alphaE,
                qe,
                a_old,
                ds_vent,
                ds_skull,
                n,
                1 / dt,
                param,
            )
            + (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaE,
                n,
                time,
                period,
                mesh,
            )
            - (theta * gamma)
            / beta
            * VFE.Fb_DG_D(
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                dt,
                alphaE,
                n,
                time_prev,
                period,
                mesh,
            )
        )

        # FINAL RHS AND LHS TERMS CONSTRUCTION
    a = aA + aC + aV + aE + aU
    L = LA + LC + LV + LE + LU

    return a, L


##############################################################################
# 						Constructor of Initial Condition					 #
##############################################################################


def InitialConditionConstructor(
    param, mesh, X, x, pa_old, pc_old, pv_old, pe_old, u_old, time, conv
):

    U = param["Scaling Parameters"]["Characteristic Length"]
    P = param["Scaling Parameters"]["Characteristic Pressure"]

    # Constant Initial Condition Construction
    pa0 = param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Initial Condition (Pressure)"
    ]
    pc0 = param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Initial Condition (Pressure)"
    ]
    pv0 = param["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Initial Condition (Pressure)"
    ]
    pe0 = param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Initial Condition (Pressure)"
    ]
    u0 = param["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition (Displacement)"
    ]

    if mesh.ufl_cell() == triangle:
        x0 = Constant(
            (pc0 / P, pa0 / P, pv0 / P, pe0 / P, u0[0] / U, u0[1] / U)
        )

    else:
        x0 = Constant(
            (
                pc0 / P,
                pa0 / P,
                pv0 / P,
                pe0 / P,
                u0[0] / U,
                u0[1] / U,
                u0[2] / U,
            )
        )

    # Analytical Initial Condition Construction
    if conv:
        ux = param["Convergence Test"][
            "Displacement Exact Solution (x-component)"
        ]
        uy = param["Convergence Test"][
            "Displacement Exact Solution (y-component)"
        ]
        uz = param["Convergence Test"][
            "Displacement Exact Solution (z-component)"
        ]
        pa = param["Convergence Test"]["Arterial Pressure Exact Solution"]
        pc = param["Convergence Test"]["Capillary Pressure Exact Solution"]
        pv = param["Convergence Test"]["Venous Pressure Exact Solution"]
        pe = param["Convergence Test"]["CSF-ISF Pressure Exact Solution"]

        if mesh.ufl_cell() == triangle:
            x0 = Expression(
                (pc, pa, pv, pe, ux, uy),
                degree=int(
                    param["Spatial Discretization"][
                        "Polynomial Degree for Displacement"
                    ]
                ),
                t=time,
            )

        else:
            x0 = Expression(
                (pc, pa, pv, pe, ux, uy, uz),
                degree=int(
                    param["Spatial Discretization"][
                        "Polynomial Degree for Displacement"
                    ]
                ),
                t=time,
            )

    x = interpolate(x0, X)

    # Initial Condition Importing from Files
    if (
        param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pc_old = HDF5_handler.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pc_old,
            param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
                "Name of IC Function in File"
            ],
        )
        pc_old.vector()[:] = pc_old.vector()[:] / P
        assign(x.sub(0), pc_old)

    if (
        param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pa_old = HDF5_handler.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pa_old,
            param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
                "Name of IC Function in File"
            ],
        )
        pa_old.vector()[:] = pa_old.vector()[:] / P
        assign(x.sub(1), pa_old)

    if (
        param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pv_old = HDF5_handler.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["Venous Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pv_old,
            param["Model Parameters"]["Fluid Networks"]["Venous Network"][
                "Name of IC Function in File"
            ],
        )
        pv_old.vector()[:] = pv_old.vector()[:] / P
        assign(x.sub(2), pv_old)

    if (
        param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pe_old = HDF5_handler.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pe_old,
            param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
                "Name of IC Function in File"
            ],
        )
        pe_old.vector()[:] = pe_old.vector()[:] / P
        assign(x.sub(3), pe_old)

    if (
        param["Model Parameters"]["Elastic Solid Tissue"][
            "Initial Condition from File (Displacement)"
        ]
        == "Yes"
    ):

        u_old = HDF5_handler.ImportICfromFile(
            param["Model Parameters"]["Elastic Solid Tissue"][
                "Initial Condition File Name"
            ],
            mesh,
            u_old,
            param["Model Parameters"]["Elastic Solid Tissue"][
                "Name of IC Function in File"
            ],
        )
        u_old.vector()[:] = u_old.vector()[:] / U
        assign(x.sub(4), u_old)

    return x


##############################################################################
# 				Constructor of Velocity Initial Condition					 #
##############################################################################


def VelocityInitialConditionConstructor(param, mesh, X, z, z_old, conv, time):

    # Constant Initial Condition Construction
    v0 = param["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition (Velocity)"
    ]
    U = param["Scaling Parameters"]["Characteristic Length"]

    if mesh.ufl_cell() == triangle:
        v0 = Constant((0, 0, 0, 0, v0[0] / U, v0[1] / U))

    else:
        v0 = Constant((0, 0, 0, 0, v0[0] / U, v0[1] / U, v0[2] / U))

    # Analytical Initial Condition Construction
    if conv:
        vx = param["Convergence Test"]["Velocity Exact Solution (x-component)"]
        vy = param["Convergence Test"]["Velocity Exact Solution (y-component)"]
        vz = param["Convergence Test"]["Velocity Exact Solution (z-component)"]

        if mesh.ufl_cell() == triangle:
            v0 = Expression(
                ("0", "0", "0", "0", vx, vy),
                degree=int(
                    param["Spatial Discretization"][
                        "Polynomial Degree for Displacement"
                    ]
                ),
                t=time,
            )

        else:
            v0 = Expression(
                ("0", "0", "0", "0", vx, vy, vz),
                degree=int(
                    param["Spatial Discretization"][
                        "Polynomial Degree for Displacement"
                    ]
                ),
                t=time,
            )

    z = interpolate(v0, X)

    # Initial Condition Importing from File
    if (
        param["Model Parameters"]["Elastic Solid Tissue"][
            "Initial Condition from File (Velocity)"
        ]
        == "Yes"
    ):

        z_old = HDF5.ImportICfromFile(
            param["Model Parameters"]["Elastic Solid Tissue"][
                "Initial Condition File Name (Velocity)"
            ],
            mesh,
            z_old,
            param["Model Parameters"]["Elastic Solid Tissue"][
                "Name of IC Function in File (Velocity)"
            ],
        )
        z_old.vector()[:] = z_old.vector()[:] / U
        assign(v.sub(4), z_old)

    return z


##############################################
# 					Main 				     #
##############################################

if __name__ == "__main__":

    Common_main.main(sys.argv[1:], cwd, "/../../physics/BrainMPET")

    if MPI.comm_world.Get_rank() == 0:
        print("Problem Solved!")

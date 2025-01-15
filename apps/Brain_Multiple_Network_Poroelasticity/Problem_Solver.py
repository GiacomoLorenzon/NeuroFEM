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
import TensorialDiffusion_handler
import DarcyIC_handler
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
            "Error_H1_pC",
            "Error_H1_pA",
            "Error_H1_pV",
            "Error_H1_pE",
            "Error_H1_u",
        ]
    )

    for it in range(0, conv):
        # Convergence iteration solver
        errors = problemsolver(filename, it, True, errors)
        errors.to_csv("solution/ErrorsDGP3P4.csv")


# PROBLEM SOLVER
def problemsolver(filename, iteration=0, conv=False, errors=False):

    # Import the parameters given the filename
    param = prmh.readprmfile(filename)

    parameters["ghost_mode"] = "shared_facet"

    # Handling of the mesh
    mesh = Mesh_handler.MeshHandler(param, iteration)

    # Importing the Boundary Identifier
    BoundaryID = BCs_handler.ImportFaceFunction(param, mesh)

    # Computing the mesh dimensionality
    D = mesh.topology().dim()

    # Pressures and Displacement Functional Spaces
    if param["Spatial Discretization"]["Method"] == "DG-FEM":
        P1 = FiniteElement(
            "DG",
            mesh.ufl_cell(),
            int(param["Spatial Discretization"]["Polynomial Degree"]),
        )
        P2 = VectorElement(
            "DG",
            mesh.ufl_cell(),
            int(param["Spatial Discretization"]["Polynomial Degree"] + 1),
        )

    elif param["Spatial Discretization"]["Method"] == "CG-FEM":
        P1 = FiniteElement(
            "CG",
            mesh.ufl_cell(),
            int(param["Spatial Discretization"]["Polynomial Degree"]),
        )
        P2 = VectorElement(
            "CG",
            mesh.ufl_cell(),
            int(param["Spatial Discretization"]["Polynomial Degree"] + 1),
        )

    # Mixed FEM Spaces
    element = MixedElement([P1, P1, P1, P1, P2])

    # Connecting the FEM element to the mesh discretization
    X = FunctionSpace(mesh, element)

    # Construction of tensorial space
    X9 = TensorFunctionSpace(mesh, "DG", 0)

    # Permeability tensors definition
    if (
        param["Model Parameters"]["Fluid Networks"]["Isotropic Permeability"]
        == "No"
    ):

        K = Function(X9)
        K = TensorialDiffusion_handler.ImportPermeabilityTensor(param, mesh, K)

    else:
        K = False

    # Time step and normal definitions
    dt = Constant(param["Temporal Discretization"]["Time Step"])
    T = param["Temporal Discretization"]["Final Time"]

    n = FacetNormal(mesh)

    # Solution functions definition
    x = Function(X)
    pc, pa, pv, pe, u = TrialFunctions(X)

    # Test functions definition
    qc, qa, qv, qe, v = TestFunctions(X)

    # Previous timestep functions definition
    x_old = Function(X)
    pc_old, pa_old, pv_old, pe_old, u_old = x_old.split(deepcopy=True)

    # Measure with boundary distinction definition
    ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

    # Time Initialization
    t = 0.0

    # Initial Condition Construction
    x = InitialConditionConstructor(
        param, mesh, X, x, pa_old, pc_old, pv_old, pe_old, u_old
    )

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
        OutputFN, x, t, param["Temporal Discretization"]["Time Step"]
    )

    # Time advancement of the solution
    x_old.assign(x)
    pc_old, pa_old, pv_old, pe_old, u_old = split(x_old)

    # Problem Resolution Cicle
    while t < T:

        # Temporal advancement
        t += param["Temporal Discretization"]["Time Step"]

        # Dirichlet Boundary Conditions vector construction
        bc = VFE.DirichletBoundary(X, param, BoundaryID, t, mesh)

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
            K,
            t,
            mesh,
            ds_vent,
            ds_skull,
        )

        # Problem Solver Definition
        A = assemble(a)
        b = assemble(L)

        if param["Spatial Discretization"]["Method"] == "CG-FEM":
            [bci.apply(A) for bci in bc]
            [bci.apply(b) for bci in bc]

        # Linear System Resolution
        x = LinearSolver_handler.LinearSolver(A, x, b, param)

        # Save the solution at time t
        XDMF_handler.MPETSolutionSave(
            OutputFN, x, t, param["Temporal Discretization"]["Time Step"]
        )

        if MPI.comm_world.Get_rank() == 0:
            print("Problem at time {:.6f}".format(t), "solved")

        # Time advancement of the solution
        x_old.assign(x)
        pc_old, pa_old, pv_old, pe_old, u_old = x_old.split(deepcopy=True)

    # Error of approximation
    if conv:
        errors = ErrorComputation_handler.MPET_Errors(
            param, x, errors, mesh, iteration, t
        )

        if MPI.comm_world.Get_rank() == 0:
            print(errors)

        return errors


#########################################################################################################################
# 						Variational Formulation Definition					#
#########################################################################################################################


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
    K,
    time,
    mesh,
    ds_vent,
    ds_skull,
):

    period = param["Temporal Discretization"]["Problem Periodicity"]
    time_prev = time - param["Temporal Discretization"]["Time Step"]
    theta = param["Temporal Discretization"]["Theta-Method Parameter"]
    degP = param["Spatial Discretization"]["Polynomial Degree"]
    degU = degP + 1

    h = CellDiameter(mesh)

    # Lamè Parameters Extraction
    G = Constant(
        param["Model Parameters"]["Elastic Solid Tissue"][
            "First Lamé Parameter"
        ]
    )
    l = Constant(
        param["Model Parameters"]["Elastic Solid Tissue"][
            "Second Lamé Parameter"
        ]
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
        param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Arterial-Capillary Coupling Parameter"
        ]
    )
    wAV = Constant(
        param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Arterial-Venous Coupling Parameter"
        ]
    )
    wEA = Constant(
        param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "CSF-Arterial Coupling Parameter"
        ]
    )
    wEV = Constant(
        param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "CSF-Venous Coupling Parameter"
        ]
    )
    wEC = Constant(
        param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "CSF-Capillary Coupling Parameter"
        ]
    )
    wVC = Constant(
        param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Venous-Capillary Coupling Parameter"
        ]
    )

    # Permeability Parameters Extraction
    KC = Constant(
        param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Permeability"
        ]
    )
    KA = Constant(
        param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Permeability"
        ]
    )
    KV = Constant(
        param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Permeability"
        ]
    )
    KE = Constant(
        param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Permeability"
        ]
    )

    # Time Derivative Terms Extraction
    cC = Constant(
        param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Time Derivative Coefficient"
        ]
    )
    cA = Constant(
        param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Time Derivative Coefficient"
        ]
    )
    cV = Constant(
        param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Time Derivative Coefficient"
        ]
    )
    cE = Constant(
        param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Time Derivative Coefficient"
        ]
    )

    # LINEAR POROELASTICITY MOMENTUM EQUAION
    aU = (
        VFE.a_el(G, l, u, v)
        - VFE.bp(alphaA, v, pa, 1)
        - VFE.bp(alphaC, v, pc, 1)
        - VFE.bp(alphaV, v, pv, 1)
        - VFE.bp(alphaE, v, pe, 1)
    )

    LU = VFE.F_el(v, time, param, mesh) + VFE.F_N_el(
        v, param, ds_vent, ds_skull, time, period
    )

    # CAPILLARY PRESSURE BALANCE EQUATION
    aC = (
        VFE.tP_der(cC, pc, qc, dt)
        + VFE.bp(alphaC, u, qc, dt)
        + theta * VFE.ap(KC, False, pc, qc)
        + theta * VFE.C_coupl(wAC, pc, pa, qc)
        + theta * VFE.C_coupl(wVC, pc, pv, qc)
        + theta * VFE.C_coupl(wEC, pc, pe, qc)
    )

    LC = (
        VFE.tP_der(cC, pc_old, qc, dt)
        + VFE.bp(alphaC, u_old, qc, dt)
        - (1 - theta) * VFE.ap(KC, False, pc_old, qc)
        - (1 - theta) * VFE.C_coupl(wAC, pc_old, pa_old, qc)
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
    )

    # ARTERIAL PRESSURE BALANCE EQUATION
    aA = (
        VFE.tP_der(cA, pa, qa, dt)
        + VFE.bp(alphaA, u, qa, dt)
        + theta * VFE.ap(KA, K, pa, qa)
        + theta * VFE.C_coupl(wAC, pa, pc, qa)
        + theta * VFE.C_coupl(wAV, pa, pv, qa)
        + theta * VFE.C_coupl(wEA, pa, pe, qa)
    )

    LA = (
        VFE.tP_der(cA, pa_old, qa, dt)
        + VFE.bp(alphaA, u_old, qa, dt)
        - (1 - theta) * VFE.ap(KA, K, pa_old, qa)
        - (1 - theta) * VFE.C_coupl(wAC, pa_old, pc_old, qa)
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
    )

    # VENOUS PRESSURE BALANCE EQUATION
    aV = (
        VFE.tP_der(cV, pv, qv, dt)
        + VFE.bp(alphaV, u, qv, dt)
        + theta * VFE.ap(KV, K, pv, qv)
        + theta * VFE.C_coupl(wVC, pv, pc, qv)
        + theta * VFE.C_coupl(wAV, pv, pa, qv)
        + theta * VFE.C_coupl(wEV, pv, pe, qv)
    )

    LV = (
        VFE.tP_der(cV, pv_old, qv, dt)
        + VFE.bp(alphaV, u_old, qv, dt)
        - (1 - theta) * VFE.ap(KV, K, pv_old, qv)
        - (1 - theta) * VFE.C_coupl(wVC, pv_old, pc_old, qv)
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
    )

    # CSF-ISF PRESSURE BALANCE EQUATION
    aE = (
        VFE.tP_der(cE, pe, qe, dt)
        + VFE.bp(alphaE, u, qe, dt)
        + theta * VFE.ap(KE, K, pe, qe)
        + theta * VFE.C_coupl(wEC, pe, pc, qe)
        + theta * VFE.C_coupl(wEA, pe, pa, qe)
        + theta * VFE.C_coupl(wEV, pe, pv, qe)
    )

    # CSF-ISF Pressure RHS
    LE = (
        VFE.tP_der(cE, pe_old, qe, dt)
        + VFE.bp(alphaE, u_old, qe, dt)
        - (1 - theta) * VFE.ap(KE, False, pe_old, qe)
        - (1 - theta) * VFE.C_coupl(wEC, pe_old, pc_old, qe)
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
    )

    # DISCONTINUOUS GALERKIN TERMS
    if param["Spatial Discretization"]["Method"] == "DG-FEM":

        # Definition of the stabilization parameters
        etaC = 10
        etaA = 10
        etaV = 10
        etaE = 10
        etaU = 10

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
            + theta * VFE.ap_DG(KC, False, pc, qc, etaC, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pc,
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                False,
                etaC,
                degP,
                h,
                n,
            )
            + VFE.b_DG(alphaC, u, n, qc, dt)
            + VFE.b_DG_D(
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
            - (1 - theta) * VFE.ap_DG(KC, False, pc_old, qc, etaC, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pc_old,
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                False,
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
                False,
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
                False,
                etaC,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + VFE.b_DG_D(
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
            + VFE.Fb_DG_D(
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
            - VFE.Fb_DG_D(
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
            + theta * VFE.ap_DG(KA, K, pa, qa, etaA, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pa,
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K,
                etaA,
                degP,
                h,
                n,
            )
            + VFE.b_DG(alphaA, u, n, qa, dt)
            + VFE.b_DG_D(
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
            - (1 - theta) * VFE.ap_DG(KA, K, pa_old, qa, etaA, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pa_old,
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K,
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
                K,
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
                K,
                etaA,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + VFE.b_DG_D(
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
            + VFE.Fb_DG_D(
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
            - VFE.Fb_DG_D(
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
            + theta * VFE.ap_DG(KV, K, pv, qv, etaV, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pv,
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K,
                etaV,
                degP,
                h,
                n,
            )
            + VFE.b_DG(alphaV, u, n, qv, dt)
            + VFE.b_DG_D(
                "Venous Network", alphaV, qv, u, ds_vent, ds_skull, n, dt, param
            )
        )

        LV = (
            LV
            - (1 - theta) * VFE.ap_DG(KV, K, pv_old, qv, etaV, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pv_old,
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K,
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
                K,
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
                K,
                etaV,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + VFE.b_DG_D(
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
            + VFE.Fb_DG_D(
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
            - VFE.Fb_DG_D(
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
            + theta * VFE.ap_DG(KE, False, pe, qe, etaE, degP, h, n)
            + theta
            * VFE.aP_DG_D(
                pe,
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                KE,
                False,
                etaE,
                degP,
                h,
                n,
            )
            + VFE.b_DG(alphaE, u, n, qe, dt)
            + VFE.b_DG_D(
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
            - (1 - theta) * VFE.ap_DG(KE, K, pe_old, qe, etaE, degP, h, n)
            - (1 - theta)
            * VFE.aP_DG_D(
                pe_old,
                qe,
                "CSF-ISF Network",
                param,
                ds_vent,
                ds_skull,
                KE,
                False,
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
                False,
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
                False,
                etaE,
                degP,
                h,
                n,
                time_prev,
                period,
            )
            + VFE.b_DG_D(
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
            + VFE.Fb_DG_D(
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
            - VFE.Fb_DG_D(
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


##############################################################################################################################
# 				Constructor of Initial Condition from file or constant values 				     #
##############################################################################################################################


def InitialConditionConstructor(
    param, mesh, X, x, pa_old, pc_old, pv_old, pe_old, u_old
):

    # Solution Initialization
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
        x0 = Constant((pc0, pa0, pv0, pe0, u0[0], u0[1]))

    else:
        x0 = Constant((pc0, pa0, pv0, pe0, u0[0], u0[1], u0[2]))

    x = interpolate(x0, X)

    # Initial Condition Importing from Files
    if (
        param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pa_old = HDF5.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pc_old,
            param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
                "Name of IC Function in File"
            ],
        )
        assign(x.sub(0), pc_old)

    if (
        param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pa_old = HDF5.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pa_old,
            param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
                "Name of IC Function in File"
            ],
        )
        assign(x.sub(1), pa_old)

    if (
        param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pa_old = HDF5.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["Venous Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pv_old,
            param["Model Parameters"]["Fluid Networks"]["Venous Network"][
                "Name of IC Function in File"
            ],
        )
        assign(x.sub(2), pv_old)

    if (
        param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
            "Initial Condition from File (Pressure)"
        ]
        == "Yes"
    ):

        pa_old = HDF5.ImportICfromFile(
            param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
                "Initial Condition File Name"
            ],
            mesh,
            pe_old,
            param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
                "Name of IC Function in File"
            ],
        )
        assign(x.sub(3), pe_old)

    if (
        param["Model Parameters"]["Elastic Solid Tissue"][
            "Initial Condition from File (Displacement)"
        ]
        == "Yes"
    ):

        pa_old = HDF5.ImportICfromFile(
            param["Model Parameters"]["Elastic Solid Tissue"][
                "Initial Condition File Name"
            ],
            mesh,
            u_old,
            param["Model Parameters"]["Elastic Solid Tissue"][
                "Name of IC Function in File"
            ],
        )
        assign(x.sub(4), u_old)

    return x


######################################################################
# 				Main 				     #
######################################################################

if __name__ == "__main__":

    Common_main.main(sys.argv[1:], cwd, "/../../physics/BrainMPET")

    if MPI.comm_world.Get_rank() == 0:
        print("Problem Solved!")

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
import scipy.io

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/../../utilities")

import ParameterFile_handler as prmh
import BCs_handler
import Mesh_handler
import XDMF_handler
import TensorialDiffusion_handler
import LinearSolver_handler
import HDF5_handler
import Common_main
import Formulation_Elements as VFE
import ErrorComputation_handler
import Preconditioners_handler

# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

    errors = pd.DataFrame(
        columns=[
            "Error_L2_pC",
            "Error_L2_pA",
            "Error_L2_pV",
            "Error_H1_pC",
            "Error_H1_pA",
            "Error_H1_pV",
        ]
    )

    for it in range(0, conv):
        # Convergence iteration solver
        errors = problemsolver(filename, it, True, errors)

    errors.to_csv("solution/Errors.csv")


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

    # Pressures Functional Spaces
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

    # Mixed FEM Spaces
    element = MixedElement([Pp, Pp, Pp])

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

    n = FacetNormal(mesh)

    # Solution functions definition
    x = Function(X)
    pc, pa, pv = TrialFunctions(X)

    # Test functions definition
    qc, qa, qv = TestFunctions(X)

    # Measure with boundary distinction definition
    ds_vent, ds_skull = BCs_handler.MeasuresDefinition(param, mesh, BoundaryID)

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

    OutputFNh5 = OutputFN + "_DarcyIC.h5"

    # Variational Formulation Construction
    a, L = VariationalFormulation(
        param, pc, pa, pv, qc, qa, qv, n, K, mesh, ds_vent, ds_skull
    )

    # Problem Solver Definition
    A = assemble(a)
    b = assemble(L)

    # Dirichlet Boundary Conditions vector construction and imposition
    if param["Spatial Discretization"]["Method"] == "CG-FEM":
        bc = VFE.DirichletBoundary(X, param, BoundaryID, mesh)

        [bci.apply(A) for bci in bc]
        [bci.apply(b) for bci in bc]

    # Preconditioner Construction
    if param["Linear Solver"]["User-Defined Preconditioner"] == "Yes":

        aPrec = PreconditionerVariationalFormulation(
            param, pc, pa, pv, qc, qa, qv, K, n, mesh, ds_vent, ds_skull
        )
        P = assemble(aPrec)
        [bci.apply(P) for bci in bc]

    else:
        P = False

    # Problem Resolution
    x = LinearSolver_handler.LinearSolver(A, x, b, param, P)

    # Save the solution
    XDMF_handler.BloodDarcySolutionSave(
        OutputFN, x, param["Scaling Parameters"]["Characteristic Pressure"]
    )

    HDF5_handler.SaveDarcySolution(
        OutputFNh5,
        mesh,
        x,
        param["Scaling Parameters"]["Characteristic Pressure"],
    )

    # Computing the errors
    if conv:
        errors = ErrorComputation_handler.Darcy_Errors(
            param, x, errors, mesh, iteration
        )

        if MPI.comm_world.Get_rank() == 0:
            print(errors)

    return errors


# VARIATIONAL FORMULATION DEFINITION
def PreconditionerVariationalFormulation(
    param, pc, pa, pv, qc, qa, qv, K, n, mesh, ds_vent, ds_skull
):

    # Preconditioning Parameters
    Ktilde, Wtilde = Preconditioners_handler.BloodDarcyPreconditioner(param)

    # CAPILLARY PRESSURE BALANCE EQUATION
    aPrec_C = VFE.ap(Constant(Ktilde[0]), False, pc, qc) + (
        Constant(Wtilde[0]) * pc * qc * dx
    )

    # ARTERIAL PRESSURE BALANCE EQUATION
    aPrec_A = VFE.ap(Constant(Ktilde[1]), False, pa, qa) + (
        Constant(Wtilde[1]) * pa * qa * dx
    )

    # ARTERIAL PRESSURE BALANCE EQUATION
    aPrec_V = VFE.ap(Constant(Ktilde[2]), False, pv, qv) + (
        Constant(Wtilde[2]) * pv * qv * dx
    )

    aPrec = aPrec_C + aPrec_A + aPrec_V

    return aPrec


# VARIATIONAL FORMULATION DEFINITION
def VariationalFormulation(
    param, pc, pa, pv, qc, qa, qv, n, K, mesh, ds_vent, ds_skull
):

    deg = param["Spatial Discretization"]["Polynomial Degree for Pressure"]
    h = CellDiameter(mesh)

    # Scaling Parameters
    U = param["Scaling Parameters"]["Characteristic Length"]
    P = param["Scaling Parameters"]["Characteristic Pressure"]
    tau = param["Scaling Parameters"]["Characteristic Time"]

    # Coupling Parameters Extraction
    wAC = Constant(
        tau
        * P
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Arterial-Capillary Coupling Parameter"
        ]
    )
    wAV = Constant(
        tau
        * P
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Arterial-Venous Coupling Parameter"
        ]
    )
    wVC = Constant(
        tau
        * P
        * param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
            "Venous-Capillary Coupling Parameter"
        ]
    )

    # Permeability Parameters Extraction
    KC = Constant(
        (tau * P / (U * U))
        * param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
            "Permeability"
        ]
    )
    KA = Constant(
        (tau * P / (U * U))
        * param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
            "Permeability"
        ]
    )
    KV = Constant(
        (tau * P / (U * U))
        * param["Model Parameters"]["Fluid Networks"]["Venous Network"][
            "Permeability"
        ]
    )

    # CAPILLARY PRESSURE BALANCE EQUATION
    aC = (
        VFE.ap(KC, K_Ct, pc, qc)
        + VFE.C_coupl(wAC, pc, pa, qc)
        + VFE.C_coupl(wVC, pc, pv, qc)
    )
    LC = VFE.F("Capillary", qc, param) + VFE.F_N(
        qc, "Capillary Network", param, ds_vent, ds_skull
    )

    # ARTERIAL PRESSURE BALANCE EQUATION
    aA = (
        VFE.ap(KA, K_At, pa, qa)
        + VFE.C_coupl(wAC, pa, pc, qa)
        + VFE.C_coupl(wAV, pa, pv, qa)
    )
    LA = VFE.F("Arterial", qc, param) + VFE.F_N(
        qa, "Arterial Network", param, ds_vent, ds_skull
    )

    # ARTERIAL PRESSURE BALANCE EQUATION
    aV = (
        VFE.ap(KV, K_Vt, pv, qv)
        + VFE.C_coupl(wVC, pv, pc, qv)
        + VFE.C_coupl(wAV, pv, pa, qv)
    )
    LV = VFE.F("Venous", qc, param) + VFE.F_N(
        qv, "Venous Network", param, ds_vent, ds_skull
    )

    # DISCONTINUOUS GALERKIN TERMS
    if param["Spatial Discretization"]["Method"] == "DG-FEM":

        # Stabilization Parameters
        etaC = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Capillary Pressure"
        ]
        etaA = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Arterial Pressure"
        ]
        etaV = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter for Venous Pressure"
        ]

        # CAPILLARY PRESSURE BALANCE EQUATION
        aC = (
            aC
            + VFE.ap_DG(KC, K_Ct, pc, qc, etaC, deg, h, n)
            + VFE.aP_DG_D(
                pc,
                qc,
                "Capillary Network",
                param,
                ds_vent,
                ds_skull,
                KC,
                K_Ct,
                etaC,
                deg,
                h,
                n,
            )
        )
        LC = LC + VFE.F_DG_D(
            qc,
            "Capillary Network",
            param,
            ds_vent,
            ds_skull,
            KC,
            K_Ct,
            etaC,
            deg,
            h,
            n,
        )

        # ARTERIAL PRESSURE BALANCE EQUATION
        aA = (
            aA
            + VFE.ap_DG(KA, K_At, pa, qa, etaA, deg, h, n)
            + VFE.aP_DG_D(
                pa,
                qa,
                "Arterial Network",
                param,
                ds_vent,
                ds_skull,
                KA,
                K_At,
                etaA,
                deg,
                h,
                n,
            )
        )
        LA = LA + VFE.F_DG_D(
            qa,
            "Arterial Network",
            param,
            ds_vent,
            ds_skull,
            KA,
            K_At,
            etaA,
            deg,
            h,
            n,
        )

        # VENOUS PRESSURE BALANCE EQUATION
        aV = (
            aV
            + VFE.ap_DG(KV, K_Vt, pv, qv, etaV, deg, h, n)
            + VFE.aP_DG_D(
                pv,
                qv,
                "Venous Network",
                param,
                ds_vent,
                ds_skull,
                KV,
                K_Vt,
                etaV,
                deg,
                h,
                n,
            )
        )
        LV = LV + VFE.F_DG_D(
            qv,
            "Venous Network",
            param,
            ds_vent,
            ds_skull,
            KV,
            K_Vt,
            etaV,
            deg,
            h,
            n,
        )

        # FINAL RHS AND LHS TERMS CONSTRUCTION
    a = aA + aC + aV
    L = LA + LC + LV

    return a, L


# MAIN SOLVER
if __name__ == "__main__":

    Common_main.main(sys.argv[1:], cwd, "/../../physics/BrainMPET")

    if MPI.comm_world.Get_rank() == 0:
        print("Problem Solved!")

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
import DarcyIC_handler
import LinearSolver_handler
import HDF5_handler
import Common_main
import Formulation_Elements as VFE


# Pressures Diffusion Term
def ap(Kval, K, p, q):

    if K == False:
        return Kval * dot(grad(p), grad(q)) * dx

    else:
        return Kval * dot(dot(K, grad(p)), grad(q)) * dx


# Pressures Diffusion Term
def ap_DG(Kval, K, p, q, eta, deg, h, n):

    h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))

    if K == False:
        return (
            (Kval * eta * deg * deg / h_avg * dot(jump(p, n), jump(q, n)) * dS)
            - (Kval * dot(avg(grad(p)), jump(q, n)) * dS)
            - (Kval * dot(avg(grad(q)), jump(p, n)) * dS)
        )

    else:
        return (
            (Kval * eta * deg * deg / h_avg * dot(jump(p, n), jump(q, n)) * dS)
            - (Kval * dot(avg(dot(K, grad(p))), jump(q, n)) * dS)
            - (Kval * dot(avg(dot(K, grad(q))), jump(p, n)) * dS)
        )


# Mass Forcing Term
def F(network, q, param):

    tau = param["Scaling Parameters"]["Characteristic Time"]

    # Extraction of forcing term
    forcing = network + " Pressure Forcing Term"
    f = Expression(
        param["Model Parameters"]["Forcing Terms"][forcing[:]],
        degree=int(
            param["Spatial Discretization"]["Polynomial Degree for Pressure"]
        ),
    )

    return tau * f * q * dx


# Neumann Boundary Term
def F_N(q, network, param, ds_vent, ds_skull, time=0, period=1):

    U = param["Scaling Parameters"]["Characteristic Length"]
    tau = param["Scaling Parameters"]["Characteristic Time"]

    L_BCs = Constant("0") * q * ds

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles BCs"
        ]
        == "Neumann"
    ):

        BCsType = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Input for Ventricles BCs"
        ]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles Neumann BCs Value (Flux)"
        ]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            network[:]
        ]["File Column Name Ventricles BCs"]

        gCv = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        L_BCs = L_BCs + ((tau / (U * U)) * gCv * q * ds_vent)

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]]["Skull BCs"]
        == "Neumann"
    ):

        BCsType = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Input for Skull BCs"
        ]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Skull Neumann BCs Value (Flux)"
        ]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            network[:]
        ]["File Column Name Skull BCs"]

        gCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        L_BCs = L_BCs + ((tau / (U * U)) * gCs * q * ds_skull)

    return L_BCs


# Dirichlet Boundary Conditions for DG problem
def F_DG_D(
    q,
    network,
    param,
    ds_vent,
    ds_skull,
    Kval,
    K,
    eta,
    deg,
    h,
    n,
    time=0,
    period=1,
):

    L_BCs = Constant("0") * q * ds

    # Scaling BCs
    P = param["Scaling Parameters"]["Characteristic Pressure"]

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):

        BCsType = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Input for Ventricles BCs"
        ]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles Dirichlet BCs Value (Pressure)"
        ]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            network[:]
        ]["File Column Name Ventricles BCs"]

        p_vent = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        if K == False:
            L_BCs = (
                L_BCs
                + (Kval * eta * deg * deg / h * p_vent / P * q * ds_vent)
                - (Kval * inner(grad(q), n) * p_vent / P * ds_vent)
            )

        else:
            L_BCs = (
                L_BCs
                + (Kval * eta * deg * deg / h * p_vent / P * q * ds_vent)
                - (Kval * inner(dot(K, grad(q)), n) * p_vent / P * ds_vent)
            )

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]]["Skull BCs"]
        == "Dirichlet"
    ):

        BCsType = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Input for Skull BCs"
        ]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Skull Dirichlet BCs Value (Pressure)"
        ]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            network[:]
        ]["File Column Name Skull BCs"]

        p_skull = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        if K == False:
            L_BCs = (
                L_BCs
                + (Kval * eta * deg * deg / h * p_skull / P * q * ds_skull)
                - (Kval * inner(grad(q), n) * p_skull / P * ds_skull)
            )

        else:
            L_BCs = (
                L_BCs
                + (Kval * eta * deg * deg / h * p_skull / P * q * ds_skull)
                - (Kval * inner(dot(K, grad(q)), n) * p_skull / P * ds_skull)
            )

    return L_BCs


# Dirichlet Boundary Conditions for DG problem
def aP_DG_D(
    p,
    q,
    network,
    param,
    ds_vent,
    ds_skull,
    Kval,
    K,
    eta,
    deg,
    h,
    n,
    time=0,
    period=1,
):

    ap_DG_val = Constant("0") * p * q * ds

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):
        if K == False:
            ap_DG_val = (
                ap_DG_val
                - (Kval * inner(grad(p), n) * q * ds_vent)
                - (Kval * inner(grad(q), n) * p * ds_vent)
                + (Kval * eta * deg * deg / h * p * q * ds_vent)
            )

        else:
            ap_DG_val = (
                ap_DG_val
                - (Kval * inner(dot(K, grad(p)), n) * q * ds_vent)
                - (Kval * inner(dot(K, grad(q)), n) * p * ds_vent)
                + (Kval * eta * deg * deg / h * p * q * ds_vent)
            )

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]]["Skull BCs"]
        == "Dirichlet"
    ):

        if K == False:
            ap_DG_val = (
                ap_DG_val
                - (Kval * inner(grad(p), n) * q * ds_skull)
                - (Kval * inner(grad(q), n) * p * ds_skull)
                + (Kval * eta * deg * deg / h * p * q * ds_skull)
            )

        else:
            ap_DG_val = (
                ap_DG_val
                - (Kval * inner(dot(K, grad(p)), n) * q * ds_skull)
                - (Kval * inner(dot(K, grad(q)), n) * p * ds_skull)
                + (Kval * eta * deg * deg / h * p * q * ds_skull)
            )

    return ap_DG_val


# Pressure Coupling
def C_coupl(w12, p1, p2, q):
    return w12 * (p1 - p2) * q * dx


###########################################
# DIRICHLET BOUNDARY CONDITION FOR CG-FEM #
###########################################
def DirichletBoundary(X, param, BoundaryID, mesh):

    # Vector initialization
    bc = []

    # Skull Dirichlet BCs Imposition
    period = param["Temporal Discretization"]["Problem Periodicity"]

    if (
        param["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
            "Skull BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "Capillary Network"
        ]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "Capillary Network"
        ]["Skull Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "Capillary Network"
        ]["File Column Name Skull BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, 0, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(0), BCs, BoundaryID, 1))

    if (
        param["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
            "Skull BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "Arterial Network"
        ]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "Arterial Network"
        ]["Skull Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "Arterial Network"
        ]["File Column Name Skull BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, 0, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(1), BCs, BoundaryID, 1))

    if (
        param["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
            "Skull BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "Venous Network"
        ]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "Venous Network"
        ]["Skull Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "Venous Network"
        ]["File Column Name Skull BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, 0, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(2), BCs, BoundaryID, 1))

    # Ventricles Dirichlet BCs Imposition

    if (
        param["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "Capillary Network"
        ]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "Capillary Network"
        ]["Ventricles Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "Capillary Network"
        ]["File Column Name Ventricles BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, 0, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(0), BCs, BoundaryID, 2))

    if (
        param["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "Arterial Network"
        ]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "Arterial Network"
        ]["Ventricles Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "Arterial Network"
        ]["File Column Name Ventricles BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, 0, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(1), BCs, BoundaryID, 2))

    if (
        param["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "Venous Network"
        ]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "Venous Network"
        ]["Ventricles Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "Venous Network"
        ]["File Column Name Ventricles BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, 0, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(2), BCs, BoundaryID, 2))

    # End of the procedure

    return bc

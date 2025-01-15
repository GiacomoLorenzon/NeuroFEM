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
import LinearSolver_handler
import HDF5_handler
import Common_main
import Formulation_Elements as VFE

# L2 dot product
def dot_L2(K, c, v):
    return K * c * v * dx


# Proteins Diffusion Term
def ac(d_ext, d_axn, K, c, v):

    if K == False:
        return d_ext * dot(grad(c), grad(v)) * dx

    else:
        return (d_ext * dot(grad(c), grad(v)) * dx) + (
            d_axn * dot(dot(K, grad(c)), grad(v)) * dx
        )


# Forcing Term
def F(v, time, param):

    # Extraction of forcing term
    f = Expression(param["Model Parameters"]["Forcing Term"], degree=6, t=time)

    return f * v * dx


# Proteins Diffusion Term
def ac_DG(d_ext, d_axn, K, c, v, eta, deg, h, n):

    h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))

    if K == False:
        return (
            (eta * deg * deg / h_avg * dot(jump(c, n), jump(v, n)) * dS)
            - (d_ext * dot(avg(grad(c)), jump(v, n)) * dS)
            - (d_ext * dot(avg(grad(v)), jump(c, n)) * dS)
        )

    else:
        return (
            (eta * deg * deg / h_avg * dot(jump(c, n), jump(v, n)) * dS)
            - (d_ext * dot(avg(grad(c)), jump(v, n)) * dS)
            - (d_ext * dot(avg(grad(v)), jump(c, n)) * dS)
            - (d_axn * dot(avg(dot(K, grad(c))), jump(v, n)) * dS)
            - (d_axn * dot(avg(dot(K, grad(v))), jump(c, n)) * dS)
        )


# Dirichlet Boundary Conditions for DG problem
def ac_DG_D(c, v, param, ds_vent, ds_skull, d_ext, d_axn, K, eta, deg, h, n):

    ac_DG_val = Constant("0") * c * v * ds

    if param["Boundary Conditions"]["Ventricles BCs"] == "Dirichlet":
        if K == False:
            ac_DG_val = (
                ac_DG_val
                + (eta * deg * deg / h * c * v * ds_vent)
                - (d_ext * v * dot(grad(c), n) * ds_vent)
                - (d_ext * c * dot(grad(v), n) * ds_vent)
            )

        else:
            ac_DG_val = (
                ac_DG_val
                + (eta * deg * deg / h * c * v * ds_vent)
                - (d_ext * v * dot(grad(c), n) * ds_vent)
                - (d_ext * c * dot(grad(v), n) * ds_vent)
                - (d_axn * dot(dot(K, grad(c)), n) * v * ds_vent)
                - (d_axn * c * dot(dot(K, grad(v)), n) * ds_vent)
            )

    if param["Boundary Conditions"]["Skull BCs"] == "Dirichlet":

        if K == False:
            ac_DG_val = (
                ac_DG_val
                + (eta * deg * deg / h * c * v * ds_skull)
                - (d_ext * v * dot(grad(c), n) * ds_skull)
                - (d_ext * c * dot(grad(v), n) * ds_skull)
            )

        else:
            ac_DG_val = (
                ac_DG_val
                + (eta * deg * deg / h * c * v * ds_skull)
                - (d_ext * v * dot(grad(c), n) * ds_skull)
                - (d_ext * c * dot(grad(v), n) * ds_skull)
                - (d_axn * dot(dot(K, grad(c)), n) * v * ds_skull)
                - (d_axn * c * dot(dot(K, grad(v)), n) * ds_skull)
            )

    return ac_DG_val


# Dirichlet Boundary Conditions for DG problem
def F_DG_D(
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
    time=0,
    period=1,
):

    L_BCs = Constant("0") * v * ds

    if param["Boundary Conditions"]["Ventricles BCs"] == "Dirichlet":

        BCsType = param["Boundary Conditions"]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"][
            "Ventricles Dirichlet BCs Value"
        ]
        BCsColumnName = param["Boundary Conditions"][
            "File Column Name Ventricles BCs"
        ]

        c_vent = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        if K == False:
            L_BCs = (
                L_BCs
                + (eta * deg * deg / h * c_vent * v * ds_vent)
                - (d_ext * c_vent * dot(grad(v), n) * ds_vent)
            )

        else:
            L_BCs = (
                L_BCs
                + (eta * deg * deg / h * c_vent * v * ds_vent)
                - (d_ext * c_vent * dot(grad(v), n) * ds_vent)
                - (d_axn * c_vent * dot(dot(K, grad(v)), n) * ds_vent)
            )

    if param["Boundary Conditions"]["Skull BCs"] == "Dirichlet":

        BCsType = param["Boundary Conditions"]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Skull Dirichlet BCs Value"]
        BCsColumnName = param["Boundary Conditions"][
            "File Column Name Skull BCs"
        ]

        c_skull = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        if K == False:
            L_BCs = (
                L_BCs
                + (eta * deg * deg / h * c_skull * v * ds_skull)
                - (d_ext * c_skull * dot(grad(v), n) * ds_skull)
            )

        else:
            L_BCs = (
                L_BCs
                + (eta * deg * deg / h * c_skull * v * ds_skull)
                - (d_ext * c_skull * dot(grad(v), n) * ds_skull)
                - (d_axn * c_skull * dot(dot(K, grad(v)), n) * ds_skull)
            )

    return L_BCs


# Neumann Boundary Term
def F_N(v, param, ds_vent, ds_skull, time=0, period=1):

    L_BCs = Constant("0") * v * ds

    if param["Boundary Conditions"]["Ventricles BCs"] == "Neumann":

        BCsType = param["Boundary Conditions"]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"]["Ventricles Neumann BCs Value"]
        BCsColumnName = param["Boundary Conditions"][
            "File Column Name Ventricles BCs"
        ]

        gCv = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        L_BCs = L_BCs + (gCv * v * ds_vent)

    if param["Boundary Conditions"]["Skull BCs"] == "Neumann":

        BCsType = param["Boundary Conditions"]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Skull Neumann BCs Value"]
        BCsColumnName = param["Boundary Conditions"][
            "File Column Name Skull BCs"
        ]

        gCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        L_BCs = L_BCs + (gCs * v * ds_skull)

    return L_BCs


###########################################
# DIRICHLET BOUNDARY CONDITION FOR CG-FEM #
###########################################
def DirichletBoundary(X, param, BoundaryID, time, mesh):

    # Vector initialization
    bc = []

    # Skull Dirichlet BCs Imposition
    period = param["Temporal Discretization"]["Problem Periodicity"]

    if param["Boundary Conditions"]["Skull BCs"] == "Dirichlet":

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Skull Dirichlet BCs Value"]
        BCsColumnName = param["Boundary Conditions"][
            "File Column Name Skull BCs"
        ]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X, BCs, BoundaryID, 1))

    # Ventricle Dirichlet BCs Imposition
    if param["Boundary Conditions"]["Ventricles BCs"] == "Dirichlet":

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"][
            "Ventricles Dirichlet BCs Value (Pressure)"
        ]
        BCsColumnName = param["Boundary Conditions"][
            "File Column Name Ventricles BCs"
        ]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(2), BCs, BoundaryID, 2))

    return bc

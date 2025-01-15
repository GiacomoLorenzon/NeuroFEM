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

# Tensor Jump Operator
def tensor_jump(u, n):
    return (outer(u("+"), n("+")) + outer(n("+"), u("+"))) / 2 + (
        outer(n("-"), u("-")) + outer(u("-"), n("-"))
    ) / 2


# L2 dot product
def dot_L2(c, u, v):
    return c * dot(u, v) * dx


# Momentum Forcing Term
def F_el(v, time, param, mesh):

    fx = param["Model Parameters"]["Forcing Terms"][
        "Momentum Body Forces (x-component)"
    ]
    fy = param["Model Parameters"]["Forcing Terms"][
        "Momentum Body Forces (y-component)"
    ]
    fz = param["Model Parameters"]["Forcing Terms"][
        "Momentum Body Forces (z-component)"
    ]

    if mesh.ufl_cell() == triangle:
        f = Expression((fx, fy), degree=6, t=time)

    else:
        f = Expression((fx, fy, fz), degree=6, t=time)

    return dot(f, v) * dx


# Elasticity Bilinear Form
def a_el(G, l, u, v):
    return (2 * G * inner(sym(grad(u)), sym(grad(v))) * dx) + (
        l * div(u) * div(v) * dx
    )


# Elasticity DG Bilinear Form
def a_el_DG(G, l, u, v, etaU, deg, h, n):

    h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))

    a_DG = (
        (
            (2 * l + 5 * G)
            * etaU
            * deg
            * deg
            / h_avg
            * inner(tensor_jump(u, n), tensor_jump(v, n))
            * dS
        )
        - (2 * G * inner(avg(sym(grad(u))), tensor_jump(v, n)) * dS)
        - (2 * G * inner(avg(sym(grad(v))), tensor_jump(u, n)) * dS)
        - (l * avg(div(u)) * jump(v, n) * dS)
        - (l * avg(div(v)) * jump(u, n) * dS)
    )

    return a_DG


# Elasticity DG Bilinear Form on Dirichlet Boundary
def a_el_DG_D(G, l, u, v, ds_skull, ds_vent, etaU, deg, h, n, param):

    a_el_DG_val = Constant("0") * dot(u, v) * ds

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Ventricles BCs"]
        == "Dirichlet"
    ):

        a_el_DG_val = (
            a_el_DG_val
            + (
                (2 * l + 5 * G)
                * etaU
                * deg
                * deg
                / h
                * inner(
                    (outer(u, n) + outer(n, u)) / 2,
                    (outer(v, n) + outer(n, v)) / 2,
                )
                * ds_vent
            )
            - (
                2
                * G
                * inner(sym(grad(u)), (outer(v, n) + outer(n, v)) / 2)
                * ds_vent
            )
            - (
                2
                * G
                * inner(sym(grad(v)), (outer(u, n) + outer(n, u)) / 2)
                * ds_vent
            )
            - (l * div(u) * dot(v, n) * ds_vent)
            - (l * div(v) * dot(u, n) * ds_vent)
        )

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Skull BCs"]
        == "Dirichlet"
    ):

        a_el_DG_val = (
            a_el_DG_val
            + (
                (2 * l + 5 * G)
                * etaU
                * deg
                * deg
                / h
                * inner(
                    (outer(u, n) + outer(n, u)) / 2,
                    (outer(v, n) + outer(n, v)) / 2,
                )
                * ds_skull
            )
            - (
                2
                * G
                * inner(sym(grad(u)), (outer(v, n) + outer(n, v)) / 2)
                * ds_skull
            )
            - (
                2
                * G
                * inner(sym(grad(v)), (outer(u, n) + outer(n, u)) / 2)
                * ds_skull
            )
            - (l * div(u) * dot(v, n) * ds_skull)
            - (l * div(v) * dot(u, n) * ds_skull)
        )

    return a_el_DG_val


def F_el_DG_D(
    G, l, u, v, ds_skull, ds_vent, etaU, deg, h, n, param, mesh, time, period
):

    U = param["Scaling Parameters"]["Characteristic Length"]

    L_el_DG_val = Constant("0") * div(v) * ds

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Ventricles BCs"]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Ventricles BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            uv = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            uv = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        L_el_DG_val = (
            L_el_DG_val
            + (
                (2 * l + 5 * G)
                * etaU
                * deg
                * deg
                / h
                * inner(
                    (outer(uv, n) + outer(n, uv)) / 2 / U,
                    (outer(v, n) + outer(n, v)) / 2,
                )
                * ds_vent
            )
            - (
                2
                * G
                * inner(sym(grad(v)), (outer(uv, n) + outer(n, uv)) / 2 / U)
                * ds_vent
            )
            - (l * div(v) * dot(uv, n) / U * ds_vent)
        )

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Skull BCs"]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Skull BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            us = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            us = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        L_el_DG_val = (
            L_el_DG_val
            + (
                (2 * l + 5 * G)
                * etaU
                * deg
                * deg
                / h
                * inner(
                    (outer(us, n) + outer(n, us)) / 2 / U,
                    (outer(v, n) + outer(n, v)) / 2,
                )
                * ds_skull
            )
            - (
                2
                * G
                * inner(sym(grad(v)), (outer(us, n) + outer(n, us)) / 2 / U)
                * ds_skull
            )
            - (l * div(v) * dot(us, n) / U * ds_skull)
        )

        return L_el_DG_val


# Neumann Boundary Term
def F_N_el(v, param, ds_vent, ds_skull, mesh, time=0, period=1):

    U = param["Scaling Parameters"]["Characteristic Length"]
    P = param["Scaling Parameters"]["Characteristic Pressure"]

    L_BCs = Constant("0") * div(v) * ds

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Ventricles BCs"]
        == "Neumann"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Ventricles BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Neumann BCs Value (Stress x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Neumann BCs Value (Stress y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Neumann BCs Value (Stress z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            g = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            g = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        L_BCs = L_BCs + ((1 / (P * U)) * dot(g, v) * ds_vent)

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Ventricles BCs"]
        == "Neumann with CSF Pressure"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Ventricles Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["File Column Name Ventricles BCs"]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            pE_Vent = BCs_handler.FindBoundaryConditionValue1D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            pE_Vent = BCs_handler.FindBoundaryConditionValue1D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        L_BCs = L_BCs - ((1 / P) * pE_Vent * dot(n, v) * ds_vent)

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Skull BCs"]
        == "Neumann"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Skull BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Neumann BCs Value (Stress x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Neumann BCs Value (Stress y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Neumann BCs Value (Stress z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            g = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            g = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        L_BCs = L_BCs + ((1 / (P * U)) * dot(g, v) * ds_skull)

    return L_BCs


# Pressures Diffusion Term
def ap(Kval, K, p, q):

    if K == False:
        return Kval * dot(grad(p), grad(q)) * dx

    else:
        return Kval * dot(dot(K, grad(p)), grad(q)) * dx


# Pressures Diffusion Term
def ap_DG(Kval, K, p, q, eta, deg, h, n):

    h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))
    KvalCorr = max(Kval, 1)
    if K == False:
        return (
            (
                KvalCorr
                * eta
                * deg
                * deg
                / h_avg
                * dot(jump(p, n), jump(q, n))
                * dS
            )
            - (Kval * dot(avg(grad(p)), jump(q, n)) * dS)
            - (Kval * dot(avg(grad(q)), jump(p, n)) * dS)
        )

    else:
        return (
            (
                KvalCorr
                * eta
                * deg
                * deg
                / h_avg
                * dot(jump(p, n), jump(q, n))
                * dS
            )
            - (Kval * dot(avg(dot(K, grad(p))), jump(q, n)) * dS)
            - (Kval * dot(avg(dot(K, grad(q))), jump(p, n)) * dS)
        )


# Mass Forcing Term
def F(network, q, time, param):

    # Extraction of forcing term
    forcing = network + " Pressure Forcing Term"
    f = Expression(
        param["Model Parameters"]["Forcing Terms"][forcing[:]], degree=6, t=time
    )

    return f * q * dx


# Discontinuous Galkerin Coupling Term
def b_DG(alpha, u, n, q, dt):
    return -alpha * inner(jump(u, n), avg(q)) / dt * dS


# Discontinuous Galkerin Coupling Term on Dirichlet Boundary
def b_DG_D(network, alpha, q, u, ds_vent, ds_skull, n, dt, param):

    b_BCs = Constant("0") * inner(u, n) * q * ds

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):
        b_BCs = b_BCs - (alpha * inner(u, n) * q / dt * ds_vent)

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]]["Skull BCs"]
        == "Dirichlet"
    ):
        b_BCs = b_BCs - (alpha * inner(u, n) * q / dt * ds_skull)

    return b_BCs


# Discontinuous Galkerin Coupling Term on Dirichlet Boundary
def Fbel_DG_D(v, network, param, ds_vent, ds_skull, alpha, n, time=0, period=1):

    P = param["Scaling Parameters"]["Characteristic Pressure"]

    L_BCs = Constant("0") * div(v) * ds

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

        L_BCs = L_BCs - (alpha * inner(v, n) * p_vent / P * ds_vent)

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

        L_BCs = L_BCs - (alpha * inner(v, n) * p_skull / P * ds_skull)

    return L_BCs


# Discontinuous Galkerin Coupling Term on Dirichlet Boundary
def Fb_DG_D(
    q, network, param, ds_vent, ds_skull, dt, alpha, n, time, period, mesh
):

    U = param["Scaling Parameters"]["Characteristic Length"]

    L_BCs = Constant("0") * q * ds

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Ventricles BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            uv = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            uv = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        L_BCs = L_BCs - (alpha * inner(uv, n) / U * q / dt * ds_vent)

    if (
        param["Boundary Conditions"]["Fluid Networks"][network[:]]["Skull BCs"]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Skull BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            us = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            us = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        L_BCs = L_BCs - (alpha * inner(us, n) / U * q / dt * ds_skull)

    return L_BCs


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

    P = param["Scaling Parameters"]["Characteristic Pressure"]

    L_BCs = Constant("0") * q * ds
    KvalCorr = max(Kval, 1)

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
                + (KvalCorr * eta * deg * deg / h * p_vent / P * q * ds_vent)
                - (Kval * inner(grad(q), n) * p_vent / P * ds_vent)
            )

        else:
            L_BCs = (
                L_BCs
                + (KvalCorr * eta * deg * deg / h * p_vent / P * q * ds_vent)
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
                + (KvalCorr * eta * deg * deg / h * p_skull / P * q * ds_skull)
                - (Kval * inner(grad(q), n) * p_skull / P * ds_skull)
            )

        else:
            L_BCs = (
                L_BCs
                + (KvalCorr * eta * deg * deg / h * p_skull / P * q * ds_skull)
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
    KvalCorr = max(Kval, 1)
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
                + (KvalCorr * eta * deg * deg / h * p * q * ds_vent)
            )

        else:
            ap_DG_val = (
                ap_DG_val
                - (Kval * inner(dot(K, grad(p)), n) * q * ds_vent)
                - (Kval * inner(dot(K, grad(q)), n) * p * ds_vent)
                + (KvalCorr * eta * deg * deg / h * p * q * ds_vent)
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
                + (KvalCorr * eta * deg * deg / h * p * q * ds_skull)
            )

        else:
            ap_DG_val = (
                ap_DG_val
                - (Kval * inner(dot(K, grad(p)), n) * q * ds_skull)
                - (Kval * inner(dot(K, grad(q)), n) * p * ds_skull)
                + (KvalCorr * eta * deg * deg / h * p * q * ds_skull)
            )

    return ap_DG_val


# Pressure Coupling
def C_coupl(w12, p1, p2, q):
    return w12 * (p1 - p2) * q * dx


# Pressure External Coupling
def C_ext_coupl(wE, p, q):
    return wE * p * q * dx


# Time Pressure Derivative
def tP_der(c, p, q, dt):
    return c * p * q / dt * dx


# Time Derivative of Biot's Coupling Term
def bp(alpha, u, q, dt):
    return alpha * div(u) * q / dt * dx


###########################################
# DIRICHLET BOUNDARY CONDITION FOR CG-FEM #
###########################################
def DirichletBoundary(X, param, BoundaryID, time, mesh):

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
            BCsType, BCsValue, BCsColumnName, time, period
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
            BCsType, BCsValue, BCsColumnName, time, period
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
            BCsType, BCsValue, BCsColumnName, time, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(2), BCs, BoundaryID, 1))

    if (
        param["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
            "Skull BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Input for Skull BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Skull Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["File Column Name Skull BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(3), BCs, BoundaryID, 1))

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Skull BCs"]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Skull BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Skull Dirichlet BCs Value (Displacement z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Skull BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            BCs = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            BCs = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(4), BCs, BoundaryID, 1))

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
            BCsType, BCsValue, BCsColumnName, time, period
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
            BCsType, BCsValue, BCsColumnName, time, period
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
            BCsType, BCsValue, BCsColumnName, time, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(2), BCs, BoundaryID, 2))

    if (
        param["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
            "Ventricles BCs"
        ]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Input for Ventricles BCs"]
        BCsValue = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["Ventricles Dirichlet BCs Value (Pressure)"]
        BCsColumnName = param["Boundary Conditions"]["Fluid Networks"][
            "CSF-ISF Network"
        ]["File Column Name Ventricles BCs"]

        BCs = BCs_handler.FindBoundaryConditionValue1D(
            BCsType, BCsValue, BCsColumnName, time, period
        )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(3), BCs, BoundaryID, 2))

    if (
        param["Boundary Conditions"]["Elastic Solid Tissue"]["Ventricles BCs"]
        == "Dirichlet"
    ):

        # Boundary Condition Extraction Value
        BCsType = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Input for Ventricles BCs"
        ]
        BCsValueX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement x-component)"
        ]
        BCsValueY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement y-component)"
        ]
        BCsValueZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "Ventricles Dirichlet BCs Value (Displacement z-component)"
        ]
        BCsColumnNameX = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (x-component)"
        ]
        BCsColumnNameY = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (y-component)"
        ]
        BCsColumnNameZ = param["Boundary Conditions"]["Elastic Solid Tissue"][
            "File Column Name Ventricles BCs (z-component)"
        ]

        # Control of problem dimensionality
        if mesh.ufl_cell() == triangle:
            BCs = BCs_handler.FindBoundaryConditionValue2D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsColumnNameX,
                BCsColumnNameY,
                time,
                period,
            )

        else:
            BCs = BCs_handler.FindBoundaryConditionValue3D(
                BCsType,
                BCsValueX,
                BCsValueY,
                BCsValueZ,
                BCsColumnNameX,
                BCsColumnNameY,
                BCsColumnNameZ,
                time,
                period,
            )

        # Boundary Condition Imposition
        bc.append(DirichletBC(X.sub(4), BCs, BoundaryID, 2))

    # End of the procedure

    return bc

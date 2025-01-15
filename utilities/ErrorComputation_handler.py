import dolfin
import numpy as np
import pandas as pd
from fenics import *

# TODO - make this file more general. apps/Heterodimer up to now has its own
# ErrorComputationHandler

############################################################################################################
# Control the input parameters about the type of mesh to be used and call the generation/import procedures #
############################################################################################################


def Darcy_Errors(param, x, errors, mesh, iteration):
    # Importing the exact solutions
    pc_ex = Expression(
        param["Convergence Test"]["Capillary Pressure Exact Solution"], degree=6
    )
    pa_ex = Expression(
        param["Convergence Test"]["Arterial Pressure Exact Solution"], degree=6
    )
    pv_ex = Expression(
        param["Convergence Test"]["Venous Pressure Exact Solution"], degree=6
    )

    # Importing the FEM solution
    pc, pa, pv = x.split(deepcopy=True)

    # Computing the errors
    Error_L2_pc = errornorm(
        pc_ex,
        pc,
        "L2",
        int(param["Spatial Discretization"]["Polynomial Degree"]) + 1,
        mesh,
    )
    Error_L2_pa = errornorm(
        pa_ex,
        pa,
        "L2",
        int(param["Spatial Discretization"]["Polynomial Degree"]) + 1,
        mesh,
    )
    Error_L2_pv = errornorm(
        pv_ex,
        pv,
        "L2",
        int(param["Spatial Discretization"]["Polynomial Degree"]) + 1,
        mesh,
    )
    Error_H1_pc = errornorm(
        pc_ex,
        pc,
        "H1",
        int(param["Spatial Discretization"]["Polynomial Degree"]) + 1,
        mesh,
    )
    Error_H1_pa = errornorm(
        pa_ex,
        pa,
        "H1",
        int(param["Spatial Discretization"]["Polynomial Degree"]) + 1,
        mesh,
    )
    Error_H1_pv = errornorm(
        pv_ex,
        pv,
        "H1",
        int(param["Spatial Discretization"]["Polynomial Degree"]) + 1,
        mesh,
    )

    errorsnew = pd.DataFrame(
        {
            "Error_L2_pC": Error_L2_pc,
            "Error_L2_pA": Error_L2_pa,
            "Error_L2_pV": Error_L2_pv,
            "Error_H1_pC": Error_H1_pc,
            "Error_H1_pA": Error_H1_pa,
            "Error_H1_pV": Error_H1_pv,
        },
        index=[iteration],
    )

    if iteration == 0:
        errors = errorsnew

    else:
        errors = pd.concat([errors, errorsnew])

    return errors


def MPET_Errors(param, x, errors, mesh, iteration, T, n):
    # Importing the exact solutions
    pc_ex = Expression(
        param["Convergence Test"]["Capillary Pressure Exact Solution"],
        degree=6,
        t=T,
    )
    pa_ex = Expression(
        param["Convergence Test"]["Arterial Pressure Exact Solution"],
        degree=6,
        t=T,
    )
    pe_ex = Expression(
        param["Convergence Test"]["CSF-ISF Pressure Exact Solution"],
        degree=6,
        t=T,
    )
    pv_ex = Expression(
        param["Convergence Test"]["Venous Pressure Exact Solution"],
        degree=6,
        t=T,
    )

    if mesh.topology().dim() == 2:
        u_ex = Expression(
            (
                param["Convergence Test"][
                    "Displacement Exact Solution (x-component)"
                ],
                param["Convergence Test"][
                    "Displacement Exact Solution (y-component)"
                ],
            ),
            degree=6,
            t=T,
        )

    else:
        u_ex = Expression(
            (
                param["Convergence Test"][
                    "Displacement Exact Solution (x-component)"
                ],
                param["Convergence Test"][
                    "Displacement Exact Solution (y-component)"
                ],
                param["Convergence Test"][
                    "Displacement Exact Solution (z-component)"
                ],
            ),
            degree=6,
            t=T,
        )

    # Importing the FEM solution
    pc, pa, pv, pe, u = x.split(deepcopy=True)

    # Computing the errors il L2-norm
    Error_L2_pc = errornorm(pc_ex, pc, "L2", 0, mesh)
    Error_L2_pa = errornorm(pa_ex, pa, "L2", 0, mesh)
    Error_L2_pv = errornorm(pv_ex, pv, "L2", 0, mesh)
    Error_L2_pe = errornorm(pe_ex, pe, "L2", 0, mesh)
    Error_L2_u = errornorm(u_ex, u, "L2", 0, mesh)

    # Computing the errors in H1-norm
    Error_H1_pc = errornorm(pc_ex, pc, "H10", 0, mesh)
    Error_H1_pa = errornorm(pa_ex, pa, "H10", 0, mesh)
    Error_H1_pv = errornorm(pv_ex, pv, "H10", 0, mesh)
    Error_H1_pe = errornorm(pe_ex, pe, "H10", 0, mesh)
    Error_H1_u = errornorm(u_ex, u, "H10", 0, mesh)

    if param["Spatial Discretization"]["Method"] == "DG-FEM":
        # Polynomial Degree of Approximation
        degP = param["Spatial Discretization"]["Polynomial Degree for Pressure"]
        degU = param["Spatial Discretization"][
            "Polynomial Degree for Displacement"
        ]

        # Penalty parameters
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

        # Cell diameter computation
        h = CellDiameter(mesh)
        h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))

        # Lamè parameters
        G = param["Model Parameters"]["Elastic Solid Tissue"][
            "First Lamé Parameter"
        ]
        l = param["Model Parameters"]["Elastic Solid Tissue"][
            "Second Lamé Parameter"
        ]

        # Permeability Constant
        KC = max(
            1,
            param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
                "Permeability"
            ],
        )
        KA = max(
            1,
            param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
                "Permeability"
            ],
        )
        KV = max(
            1,
            param["Model Parameters"]["Fluid Networks"]["Venous Network"][
                "Permeability"
            ],
        )
        KE = max(
            1,
            param["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
                "Permeability"
            ],
        )

        # Computing the errors in DG-norm
        I_pc = (KC * degP * degP * etaC / h * (pc - pc_ex) ** 2 * ds) + (
            KC
            * degP
            * degP
            * etaC
            / h_avg
            * dot(jump(pc - pc_ex, n), jump(pc - pc_ex, n))
            * dS
        )
        Error_H1_pc = sqrt(assemble(I_pc)) + Error_H1_pc

        I_pa = (KA * degP * degP * etaA / h * (pa - pa_ex) ** 2 * ds) + (
            KA
            * degP
            * degP
            * etaA
            / h_avg
            * dot(jump(pa - pa_ex, n), jump(pa - pa_ex, n))
            * dS
        )
        Error_H1_pa = sqrt(assemble(I_pa)) + Error_H1_pa

        I_pv = (KV * degP * degP * etaV / h * (pv - pv_ex) ** 2 * ds) + (
            KV
            * degP
            * degP
            * etaV
            / h_avg
            * dot(jump(pv - pv_ex, n), jump(pv - pv_ex, n))
            * dS
        )
        Error_H1_pv = sqrt(assemble(I_pv)) + Error_H1_pv

        I_pe = (KE * degP * degP * etaE / h * (pe - pe_ex) ** 2 * ds) + (
            KA
            * degP
            * degP
            * etaE
            / h_avg
            * dot(jump(pe - pe_ex, n), jump(pe - pe_ex, n))
            * dS
        )
        Error_H1_pe = sqrt(assemble(I_pe)) + Error_H1_pe

        I_u = (
            (5 * G + 2 * l)
            * degU
            * degU
            * etaU
            / h
            * dot((u - u_ex), (u - u_ex))
            * ds
        ) + (
            (5 * G + 2 * l)
            * degU
            * degU
            * etaU
            / h_avg
            * jump(u - u_ex, n)
            * jump(u - u_ex, n)
            * dS
        )
        Error_H1_u = sqrt(assemble(I_u)) + sqrt(5 * G + 2 * l) * Error_H1_u

    errorsnew = pd.DataFrame(
        {
            "Error_L2_pC": Error_L2_pc,
            "Error_L2_pA": Error_L2_pa,
            "Error_L2_pV": Error_L2_pv,
            "Error_L2_pE": Error_L2_pe,
            "Error_L2_u": Error_L2_u,
            "Error_DG_pC": Error_H1_pc,
            "Error_DG_pA": Error_H1_pa,
            "Error_DG_pV": Error_H1_pv,
            "Error_DG_pE": Error_H1_pe,
            "Error_DG_u": Error_H1_u,
        },
        index=[iteration],
    )

    if iteration == 0:
        errors = errorsnew

    else:
        errors = pd.concat([errors, errorsnew])

    return errors


def FK_Errors(param, c, errors, mesh, iteration, T, n):
    # Importing the exact solutions
    c_ex = Expression(
        param["Convergence Test"]["Exact Solution"], degree=6, t=T
    )

    if param["Spatial Discretization"]["Formulation"] == "Exponential":
        c.vector()[:] = np.exp(c.vector()[:])

    # Computing the errors il L2-norm
    Error_L2_c = errornorm(c_ex, c, "L2", 0, mesh)

    # Computing the errors in H1-norm
    Error_H1_c = errornorm(c_ex, c, "H10", 0, mesh)

    if param["Spatial Discretization"]["Method"] == "DG-FEM":
        # Polynomial Degree of Approximation and Penalty parameters
        deg = param["Spatial Discretization"]["Polynomial Degree"]
        eta = param["Spatial Discretization"]["Discontinuous Galerkin"][
            "Penalty Parameter"
        ]

        # Cell diameter computation
        h = CellDiameter(mesh)
        h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))

        # Computing the errors in DG-norm
        I_c = (deg * deg * eta / h * (c - c_ex) ** 2 * ds) + (
            deg
            * deg
            * eta
            / h_avg
            * dot(jump(c - c_ex, n), jump(c - c_ex, n))
            * dS
        )
        Error_H1_c = sqrt(assemble(I_c)) + Error_H1_c

    errorsnew = pd.DataFrame(
        {"Error_L2_c": Error_L2_c, "Error_DG_c": Error_H1_c}, index=[iteration]
    )

    if iteration == 0:
        errors = errorsnew

    else:
        errors = pd.concat([errors, errorsnew])

    return errors

import mshr
import dolfin
from fenics import *
import numpy as np


############################################################################################################
# 			  Construction of the parameters for the MPT preconditioning	 		   #
############################################################################################################


def BloodDarcyPreconditioner(param):
    # Coupling Parameters Extraction
    wAC = param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "Arterial-Capillary Coupling Parameter"
    ]
    wAV = param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "Arterial-Venous Coupling Parameter"
    ]
    wVC = param["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "Venous-Capillary Coupling Parameter"
    ]

    # Permeability Parameters Extraction
    KC = param["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Permeability"
    ]
    KA = param["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Permeability"
    ]
    KV = param["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Permeability"
    ]

    Kinv = np.array([[1 / KC, 0, 0], [0, 1 / KA, 0], [0, 0, 1 / KV]])
    K = np.array([[KC, 0, 0], [0, KA, 0], [0, 0, KV]])
    W = np.array(
        [
            [wVC + wAC, -wAC, -wVC],
            [-wAC, wAV + wAC, -wAV],
            [-wVC, -wAV, wAV + wVC],
        ]
    )

    C = np.matmul(Kinv, W)
    eval, evec = np.linalg.eig(C)

    Ktilde = np.diag(np.matmul(np.transpose(evec), np.matmul(K, evec)))
    Wtilde = np.diag(np.matmul(np.transpose(evec), np.matmul(W, evec)))

    return Ktilde, Wtilde

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
import ErrorComputation_handler
import Preconditioners_handler


# PROBLEM CONVERGENCE ITERATIONS
def problemconvergence(filename, conv):

    if conv > 0:
        print(
            "Convergence test not available for this solver! Problem resolution is starting"
        )

    problemsolver(filename)


# PROBLEM SOLVER
def problemsolver(filename, iteration=0):

    # Import the parameters given the filename
    param = prmh.readprmfile(filename)

    parameters["ghost_mode"] = "shared_facet"

    # Handling of the mesh
    mesh = Mesh_handler.MeshHandler(param, iteration)

    mesh.scale(1 / param["Scaling Parameters"]["Characteristic Length"])

    # Importing the Boundary Identifier
    BoundaryID = BCs_handler.ImportFaceFunctionFromFile(param, mesh)

    # Functional Space
    P = FiniteElement("CG", tetrahedron, 1)

    # Connecting the FEM element to the mesh discretization
    X = FunctionSpace(mesh, P)

    X1 = FunctionSpace(mesh, "DG", 0)
    X9 = TensorFunctionSpace(mesh, "DG", 0)

    # Trial functions definition
    t = TrialFunction(X)

    # Test functions definition
    v = TestFunction(X)

    # Dirichlet Boundary Condition Imposition
    bc = [
        DirichletBC(X, Constant("0"), BoundaryID, 1),
        DirichletBC(X, Constant("1"), BoundaryID, 2),
    ]

    # Variational Formulation Construction
    a = dot(grad(t), grad(v)) * dx
    L = Constant("0") * v * dx

    # Solution functions definition
    t = Function(X)

    # Problem Resolution
    solve(a == L, t, bc)

    Dxt = Function(X1)
    Dyt = Function(X1)
    Dzt = Function(X1)
    D = Function(X9)

    Dxt = project(
        t.dx(0) / (sqrt(t.dx(0) ** 2 + t.dx(1) ** 2 + t.dx(2) ** 2) + 1e-30),
        X1,
        solver_type="gmres",
    )
    Dyt = project(
        t.dx(1) / (sqrt(t.dx(0) ** 2 + t.dx(1) ** 2 + t.dx(2) ** 2) + 1e-30),
        X1,
        solver_type="gmres",
    )
    Dzt = project(
        t.dx(2) / (sqrt(t.dx(0) ** 2 + t.dx(1) ** 2 + t.dx(2) ** 2) + 1e-30),
        X1,
        solver_type="gmres",
    )

    print("Construction of Permeability Tensor is started!")

    dapp = D.vector()[:]
    coord = X1.tabulate_dof_coordinates()

    for ii in range(len(Dxt.vector()[:])):

        if (
            Dxt.vector()[:][ii] ** 2
            + Dyt.vector()[:][ii] ** 2
            + Dzt.vector()[:][ii] ** 2
        ) < 0.0001:
            dist = np.linalg.norm(coord - coord[ii], axis=1)
            dist = dist * (dist != 0) + (dist.max()) * (dist == 0)
            min_dist = dist.min()
            jj = np.where(dist == min_dist)[0][0]

            while (
                Dxt.vector()[:][jj] ** 2
                + Dyt.vector()[:][jj] ** 2
                + Dzt.vector()[:][jj] ** 2
            ) < 0.0001:
                dist[jj] = dist.max()
                min_dist = dist.min()
                jj = np.where(dist == min_dist)[0][0]

        else:
            jj = ii

        dapp[9 * ii] = Dxt.vector()[:][jj] * Dxt.vector()[:][jj]
        dapp[9 * ii + 1] = Dxt.vector()[:][jj] * Dyt.vector()[:][jj]
        dapp[9 * ii + 2] = Dxt.vector()[:][jj] * Dzt.vector()[:][jj]
        dapp[9 * ii + 3] = dapp[9 * ii + 1]
        dapp[9 * ii + 4] = Dyt.vector()[:][jj] * Dyt.vector()[:][jj]
        dapp[9 * ii + 5] = Dyt.vector()[:][jj] * Dzt.vector()[:][jj]
        dapp[9 * ii + 6] = dapp[9 * ii + 2]
        dapp[9 * ii + 7] = dapp[9 * ii + 5]
        dapp[9 * ii + 8] = Dzt.vector()[:][jj] * Dzt.vector()[:][jj]

        if ii % 10000 == 0:
            advancement = (
                "- Status (Rank "
                + str(MPI.comm_world.Get_rank())
                + "): "
                + str(math.trunc(ii / len(Dxt.vector()[:]) * 10000) / 100)
                + " %"
            )
            print(advancement)

    D.vector()[:] = dapp

    print("Construction of Permeability Tensor is finished!")

    XDMF_handler.AVDirSolutionSave(param["Output"]["Output XDMF File Name"], D)

    HDF5_handler.SavePermeabilityTensor(
        param["Output"]["Output h5 File Name"], mesh, BoundaryID, D, "K_AV"
    )


# MAIN SOLVER
if __name__ == "__main__":

    Common_main.main(sys.argv[1:], cwd, "/../../physics/BrainAVDir")

    if MPI.comm_world.Get_rank() == 0:
        print("Problem Solved!")

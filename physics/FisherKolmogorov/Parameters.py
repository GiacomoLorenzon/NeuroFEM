import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pyrameters import PRM

cwd = os.getcwd()
sys.path.append(cwd + "/../../utilities/")

import ParameterFile_handler as pfh


#################################################################################
# 	   Generator of parameter file for the Brain MPET problem 		#
#################################################################################


def Createprm(filename, conv=False):
    # Generate default parameters
    prm = parametersBrainFK(conv)

    if pfh.generateprmfile(prm, filename) == 1:
        Lines = pfh.readlinesprmfile(filename)
        commentedlines = commentingprmBrainFK(Lines)

        if pfh.writelinesprmfile(commentedlines, filename) == 1:
            print("Parameter file with default values generated")

        else:
            print("Error in parameter file generation!")

    else:
        print("Error in parameter file generation!")


#################################################################################
# 	Definition of the default parameters for the Brain MPET problem 	#
#################################################################################


def parametersBrainFK(conv):
    # Create the prm file object
    prm = PRM()

    # PARAMETERS DEFINITION

    # SUBSECTION OF SPATIAL DISCRETIZATION

    prm.add_subsection("Spatial Discretization")

    prm["Spatial Discretization"]["Method"] = "DG-FEM"
    prm["Spatial Discretization"]["Formulation"] = "Classical"
    prm["Spatial Discretization"]["Polynomial Degree"] = 1

    # SUBSECTION OF DISCONTINUOUS GALERKIN METHOD

    prm["Spatial Discretization"].add_subsection("Discontinuous Galerkin")

    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter"
    ] = 10

    # SUBSECTION SCALING PARAMETERS FOR DIMENSIONLESS SOLVER
    prm.add_subsection("Scaling Parameters")

    prm["Scaling Parameters"]["Characteristic Length"] = 1
    prm["Scaling Parameters"]["Characteristic Time"] = 1

    # SUBSECTION OF MESH INFORMATION

    prm.add_subsection("Domain Definition")

    prm["Domain Definition"]["Type of Mesh"] = "File"
    prm["Domain Definition"]["ID for Ventricles"] = 2
    prm["Domain Definition"]["ID for Skull"] = 1
    prm["Domain Definition"]["Boundary ID Function Name"] = "boundaries"
    prm["Domain Definition"]["Subdomain ID Function Name"] = "subdomains"

    # Definition of a built-in mesh
    prm["Domain Definition"].add_subsection("Built-in Mesh")

    prm["Domain Definition"]["Built-in Mesh"]["Geometry Type"] = "Cube"
    prm["Domain Definition"]["Built-in Mesh"]["Mesh Refinement"] = 20

    # Definition of a cubic mesh
    prm["Domain Definition"]["Built-in Mesh"].add_subsection("Cubic Mesh")

    prm["Domain Definition"]["Built-in Mesh"]["Cubic Mesh"][
        "External Edge Length"
    ] = 0.1
    prm["Domain Definition"]["Built-in Mesh"]["Cubic Mesh"][
        "Internal Edge Length"
    ] = 0.01

    # Definition of a cubic mesh
    prm["Domain Definition"]["Built-in Mesh"].add_subsection("Spherical Mesh")

    prm["Domain Definition"]["Built-in Mesh"]["Spherical Mesh"][
        "External Radius"
    ] = 0.1
    prm["Domain Definition"]["Built-in Mesh"]["Spherical Mesh"][
        "Internal Radius"
    ] = 0.01

    # Definition of file information
    prm["Domain Definition"].add_subsection("Mesh from File")

    prm["Domain Definition"]["Mesh from File"][
        "File Name"
    ] = "../../mesh/Brain.h5"

    # SUBSECTION OF TEMPORAL DISCRETIZATION

    prm.add_subsection("Temporal Discretization")

    # Definition of final time and timestep
    prm["Temporal Discretization"]["Final Time"] = 1.0
    prm["Temporal Discretization"]["Time Step"] = 0.01
    prm["Temporal Discretization"]["Problem Periodicity"] = 1.0
    prm["Temporal Discretization"]["Theta-Method Parameter"] = 1.0

    # SUBSECTION PARAMETERS OF THE MODEL

    prm.add_subsection("Model Parameters")

    # SUBSECTION OF ARTERIAL BLOOD NETWORK

    prm["Model Parameters"]["Extracellular diffusion"] = 8e-6
    prm["Model Parameters"]["Axonal diffusion"] = 8e-5
    prm["Model Parameters"]["Reaction Coefficient"] = 0.9
    prm["Model Parameters"]["Initial Condition"] = 9250.57
    prm["Model Parameters"]["Initial Condition from File"] = "No"
    prm["Model Parameters"]["Initial Condition File Name"] = "..."
    prm["Model Parameters"]["Name of IC Function in File"] = "c0"
    prm["Model Parameters"]["Isotropic Diffusion"] = "Yes"
    prm["Model Parameters"]["Axonal Diffusion Tensor File Name"] = "..."
    prm["Model Parameters"]["Name of Axonal Diffusion Tensor in File"] = "D"
    prm["Model Parameters"]["Forcing Term"] = "0*x[0]"

    # SUBSECTION OF BOUNDARY CONDITIONS

    prm.add_subsection("Boundary Conditions")

    prm["Boundary Conditions"]["Ventricles BCs"] = "Neumann"
    prm["Boundary Conditions"]["Skull BCs"] = "Neumann"
    prm["Boundary Conditions"]["Input for Ventricles BCs"] = "Constant"
    prm["Boundary Conditions"]["Input for Skull BCs"] = "Constant"
    prm["Boundary Conditions"]["File Column Name Ventricles BCs"] = "c_vent"
    prm["Boundary Conditions"]["File Column Name Skull BCs"] = "c_skull"
    prm["Boundary Conditions"]["Ventricles Dirichlet BCs Value"] = 0
    prm["Boundary Conditions"]["Skull Dirichlet BCs Value"] = 0
    prm["Boundary Conditions"]["Ventricles Neumann BCs Value"] = 0
    prm["Boundary Conditions"]["Skull Neumann BCs Value"] = 0

    # SUBSECTION OF LINEAR SOLVER

    prm.add_subsection("Linear Solver")

    prm["Linear Solver"]["Type of Solver"] = "..."
    prm["Linear Solver"]["Iterative Solver"] = "..."
    prm["Linear Solver"]["Preconditioner"] = "..."
    prm["Linear Solver"]["User-Defined Preconditioner"] = "No"

    prm["Linear Solver"].add_subsection("Iterative Solver Options")
    prm["Linear Solver"]["Iterative Solver Options"][
        "Relative tolerance"
    ] = 1e-10
    prm["Linear Solver"]["Iterative Solver Options"][
        "Absolute tolerance"
    ] = 1e-10
    prm["Linear Solver"]["Iterative Solver Options"][
        "Non-zero initial guess"
    ] = True
    prm["Linear Solver"]["Iterative Solver Options"][
        "Monitor convergence"
    ] = True
    prm["Linear Solver"]["Iterative Solver Options"]["Report"] = True
    prm["Linear Solver"]["Iterative Solver Options"][
        "Maximum iterations"
    ] = 100000

    # SUBSECTION OF OUTPUT FILE

    prm.add_subsection("Output")

    prm["Output"]["Output XDMF File Name"] = "..."
    prm["Output"]["Timestep of File Save"] = 0.1

    # SUBSECTION OF CONVERGENCE TEST
    if conv:
        prm.add_subsection("Convergence Test")

        prm["Convergence Test"]["Exact Solution"] = 0

    return prm


#################################################################################################################
# 	   		Generator of comments for the Brain MPET parameters file		 		#
#################################################################################################################


def commentingprmBrainFK(Lines):
    commentedlines = []

    for line in Lines:
        comm = pfh.findinitialspaces(line)

        if not (line.find("set Method") == -1):
            comm = (
                comm
                + "# Decide the type of spatial discretization method to apply (DG-FEM/CG-FEM)"
            )

        if not (line.find("set Formulation") == -1):
            comm = (
                comm
                + "# Decide the type of problem formulation (Classical/Exponential)"
            )

        if not (line.find("set Polynomial Degree") == -1):
            comm = (
                comm + "# Decide the polynomial degree of the FEM approximation"
            )

        if not (line.find("set Type of Mesh") == -1):
            comm = (
                comm
                + "# Decide the type of mesh to use in your simulation: File/Built-in"
            )

        if not (line.find("set Characteristic Length") == -1):
            comm = (
                comm
                + "# Set the characteristic length scale of the problem [m]"
            )

        if not (line.find("set Characteristic Time") == -1):
            comm = (
                comm + "# Set the characteristic time scale of the problem [s]"
            )

        if not (line.find("set Characteristic Pressure") == -1):
            comm = (
                comm
                + "# Set the characteristic pressure scale of the problem [Pa]"
            )

        if not (line.find("set ID for Ventricles") == -1):
            comm = comm + "# Set the value of boundary ID of ventricles"

        if not (line.find("set ID for Skull") == -1):
            comm = comm + "# Set the value of boundary ID of skull"

        if not (line.find("set Boundary ID Function Name") == -1):
            comm = (
                comm
                + "# Set the name of the function containing the boundary ID"
            )

        if not (line.find("set Subdomain ID Function Name") == -1):
            comm = (
                comm
                + "# Set the name of the function containing the subdomain ID"
            )

        if not (line.find("set Geometry Type") == -1):
            comm = (
                comm
                + "# Decide the type of geometrical built-in object: Cube/Sphere/Square"
            )

        if not (line.find("set Mesh Refinement") == -1):
            comm = comm + "# Refinement value of the mesh"

        if not (line.find("set External Edge Length") == -1):
            comm = comm + "# Length of the external cube edge [m]"

        if not (line.find("set Internal Edge Length") == -1):
            comm = comm + "# Length of the internal cube edge [m]"

        if not (line.find("set External Radius") == -1):
            comm = comm + "# Length of the external sphere radius [m]"

        if not (line.find("set Internal Radius") == -1):
            comm = comm + "# Length of the internal sphere radius [m]"

        if not (line.find("set File Name") == -1):
            comm = (
                comm
                + "# Name of the file containing the mesh. Possible extensions: .h5"
            )

        if not (line.find("set Final Time") == -1):
            comm = comm + "# Final time of the simulation [s]"

        if not (line.find("set Time Step") == -1):
            comm = comm + "# Time step of the problem [s]"

        if not (line.find("set Problem Periodicity") == -1):
            comm = comm + "# Periodicity of the BCs [s]"

        if not (line.find("set Extracellular diffusion") == -1):
            comm = comm + "# Extracellular diffusion constant [m^2/years]"

        if not (line.find("set Axonal diffusion") == -1):
            comm = comm + "# Axonal diffusion constant [m^2/years]"

        if not (line.find("set Reaction Coefficient") == -1):
            comm = comm + "# Reaction Coefficient of the Proteins [1/years]"

        if not (line.find("set Isotropic Diffusion") == -1):
            comm = comm + "# Isotropic Diffusion Tensors assumption: Yes/No"

        if not (line.find("set Axonal Diffusion Tensor File Name") == -1):
            comm = comm + "# Name of the file containing the tensor"

        if not (line.find("set Name of Axonal Diffusion Tensor in File") == -1):
            comm = (
                comm
                + "# Name of the field in which the axonal diffusion tensor is stored"
            )

        if not (line.find("Forcing Term") == -1):
            comm = comm + "# Forcing Term [1/s]"

        if not (line.find("set Ventricles BCs") == -1):
            comm = (
                comm
                + "# Type of Boundary Condition imposed on the Ventricular Surface: Dirichlet/Neumann/Neumann with CSF Pressure"
            )

        if not (line.find("set Skull BCs") == -1):
            comm = (
                comm
                + "# Type of Boundary Condition imposed on the Skull Surface: Dirichlet/Neumann"
            )

        if not (line.find("set Initial Condition from File") == -1):
            comm = (
                comm + "# Enable the reading of an initial condition from file"
            )

        if not (line.find("set Initial Condition File Name") == -1):
            comm = comm + "# Name of the file containing the initial condition"

        if not (line.find("set Name of IC Function in File") == -1):
            comm = (
                comm
                + "# Name of the function containing the initial condition in the file"
            )

        if not (line.find("set Input for Ventricles BCs") == -1):
            comm = (
                comm
                + "# Type of Input for the imposition of Boundary Condition on the Ventricular Surface: Constant/File/Expression"
            )

        if not (line.find("set Input for Skull BCs") == -1):
            comm = (
                comm
                + "# Type of Input for the imposition of Boundary Condition imposed on the Skull Surface: Constant/File/Expression"
            )

        if not (line.find("set File Column Name") == -1):
            comm = (
                comm
                + "# Set the Column Name where is stored the BCs in the .csv file (associated to a column time)"
            )

        if not (line.find("set") == -1) and not (
            line.find("Dirichlet BCs Value") == -1
        ):
            comm = comm + "# Boundary Condition value to be imposed [-]"

        if not (line.find("set") == -1) and not (
            line.find("Neumann BCs Value") == -1
        ):
            comm = comm + "# Boundary Condition value to be imposed [1/m]"

        if not (line.find("set Output XDMF File Name") == -1):
            comm = (
                comm
                + "# Output file name (The relative/absolute path must be indicated!)"
            )

        if not (line.find("set Initial Condition") == -1):
            comm = comm + "# Initial condition of a concentration value"

        if not (line.find("set") == -1) and not (
            line.find("Exact Solution") == -1
        ):
            comm = comm + "# Exact solution of the test problem"

        if not (line.find("set Type of Solver") == -1):
            comm = (
                comm
                + "# Choice of linear solver type: Default/Iterative Solver/MUMPS"
            )

        if not (line.find("set Iterative Solver") == -1):
            comm = (
                comm
                + "# Choice of iterative solver type. The available options are: \n"
            )
            comm = (
                comm
                + "  "
                + "#   gmres - cg - minres - tfqmr - richardson - bicgstab - nash - stcg"
            )

        if not (line.find("set Preconditioner") == -1):
            comm = (
                comm
                + "# Choice of preconditioner type. The available options are: \n"
            )
            comm = (
                comm
                + "  "
                + "#   ilu - icc - jacobi - bjacobi - sor - additive_schwarz - petsc_amg - hypre_amg - \n"
            )
            comm = (
                comm
                + "  "
                + "#   hypre_euclid - hypre_parasails - amg - ml_amg - none"
            )

        if not (line.find("set User-Defined Preconditioner") == -1):
            comm = (
                comm
                + "# Choice of using the user defined block preconditioner: Yes/No"
            )

        if not (line.find("set Theta-Method Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the parameter theta: IE(1) - CN(0.5) - EE(0)"
            )

        if not (line.find("set Penalty Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the penalty parameter for the DG discretization"
            )

        if not (line.find("set Timestep of File Save") == -1):
            comm = comm + "# Temporal distance between saving two files"

        commentedlines = pfh.addcomment(comm, line, commentedlines)

    return commentedlines

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
    prm = parametersBrainMPET(conv)

    if pfh.generateprmfile(prm, filename) == 1:

        Lines = pfh.readlinesprmfile(filename)
        commentedlines = commentingprmBrainMPET(Lines)

        if pfh.writelinesprmfile(commentedlines, filename) == 1:

            print("Parameter file with default values generated")

        else:
            print("Error in parameter file generation!")

    else:

        print("Error in parameter file generation!")


#################################################################################
# 	Definition of the default parameters for the Brain MPET problem 	#
#################################################################################


def parametersBrainMPET(conv):

    # Create the prm file object
    prm = PRM()

    # PARAMETERS DEFINITION

    # SUBSECTION OF SPATIAL DISCRETIZATION

    prm.add_subsection("Spatial Discretization")

    prm["Spatial Discretization"]["Method"] = "DG-FEM"
    prm["Spatial Discretization"]["Polynomial Degree for Displacement"] = 2
    prm["Spatial Discretization"]["Polynomial Degree for Pressure"] = 1

    # SUBSECTION OF DISCONTINUOUS GALERKIN METHOD

    prm["Spatial Discretization"].add_subsection("Discontinuous Galerkin")

    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter for Arterial Pressure"
    ] = 10
    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter for Capillary Pressure"
    ] = 10
    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter for CSF-ISF Pressure"
    ] = 10
    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter for Venous Pressure"
    ] = 10
    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter for Displacement"
    ] = 100

    # SUBSECTION SCALING PARAMETERS FOR DIMENSIONLESS SOLVER
    prm.add_subsection("Scaling Parameters")

    prm["Scaling Parameters"]["Characteristic Length"] = 1e-03
    prm["Scaling Parameters"]["Characteristic Pressure"] = 1e04
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
    prm["Temporal Discretization"]["Newmark Beta Parameter"] = 0.25
    prm["Temporal Discretization"]["Newmark Gamma Parameter"] = 0.5

    # SUBSECTION PARAMETERS OF THE MODEL

    prm.add_subsection("Model Parameters")

    # SUBSECTION OF FLUID NETWORKS MODELING

    prm["Model Parameters"].add_subsection("Fluid Networks")

    # SUBSECTION OF ARTERIAL BLOOD NETWORK

    prm["Model Parameters"]["Fluid Networks"].add_subsection("Arterial Network")

    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Permeability"
    ] = 1.234e-09
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "External Coupling Parameter"
    ] = 0
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Biot Coefficient"
    ] = 0.25
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Time Derivative Coefficient"
    ] = 2.9e-4
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Initial Condition (Pressure)"
    ] = 9250.57
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Initial Condition from File (Pressure)"
    ] = "No"
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Initial Condition File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Name of IC Function in File"
    ] = "pa0"
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Isotropic Permeability"
    ] = "Yes"
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Permeability Tensor File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["Arterial Network"][
        "Name of Permeability Tensor in File"
    ] = "K_AV"

    # SUBSECTION OF CAPILLARY BLOOD NETWORK

    prm["Model Parameters"]["Fluid Networks"].add_subsection(
        "Capillary Network"
    )

    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Permeability"
    ] = 4.28e-13
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "External Coupling Parameter"
    ] = 0
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Biot Coefficient"
    ] = 0.25
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Time Derivative Coefficient"
    ] = 2.9e-4
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Initial Condition (Pressure)"
    ] = 5066.25
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Initial Condition from File (Pressure)"
    ] = "No"
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Initial Condition File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Name of IC Function in File"
    ] = "pc0"
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Isotropic Permeability"
    ] = "Yes"
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Permeability Tensor File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["Capillary Network"][
        "Name of Permeability Tensor in File"
    ] = "K_AV"

    # SUBSECTION OF VENOUS BLOOD NETWORK

    prm["Model Parameters"]["Fluid Networks"].add_subsection("Venous Network")

    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Permeability"
    ] = 2.468e-9
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "External Coupling Parameter"
    ] = 0
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Biot Coefficient"
    ] = 0.01
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Time Derivative Coefficient"
    ] = 1.5e-5
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Initial Condition (Pressure)"
    ] = 1333.22
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Initial Condition from File (Pressure)"
    ] = "No"
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Initial Condition File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Name of IC Function in File"
    ] = "pv0"
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Isotropic Permeability"
    ] = "Yes"
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Permeability Tensor File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["Venous Network"][
        "Name of Permeability Tensor in File"
    ] = "K_AV"

    # SUBSECTION OF CSF-ISF NETWORK

    prm["Model Parameters"]["Fluid Networks"].add_subsection("CSF-ISF Network")

    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Permeability"
    ] = 1.1e-16
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "External Coupling Parameter"
    ] = 0
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Biot Coefficient"
    ] = 0.49
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Time Derivative Coefficient"
    ] = 3.9e-4
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Initial Condition (Pressure)"
    ] = 1066.58
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Initial Condition from File (Pressure)"
    ] = "No"
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Initial Condition File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Name of IC Function in File"
    ] = "pe0"
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Isotropic Permeability"
    ] = "Yes"
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Permeability Tensor File Name"
    ] = "..."
    prm["Model Parameters"]["Fluid Networks"]["CSF-ISF Network"][
        "Name of Permeability Tensor in File"
    ] = "K_AV"

    # SUBSECTION OF COUPLING PARAMETERS

    prm["Model Parameters"]["Fluid Networks"].add_subsection(
        "Coupling Parameters"
    )

    prm["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "Arterial-Capillary Coupling Parameter"
    ] = 1.0e-6
    prm["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "Venous-Capillary Coupling Parameter"
    ] = 3.0e-6
    prm["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "Arterial-Venous Coupling Parameter"
    ] = 0.0
    prm["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "CSF-Arterial Coupling Parameter"
    ] = 0.0
    prm["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "CSF-Capillary Coupling Parameter"
    ] = 1.0e-6
    prm["Model Parameters"]["Fluid Networks"]["Coupling Parameters"][
        "CSF-Venous Coupling Parameter"
    ] = 1.0e-6

    # SUBSECTION OF SOLID TISSUE MODELING

    prm["Model Parameters"].add_subsection("Elastic Solid Tissue")

    prm["Model Parameters"]["Elastic Solid Tissue"][
        "First Lamé Parameter"
    ] = 2000
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Second Lamé Parameter"
    ] = 2400000
    prm["Model Parameters"]["Elastic Solid Tissue"]["Tissue Density"] = 1000
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition (Displacement)"
    ] = [0, 0, 0]
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition from File (Displacement)"
    ] = "No"
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition File Name (Displacement)"
    ] = "..."
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Name of IC Function in File (Displacement)"
    ] = "u0"
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition (Velocity)"
    ] = [0, 0, 0]
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition from File (Velocity)"
    ] = "No"
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Initial Condition File Name (Velocity)"
    ] = "..."
    prm["Model Parameters"]["Elastic Solid Tissue"][
        "Name of IC Function in File (Velocity)"
    ] = "u0"

    prm["Model Parameters"].add_subsection("Forcing Terms")

    prm["Model Parameters"]["Forcing Terms"][
        "Momentum Body Forces (x-component)"
    ] = "0*x[0]"
    prm["Model Parameters"]["Forcing Terms"][
        "Momentum Body Forces (y-component)"
    ] = "0*x[0]"
    prm["Model Parameters"]["Forcing Terms"][
        "Momentum Body Forces (z-component)"
    ] = "0*x[0]"
    prm["Model Parameters"]["Forcing Terms"][
        "Capillary Pressure Forcing Term"
    ] = "0*x[0]"
    prm["Model Parameters"]["Forcing Terms"][
        "CSF-ISF Pressure Forcing Term"
    ] = "0*x[0]"
    prm["Model Parameters"]["Forcing Terms"][
        "Arterial Pressure Forcing Term"
    ] = "0*x[0]"
    prm["Model Parameters"]["Forcing Terms"][
        "Venous Pressure Forcing Term"
    ] = "0*x[0]"

    # SUBSECTION OF BOUNDARY CONDITIONS

    prm.add_subsection("Boundary Conditions")

    prm["Boundary Conditions"].add_subsection("Elastic Solid Tissue")

    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles BCs"
    ] = "Neumann"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull BCs"
    ] = "Dirichlet"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Input for Skull BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "File Column Name Ventricles BCs (x-component)"
    ] = "ux_vent"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "File Column Name Ventricles BCs (y-component)"
    ] = "uy_vent"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "File Column Name Ventricles BCs (z-component)"
    ] = "uz_vent"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "File Column Name Skull BCs (x-component)"
    ] = "ux_skull"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "File Column Name Skull BCs (y-component)"
    ] = "uy_skull"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "File Column Name Skull BCs (z-component)"
    ] = "uz_skull"
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles Dirichlet BCs Value (Displacement x-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles Dirichlet BCs Value (Displacement y-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles Dirichlet BCs Value (Displacement z-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull Dirichlet BCs Value (Displacement x-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull Dirichlet BCs Value (Displacement y-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull Dirichlet BCs Value (Displacement z-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles Neumann BCs Value (Stress x-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles Neumann BCs Value (Stress y-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Ventricles Neumann BCs Value (Stress z-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull Neumann BCs Value (Stress x-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull Neumann BCs Value (Stress y-component)"
    ] = 0
    prm["Boundary Conditions"]["Elastic Solid Tissue"][
        "Skull Neumann BCs Value (Stress z-component)"
    ] = 0

    prm["Boundary Conditions"].add_subsection("Fluid Networks")
    prm["Boundary Conditions"]["Fluid Networks"].add_subsection(
        "Arterial Network"
    )

    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Ventricles BCs"
    ] = "Neumann"
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Skull BCs"
    ] = "Dirichlet"
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Input for Skull BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "File Column Name Ventricles BCs"
    ] = "pA_vent"
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "File Column Name Skull BCs"
    ] = "pA_skull"
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Ventricles Dirichlet BCs Value (Pressure)"
    ] = 700
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Skull Dirichlet BCs Value (Pressure)"
    ] = 10000
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Ventricles Neumann BCs Value (Flux)"
    ] = 0
    prm["Boundary Conditions"]["Fluid Networks"]["Arterial Network"][
        "Skull Neumann BCs Value (Flux)"
    ] = "PAFlux.csv"

    prm["Boundary Conditions"]["Fluid Networks"].add_subsection(
        "Capillary Network"
    )

    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Ventricles BCs"
    ] = "Neumann"
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Skull BCs"
    ] = "Neumann"
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Input for Skull BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "File Column Name Ventricles BCs"
    ] = "pC_vent"
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "File Column Name Skull BCs"
    ] = "pC_skull"
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Ventricles Dirichlet BCs Value (Pressure)"
    ] = 0
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Skull Dirichlet BCs Value (Pressure)"
    ] = 0
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Ventricles Neumann BCs Value (Flux)"
    ] = 0
    prm["Boundary Conditions"]["Fluid Networks"]["Capillary Network"][
        "Skull Neumann BCs Value (Flux)"
    ] = 0

    prm["Boundary Conditions"]["Fluid Networks"].add_subsection(
        "Venous Network"
    )

    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Ventricles BCs"
    ] = "Dirichlet"
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Skull BCs"
    ] = "Dirichlet"
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Input for Skull BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "File Column Name Ventricles BCs"
    ] = "pV_vent"
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "File Column Name Skull BCs"
    ] = "pV_skull"
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Ventricles Dirichlet BCs Value (Pressure)"
    ] = 700
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Skull Dirichlet BCs Value (Pressure)"
    ] = 1399.89
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Ventricles Neumann BCs Value (Flux)"
    ] = 0
    prm["Boundary Conditions"]["Fluid Networks"]["Venous Network"][
        "Skull Neumann BCs Value (Flux)"
    ] = 0

    prm["Boundary Conditions"]["Fluid Networks"].add_subsection(
        "CSF-ISF Network"
    )

    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Ventricles BCs"
    ] = "Dirichlet"
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Skull BCs"
    ] = "Dirichlet"
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Input for Skull BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "File Column Name Ventricles BCs"
    ] = "pE_vent"
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "File Column Name Skull BCs"
    ] = "pE_skull"
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Ventricles Dirichlet BCs Value (Pressure)"
    ] = 1066.58
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Skull Dirichlet BCs Value (Pressure)"
    ] = 1066.58
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Ventricles Neumann BCs Value (Flux)"
    ] = 0
    prm["Boundary Conditions"]["Fluid Networks"]["CSF-ISF Network"][
        "Skull Neumann BCs Value (Flux)"
    ] = 0

    # SUBSECTION OF LINEAR SOLVER

    prm.add_subsection("Linear Solver")

    prm["Linear Solver"]["Type of Solver"] = "..."
    prm["Linear Solver"]["Iterative Solver"] = "..."
    prm["Linear Solver"]["Preconditioner"] = "..."
    prm["Linear Solver"]["User-Defined Preconditioner"] = "No"

    # SUBSECTION OF OUTPUT FILE

    prm.add_subsection("Output")

    prm["Output"]["Output XDMF File Name"] = "..."
    prm["Output"]["Timestep of File Save"] = 0.1

    # SUBSECTION OF CONVERGENCE TEST
    if conv:
        prm.add_subsection("Convergence Test")

        prm["Convergence Test"]["Displacement Exact Solution (x-component)"] = 0
        prm["Convergence Test"]["Displacement Exact Solution (y-component)"] = 0
        prm["Convergence Test"]["Displacement Exact Solution (z-component)"] = 0
        prm["Convergence Test"]["Velocity Exact Solution (x-component)"] = 0
        prm["Convergence Test"]["Velocity Exact Solution (y-component)"] = 0
        prm["Convergence Test"]["Velocity Exact Solution (z-component)"] = 0
        prm["Convergence Test"]["Arterial Pressure Exact Solution"] = 0
        prm["Convergence Test"]["Capillary Pressure Exact Solution"] = 0
        prm["Convergence Test"]["Venous Pressure Exact Solution"] = 0
        prm["Convergence Test"]["CSF-ISF Pressure Exact Solution"] = 0

    return prm


#################################################################################################################
# 	   		Generator of comments for the Brain MPET parameters file		 		#
#################################################################################################################


def commentingprmBrainMPET(Lines):

    commentedlines = []

    for line in Lines:

        comm = pfh.findinitialspaces(line)

        if not (line.find("set Method") == -1):
            comm = (
                comm
                + "# Decide the type of spatial discretization method to apply (DG-FEM/CG-FEM)"
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

        if (not (line.find("set Permeability") == -1)) and (
            line.find("set Permeability Tensor File Name") == -1
        ):
            comm = (
                comm
                + "# Permeability Constant of the Fluid Network [m^4/(N*s)]"
            )

        if not (line.find("set Biot Coefficient") == -1):
            comm = comm + "# Biot Coefficient of the Fluid Network [-]"

        if not (line.find("set Time Derivative Coefficient") == -1):
            comm = (
                comm
                + "# Time Derivative Coefficient of the Fluid Network [m^2/N]"
            )

        if not (line.find("set Isotropic Permeability") == -1):
            comm = comm + "# Isotropic Permeability Tensors assumption: Yes/No"

        if not (line.find("set Permeability Tensor File Name") == -1):
            comm = comm + "# Name of the file containing the tensor"

        if not (line.find("set Name of Permeability Tensor in File") == -1):
            comm = (
                comm
                + "# Name of the field in which the permeability tensor is stored"
            )

        if not (line.find("set") == -1) and not (
            line.find("Coupling Parameter") == -1
        ):
            comm = (
                comm
                + "# Coupling Parameters between the Fluid Networks or External Discharge [m^2/(N*s)]"
            )

        if not (line.find("set First Lamé Parameter") == -1):
            comm = comm + "# First Lamé Parameter of the Solid Tissue [Pa]"

        if not (line.find("set Second Lamé Parameter") == -1):
            comm = comm + "# Second Lamé Parameter of the Solid Tissue [Pa]"

        if not (line.find("set Tissue Density") == -1):
            comm = comm + "# Density of the Solid Tissue [Kg/m^3]"

        if not (line.find("set Momentum Body Forces") == -1):
            comm = (
                comm + "# Body Forces in the Momentum Equation [Kg/(m^2*s^2)]"
            )

        if not (line.find("Pressure Forcing Term") == -1):
            comm = comm + "# Forcing Term in the Mass Balance Equation [1/s]"

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
            line.find("Dirichlet BCs Value (Displacement") == -1
        ):
            comm = comm + "# Boundary Condition value to be imposed [m]"

        if not (line.find("set") == -1) and not (
            line.find("Dirichlet BCs Value (Pressure)") == -1
        ):
            comm = (
                comm
                + "# Boundary Condition value to be imposed [Pa]: insert the constant value or the file name"
            )

        if not (line.find("set") == -1) and not (
            line.find("Neumann BCs Value (Stress") == -1
        ):
            comm = comm + "# Boundary Condition value to be imposed [Pa]"

        if not (line.find("set") == -1) and not (
            line.find("Neumann BCs Value (Flux)") == -1
        ):
            comm = (
                comm
                + "# Boundary Condition value to be imposed [m^3/s]: insert the constant value or the file name"
            )

        if not (line.find("set Output XDMF File Name") == -1):
            comm = (
                comm
                + "# Output file name (The relative/absolute path must be indicated!)"
            )

        if not (line.find("set Initial Condition (Pressure)") == -1):
            comm = comm + "# Initial condition of a pressure value [Pa]"

        if not (line.find("set Initial Condition (Displacement)") == -1):
            comm = comm + "# Initial condition of a displacement value [m]"

        if not (line.find("set Initial Condition (Velocity)") == -1):
            comm = comm + "# Initial condition of a velocity value [m/s]"

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

        if not (line.find("set Newmark Beta Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the parameter beta for the Newmark method"
            )

        if not (line.find("set Newmark Gamma Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the parameter gamma for the Newmark method"
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

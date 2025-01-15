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


def Createprm(filename):

    # Generate default parameters
    prm = parametersBrainAVDir()

    if pfh.generateprmfile(prm, filename) == 1:

        Lines = pfh.readlinesprmfile(filename)
        commentedlines = commentingprmBrainAVDir(Lines)

        if pfh.writelinesprmfile(commentedlines, filename) == 1:

            print("Parameter file with default values generated")

        else:
            print("Error in parameter file generation!")

    else:

        print("Error in parameter file generation!")


#################################################################################
# 	Definition of the default parameters for the Brain MPET problem 	#
#################################################################################


def parametersBrainAVDir():

    # Create the prm file object
    prm = PRM()

    # SUBSECTION SCALING PARAMETERS FOR DIMENSIONLESS SOLVER
    prm.add_subsection("Scaling Parameters")

    prm["Scaling Parameters"]["Characteristic Length"] = 1e-03

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

    # Definition of output files properties
    prm.add_subsection("Output")

    prm["Output"]["Output XDMF File Name"] = "..."

    prm["Output"]["Output h5 File Name"] = "..."

    return prm


#################################################################################################################
# 	   		Generator of comments for the Brain MPET parameters file		 		#
#################################################################################################################


def commentingprmBrainAVDir(Lines):

    commentedlines = []

    for line in Lines:

        comm = pfh.findinitialspaces(line)

        if not (line.find("set External Edge Length") == -1):
            comm = comm + "# Length of the external cube edge [m]"

        if not (line.find("set Internal Edge Length") == -1):
            comm = comm + "# Length of the internal cube edge [m]"

        if not (line.find("set External Radius") == -1):
            comm = comm + "# Length of the external sphere radius [m]"

        if not (line.find("set Internal Radius") == -1):
            comm = comm + "# Length of the internal sphere radius [m]"

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

        if not (line.find("set Characteristic Length") == -1):
            comm = (
                comm
                + "# Set the characteristic length scale of the problem [m]"
            )

        if not (line.find("set Type of Mesh") == -1):
            comm = (
                comm
                + "# Decide the type of mesh to use in your simulation: File/Built-in"
            )

        if not (line.find("set File Name") == -1):
            comm = (
                comm
                + "# Name of the file containing the mesh. Possible extensions: .h5"
            )

        if not (line.find("set Output XDMF File Name") == -1):
            comm = (
                comm
                + "# Output file name (The relative/absolute path must be indicated!)"
            )

        if not (line.find("set Output h5 File Name") == -1):
            comm = (
                comm
                + "# Output file name for the mesh with tensorial field (The relative/absolute path must be indicated!)"
            )

        commentedlines = pfh.addcomment(comm, line, commentedlines)

    return commentedlines

import math
from copy import deepcopy

import numpy as np
from fenics import *

# TODO - make this file more general. apps/Heterodimer up to now has its own
# XDMFHandler

#####################################################
# 		Generation of XDMF Solution File		    #
#####################################################


def SolutionFileCreator(file_name):
    # XDMF File Creation
    xdmf = XDMFFile(MPI.comm_world, file_name)

    # Allow the sharing of a mesh for different functions
    xdmf.parameters["functions_share_mesh"] = True

    # Not allowing the overwriting of functions
    xdmf.parameters["rewrite_function_mesh"] = False

    return xdmf


#####################################################
# 		Reconstruction of the Output File name		#
#####################################################


def FileNameReconstruction(time, dt, filename):
    # Time Step Reconstruction
    timestep = math.ceil(time / dt)

    # Filename reconstruction
    filenameout = filename + "_"

    if timestep < 1000000:
        filenameout = filenameout + "0"
    if timestep < 100000:
        filenameout = filenameout + "0"
    if timestep < 10000:
        filenameout = filenameout + "0"
    if timestep < 1000:
        filenameout = filenameout + "0"
    if timestep < 100:
        filenameout = filenameout + "0"
    if timestep < 10:
        filenameout = filenameout + "0"

    filenameout = filenameout + str(timestep) + ".xdmf"

    return filenameout


#####################################################
# 		Save MPET Solution at given time		    #
#####################################################


def FKSolutionSave(filename, x, time, dt, method, mesh, X):
    filenameout = FileNameReconstruction(time, dt, filename)

    xdmf = SolutionFileCreator(filenameout)

    c = Function(X)

    if method == "Classical":
        c.assign(x)
    elif method == "Exponential":
        c.vector()[:] = np.exp(x.vector()[:])

    c.rename("c", "Concentration")
    xdmf.write(c, time)
    xdmf.close()


#####################################################
# 		Save MPET Solution at given time		    #
#####################################################


def MPETSolutionSave(filename, x, time, dt, P, U):
    filenameout = FileNameReconstruction(time, dt, filename)

    xdmf = SolutionFileCreator(filenameout)

    # Splitting of the solution in the complete space
    pc, pa, pv, pe, u = x.split(deepcopy=True)

    pc.vector()[:] = P * pc.vector()[:]
    pa.vector()[:] = P * pa.vector()[:]
    pv.vector()[:] = P * pv.vector()[:]
    pe.vector()[:] = P * pe.vector()[:]
    u.vector()[:] = U * u.vector()[:]

    # Rename the functions to improve the understanding in save
    pc.rename("pC", "Capillary Pressure")
    pa.rename("pA", "Arterial Pressure")
    pv.rename("pV", "Venous Pressure")
    pe.rename("pE", "CSF Pressure")
    u.rename("u", "Displacement")

    # File Update
    xdmf.write(pc, time)
    xdmf.write(pa, time)
    xdmf.write(pv, time)
    xdmf.write(pe, time)
    xdmf.write(u, time)

    xdmf.close()


#####################################################################
# 			Save Blood Darcy Solution		    #
#####################################################################


def BloodDarcySolutionSave(filename, x, P):
    filenameout = filename + ".xdmf"

    xdmf = SolutionFileCreator(filenameout)

    # Splitting of the solution in the complete space
    (
        pc,
        pa,
        pv,
    ) = x.split(deepcopy=True)

    pc.vector()[:] = P * pc.vector()[:]
    pa.vector()[:] = P * pa.vector()[:]
    pv.vector()[:] = P * pv.vector()[:]

    # Rename the functions to improve the understanding in save
    pc.rename("pC", "Capillary Pressure")
    pa.rename("pA", "Arterial Pressure")
    pv.rename("pV", "Venous Pressure")

    # File Update
    xdmf.write(pc, 0)
    xdmf.write(pa, 0)
    xdmf.write(pv, 0)

    xdmf.close()


#####################################################################
# 	     Save Arterioles-Venules Direction Solution		    #
#####################################################################


def AVDirSolutionSave(filename, x):
    # Filename reconstruction
    filenameout = filename + ".xdmf"

    xdmf = SolutionFileCreator(filenameout)

    x.rename("t", "Normalized Thickness")

    # File Update
    xdmf.write(x, 0)

    xdmf.close()

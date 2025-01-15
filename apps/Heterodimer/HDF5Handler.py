#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Generalisation of the file utilities/HDF5_handler.py. To merge.
#
# Ok di Mattia, dopo tesi e al suo ritorno, eventualmente.
#
# ------------------------------------------------------------------------------
# Authors:
# Mattia Corti <mattia.corti@mail.polimi.it>
# Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################


import mshr
import dolfin
from fenics import *


def import_tensor(filename, K, KName):
    """import_tensor import the tensor encoded in a `.h5` file.

    Args:
        filename (str): name of the file containing the binary encoding of the
        tensor.
        K : tensor object.
        KName : group name to be searched into the `.h5` file.

    Returns:
        tensor
    """
    file = HDF5File(MPI.comm_world, filename, "r")

    # Read the function from file.
    KName_imp = "/" + KName
    print(KName_imp)
    print(file)
    file.read(K, KName_imp)

    file.close()

    print("Tensor Imported.")
    return K


def save_tensor(output_filename, mesh, K, KName):
    # Output file for tensorial diffusion
    hdf5_file = HDF5File(MPI.comm_world, output_filename, "w")

    # Writing mesh to the HDF5 file
    hdf5_file.write(mesh, "/mesh")
    KName_imp = "/" + KName
    hdf5_file.write(K, KName_imp)

    hdf5_file.close()

    # Create XDMF file
    xdmf_file = XDMFFile(
        MPI.comm_world, output_filename.replace(".h5", ".xdmf")
    )
    xdmf_file.write(mesh)
    xdmf_file.write(K)
    xdmf_file.close()


def import_IC_from_file(filename, x, x_name):
    """import_IC_from_file import initial conditions of the problem for all the
    solution's components from a HDF5 `.h5` file.

    Read the file `filename.h5` in parallel in read mode and assign to the
    solution's component `x` the values stored in the group section named
    `x_name`.

    Notice that the values stored in `filename.h5` should be linked to the
    mesh topology to ensure a proper problem initialisation.

    Args:
        filename (str): name of the file containing the initial solution's
        components.
        x : solution's component;
        x_name (str): group name in the file `filename.ht`.

    Returns:
        Initial solution's component filled with values read from file
        `filename` in group `x_name`.
    """
    # File name recover from parameters
    file = HDF5File(MPI.comm_world, filename, "r")

    # Read the function from the file
    x_name_imp = "/" + x_name
    file.read(x, x_name_imp)

    file.close()

    if MPI.comm_world.Get_rank() == 0:
        print("Initial Condition " + x_name + " imported from File!")

    return x

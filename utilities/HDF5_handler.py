import mshr
import dolfin
from fenics import *

# SAVE MPT SOLUTION AS INITIAL CONDITION


def SaveDarcySolution(outputfilename, mesh, x, P):
    # Output file for tensorial diffusion
    file = HDF5File(MPI.comm_world, outputfilename, "w")

    x.vector()[:] = x.vector()[:] * P
    pc, pa, pv = x.split(deepcopy=True)

    # Writing mesh and functions on the outputfile
    file.write(mesh, "/mesh")
    file.write(pc, "/pC0")
    file.write(pa, "/pA0")
    file.write(pv, "/pV0")

    file.close()


# IMPORT INITIAL CONDITIO FROM FILE


def ImportICfromFile(filename, mesh, x, x_name):
    # File name recover from parameters
    file = HDF5File(MPI.comm_world, filename, "r")

    # Read the function from the file
    x_name_imp = "/" + x_name
    file.read(x, x_name_imp)

    file.close()

    if MPI.comm_world.Get_rank() == 0:
        mess = "Initial Condition " + x_name + " imported from File!"
        print(mess)

    return x


# IMPORT INITIAL CONDITIO FROM FILE


def SavePermeabilityTensor(outputfilename, mesh, boundaries, K, KName):
    # Output file for tensorial diffusion
    file = HDF5File(MPI.comm_world, outputfilename, "w")

    # Writing mesh and functions on the outputfile
    file.write(mesh, "/mesh")
    KName_imp = "/" + KName
    file.write(K, KName_imp)

    file.close()


def ImportPermeabilityTensor(filename, mesh, K, KName):
    file = HDF5File(MPI.comm_world, filename, "r")

    # Read the function from the file
    KName_imp = "/" + KName
    file.read(K, KName_imp)

    file.close()

    print("Permeability Tensor Imported!")
    return K

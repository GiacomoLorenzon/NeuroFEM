import mshr
import meshio
import math
import dolfin
from mpi4py import MPI
from fenics import *
import numpy as np
import sys
import getopt
import os

cwd = os.getcwd()
sys.path.append(cwd + "/../utilities")
import Mesh_conversions as meshingtools

#######################################################################################################################################
#                                                         HELP COMMAND INSTRUCTION                                                    #
#######################################################################################################################################


def helpcommand():
    print("The following sintax needs to be provided in input for the problem:")
    print("\nInput data:\n")
    print(" -i (--ifile) \t<inputfile> \t MeshFile format .xdmf")
    print(
        " -s (--sfile) \t<subdomainsfile> \t File containing subdomains division .xdmf"
    )
    print(
        " -b (--bfile) \t<boundaryfile> \t File containing boundary's IDs .xdmf"
    )
    print(" -o (--ofile) \t<outputfile> \t MeshFile format .h5")


#######################################################################################################################################
#                                                             INPUT DATA READING                                                      #
#######################################################################################################################################


def main(argv):

    # Control of no errors in calling the function of conversion
    try:
        opts, args = getopt.getopt(
            argv, "hi:o:s:b:", ["help", "ifile=", "ofile=", "sfile=", "bfile="]
        )
    except getopt.GetoptError:
        print("Error in calling the script MeshConversion!")
        helpcommand()
        sys.exit(2)

    # Allocation of information dictionary with default data
    param = {
        "InputFileName": "",
        "OutputFileName": "",
        "SubdomainsFileName": "",
        "BoundariesFileName": "",
    }

    # Extraction of information from function call
    for opt, arg in opts:

        # Help command execution
        if opt in ("-h", "--help"):
            helpcommand()
            sys.exit()

        # Input file name extraction
        elif opt in ("-i", "--ifile"):
            param["InputFileName"] = arg
            print("Input File: ", param["InputFileName"])

        # Output file name extraction
        elif opt in ("-o", "--ofile"):
            param["OutputFileName"] = arg
            print("Output File: ", param["OutputFileName"])

        # Subdomains file name extraction
        elif opt in ("-s", "--sfile"):
            param["SubdomainsFileName"] = arg
            print("Subdomains File: ", param["SubdomainsFileName"])

        # Subdomains file name extraction
        elif opt in ("-b", "--bfile"):
            param["BoundariesFileName"] = arg
            print("Boundaries File: ", param["BoundariesFileName"])

    # Control of existence of both input and output file names
    if param["InputFileName"] == "":
        print("\n Input file name needs to be provided!")
        sys.exit(0)

    if param["SubdomainsFileName"] == "":
        print("\n Subdomains file name needs to be provided!")
        sys.exit(0)

    if param["BoundariesFileName"] == "":
        print("\n Boundaries file name needs to be provided!")
        sys.exit(0)

    if param["OutputFileName"] == "":
        print("\n Output file name needs to be provided!")
        sys.exit(0)

    return param


#############################################################################################################################
# 							Mesh constructor function					    #
#############################################################################################################################


def meshconstruction(param):
    totlen = len(param["InputFileName"])

    # Control of extension of input mesh file
    if param["InputFileName"][totlen - 5 : totlen] == ".xdmf":
        meshingtools.xdmftoh5_heterog(
            param["InputFileName"],
            param["SubdomainsFileName"],
            param["BoundariesFileName"],
            param["OutputFileName"],
        )

    # Error for invalid mesh file name
    else:
        print(
            "\n The extension of the input file name is not valid! Insert a .xdmf file"
        )
        sys.exit(0)


#############################################################################################################################
# 							Main Function							    #
#############################################################################################################################

if __name__ == "__main__":
    param = main(sys.argv[1:])
    meshconstruction(param)

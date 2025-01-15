#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Parameter file generation for the heterodimer model.
#
# ------------------------------------------------------------------------------
# Author: Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

# sys and os are needed to manage paths for finding files.
import os
import sys

from pyrameters import PRM

# Retrieve the directory from where this script is executed.
cwd = os.getcwd()
# Allows search for modules both in the current directory and in "utilities".
sys.path.append(cwd + "/../../utilities/")

import ParameterFile_handler as pfh


def Createprm(filename, conv=False):
    """Createprm.   Parameter file generation for the heterodimer model.

    Wrap function that manages the generation of the parameter file for the
    heterodimer model with default values. It relies on the problem-specific
    function `parameters_brain_heterodimer`.

    Args:
        filename (str): `filename.prm` is the file containing the data
        necessary to run the app, i.e. boundary conditions, constant values,
        numerical solver options, etc. .
        conv (bool): number of iterations desired for the convergence test.
    """
    # Generate default parameters specific to the physical model.
    prm = parameters_brain_heterodimer(conv)

    # If the reading is succesful, read lines and generate comments.
    assert (
        pfh.generateprmfile(prm, filename) == 1
    ), "Error in parameter file generation."
    Lines = pfh.readlinesprmfile(filename)
    commentedlines = commentingprm_brain_heterodimer(Lines)

    assert (
        pfh.writelinesprmfile(commentedlines, filename) == 1
    ), "Error in parameter file generation."
    print(
        f"Parameter file has been generated with default values.\n"
        + f"   Name:    {filename}\n"
        + f"   Path:    {cwd}\n\n"
    )
    if conv:
        print(
            f"You may want to run the simulation by typing:\n"
            + f"   python3 Problem_Solver.py -f {filename} -c 3\n"
        )
    else:
        print(
            f"You may want to run the simulation by typing:\n"
            + f"   python3 Problem_Solver.py -f {filename}\n"
        )


def parameters_brain_heterodimer(conv):
    """parameters_brain_heterodimer creation and default defintion of all the
    parameters needed for this application.

    Args:
        conv (bool): `True` if a convergence test is performed.

    Returns:
        dictionary: dictionary containing all the parameters read in input.
    """
    # Create the prm file object.
    prm = PRM()

    # PARAMETERS DEFINITION.
    # Subsection of spatial definition.
    prm.add_subsection("Spatial Discretization")

    prm["Spatial Discretization"]["Polynomial Degree"] = 2

    # Subsection specific to Discontinuos Galerkine.
    prm["Spatial Discretization"].add_subsection("Discontinuous Galerkin")

    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Penalty Parameter"
    ] = 10
    prm["Spatial Discretization"]["Discontinuous Galerkin"][
        "Interior Penalty Parameter"
    ] = 1

    # Subsection scaling parameters for dimensionless sovler.
    prm.add_subsection("Scaling Parameters")

    prm["Scaling Parameters"]["Characteristic Length"] = 1
    prm["Scaling Parameters"]["Characteristic Time"] = 1
    prm["Scaling Parameters"]["Characteristic Concentration"] = 1

    # Subsection for mesh information.
    prm.add_subsection("Domain Definition")

    prm["Domain Definition"]["ID for Skull"] = 1
    prm["Domain Definition"]["ID for Ventricles"] = 2
    prm["Domain Definition"][
        "Boundary ID Function Name"
    ] = "..."  # "boundaries"
    prm["Domain Definition"][
        "Subdomain ID Function Name"
    ] = "..."  # "subdomains"

    prm["Domain Definition"]["Type of Mesh"] = "Built-in"
    # Definition of a built-in mesh.
    prm["Domain Definition"].add_subsection("Built-in Mesh")
    prm["Domain Definition"]["Built-in Mesh"]["Geometry Type"] = "Square"
    prm["Domain Definition"]["Built-in Mesh"]["Mesh Refinement"] = 8

    if (
        prm["Domain Definition"]["Built-in Mesh"]["Geometry Type"]
        == "Holed-Cube"
    ):
        # Definition of a Holed-Cubic mesh.
        prm["Domain Definition"]["Built-in Mesh"].add_subsection(
            "Holed-Cube Mesh"
        )

        prm["Domain Definition"]["Built-in Mesh"]["Holed-Cube Mesh"][
            "External Edge Length"
        ] = 0.1
        prm["Domain Definition"]["Built-in Mesh"]["Holed-Cube Mesh"][
            "Internal Edge Length"
        ] = 0.01
    elif prm["Domain Definition"]["Built-in Mesh"]["Geometry Type"] == "Sphere":
        # Definition of a spherical mesh.
        prm["Domain Definition"]["Built-in Mesh"].add_subsection(
            "Spherical Mesh"
        )

        prm["Domain Definition"]["Built-in Mesh"]["Spherical Mesh"][
            "External Radius"
        ] = 0.1
        prm["Domain Definition"]["Built-in Mesh"]["Spherical Mesh"][
            "Internal Radius"
        ] = 0.01

    # Definition of file information.
    prm["Domain Definition"].add_subsection("Mesh from File")

    prm["Domain Definition"]["Mesh from File"][
        "File Name"
    ] = "..."  # "../../mesh/Brain.h5"

    # SUBSECTION OF LINEAR SOLVER
    prm.add_subsection("Linear Solver")

    prm["Linear Solver"]["Type of Solver"] = "Iterative Solver"
    prm["Linear Solver"]["Iterative Solver"] = "gmres"
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
    prm["Linear Solver"]["Preconditioner"] = "sor"
    prm["Linear Solver"]["User-Defined Preconditioner"] = "No"

    # SUBSECTION: MODEL'S PARAMETERS
    prm.add_subsection("Model Parameters")

    prm["Model Parameters"]["Isotropic Diffusion"] = "Yes"
    prm["Model Parameters"]["Extracellular diffusion: healthy proteins"] = 1.0
    prm["Model Parameters"]["Extracellular diffusion: misfolded proteins"] = 1.0
    prm["Model Parameters"]["Axonal diffusion: healthy proteins"] = 0.0
    prm["Model Parameters"]["Axonal diffusion: misfolded proteins"] = 0.0
    prm["Model Parameters"]["Axonal Diffusion Tensor File Name"] = "..."
    prm["Model Parameters"]["Name of Axonal Diffusion Tensor in File"] = "..."

    prm["Model Parameters"]["Healthy proteins Producing Rate"] = 1.0
    prm["Model Parameters"]["Reaction Coefficient: healthy proteins"] = 1.0
    prm["Model Parameters"]["Reaction Coefficient: misfolded proteins"] = 1.0
    prm["Model Parameters"]["Reaction Coefficient: non-linear"] = 1.0

    prm["Model Parameters"][
        "Input for forcing term: healthy proteins"
    ] = "Constant"
    prm["Model Parameters"]["Forcing Term: healthy proteins"] = "0.0"
    prm["Model Parameters"][
        "Input for forcing term: misfolded proteins"
    ] = "Constant"
    prm["Model Parameters"]["Forcing Term: misfolded proteins"] = "0.0"

    prm["Model Parameters"]["Initial Condition from File"] = "No"

    prm["Model Parameters"]["Initial Condition: healthy proteins"] = "1.0*x[0]"
    prm["Model Parameters"][
        "Initial Condition: misfolded proteins"
    ] = "0.1*(t+1)"

    prm["Model Parameters"]["Initial Condition File Name"] = "..."
    prm["Model Parameters"][
        "Name of IC Function in File: healthy proteins"
    ] = "..."  # "c_0"
    prm["Model Parameters"][
        "Name of IC Function in File: misfolded proteins"
    ] = "..."  # "q_0"

    # SUBSECTION OF OUTPUT FILE
    prm.add_subsection("Output")

    prm["Output"]["Output XDMF File Name"] = "./solution"
    prm["Output"]["Timestep of File Save"] = 0.01
    prm["Output"]["Multiple output files"] = True
    prm["Output"]["Limit print messages"] = False

    # Subsection for temporal discretisation.
    prm.add_subsection("Temporal Discretization")

    # Definition of final time and timestep.
    prm["Temporal Discretization"]["Final Time"] = 5e-5
    prm["Temporal Discretization"]["Time Step"] = 1e-5
    prm["Temporal Discretization"]["Problem Periodicity"] = 1.0
    prm["Temporal Discretization"]["Theta-Method Parameter"] = 0.5

    # SUBSECTION OF BOUNDARY CONDITIONS
    prm.add_subsection("Boundary Conditions")
    prm["Boundary Conditions"].add_subsection("Healthy proteins")

    prm["Boundary Conditions"]["Healthy proteins"]["Ventricles BCs"] = "Neumann"
    prm["Boundary Conditions"]["Healthy proteins"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Healthy proteins"][
        "File Column Name Ventricles BCs"
    ] = "..."  # "c_vent"
    prm["Boundary Conditions"]["Healthy proteins"][
        "Ventricles BCs Value"
    ] = "0.0"

    prm["Boundary Conditions"]["Healthy proteins"]["Skull BCs"] = "Neumann"
    prm["Boundary Conditions"]["Healthy proteins"][
        "Input for Skull BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Healthy proteins"][
        "File Column Name Skull BCs"
    ] = "..."  # "c_skull"
    prm["Boundary Conditions"]["Healthy proteins"]["Skull BCs Value"] = "0.0"

    prm["Boundary Conditions"].add_subsection("Misfolded proteins")

    prm["Boundary Conditions"]["Misfolded proteins"][
        "Ventricles BCs"
    ] = "Neumann"
    prm["Boundary Conditions"]["Misfolded proteins"][
        "Input for Ventricles BCs"
    ] = "Constant"
    prm["Boundary Conditions"]["Misfolded proteins"][
        "File Column Name Ventricles BCs"
    ] = "..."  # "q_vent"
    prm["Boundary Conditions"]["Misfolded proteins"][
        "Ventricles BCs Value"
    ] = "0.0"

    prm["Boundary Conditions"]["Misfolded proteins"]["Skull BCs"] = "Neumann"
    prm["Boundary Conditions"]["Misfolded proteins"][
        "Input for Skull BCs"
    ] = "Constant"  # or File
    prm["Boundary Conditions"]["Misfolded proteins"][
        "File Column Name Skull BCs"
    ] = "..."  # "q_skull"
    prm["Boundary Conditions"]["Misfolded proteins"]["Skull BCs Value"] = "0.0"

    # SUBSECTION OF CONVERGENCE TEST
    prm.add_subsection("Convergence Test")
    if conv:
        prm["Model Parameters"][
            "Input for forcing term: healthy proteins"
        ] = "Expression"
        prm["Model Parameters"][
            "Input for forcing term: misfolded proteins"
        ] = "Expression"

        prm["Boundary Conditions"]["Healthy proteins"][
            "Ventricles BCs"
        ] = "Dirichlet"
        prm["Boundary Conditions"]["Healthy proteins"][
            "Input for Ventricles BCs"
        ] = "Expression"

        prm["Boundary Conditions"]["Healthy proteins"][
            "Skull BCs"
        ] = "Dirichlet"
        prm["Boundary Conditions"]["Healthy proteins"][
            "Input for Skull BCs"
        ] = "Expression"

        prm["Boundary Conditions"]["Misfolded proteins"][
            "Ventricles BCs"
        ] = "Dirichlet"
        prm["Boundary Conditions"]["Misfolded proteins"][
            "Input for Ventricles BCs"
        ] = "Expression"

        prm["Boundary Conditions"]["Misfolded proteins"][
            "Skull BCs"
        ] = "Dirichlet"
        prm["Boundary Conditions"]["Misfolded proteins"][
            "Input for Skull BCs"
        ] = "Expression"

        if (
            prm["Domain Definition"]["Built-in Mesh"]["Geometry Type"]
            == "Square"
        ):
            prm["Convergence Test"][
                "Exact Solution: healthy proteins"
            ] = "(cos(pi * x[0]) + cos(pi * x[1])) * cos(t)"
            prm["Convergence Test"][
                "Exact Solution: misfolded proteins"
            ] = "(cos(4 * pi * x[0]) * cos(4 * pi * x[1]) + 2) * exp(-t)"

            prm["Model Parameters"][
                "Forcing Term: healthy proteins"
            ] = "-sin(t)*(cos(pi*x[0])+cos(pi*x[1]))+pi*pi*cos(pi*x[0])*cos(t)+pi*pi*cos(pi*x[1])*cos(t)+(cos(pi*x[0])+cos(pi*x[1]))*cos(t)+(cos(pi*x[0])+cos(pi*x[1]))*cos(t)*(cos(4*pi*x[0])*cos(4*pi*x[1])+2)*exp(-t)"
            prm["Model Parameters"][
                "Forcing Term: misfolded proteins"
            ] = "-(cos(4*pi*x[0])*cos(4*pi*x[1])+2)*exp(-t)+4*pi*4*pi*cos(4*pi*x[0])*cos(4*pi*x[1])*exp(-t)+4*pi*4*pi*cos(4*pi*x[1])*cos(4*pi*x[0])*exp(-t)+(cos(4*pi*x[0])*cos(4*pi*x[1])+2)*exp(-t)+(cos(4*pi*x[0])*cos(4*pi*x[1])+2)*exp(-t)-(cos(4*pi*x[0])*cos(4*pi*x[1])+2)*exp(-t)*(cos(pi*x[0])+cos(pi*x[1]))*cos(t)"

            prm["Model Parameters"][
                "Initial Condition: healthy proteins"
            ] = "(cos(pi*x[0])+cos(pi*x[1]))*cos(t)"
            prm["Model Parameters"][
                "Initial Condition: misfolded proteins"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])+2)*exp(-t)"

            prm["Boundary Conditions"]["Healthy proteins"][
                "Ventricles BCs Value"
            ] = "(cos(pi*x[0])+cos(pi*x[1]))*cos(t)"

            prm["Boundary Conditions"]["Healthy proteins"][
                "Skull BCs Value"
            ] = "(cos(pi*x[0])+cos(pi*x[1]))*cos(t)"

            prm["Boundary Conditions"]["Misfolded proteins"][
                "Ventricles BCs Value"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])+2)*exp(-t)"

            prm["Boundary Conditions"]["Misfolded proteins"][
                "Skull BCs Value"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])+2)*exp(-t)"

        elif (
            prm["Domain Definition"]["Built-in Mesh"]["Geometry Type"] == "Cube"
        ):
            prm["Convergence Test"][
                "Exact Solution: healthy proteins"
            ] = "(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t)"
            prm["Convergence Test"][
                "Exact Solution: misfolded proteins"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t)"

            prm["Model Parameters"][
                "Forcing Term: healthy proteins"
            ] = "-sin(t)*(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))+pi*pi*cos(t)*(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))+(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t)+((cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t))*((cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t))"
            prm["Model Parameters"][
                "Forcing Term: misfolded proteins"
            ] = "-(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t)+3*2*2*pi*pi*cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])*exp(-t)+(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t)-((cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t))*((cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t))"

            prm["Model Parameters"][
                "Initial Condition: healthy proteins"
            ] = "(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t)"
            prm["Model Parameters"][
                "Initial Condition: misfolded proteins"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t)"

            prm["Boundary Conditions"]["Healthy proteins"][
                "Ventricles BCs Value"
            ] = "(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t)"

            prm["Boundary Conditions"]["Healthy proteins"][
                "Skull BCs Value"
            ] = "(cos(pi*x[0])+cos(pi*x[1])+cos(pi*x[2]))*cos(t)"

            prm["Boundary Conditions"]["Misfolded proteins"][
                "Ventricles BCs Value"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t)"

            prm["Boundary Conditions"]["Misfolded proteins"][
                "Skull BCs Value"
            ] = "(cos(2*pi*x[0])*cos(2*pi*x[1])*cos(2*pi*x[2])+3)*exp(-t)"

    else:
        prm["Convergence Test"]["Exact Solution: healthy proteins"] = "..."
        prm["Convergence Test"]["Exact Solution: misfolded proteins"] = "..."

    return prm


def commentingprm_brain_heterodimer(Lines):
    """commentingprm_brain_heterodimer provides some comments in the .prm file."""
    commentedlines = []

    for line in Lines:
        comm = pfh.findinitialspaces(line)

        # Boundary conditions.
        if not (line.find("set Input for Skull BCs") == -1):
            comm = (
                comm
                + "# Input type for boundary conditions on the Skull Surface: Constant/Expression"
            )
        if not (line.find("set Input for Ventricles BCs") == -1):
            comm = (
                comm
                + "# Input type for boundary conditions on the Ventricular Surface: Constant/Expression"
            )
        if not (line.find("set Skull BCs") == -1):
            comm = (
                comm
                + "# Input type for boundary conditions on the Skull Surface: Dirichlet/Neumann"
            )
        if not (line.find("set Ventricles BCs") == -1):
            comm = (
                comm
                + "# Input type for boundary conditions on the Ventricular Surface: Dirichlet/Neumann"
            )

        # Domain definition.
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
        if not (line.find("set ID for Skull") == -1):
            comm = comm + "# Set the value of boundary ID of skull"
        if not (line.find("set ID for Ventricles") == -1):
            comm = comm + "# Set the value of boundary ID of ventricles"
        if not (line.find("set Type of Mesh") == -1):
            comm = comm + "# Mesh type used in your simulation: File/Built-in"
        if not (line.find("set Geometry Type") == -1):
            comm = (
                comm
                + "# Built-in geometry: Interval/Square/Square crossed/Holed-Cube/Cube/Sphere"
            )
        if not (line.find("set Mesh Refinement") == -1):
            comm = comm + "# Refinements of the mesh"
        if not (line.find("set File Name") == -1):
            comm = comm + "# Mesh file name. Possible extensions: .h5"

        # Linear solver
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
        if not (line.find("set Type of Solver") == -1):
            comm = (
                comm
                + "# Choice of linear solver type: Default/Iterative Solver/MUMPS"
            )
        if not (line.find("set User-Defined Preconditioner") == -1):
            comm = (
                comm
                + "# Choice of using the user defined block preconditioner: Yes/No"
            )

        # Model parameters.
        if not (line.find("set Isotropic Diffusion") == -1):
            comm = comm + "# Isotropic Diffusion Tensors assumption: Yes/No"
        if not (line.find("set Axonal diffusion: healthy proteins") == -1):
            comm = comm + "# Axonal diffusion constant [m^2/years]"
        if not (line.find("set Axonal Diffusion Tensor File Name") == -1):
            comm = comm + "# Name of the file containing the tensor"
        if not (line.find("set Name of Axonal Diffusion Tensor in File") == -1):
            comm = (
                comm
                + "# Field name in which the axonal diffusion tensor is stored"
            )

        if not (line.find("Extracellular diffusion: healthy proteins") == -1):
            comm = comm + "# Extracellular diffusion constant [m^2/years]"

        if not (
            line.find("set Input for forcing term: healthy proteins") == -1
        ):
            comm = comm + "# Input type for forcing term: Constant/Expression"
        if not (line.find("set Forcing Term: healthy proteins") == -1):
            comm = comm + "# Forcing Term [1/years]"

        if not (
            line.find("set Input for forcing term: misfolded proteins") == -1
        ):
            comm = comm + "# Input type for forcing term: Constant/Expression"
        if not (line.find("set Forcing Term: misfolded proteins") == -1):
            comm = comm + "# Forcing Term [1/years]"

        if not (line.find("set Healthy proteins Producing Rate") == -1):
            comm = comm + "# Production rate constant [1/years]"
        if not (line.find("set Reaction Coefficient: healthy proteins") == -1):
            comm = (
                comm
                + "# Reaction Coefficient of the healthy Proteins [1/years]"
            )
        if not (
            line.find("set Reaction Coefficient: misfolded proteins") == -1
        ):
            comm = (
                comm
                + "# Reaction Coefficient of the misfolded Proteins [1/years]"
            )
        if not (line.find("set Reaction Coefficient: non-linear") == -1):
            comm = comm + "# Non-linear reaction Coefficient [1/years]"

        if not (line.find("set Initial Condition from File") == -1):
            comm = comm + "# Initial condition read from file."

        # Output
        if not (line.find("set Limit print messages") == -1):
            comm = (
                comm
                + "# Limit to the bare minimum the print messages to video for high performance."
            )
        if not (line.find("set Multiple output files") == -1):
            comm = (
                comm
                + "# Save each solution time step in a different file or in a single one."
            )
        if not (line.find("set Output XDMF File Name") == -1):
            comm = (
                comm
                + "# Output file name (The relative/absolute path must be indicated.)"
            )
        if not (line.find("set Timestep of File Save") == -1):
            comm = comm + "# Temporal distance between saving two files"

        # Scaling parameters.
        if not (line.find("set Characteristic Concentration") == -1):
            comm = (
                comm
                + "# Set the characteristic concentration value of the problem [g/m^3]"
            )
        if not (line.find("set Characteristic Time") == -1):
            comm = (
                comm
                + "# Set the characteristic time scale of the problem [years]"
            )
        if not (line.find("set Characteristic Length") == -1):
            comm = (
                comm
                + "# Set the characteristic length scale of the problem [m]"
            )

        # Spatial discretisation.
        if not (line.find("set Polynomial Degree") == -1):
            comm = comm + "# Polynomial degree of the FEM approximation"

        # Discontinuous Galerkin.
        if not (line.find("set Interior Penalty Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the interior penalty parameter for the DG discretization. The choices {-1, 0, 1} corresponds respectively to {NIP, IIP, SIP}."
            )
        if not (line.find("set Penalty Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the penalty parameter for the DG discretization"
            )

        # Temporal discretisation.
        if not (line.find("set Final Time") == -1):
            comm = comm + "# Final time of the simulation [s]"
        if not (line.find("set Problem Periodicity") == -1):
            comm = comm + "# Periodicity of the BCs [s]"
        if not (line.find("set Theta-Method Parameter") == -1):
            comm = (
                comm
                + "# Choice of the value of the parameter theta: IE(1) - CN(0.5) - EE(0)"
            )
        if not (line.find("set Time Step") == -1):
            comm = comm + "# Time step of the problem [s]"

        # Exact solution
        if not (line.find("set") == -1) and not (
            line.find("Exact Solution: healthy proteins") == -1
        ):
            comm = (
                comm
                + "# Exact solution of the test problem: healthy proteins component."
            )

        if not (line.find("set") == -1) and not (
            line.find("Exact Solution: misfolded proteins") == -1
        ):
            comm = (
                comm
                + "# Exact solution of the test problem: misfolded proteins component."
            )

        commentedlines = pfh.addcomment(comm, line, commentedlines)

    return commentedlines

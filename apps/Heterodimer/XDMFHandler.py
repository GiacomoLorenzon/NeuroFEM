#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# File handler containing utilites to export solution data in output.
#
# ------------------------------------------------------------------------------
# Authors:
# Mattia Corti <mattia.corti@mail.polimi.it>
# Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

import numpy as np
from fenics import *

from dolfin import MPI

import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/../../utilities")
sys.path.append("../../utilities")

import XDMF_handler as XDMF_H


class SolutionSaver:
    """_summary_
    ------------
    Utility class useful to save time-dependent solutions in a `.xdmf` file. It
    works both with the Classical and the Exponential formulation of the
    variational problem.

    After defining an object `SolutionSaver`, one has to initialise
    the object. This creates a file with `.xdmf` extension which points to the
    respective `.h5` file. At each time step the numerical solution might be
    saved with the method `save_solution(self, x, time)`. Depending on the
    option chosen, one can save each solution frame in a different file, or
    append it in the same one.
    All the settings should be passed while defining the objecct.
    The desctructor closes the file(s).

    Usage example:
    ```
    import XDMF_Handler
    ...
    xdmf_saver = XDMF_handler.SolutionSaver(...)
    # Save the initial solution.
    xdmf_saver.save_solution(u, t0)
    # Iterate on time.
    while t < T:
        # Save the current solution.
        xdmf_saver.save_solution(u, t)
    ```
    """

    def __init__(
        self,
        filename,
        mesh,
        Xs,
        names,
        symbols,
        save_multiple_files=False,
        method="",
    ):
        """__init__
        Constructor.

        Args:
            filename (str): name of the `.xdmf` file that will store the
            solution.
            mesh (MeshHandler): mesh of the geometry of the problem.
            Xs (FunctionSpace): function space of the solution, possibly vector-
            valued.
            names (str): names of the functions' components that will be saved.
            symbols (str): symbols of the functions' component that will be
            saved, and displayed in the plots generated with e.g. ParaView.
            method (str, optional): Classical or. Defaults to "".
        """
        self.filename = filename
        self.mesh = mesh
        self.Xs = Xs
        self.names = names
        self.symbols = symbols
        self.xdmf = None
        self.method = method
        self.multiple_output = False

        if save_multiple_files == True:
            self.multiple_output = True
        elif save_multiple_files == False:
            self.multiple_output = False
            self.xdmf = XDMF_H.SolutionFileCreator(self.filename + ".xdmf")

        if self.method == "":
            self.method = "Classical"
        assert (
            len(self.names) == len(self.symbols) == len(self.Xs)
        ), "The number of function components or functions passed must equal the number of labels and names."
        assert (
            self.method == "Classical" or self.method == "Exponential"
        ), "The option 'method' can be set to 'Classical' or 'Exponential'. If left empty it is automatically set to 'Classical'."

    def append_solution(self, x, time):
        """append_solution append the solution provided at the given time to the
        file, open, containing the numerical solution output time series.

        It append in the `.xdmf` file the numerical solution `x` with time label
        `time` to the numerical solutions computed at previous time steps. It
        adds the names and the symbol labels provided to the constructor.

        Args:
            x (Function): solution.
            time (Float): current time.
        """
        for i in range(len(x)):
            out_function = Function((self.Xs)[i])

            if self.method == "Exponential":
                out_function.vector()[:] = np.exp(x[i].vector()[:])
            else:
                out_function.assign(x[i])

            out_function.rename((self.symbols)[i], (self.names)[i])
            self.xdmf.write(out_function, time)

        if MPI.comm_world.Get_rank() == 0:
            print("Solution at time {:.6f}".format(time), "appended.")

    def save_solution_per_timestep(self, x, time):
        """save_solution_per_timestep save the solution at the given time in a
        separate `.xdmf` file.

        It creates a new `.xdmf` file for each time step and writes the solution
        `x` with time label `time`. It adds the names and the symbol labels
        provided to the constructor.

        Args:
            x (Function): solution.
            time (float): current time.
        """
        # Create a new XDMF file for each time step
        xdmf_per_timestep = XDMF_H.SolutionFileCreator(
            f"{self.filename}_time_{time:.5f}.xdmf"
        )

        for i in range(len(x)):
            out_function = Function(self.Xs[i])

            if self.method == "Exponential":
                out_function.vector()[:] = np.exp(x[i].vector()[:])
            else:
                out_function.assign(x[i])

            out_function.rename(self.symbols[i], self.names[i])
            xdmf_per_timestep.write(out_function, time)

        # Close the file for this time step
        if MPI.comm_world.Get_rank() == 0:
            xdmf_per_timestep.close()
            print("Solution at time {:.6f}".format(time), "saved.")

    def save_solution(self, x, time):
        """save_solution save the solution in output. Its behaviour varies
        depending on the choice passede to the constructor.

        It appends the solution to the already existing time series or it
        creates a new file containing the solution frame.

        Args:
            x : numerical solution's components.
            time : current time.
        """
        assert len(x) == len(
            self.names
        ), "The number of function components or functions passed must equal the number of labels and names."

        if self.multiple_output == False:
            self.append_solution(x, time)
        elif self.multiple_output == True:
            self.save_solution_per_timestep(x, time)

    def __del__(self):
        """Destructor. It closes the temporay `.xdmf` file."""

        if (
            self.xdmf is not None
            and MPI.comm_world.Get_rank() == 0
            and self.multiple_output == False
        ):
            self.xdmf.close()
            print("File closed.")

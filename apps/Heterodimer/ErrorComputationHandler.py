#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# File handler containing utilites to compute errors during convergence tests
# against exact solutions, and to autmatically visualise them.
#
# ------------------------------------------------------------------------------
# Authors:
# Mattia Corti <mattia.corti@mail.polimi.it>
# Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fenics import *


# @TODO generalise, then merge with other error computing scripts.
# Ok di Mattia, dopo tesi.
class ComputeErrorsHeterodimer:
    """Class that manages the computation of the errors for the Heterodimer
    problem in the L2 and DG norm, as defined in Antonietti, Bonizzoni, Corti,
    Dall'Olio (2023). This can be an helpful tool to test the convergence rate
    of the scheme implemented.

    Example:
    In the `problemconvergence` function:
    ```
    def problemconvergence(filename, conv):
        errors = ErrComp.ComputeErrorsHeterodimer()

        for it in range(0, conv):
            problemsolver(filename, it, True, errors)

        errors.save_to_csv("errors_convergence_test")
    ```

    In the `problemsolver` function:
    ```
    def problemsolver(filename, iteration=0, conv=False, errors=False):
        ...
        while t < T:
            ...
            errors.compute_errors(x_c, x_q, iteration, t, it)
    ```
    """

    def __init__(self):
        """__init__ Initialise errors' tables.

        It creates an empty table with L2 and DG error.
        """

        # L2 and DG just for the last iteration.
        self.erros_L2_DG_final = pd.DataFrame(
            columns=["L2_c", "2DG_c", "3DG_c", "L2_q", "2DG_q", "3DG_q"]
        )

        # Energy norms.
        self.errors_energy_final = pd.DataFrame(
            columns=["2E_c", "3E_c", "2E_q", "3E_q"]
        )

    def initialise(self, X, mesh, h, n, param):
        """initialise function that retrieves data for error computations.

        The following data are retrieved from the user-defined parameter file:
        + the exact solution;
        + the polynomial degree of approximation;
        + the penalty parameter;
        + the final time of simulation;
        + the time step.
        Then the L2 and DG errors are initialised to 0.

        Args:
            X : functional space.
            mesh : mesh of the domain.
            h : cells' diameter sizes.
            n : face cells' normal vectors.
            param : parameter list.
        """
        # Functional space.
        self.X = X
        # Exact solutions.
        self.c_ex_expr = param["Convergence Test"][
            "Exact Solution: healthy proteins"
        ]
        self.q_ex_expr = param["Convergence Test"][
            "Exact Solution: misfolded proteins"
        ]
        # Mesh-related info.
        self.h = h
        self.h_avg = (2 * self.h("+") * self.h("-")) / (
            self.h("+") + self.h("-")
        )
        self.n = n
        self.mesh = mesh
        # Polynomial degree.
        self.deg = param["Spatial Discretization"]["Polynomial Degree"]
        # Scaling parameters.
        self.Tau = param["Scaling Parameters"]["Characteristic Time"]
        self.Xi = param["Scaling Parameters"]["Characteristic Concentration"]
        self.S = param["Scaling Parameters"]["Characteristic Length"]
        # Coefficients.
        # - production.
        self.k_0 = Constant(
            param["Model Parameters"]["Healthy proteins Producing Rate"]
            / (self.Tau * self.Xi)
        )
        # - reaction.
        self.k_1 = Constant(
            param["Model Parameters"]["Reaction Coefficient: healthy proteins"]
            / self.Tau
        )
        self.k_1_tilde = Constant(
            param["Model Parameters"][
                "Reaction Coefficient: misfolded proteins"
            ]
            / self.Tau
        )
        self.k_12 = Constant(
            param["Model Parameters"]["Reaction Coefficient: non-linear"]
            / (self.Tau * self.Xi)
        )
        # - diffusion.
        self.d_ext_c = Constant(
            param["Model Parameters"][
                "Extracellular diffusion: healthy proteins"
            ]
        )
        self.d_ext_q = Constant(
            param["Model Parameters"][
                "Extracellular diffusion: misfolded proteins"
            ]
        )
        # Size parameters.
        self.T = param["Temporal Discretization"]["Final Time"]
        self.dt = param["Temporal Discretization"]["Time Step"]
        self.it_tot = int(self.T / self.dt)
        self.N = self.T / self.dt

        # Penalty parameter.
        # - jump.
        self.gamma_0 = param["Spatial Discretization"][
            "Discontinuous Galerkin"
        ]["Penalty Parameter"]
        # - interior face.
        k_K = (1 + self.k_12) * (self.k_1 + self.k_1_tilde)
        d_K_c = self.d_ext_c
        d_K_q = self.d_ext_q

        self.gamma_I_c = conditional(
            avg(d_K_c) > k_K,
            self.gamma_0 * avg(d_K_c) * self.deg * self.deg / self.h_avg,
            self.gamma_0 * k_K * self.deg * self.deg / self.h_avg,
        )
        self.gamma_I_q = conditional(
            avg(d_K_q) > k_K,
            self.gamma_0 * avg(d_K_q) * self.deg * self.deg / self.h_avg,
            self.gamma_0 * k_K * self.deg * self.deg / self.h_avg,
        )
        # - boundary face.
        self.gamma_F_c = conditional(
            d_K_c > k_K,
            self.gamma_0 * d_K_c * self.deg * self.deg / h,
            self.gamma_0 * k_K * self.deg * self.deg / h,
        )
        self.gamma_F_q = conditional(
            d_K_q > k_K,
            self.gamma_0 * d_K_q * self.deg * self.deg / h,
            self.gamma_0 * k_K * self.deg * self.deg / h,
        )
        # - interior facets.
        self.theta_IP = param["Spatial Discretization"][
            "Discontinuous Galerkin"
        ]["Interior Penalty Parameter"]

        self.int_2DG_norm2_c = 0.0
        self.int_2DG_norm2_q = 0.0
        self.int_3DG_norm2_c = 0.0
        self.int_3DG_norm2_q = 0.0

    def compute_errors(self, c_k, q_k, convergence_iteration, time, it):
        """compute_errors function that compute the L2 and DG error of both
        function component.

        Since the DG norm involves and integral in time, this function should be
        called each time iteration. Automatically, at the last time iteration,
        this function evaluates the DG norm, and also the L2 norm for both
        function components at the last time istant.

        Args:
            c_k : numerical solution for the healthy proteins.
            q_k : numerical solution for the misfolded proteins.
            convergence_iteration : iteration number of the convergence test.
            time : current time.
            it : time iteration number.
        """

        # Evaluate the exact solutions at current time.
        c_ex = Expression(
            self.c_ex_expr, degree=int(self.deg + 4), t=time, domain=self.mesh
        )
        q_ex = Expression(
            self.q_ex_expr, degree=int(self.deg + 4), t=time, domain=self.mesh
        )

        # Compute the errors in L2-norm
        error_L2_c_k = self.errornorm_L2(c_ex, c_k)
        error_L2_q_k = self.errornorm_L2(q_ex, q_k)

        # Compute the errors in DG norm.
        error_2DG_c_k = self.errornorm_DG_c(c_ex, c_k, 2)
        error_3DG_c_k = self.errornorm_DG_c(c_ex, c_k, 3)
        error_2DG_q_k = self.errornorm_DG_q(q_ex, q_k, 2)
        error_3DG_q_k = self.errornorm_DG_q(q_ex, q_k, 3)

        # Update integrals for the energy norm.
        self.int_2DG_norm2_c = (
            self.int_2DG_norm2_c + self.dt * error_2DG_c_k * error_2DG_c_k
        )
        self.int_3DG_norm2_c = (
            self.int_3DG_norm2_c + self.dt * error_3DG_c_k * error_3DG_c_k
        )
        self.int_2DG_norm2_q = (
            self.int_2DG_norm2_q + self.dt * error_2DG_q_k * error_2DG_q_k
        )
        self.int_3DG_norm2_q = (
            self.int_3DG_norm2_q + self.dt * error_3DG_q_k * error_3DG_q_k
        )

        # Save the final-time errors.
        if abs(it - self.it_tot) < 0.5:
            # L2 and DG.
            errors_tmp_c = pd.DataFrame(
                {
                    "L2_c": error_L2_c_k,
                    "2DG_c": error_2DG_c_k,
                    "3DG_c": error_3DG_c_k,
                },
                index=[convergence_iteration],
            )
            errors_tmp_q = pd.DataFrame(
                {
                    "L2_q": error_L2_q_k,
                    "2DG_q": error_2DG_q_k,
                    "3DG_q": error_3DG_q_k,
                },
                index=[convergence_iteration],
            )
            errors_tmp_cq = pd.concat([errors_tmp_c, errors_tmp_q], axis=1)

            if convergence_iteration == 0:
                self.erros_L2_DG_final = errors_tmp_cq
            else:
                self.erros_L2_DG_final = pd.concat(
                    [self.erros_L2_DG_final, errors_tmp_cq]
                )

            # Energy.
            energy_2c = sqrt(error_L2_c_k**2 + self.int_2DG_norm2_c)
            energy_2q = sqrt(error_L2_q_k**2 + self.int_2DG_norm2_q)
            energy_3c = sqrt(error_L2_c_k**2 + self.int_3DG_norm2_c)
            energy_3q = sqrt(error_L2_q_k**2 + self.int_3DG_norm2_q)

            errors_tmp_energy_c = pd.DataFrame(
                {"2E_c": energy_2c, "3E_c": energy_3c},
                index=[convergence_iteration],
            )
            errors_tmp_energy_q = pd.DataFrame(
                {"2E_q": energy_2q, "3E_q": energy_3q},
                index=[convergence_iteration],
            )
            errors_tmp_energy_cq = pd.concat(
                [errors_tmp_energy_c, errors_tmp_energy_q], axis=1
            )

            if convergence_iteration == 0:
                self.errors_energy_final = errors_tmp_energy_cq
            else:
                self.errors_energy_final = pd.concat(
                    [self.errors_energy_final, errors_tmp_energy_cq]
                )

    def save_to_csv(self, name):
        """save_to_csv Save the errors computed in a csv file.

        Args:
            name : file root name.
        """
        self.errors_energy_final.to_csv(name + "_energy.csv")
        self.erros_L2_DG_final.to_csv(name + "_L2_DG.csv")

    def plot_errors_in_space(self, norm_type, normalised=False):
        """plot_errors_in_space plots the error convergence rate in a png file.

        Args:
            norm_type : available options are "Energy" or "L2DG".
            normalised (bool, optional): errors normalised. Defaults to False.
        """
        assert (
            norm_type == "Energy" or norm_type == "L2DG"
        ), "Norm types allowed are: 'L2DG' and 'Energy'. Please choose one."

        dpi_opt = 150
        # Create a plot with a semilogy axis.
        fig, ax = plt.subplots()
        # Set plot labels and title.
        ax.set_xlabel("Mesh space element size, $h$")
        ax.set_ylabel("Log(Error)")

        ax.set_yscale("log")
        ax.set_xscale("log")

        # Enable the minor grid lines.
        ax.yaxis.grid(True, which="minor", linestyle="-", linewidth=0.2)
        ax.xaxis.grid(True, which="minor", linestyle="-", linewidth=0.2)
        ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5)
        ax.xaxis.grid(True, which="major", linestyle="-", linewidth=0.5)

        iterations = range(1, len(self.erros_L2_DG_final) + 1)
        hh = [1 / (2**i) for i in iterations]

        # Add lines with slopes rates.
        rate = [1, 2, 3]
        # Normalize and add dashed lines with slopes 0, 1, and 2.
        slope_0 = np.power(10, 0) * np.ones(len(hh))
        slope_minus_a = np.power(10, 1) * np.array(hh) ** rate[0]
        slope_minus_b = np.power(10, 2) * np.array(hh) ** rate[1]
        slope_minus_c = np.power(10, 1) * np.array(hh) ** rate[2]

        if normalised == True:
            # Plot the normalised dashed lines.
            ax.plot(
                hh,
                slope_0 / slope_0[0],
                linestyle="-",
                label="$h^0$",
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                hh,
                slope_minus_a / slope_minus_a[0],
                linestyle="--",
                label=f"$h^-{rate[0]}$",
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                hh,
                slope_minus_b / slope_minus_b[0],
                linestyle="-.",
                label=f"$h^-{rate[1]}$",
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                hh,
                slope_minus_c / slope_minus_c[0],
                linestyle=":",
                label=f"$h^-{rate[2]}$",
                color="black",
                linewidth=1.0,
            )

        else:
            # Plot the dashed lines.
            ax.plot(
                hh,
                slope_0,
                linestyle="-",
                label="$h^0$",
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                hh,
                slope_minus_a,
                linestyle="--",
                label=f"$h^-{rate[0]}$",
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                hh,
                slope_minus_b,
                linestyle="-.",
                label=f"$h^-{rate[1]}$",
                color="black",
                linewidth=1.0,
            )
            ax.plot(
                hh,
                slope_minus_c,
                linestyle=":",
                label=f"$h^-{rate[2]}$",
                color="black",
                linewidth=1.0,
            )

        if norm_type == "L2DG":
            if normalised == True:
                ax.set_title("Normalised Convergence Error: $L^2, DG$")

                # Normalize each error by its value in the first iteration.
                normalised_errors = (
                    self.erros_L2_DG_final / self.erros_L2_DG_final.iloc[0]
                )

                # Plot each normalised error as a function of iteration on a logarithmic scale.
                for column in normalised_errors.columns:
                    ax.plot(
                        hh,
                        normalised_errors[column].to_numpy(),
                        marker="o",
                        label=column,
                    )
            else:  # if not normalised
                ax.set_title("Convergence Error: $L^2, DG$")
                # Plot each normalised error as a function of iteration on a logarithmic scale.
                for column in self.erros_L2_DG_final.columns:
                    ax.plot(
                        hh,
                        self.erros_L2_DG_final[column].to_numpy(),
                        marker="o",
                        label=column,
                    )
        else:  # if norm_type == "Energy"
            if normalised == True:
                ax.set_title("Normalised Convergence Error: eps")

                # Normalize each error by its value in the first iteration.
                normalised_errors = (
                    self.errors_energy_final / self.errors_energy_final.iloc[0]
                )

                # Plot each normalised error as a function of iteration on a logarithmic scale.
                for column in normalised_errors.columns:
                    ax.plot(
                        hh,
                        normalised_errors[column].to_numpy(),
                        marker="o",
                        label=column,
                    )
            else:
                ax.set_title("Convergence Error: eps")

                # Plot each normalised error as a function of iteration on a logarithmic scale.
                for column in self.errors_energy_final.columns:
                    ax.plot(
                        hh,
                        self.errors_energy_final[column].to_numpy(),
                        marker="o",
                        label=column,
                    )

        # Change the y-axis tick labels.
        ax.set_xticklabels(np.log(np.array(hh)))

        # Add a legend.
        ax.legend()

        if norm_type == "L2DG":
            if normalised == True:
                # Save the semilogy plot of normalised errors as a PNG file.
                plt.savefig("convergence_error_L2DG.png", dpi=dpi_opt)
            else:
                # Save the semilogy plot of normalised errors as a PNG file.
                plt.savefig(
                    "convergence_error_L2DG_normalised.png", dpi=dpi_opt
                )

        else:
            if normalised == True:
                # Save the semilogy plot of normalised errors as a PNG file.
                plt.savefig("convergence_error_energy.png", dpi=dpi_opt)
            else:
                # Save the semilogy plot of normalised errors as a PNG file.
                plt.savefig(
                    "convergence_error_energy_normalised.png", dpi=dpi_opt
                )

    def errornorm_L2(self, u_ex, u_k):
        tmp1 = dot((u_ex - u_k), (u_ex - u_k)) * dx
        error_L2 = sqrt(assemble(tmp1))
        return error_L2

    def errornorm_DG_c(self, u_ex, u_k, DG_norm_type):
        """errornorm_DG_c computes the DG error norm for the healthy component.

        If the DG type selected is "2", the expression reads:
        ||u||_DG = ||\sqrt{D} \grad{u}||_{L^2(\Omega)} + ||\sqrt{\gamma} \jump{u}||_{L^2(\Omega)}

        else if the DG type selected is "3", the expression reads:
        |||u|||_DG = ||u||_DG + ||\gamma^{-1/2} \avg{D} \nabla{u}||_{L^2(\Omega)}

        Args:
            u_ex : exact solution.
            u_k : numerical solution.
            DG_norm_type (int): DG norm type.

        Returns:
            Error norm
        """
        assert (
            DG_norm_type == 2 or DG_norm_type == 3
        ), "DG norm type not available."

        # Initialise.
        error_DG = 0.0

        # Error in DG norm: ||u||_DG = ||\sqrt{D} \grad{u}||_{L^2(\Omega)} + ||\sqrt{\gamma} \jump{u}||_{L^2(\Omega)}
        tmp1 = self.d_ext_c * dot(grad(u_ex - u_k), grad(u_ex - u_k)) * dx
        tmp2 = (
            self.gamma_I_c
            * dot(jump(u_k - u_ex, self.n), jump(u_k - u_ex, self.n))
            * dS
            + self.gamma_F_c * (u_k - u_ex) ** 2 * ds
        )
        error_DG = sqrt(assemble(tmp1)) + sqrt(assemble(tmp2))

        # Error in DG norm: |||u|||_DG = ||u||_DG + ||\gamma^{-1/2} \avg{D} \nabla{u}||_{L^2(\Omega)}
        if DG_norm_type == 3:
            tmp3 = (
                1
                / self.gamma_I_c
                * (avg(self.d_ext_c * grad(u_ex - u_k)) ** 2)
                * dS
                + 1
                / self.gamma_F_c
                * (self.d_ext_c * grad(u_ex - u_k)) ** 2
                * ds
            )
            error_DG = error_DG + sqrt(assemble(tmp3))

        return error_DG

    def errornorm_DG_q(self, u_ex, u_k, DG_norm_type):
        """errornorm_DG computes the DG error norm for the misfolded component.

        If the DG type selected is "2", the expression reads:
        ||u||_DG = ||\sqrt{D} \grad{u}||_{L^2(\Omega)} + ||\sqrt{\gamma} \jump{u}||_{L^2(\Omega)}

        else if the DG type selected is "3", the expression reads:
        |||u|||_DG = ||u||_DG + ||\gamma^{-1/2} \avg{D} \nabla{u}||_{L^2(\Omega)}

        Args:
            u_ex : exact solution.
            u_k : numerical solution.
            DG_norm_type (int): DG norm type.

        Returns:
            Error norm
        """
        assert (
            DG_norm_type == 2 or DG_norm_type == 3
        ), "DG norm type not available."

        # Initialise.
        error_DG = 0.0

        # Error in DG norm: ||u||_DG = ||\sqrt{D} \grad{u}||_{L^2(\Omega)} + ||\sqrt{\gamma} \jump{u}||_{L^2(\Omega)}
        tmp1 = self.d_ext_q * dot(grad(u_ex - u_k), grad(u_ex - u_k)) * dx
        tmp2 = (
            self.gamma_I_q
            * dot(jump(u_k - u_ex, self.n), jump(u_k - u_ex, self.n))
            * dS
            + self.gamma_F_q * (u_k - u_ex) ** 2 * ds
        )
        error_DG = sqrt(assemble(tmp1)) + sqrt(assemble(tmp2))

        # Error in DG norm: |||u|||_DG = ||u||_DG + ||\gamma^{-1/2} \avg{D} \nabla{u}||_{L^2(\Omega)}
        if DG_norm_type == 3:
            tmp3 = (
                1
                / self.gamma_I_q
                * (avg(self.d_ext_q * grad(u_ex - u_k)) ** 2)
                * dS
                + 1
                / self.gamma_F_q
                * (self.d_ext_q * grad(u_ex - u_k)) ** 2
                * ds
            )
            error_DG = error_DG + sqrt(assemble(tmp3))

        return error_DG

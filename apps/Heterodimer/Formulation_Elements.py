#!/usr/bin/env python3
#
################################################################################
# NeuroFEM project - 2023
#
# Helper file containing the expressions of the variational formulation of the
# Heterodimer model. Please refer to Antonietti, Bonizzoni, Corti, Dall'Olio
# (2023).
#
# ------------------------------------------------------------------------------
# Author: Giacomo Lorenzon <giacomo.lorenzon@mail.polimi.it>.
################################################################################

import os
import sys

from fenics import *

cwd = os.getcwd()
sys.path.append(cwd)
sys.path.append(cwd + "/../../utilities")

# import NeuroFEM utilites.
import HDF5Handler


################################################################################
# Variational formulation for the Heterodimer problem.
################################################################################
class HeterodimerVariationalFormulation:
    """
    This class provides the user an easy interface to initialise the variational
    formulation of the Heterodimer problem, as shown in Antonietti, Bonizzoni,
    Corti, Dall'Olio (2023).

    The constructor asks for the functional space, the measures defined along
    the boundary, the cells' diameters and normal vectors of their faces. This
    way the mesh is not passed, avoiding possibly heavy copy operations. This
    method must be called after reading the parameters of the problem instance,
    and thus after having saved them in a dictionary. This pieces of information
    must be passed as an argument to the constructor, so as to retrieve all the
    needed coefficients. If the diffusion of the problem is not isotropic, the
    tensor encoding anisotropy diffusion should be passed.
    """

    def __init__(self, X, mesh, ds1, ds2, h, n, param):
        """__init__ Constructor.

        It reads the data provided in input and retrieves all the coefficients
        needed thanks to the `param` dictionary. Then, it initialise the
        variational formulation with the DG-FEM contribution.
        It also considers the two different cases of isotropic and anisotropic
        diffusion. Please note that the variational formulation is not complete
        since the time dependent part is missing. This can be computed each time
        step by calling the method `compute(self, c_old, q_old, time)`.

        Args:
            X (FunctionSpace): function space of the, possibly vector-valued,
            function.
            mesh : domain geometry.
            ds1 (Measure): border measure with the first tag.
            ds2 (Measure): border measure with the second tag.
            h (CellDiameter): cells' diamters of the mesh.
            n (FacetNormal): facets' normal vectors of the mesh.
            param (PRM Dictionary): dictionary containing all the data parsed
            from the `.prm` file provided by the user.
        """
        # Retrieve data.

        # Polynomial degree.
        self.deg = param["Spatial Discretization"]["Polynomial Degree"]
        # Border measures.
        self.ds_vent = ds1
        self.ds_skull = ds2
        # Functinoal space.
        self.X = X
        # - test functions.
        self.w, self.v = TestFunctions(self.X)
        # - trial functions.
        self.c, self.q = TrialFunctions(self.X)
        # Mesh related.
        # - ceells' diameters.
        self.h = h
        self.h_avg = (2 * h("+") * h("-")) / (h("+") + h("-"))
        # - facets' normals.
        self.n = n
        # - time space.
        self.dt = param["Temporal Discretization"]["Time Step"]
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
        assert (
            param["Model Parameters"]["Isotropic Diffusion"] == "No"
            or param["Model Parameters"]["Isotropic Diffusion"] == "Yes"
        ), "Please choose between Yes or No"

        if param["Model Parameters"]["Isotropic Diffusion"] == "No":
            # Construction of tensorial space. Used for vectorial or tensorial
            # Diffusion tensors definition, if the diffusion is set anisotropic.
            X9 = TensorFunctionSpace(mesh, "DG", 0)
            # function over the mesh. Constnat vallues are enough.
            K = Function(X9)
            filenameK = param["Model Parameters"][
                "Axonal Diffusion Tensor File Name"
            ]
            Kname = param["Model Parameters"][
                "Name of Axonal Diffusion Tensor in File"
            ]
            d_axn_c = Constant(
                param["Model Parameters"]["Axonal diffusion: healthy proteins"]
            )
            d_ext_c = Constant(
                param["Model Parameters"][
                    "Extracellular diffusion: healthy proteins"
                ]
            )
            d_axn_q = Constant(
                param["Model Parameters"][
                    "Axonal diffusion: misfolded proteins"
                ]
            )
            d_ext_q = Constant(
                param["Model Parameters"][
                    "Extracellular diffusion: misfolded proteins"
                ]
            )
            self.K = HDF5Handler.import_tensor(filenameK, K, Kname)
            # HDF5Handler.save_tensor("anisotropy_tensor.h5", mesh, K, Kname)
            self.d_axn_c = d_axn_c / self.S
            self.d_ext_c = d_ext_c / self.S
            self.d_axn_q = d_axn_q / self.S
            self.d_ext_q = d_ext_q / self.S

        else:
            d_ext_c = Constant(
                param["Model Parameters"][
                    "Extracellular diffusion: healthy proteins"
                ]
            )
            d_ext_q = Constant(
                param["Model Parameters"][
                    "Extracellular diffusion: misfolded proteins"
                ]
            )
            self.K = False
            self.d_ext_c = d_ext_c / self.S
            self.d_ext_q = d_ext_q / self.S

        # Forcing terms.
        self.f_c = Datum(
            param["Model Parameters"][
                "Input for forcing term: healthy proteins"
            ],
            param["Model Parameters"]["Forcing Term: healthy proteins"],
            self.deg + 4,
        )
        self.f_q = Datum(
            param["Model Parameters"][
                "Input for forcing term: misfolded proteins"
            ],
            param["Model Parameters"]["Forcing Term: misfolded proteins"],
            self.deg + 4,
        )
        # Border data.
        # - healthy proteins.
        self.g_S_c = BorderDatum(
            param["Boundary Conditions"]["Healthy proteins"][
                "Input for Skull BCs"
            ],
            param["Boundary Conditions"]["Healthy proteins"]["Skull BCs Value"],
            param["Boundary Conditions"]["Healthy proteins"]["Skull BCs"],
            self.deg + 4,
            self.Xi,
        )
        self.g_V_c = BorderDatum(
            param["Boundary Conditions"]["Healthy proteins"][
                "Input for Ventricles BCs"
            ],
            param["Boundary Conditions"]["Healthy proteins"][
                "Ventricles BCs Value"
            ],
            param["Boundary Conditions"]["Healthy proteins"]["Ventricles BCs"],
            self.deg + 4,
            self.Xi,
        )
        # - misfolded proteins.
        self.g_S_q = BorderDatum(
            param["Boundary Conditions"]["Misfolded proteins"][
                "Input for Skull BCs"
            ],
            param["Boundary Conditions"]["Misfolded proteins"][
                "Skull BCs Value"
            ],
            param["Boundary Conditions"]["Misfolded proteins"]["Skull BCs"],
            self.deg + 4,
            self.Xi,
        )
        self.g_V_q = BorderDatum(
            param["Boundary Conditions"]["Misfolded proteins"][
                "Input for Ventricles BCs"
            ],
            param["Boundary Conditions"]["Misfolded proteins"][
                "Ventricles BCs Value"
            ],
            param["Boundary Conditions"]["Misfolded proteins"][
                "Ventricles BCs"
            ],
            self.deg + 4,
            self.Xi,
        )
        # Penalty parameter.
        # - jump.
        self.gamma_0 = param["Spatial Discretization"][
            "Discontinuous Galerkin"
        ]["Penalty Parameter"]
        # - interior face.
        k_K = (1 + self.k_12) * (self.k_1 + self.k_1_tilde)
        if self.K == True:
            d_K_c = self.d_ext_c + self.d_axn_c * self.K
            d_K_q = self.d_ext_q + self.d_axn_q * self.K
        else:
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

        # Theta method: integration in time.
        self.theta = param["Temporal Discretization"]["Theta-Method Parameter"]
        assert (
            abs(self.theta - 1.0) < DOLFIN_EPS
            or abs(self.theta - 0.5) < DOLFIN_EPS
            or abs(self.theta - 0.0) < DOLFIN_EPS
        ), "The only available options for the time integration scheme (Theta-method) are theta = 0.0, 0.5, 1.0. Please chose one."
        if abs(self.theta - 0.0) < DOLFIN_EPS:
            self.time_integration = "EE"  # Euler Explicit
        elif abs(self.theta - 0.5) < DOLFIN_EPS:
            self.time_integration = "CN"  # Crank Nicolson
        else:  # if abs(self.theta - 1.0) < DOLFIN_EPS:
            self.time_integration = "EI"  # Euler Implicit

        # Assemble the variational formulation.
        self.F_c = self.c * self.w * dx + self.theta * (
            self.a_c(self.c, self.w) + self.r_L(self.c, self.w)
        )
        self.F_q = self.q * self.v * dx + self.theta * (
            self.a_q(self.q, self.v) + self.r_L_tilde(self.q, self.v)
        )

        # If the scheme chosen is Discontinuos Galerkin, add these contributions.
        self.F_c = self.F_c + self.theta * (
            self.a_DG_interior_penalty_c(self.c, self.w)
            + self.a_DG_jump_penalty_c(self.c, self.w)
            - self.a_DG_c(self.c, self.w)
        )

        self.F_q = self.F_q + self.theta * (
            self.a_DG_interior_penalty_q(self.q, self.v)
            + self.a_DG_jump_penalty_q(self.q, self.v)
            - self.a_DG_q(self.q, self.v)
        )

    def compute(self, time, c_old, q_old, c_oold=False, q_oold=False):
        """compute This method computes the time-dependet part of the
        variational formulation of the problem. It adds the contribution to the
        pre-calculated costant (in time) part.

        Args:
            time (float): current nondimensional time instant.
            c_old (Function): solution component at the previous time instant.
            q_old (Function): solution component at the previous time instant.
            c_oold (Function): solution component at two time instant before.
            q_oold (Function): solution component at two time instant before.

        Returns:
            bilinear form, linear form: left hand side and right hand side of
            the variationa formulation of the heterodimer problem.
        """
        # Evaluate the forcing term functions at the current time instant.
        f_c = self.f_c.evaluate(time) * self.Xi / self.Tau
        f_q = self.f_q.evaluate(time) * self.Xi / self.Tau

        # Add the contribution to the pre-calculated variational formulation.
        F_c = (
            self.F_c
            - c_old * self.w * dx
            - self.theta * (self.force(f_c, self.w))
            - self.theta * (self.production(self.k_0, self.w))
        )
        F_q = (
            self.F_q
            - q_old * self.v * dx
            - self.theta * (self.force(f_q, self.v))
        )

        # Non linear term.
        if self.time_integration == "EI":
            F_c = F_c + self.k_12 * self.dt * self.q * c_old * self.v * dx
            F_q = F_q - self.k_12 * self.dt * q_old * self.c * self.w * dx
        elif self.time_integration == "CN":
            F_c = (
                F_c
                + self.k_12
                * self.dt
                * (1.5 * c_old - 0.5 * c_oold)
                * 0.5
                * (self.q + q_old)
                * self.v
                * dx
            )
            F_q = (
                F_q
                - self.k_12
                * self.dt
                * (1.5 * q_old - 0.5 * q_oold)
                * 0.5
                * (self.c + c_old)
                * self.w
                * dx
            )
        elif self.time_integration == "EE":
            F_c = (
                F_c + self.k_12 * self.dt * c_old * q_old * self.w * dx
            )  # @TODO do convergence tests
            F_q = (
                F_q - self.k_12 * self.dt * c_old * q_old * self.v * dx
            )  # @TODO do convergence tests

        if self.time_integration == "EI" or self.time_integration == "CN":
            # If using Discontinuos Galerkin, add terms.

            if self.g_S_c.BC_type == "Dirichlet":
                F_c = F_c + self.theta * (
                    self.a_DG_dirichlet_c(
                        self.c, self.g_S_c.evaluate(time), self.w, self.ds_skull
                    )
                )
            elif self.g_S_c.BC_type == "Neumann":
                F_c = F_c - self.theta * (
                    self.neumann_c(
                        self.c, self.g_S_c.evaluate(time), self.w, self.ds_skull
                    )
                )

            if self.g_V_c.BC_type == "Dirichlet":
                F_c = F_c + self.theta * (
                    self.a_DG_dirichlet_c(
                        self.c, self.g_V_c.evaluate(time), self.w, self.ds_vent
                    )
                )
            elif self.g_V_c.BC_type == "Neumann":
                F_c = F_c - self.theta * (
                    self.neumann_c(
                        self.c, self.g_V_c.evaluate(time), self.w, self.ds_vent
                    )
                )

            if self.g_S_q.BC_type == "Dirichlet":
                F_q = F_q + self.theta * (
                    self.a_DG_dirichlet_q(
                        self.q, self.g_S_q.evaluate(time), self.v, self.ds_skull
                    )
                )
            elif self.g_S_q.BC_type == "Neumann":
                F_q = F_q - self.theta * (
                    self.neumann_q(
                        self.q, self.g_S_q.evaluate(time), self.v, self.ds_skull
                    )
                )

            if self.g_V_q.BC_type == "Dirichlet":
                F_q = F_q + self.theta * (
                    self.a_DG_dirichlet_q(
                        self.q, self.g_V_q.evaluate(time), self.v, self.ds_vent
                    )
                )
            elif self.g_V_q.BC_type == "Neumann":
                F_q = F_q - self.theta * (
                    self.neumann_q(
                        self.q, self.g_V_q.evaluate(time), self.v, self.ds_vent
                    )
                )

        # Theta - method contribution: known terms.
        if self.time_integration == "EE" or self.time_integration == "CN":
            # Evaluate the forcing term functions at the previous time instant.
            f_c_prev = self.f_c.evaluate(time - self.dt) * self.Xi / self.Tau
            f_q_prev = self.f_q.evaluate(time - self.dt) * self.Xi / self.Tau

            # Forcing term.
            F_c = (
                F_c
                - (1 - self.theta) * self.force(f_c_prev, self.w)
                - (1 - self.theta) * self.production(self.k_0, self.w)
            )
            F_q = F_q - (1 - self.theta) * self.force(f_q_prev, self.v)

            # Linear reaction term.
            F_c = F_c + (1 - self.theta) * self.r_L(c_old, self.w)
            F_q = F_q + (1 - self.theta) * self.r_L_tilde(q_old, self.v)

            # Bilinear term.
            F_c = F_c + (1 - self.theta) * self.a_c(c_old, self.w)
            F_q = F_q + (1 - self.theta) * self.a_q(q_old, self.v)

            # If the scheme chosen is Discontinuos Galerkin, add these contributions.
            if self.g_S_c.BC_type == "Dirichlet":
                F_c = F_c + (1 - self.theta) * (
                    self.a_DG_dirichlet_c(
                        c_old,
                        self.g_S_c.evaluate(time - self.dt),
                        self.w,
                        self.ds_skull,
                    )
                )
            elif self.g_S_c.BC_type == "Neumann":
                F_c = F_c - (1 - self.theta) * (
                    self.neumann_c(
                        c_old,
                        self.g_S_c.evaluate(time - self.dt),
                        self.w,
                        self.ds_skull,
                    )
                )

            if self.g_V_c.BC_type == "Dirichlet":
                F_c = F_c + (1 - self.theta) * (
                    self.a_DG_dirichlet_c(
                        c_old,
                        self.g_V_c.evaluate(time - self.dt),
                        self.w,
                        self.ds_vent,
                    )
                )
            elif self.g_V_c.BC_type == "Neumann":
                F_c = F_c - (1 - self.theta) * (
                    self.neumann_c(
                        c_old,
                        self.g_V_c.evaluate(time - self.dt),
                        self.w,
                        self.ds_vent,
                    )
                )

            if self.g_S_q.BC_type == "Dirichlet":
                F_q = F_q + (1 - self.theta) * (
                    self.a_DG_dirichlet_q(
                        q_old,
                        self.g_S_q.evaluate(time - self.dt),
                        self.v,
                        self.ds_skull,
                    )
                )
            elif self.g_S_q.BC_type == "Neumann":
                F_q = F_q - (1 - self.theta) * (
                    self.neumann_q(
                        q_old,
                        self.g_S_q.evaluate(time - self.dt),
                        self.v,
                        self.ds_skull,
                    )
                )

            if self.g_V_q.BC_type == "Dirichlet":
                F_q = F_q + (1 - self.theta) * (
                    self.a_DG_dirichlet_q(
                        q_old,
                        self.g_V_q.evaluate(time - self.dt),
                        self.v,
                        self.ds_vent,
                    )
                )
            elif self.g_V_q.BC_type == "Neumann":
                F_q = F_q - (1 - self.theta) * (
                    self.neumann_q(
                        q_old,
                        self.g_V_q.evaluate(time - self.dt),
                        self.v,
                        self.ds_vent,
                    )
                )

            F_c = F_c + (1 - self.theta) * (
                self.a_DG_interior_penalty_c(c_old, self.w)
                + self.a_DG_jump_penalty_c(c_old, self.w)
                - self.a_DG_c(c_old, self.w)
            )

            F_q = F_q + (1 - self.theta) * (
                self.a_DG_interior_penalty_q(q_old, self.v)
                + self.a_DG_jump_penalty_q(q_old, self.v)
                - self.a_DG_q(q_old, self.v)
            )

        F = F_c + F_q

        return lhs(F), rhs(F)

    def a_c(self, u, phi):
        return self.dt * self.D_c(grad(u), grad(phi)) * dx

    def a_q(self, u, phi):
        return self.dt * self.D_q(grad(u), grad(phi)) * dx

    def a_DG_interior_penalty_c(self, u, phi):
        return (
            self.dt
            * (-self.theta_IP)
            * self.D_c(avg(grad(phi)), jump(u, self.n))
            * dS
        )

    def a_DG_interior_penalty_q(self, u, phi):
        return (
            self.dt
            * (-self.theta_IP)
            * self.D_q(avg(grad(phi)), jump(u, self.n))
            * dS
        )

    def a_DG_jump_penalty_c(self, u, phi):
        return (
            self.dt
            * self.gamma_I_c
            * dot(jump(u, self.n), jump(phi, self.n))
            * dS
        )

    def a_DG_jump_penalty_q(self, u, phi):
        return (
            self.dt
            * self.gamma_I_q
            * dot(jump(u, self.n), jump(phi, self.n))
            * dS
        )

    def a_DG_c(self, u, phi):
        return self.dt * self.D_c(avg(grad(u)), jump(phi, self.n)) * dS

    def a_DG_q(self, u, phi):
        return self.dt * self.D_q(avg(grad(u)), jump(phi, self.n)) * dS

    def a_DG_dirichlet_c(self, u, f1, phi, ds1):
        return (
            self.dt * self.gamma_F_c * (u - f1) * phi * ds1
            - self.dt * self.theta_IP * self.D_c(grad(phi), u * self.n) * ds1
            + self.dt * self.theta_IP * self.D_c(grad(phi), f1 * self.n) * ds1
            - self.dt * self.D_c(grad(u), phi * self.n) * ds1
        )

    def a_DG_dirichlet_q(self, u, f1, phi, ds1):
        return (
            self.dt * self.gamma_F_q * (u - f1) * phi * ds1
            - self.dt * self.theta_IP * self.D_q(grad(phi), u * self.n) * ds1
            + self.dt * self.theta_IP * self.D_q(grad(phi), f1 * self.n) * ds1
            - self.dt * self.D_q(grad(u), phi * self.n) * ds1
        )

    def neumann_c(self, u, g_N, phi, ds1):
        return self.dt * self.d_ext_c * g_N * phi * ds1

    def neumann_q(self, u, g_N, phi, ds1):
        return self.dt * self.d_ext_q * g_N * phi * ds1

    def r_L(self, u, phi):
        return self.k_1 * self.dt * u * phi * dx

    def r_L_tilde(self, u, phi):
        return self.k_1_tilde * self.dt * u * phi * dx

    def force(self, f, phi):
        return self.dt * f * phi * dx

    def production(self, f, phi):
        return self.dt * f * phi * dx

    def D_c(self, u, phi):
        tmp = self.d_ext_c * dot(u, phi)
        if self.K == True:
            tmp = tmp + self.d_axn_c * dot(dot(self.K, u), phi)
        return tmp

    def D_q(self, u, phi):
        tmp = self.d_ext_q * dot(u, phi)
        if self.K == True:
            tmp = tmp + self.d_axn_q * dot(dot(self.K, u), phi)
        return tmp

    def initial_conditions_constructor(
        self, param, X, c_old, q_old, time, conv
    ):
        """initial_conditions_constructor This function allows flexible
        initialisation of the initial conditions either as constants, based on
        an exact solution, or by importing them from an HDF5 file, depending on
        the simulation scenario and the parameters provided.

        The function reads the "Initial condition" from the parameters and
        initialise the variable with the expression provided. The numerical
        solution will be tested against it. Both the costant case and the exact
        expression is interpolated on the function space X. The user can also
        ask to import inital conditions from file by selecting in the parameters
        file "Initial Condition from File" = "Yes". In this case the utility
        "ImportICfromFile" reads the intial conditions from the file named in
        the parameters file "Initial Condition File Name".

        Args:
            param (Dictionary): dictionary containing simulation parameters.
            X (FunctionSpace): finite element function space where the initial
            condition will be defined.
            c_old (Function): function object to store the initial condition.
            conv (Bool): boolean flag indicating whether a convergence test is being
            conducted.
            time (Float): current time in the simulation.
            conv (bool): a convergence test is carried out.

        Returns:
            c_old (Function): function object representing the constructed initial
            conditions.
        """

        # If a convergence test is carried out, evaluate the exact solution at
        # time istant t=0 the exact solution.
        if conv == True:
            # Initial condition.
            c_IC = param["Convergence Test"]["Exact Solution: healthy proteins"]
            c_0 = Expression(
                c_IC,
                degree=int(
                    param["Spatial Discretization"]["Polynomial Degree"]
                ),
                t=time,
            )
            q_IC = param["Convergence Test"][
                "Exact Solution: misfolded proteins"
            ]
            q_0 = Expression(
                q_IC,
                degree=int(
                    param["Spatial Discretization"]["Polynomial Degree"]
                ),
                t=time,
            )

            c_old = interpolate(c_0, X)
            q_old = interpolate(q_0, X)

        else:
            if param["Model Parameters"]["Initial Condition from File"] == "No":
                # Initial condition.
                c_IC = param["Model Parameters"][
                    "Initial Condition: healthy proteins"
                ]
                c_0 = Expression(
                    c_IC,
                    degree=int(
                        param["Spatial Discretization"]["Polynomial Degree"]
                    ),
                    t=time,
                )
                q_IC = param["Model Parameters"][
                    "Initial Condition: misfolded proteins"
                ]
                q_0 = Expression(
                    q_IC,
                    degree=int(
                        param["Spatial Discretization"]["Polynomial Degree"]
                    ),
                    t=time,
                )

                c_old = interpolate(c_0, X)
                q_old = interpolate(q_0, X)

            elif (
                param["Model Parameters"]["Initial Condition from File"]
                == "Yes"
            ):
                c_old = HDF5Handler.import_IC_from_file(
                    param["Model Parameters"]["Initial Condition File Name"],
                    c_old,
                    param["Model Parameters"][
                        "Name of IC Function in File: healthy proteins"
                    ],
                )
                q_old = HDF5Handler.import_IC_from_file(
                    param["Model Parameters"]["Initial Condition File Name"],
                    q_old,
                    param["Model Parameters"][
                        "Name of IC Function in File: misfolded proteins"
                    ],
                )

        return c_old, q_old


################################################################################
# Data management.
#
# @TODO Merge and generalise utilities/BCs_handler.py
#
# Ok di Mattia. Dopo tesi.
################################################################################
class Datum:
    def __init__(self, datum_type, datum_value, deg=1):
        if (
            datum_type == "Constant"
            or datum_type == "Expression"
            or datum_type == "File"
        ):
            self.type = datum_type
            if datum_type == "Expression":
                self.deg = int(deg)
            else:
                self.deg = 1
        else:
            print("The choice of datum type provided is not available.")
            sys.exit(0)  # TODO throw a more meaningful error.

        self.value = datum_value

    def evaluate(self, time=0.0):
        if self.type == "Constant":
            datum = Constant(self.value)
        elif self.type == "Expression":
            datum = Expression(self.value, degree=self.deg, t=time)
        return datum


class BorderDatum(Datum):
    def __init__(self, datum_type, datum_value, BC_type, deg, scale=1.0):
        super().__init__(datum_type, datum_value, deg)
        self.BC_type = BC_type
        self.scale = scale

    def evaluate(self, time=0.0):
        return super().evaluate(time=time) * self.scale

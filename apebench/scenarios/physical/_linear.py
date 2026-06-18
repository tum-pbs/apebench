import exponax as ex

from ..._base_scenario import BaseScenario


class Linear(BaseScenario):
    domain_extent: float = 1.0
    dt: float = 0.1

    a_coefs: tuple[float, ...] = (0.0, -0.25, 0.0, 0.0, 0.0)
    coarse_proportion: float = 0.5

    def get_ref_stepper(self):
        return ex.stepper.generic.GeneralLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt,
            linear_coefficients=self.a_coefs,
        )

    def get_coarse_stepper(self) -> ex.BaseStepper:
        return ex.stepper.generic.GeneralLinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            domain_extent=self.domain_extent,
            num_points=self.num_points,
            dt=self.dt * self.coarse_proportion,
            linear_coefficients=self.a_coefs,
        )

    def get_scenario_name(self) -> str:
        active_indices = []
        for i, a in enumerate(self.a_coefs):
            if a != 0.0:
                active_indices.append(i)
        return f"{self.num_spatial_dims}d_phy_lin_{'_'.join(str(i) for i in active_indices)}"


class LinearSimple(Linear):
    linear_coef: float = -0.25
    linear_term_order: int = 1

    def __post_init__(self):
        self.a_coefs = (0.0,) * self.linear_term_order + (self.linear_coef,)


class Advection(Linear):
    advection_coef: float = -0.25

    def __post_init__(self):
        self.a_coefs = (0.0, self.advection_coef, 0.0, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_adv"


class Diffusion(Linear):
    diffusion_coef: float = 0.008

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, self.diffusion_coef, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_diff"


class AdvectionDiffusion(Linear):
    advection_coef: float = -0.25
    diffusion_coef: float = 0.008

    def __post_init__(self):
        self.a_coefs = (0.0, self.advection_coef, self.diffusion_coef, 0.0, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_adv_diff"


class Dispersion(Linear):
    dispersion_coef: float = 0.00025
    dt: float = 0.001  # Overwrite

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, 0.0, self.dispersion_coef, 0.0)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_disp"


class HyperDiffusion(Linear):
    hyper_diffusion_coef: float = -0.000075
    dt: float = 0.00001  # Overwrite

    def __post_init__(self):
        self.a_coefs = (0.0, 0.0, 0.0, 0.0, self.hyper_diffusion_coef)

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_hyp_diff"


class FirstFour(Linear):
    advection_coef: float = -2500.0
    diffusion_coef: float = 80.0
    dispersion_coef: float = 0.025
    hyp_diffusion_coef: float = -0.000075
    dt: float = 0.00001  # Overwrite

    def __post_init__(self):
        self.a_coefs = (
            0.0,
            self.advection_coef,
            self.diffusion_coef,
            self.dispersion_coef,
            self.hyp_diffusion_coef,
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_phy_four"

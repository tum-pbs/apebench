"""
Scenarios using the general nonlinear interface
"""

from ..._base_scenario import BaseScenario
from ...exponax import exponax as ex


class Nonlinear(BaseScenario):
    """
    Uses the single channel convection mode to not have channels grow with
    spatial dimensions.

    By default single-channel Burgers
    """

    gammas: tuple[float, ...] = (0.0, 0.0, 1.5, 0.0, 0.0)
    deltas: tuple[float, ...] = (0.0, -2.0, 0.0)

    num_substeps: int = 1

    coarse_proportion: float = 0.5

    order: int = 2
    dealiasing_fraction: float = 2 / 3
    num_circle_points: int = 16
    circle_radius: float = 1.0

    def __post_init__(self):
        pass

    def _build_stepper(self, gammas, deltas):
        substepped_gammas = tuple(g / self.num_substeps for g in gammas)
        substepped_deltas = tuple(d / self.num_substeps for d in deltas)

        substepped_stepper = ex.normalized.DifficultyGeneralNonlinearStepper(
            num_spatial_dims=self.num_spatial_dims,
            num_points=self.num_points,
            linear_difficulties=substepped_gammas,
            nonlinear_difficulties=substepped_deltas,
            order=self.order,
            dealiasing_fraction=self.dealiasing_fraction,
            num_circle_points=self.num_circle_points,
            circle_radius=self.circle_radius,
        )

        if self.num_substeps == 1:
            stepper = substepped_stepper
        else:
            stepper = ex.RepeatedStepper(substepped_stepper, self.num_substeps)

        return stepper

    def get_ref_stepper(self):
        return self._build_stepper(self.gammas, self.deltas)

    def get_coarse_stepper(self):
        return self._build_stepper(
            tuple(f * self.coarse_proportion for f in self.gammas),
            tuple(f * self.coarse_proportion for f in self.deltas),
        )

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_nonlin"

class BurgersSingleChannel(Nonlinear):
    delta_convection: float = -2.0
    gamma_diffusion: float = 1.5

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.gamma_diffusion, 0.0, 0.0)
        self.deltas = (0.0, self.delta_convection, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_burgers_sc"
    
class KortevegDeVries(Nonlinear):
    delta_convection: float = -2.0
    gamma_dispersion: float = -14.0
    gamma_hyperdiffusion: float = -9.0

    def __post_init__(self):
        self.gammas = (0.0, 0.0, 0.0, self.gamma_dispersion, self.gamma_hyperdiffusion)
        self.deltas = (0.0, self.delta_convection, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_kdv"
    
class KuramotoSivashinsky(Nonlinear):
    delta_gradient_norm: float = -6.0
    gamma_diffusion: float = -1.2  # Negative diffusion! producing energy
    gamma_hyperdiffusion: float = -15.0
    
    num_warmup_steps: int = 500  # Overwrite
    vlim: tuple[float, float] = (-6.5, 6.5)  # Overwrite

    report_metrics: str = "mean_nRMSE,mean_correlation"  # Overwrite

    def __post_init__(self):
        self.gammas = (0.0, 0.0, self.gamma_diffusion, 0.0, self.gamma_hyperdiffusion)
        self.deltas = (0.0, 0.0, self.delta_gradient_norm)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_ks"

class FisherKPP(Nonlinear):
    delta_quadratic: float = -0.02
    gamma_drag: float = 0.02
    gamma_diffusion: float = 0.2

    ic_config: str = "clamp;0.0;1.0;fourier;10;false;false"  # Overwrite

    def __post_init__(self):
        self.gammas = (self.gamma_drag, 0.0, self.gamma_diffusion)
        self.deltas = (self.delta_quadratic, 0.0, 0.0)

        super().__post_init__()

    def get_scenario_name(self) -> str:
        return f"{self.num_spatial_dims}d_diff_fisher"
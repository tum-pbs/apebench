import importlib.metadata

import exponax
import pdequinox
import trainax

from . import _scraper as scraper
from . import components, scenarios
from ._base_scenario import BaseScenario
from ._run import (
    get_experiment_name,
    melt_concat_from_list,
    melt_concat_loss_from_list,
    melt_concat_metrics_from_list,
    melt_concat_sample_rollouts_from_list,
    run_experiment,
    run_study,
    run_study_convenience,
)
from ._utils import (
    aggregate_gmean,
    check_for_nan,
    compute_pvalues_against_best,
    cumulative_aggregation,
    melt_data,
    melt_loss,
    melt_metrics,
    melt_sample_rollouts,
    read_in_kwargs,
    relative_by_config,
    split_train,
)

__version__ = importlib.metadata.version("apebench")

__all__ = [
    "exponax",
    "pdequinox",
    "get_experiment_name",
    "melt_concat_from_list",
    "melt_concat_loss_from_list",
    "melt_concat_metrics_from_list",
    "melt_concat_sample_rollouts_from_list",
    "run_experiment",
    "run_study_convenience",
    "run_study",
    "scenarios",
    "trainax",
    "melt_data",
    "melt_loss",
    "melt_metrics",
    "melt_sample_rollouts",
    "read_in_kwargs",
    "arch_extensions",
    "scraper",
    "BaseScenario",
    "aggregate_gmean",
    "relative_by_config",
    "split_train",
    "components",
    "check_for_nan",
    "cumulative_aggregation",
    "compute_pvalues_against_best",
]

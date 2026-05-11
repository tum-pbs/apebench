from typing import Literal, Union

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gmean

from ._base_scenario import BaseScenario

BASE_NAMES = [
    "seed",
    "scenario",
    "task",
    "net",
    "train",
    "scenario_kwargs",
]
BASE_NAMES_NO_TRAIN = [
    "seed",
    "scenario",
    "task",
    "net",
    "scenario_kwargs",
]


def melt_data(
    wide_data: pd.DataFrame,
    quantity_name: Union[str, list[str]],
    uniquifier_name: str,
    *,
    base_columns: list[str] = BASE_NAMES,
) -> pd.DataFrame:
    """
    Melt a wide APEBench result DataFrame into a long format suitable for
    visualization (e.g. with seaborn or plotly).

    **Arguments:**

    * `wide_data`: The wide DataFrame to melt, must contain `quantity_name` and
        `base_columns` as columns.
    * `quantity_name`: The name of the column(s) to melt.
    * `uniquifier_name`: The name of the column that will be used to uniquely
        identify the melted rows.
    * `base_columns`: The columns that should be kept as is in the melted
        DataFrame.

    **Returns:**

    * A long DataFrame with the same columns as `base_columns` and the melted
        `quantity_name`.
    """
    if isinstance(quantity_name, str):
        quantity_name = [
            quantity_name,
        ]
    data_melted = pd.wide_to_long(
        wide_data,
        stubnames=quantity_name,
        i=base_columns,
        j=uniquifier_name,
        sep="_",
    )
    data_melted = data_melted[quantity_name]
    data_melted = data_melted.reset_index()

    return data_melted


def melt_metrics(
    wide_data: pd.DataFrame,
    metric_name: Union[str, list[str]] = "mean_nRMSE",
) -> pd.DataFrame:
    """
    Melt the metrics from a wide DataFrame.
    """
    return melt_data(
        wide_data,
        quantity_name=metric_name,
        uniquifier_name="time_step",
    )


def melt_loss(wide_data: pd.DataFrame, loss_name: str = "train_loss") -> pd.DataFrame:
    """
    Melt the loss from a wide DataFrame.
    """
    return melt_data(
        wide_data,
        quantity_name=loss_name,
        uniquifier_name="update_step",
    )


def melt_sample_rollouts(
    wide_data: pd.DataFrame,
    sample_rollout_name: str = "sample_rollout",
) -> pd.DataFrame:
    """
    Melt the sample rollouts from a wide DataFrame.
    """
    return melt_data(
        wide_data,
        quantity_name=sample_rollout_name,
        uniquifier_name="sample_index",
    )


def split_train(
    metric_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Decode the `train` column into `category`, `type`, and `rollout` columns.
    """
    metric_data["category"] = metric_data["train"].apply(lambda x: x.split(";")[0])
    metric_data["type"] = metric_data["category"].apply(
        lambda x: "sup" if x in ["one", "sup"] else "div"
    )
    metric_data["rollout"] = metric_data["train"].apply(
        lambda x: int(x.split(";")[1]) if x != "one" else 1
    )

    return metric_data


def aggregate_gmean(
    metric_data: pd.DataFrame,
    *,
    up_to: int = 100,
    grouping_cols: list[str] = BASE_NAMES,
) -> pd.DataFrame:
    """
    Aggregate an error rollout over time via the geometric mean.

    Args:

    * `metric_data`: The DataFrame to aggregate, must contain `grouping_cols`
        and `mean_nRMSE` as columns. When grouped by `grouping_cols`, the groups
        shall only contain values at different time steps.
    * `up_to`: The time step up to which to aggregate. (inclusive)
    * `grouping_cols`: The columns to group by.

    Returns:

    * A DataFrame with the new column `gmean_mean_nRMSE` containing the
        geometric mean of the `mean_nRMSE` values up to `up_to` for each group.
    """
    return (
        metric_data.query(f"time_step <= {up_to}")
        .groupby(grouping_cols)
        .agg(gmean_mean_nRMSE=("mean_nRMSE", gmean))
        .reset_index()
    )


def relative_by_config(
    data: pd.DataFrame,
    *,
    grouping_cols: list[str] = BASE_NAMES_NO_TRAIN,
    norm_query: str = "train == 'one'",
    value_col: str = "mean_nRMSE",
    suffix: str = "_rel",
) -> pd.DataFrame:
    def relativate_fn(sub_df):
        rel = sub_df.query(norm_query)[value_col]
        if len(rel) != 1:
            raise ValueError(
                f"Expected exactly one row to match {norm_query}, got {len(rel)}"
            )
        col = sub_df[value_col] / rel.values[0]
        sub_df[f"{value_col}{suffix}"] = col
        return sub_df

    return data.groupby(grouping_cols).apply(relativate_fn).reset_index(drop=True)


def read_in_kwargs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse the `scenario_kwargs` column of a DataFrame and add the parsed entries
    as new columns.

    Requires that the dictionary in `scenario_kwargs` has the same keys for all
    rows.
    """
    col = df["scenario_kwargs"].apply(eval)
    entries = list(col[0].keys())
    for entry in entries:
        df[entry] = col.apply(lambda x: x[entry])
    return df


def count_nan_trjs(trjs: jax.Array) -> int:
    """
    Computes the number of trajectories that contain at least one NaN value.
    """

    def has_nan(trj):
        if jnp.sum(jnp.isnan(trj)) > 0:
            return 1
        else:
            return 0

    mask = [has_nan(trj) for trj in trjs]

    return sum(mask)


def check_for_nan(scene: BaseScenario):
    """
    Check for NaNs in the train and test data of a scenario. Also checks the
    train and test data set produced by the coarse stepper if the scenario
    supports a correction mode. Raises an AssertionError if NaNs are found.
    """
    train_data = scene.get_train_data()

    train_num_nans = count_nan_trjs(train_data)
    assert (
        train_num_nans == 0
    ), f"Train data has {train_num_nans} trajectories with NaNs"

    del train_data

    test_data = scene.get_test_data()

    test_num_nans = count_nan_trjs(test_data)
    assert test_num_nans == 0, f"Test data has {test_num_nans} trajectories with NaNs"

    del test_data

    try:
        # Some scenarios might not support a correction mode
        train_data_coarse = scene.get_train_data_coarse()

        train_num_nans_coarse = count_nan_trjs(train_data_coarse)
        assert (
            train_num_nans_coarse == 0
        ), f"Train data coarse has {train_num_nans_coarse} trajectories with NaNs"

        del train_data_coarse

        test_data_coarse = scene.get_test_data_coarse()

        test_num_nans_coarse = count_nan_trjs(test_data_coarse)
        assert (
            test_num_nans_coarse == 0
        ), f"Test data coarse has {test_num_nans_coarse} trajectories with NaNs"

        del test_data_coarse
    except NotImplementedError:
        return


def cumulative_aggregation(
    df: pd.DataFrame,
    grouping_cols: Union[str, list[str]],
    rolling_col: str,
    agg_fn_name: str = "mean",
    prefix: str = "cum",
) -> pd.DataFrame:
    """
    Apply a cumulative aggregation to a DataFrame. This can be used to
    cumulatively aggregate a metric over time.

    If you are only interested in the aggregation over some rows (but not
    cumulatively), you can use the [`apebench.aggregate_gmean`][] function.

    **Arguments:**

    - `df`: The DataFrame to aggregate.
    - `grouping_cols`: The columns to group by. Supply a list of column names
      such that if `df.groupby(grouping_cols)` is called, the groups contain
      only different time steps (or whatever you want to aggregate over).
    - `rolling_col`: The column to aggregate. Ususally, this is the name of the
      metric in a long DataFrame.
    - `agg_fn_name`: The aggregation function to use. Must be one of `"mean"`
      or, `"gmean"` (geometric mean), or `"sum"`.

    **Returns:**

    - A DataFrame with the cumulatively aggregated columns added.

    !!! example

        Train a feedforward ConvNet to emulate advection and then display the
        mean nRMSE error rollout and the cumulative mean/gmean mean nRMSE error
        rollout. The only column that varies is the `"see"` column, so we group
        by that.

        ```python
        import apebench
        advection_scenario = apebench.scenarios.difficulty.Advection()

        data, trained_net = advection_scenario(num_seeds=3)

        metric_data = apebench.melt_metrics(data)

        metric_df = apebench.cumulative_aggregation(
            metric_df, "seed", "mean_nRMSE", agg_fn_name="mean"
        )
        metric_df = apebench.cumulative_aggregation(
            metric_df, "seed", "mean_nRMSE", agg_fn_name="gmean"
        )

        fig, ax = plt.subplots()
        sns.lineplot(
            data=metric_df, x="time_step",
            y="mean_nRMSE", ax=ax, label="original"
        )
        sns.lineplot(
            data=metric_df, x="time_step",
            y="cummean_mean_nRMSE", ax=ax, label="cummean"
        )
        sns.lineplot(
            data=metric_df, x="time_step",
            y="cumgmean_mean_nRMSE", ax=ax, label="cumgmean"
        )
        ```

        ![Metric Rollout absolutely and cumulatively aggregated via mean and
        gmean](https://github.com/user-attachments/assets/ba342f26-a6ef-4a67-8a2c-2b1d94ddfde1)
    """
    agg_fn = {
        "mean": np.mean,
        "gmean": stats.gmean,
        "sum": np.sum,
    }[agg_fn_name]
    return df.groupby(grouping_cols, observed=True, group_keys=False).apply(
        lambda x: x.assign(
            **{
                f"{prefix}{agg_fn_name}_{rolling_col}": x[rolling_col]
                .expanding()
                .apply(agg_fn)
            }
        )
    )


def compute_pvalues_against_best(
    df: pd.DataFrame,
    grouping_cols: list[str],
    sorting_cols: list[str],
    value_col: str,
    performance_indicator="mean",
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    equal_var: bool = True,
    pivot: bool = False,
):
    """
    Performs a t-test of the best configuration in a group against all other
    configurations and returns the p-values. "Best" is defined as the
    configuration with the lowest aggregated value (typically the mean).

    The returned DataFrame can be interpreted in that the best configuation has
    a p-value of 1.0 against itself (or 0.5 in case `alternative` is not
    `"two-sided"`) and a lower p-value against all other configurations. Only if
    the p-value against the other configurations is below the significance level
    (typically 0.05), the best configuration can be considered significantly
    better.

    **Arguments:**

    - `df`: The DataFrame to compute the p-values for.
    - `grouping_cols`: The columns to group by. Select the `grouping_cols` such
      that when `df.groupby(grouping_cols + sorting_cols)` is called, the groups
      only contains single samples to be used for the hypothesis test. For
      example, if you want to compare network architecture and training
      methodology for all time steps and investigated scenarios, then you would
      use `grouping_cols = ["scenario", "time_step"]` together with
      `sorting_cols = ["net", "train"]`.
    - `sorting_cols`: The columns to sort by. Once the dataframe is grouped by
      `grouping_cols`, the configuration out of all combinations in
      `sorting_cols` with the lowest aggregated value (typically the mean) is
      considered the best.
    - `value_col`: The column to use for the hypothesis test. Typically, this is
      the column with a test metric. (The default metric in APEBench is
      `mean_nRMSE`.)
    - `performance_indicator`: The aggregation to determine the best
      configuration. Typically, this is the mean, which is also what is checked
      with the hypothesis test.
    - `alternative`: The alternative hypothesis to test. Must be one of
      `"two-sided"`, `"less"`, or `"greater"`.
    - `equal_var`: Whether to assume equal variance in the two samples. (This is
      a parameter of the t-test.)
    - `pivot`: Whether to pivot the DataFrame to have the p-values in a matrix
        form. This is useful for directly comparing across the `sorting_cols`.

    See also [this SciPy
    page](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html)
    for more details on the t-test.

    **Returns:**

    - A DataFrame with the p-values of the hypothesis test. If `pivot` is
      `True`,
        the DataFrame is pivoted such that the p-values are in a matrix form.


    !!! example

        Test for the significantly better architecture when emulating advection
        under `advection_gamma=2.0`.

        ```python
        CONFIGS = [
            {
                "scenario": "diff_adv",
                "task": "predict",
                "net": net,
                "train": "one",
                "start_seed": 0,
                "num_seeds": 10,
                "advection_gamma": 2.0,
            }
            for net in ["Conv;34;10;relu", "FNO;12;8;4;gelu", "UNet;12;2;relu"]
        ]

        metric_df, _, _, _ = apebench.run_study_convenience(
            CONFIGS, "conv_vs_fno_vs_unet"
        )

        p_value_df_pivoted = apebench.compute_pvalues_against_best(
            metric_df, ["time_step",], ["net",], "mean_nRMSE", pivot=True
        )

        print(
            p_value_df_pivoted.query(
                "time_step in [1, 5, 10, 50, 100, 200]"
            ).round(4).to_markdown()
        )
        ```

        |   time_step |   Conv;34;10;relu |   FNO;12;8;4;gelu |   UNet;12;2;relu |
        |------------:|------------------:|------------------:|-----------------:|
        |           1 |                 1 |            0.0017 |           0      |
        |           5 |                 1 |            0.0013 |           0      |
        |          10 |                 1 |            0.0007 |           0      |
        |          50 |                 1 |            0.0001 |           0.0457 |
        |         100 |                 1 |            0      |           0.0601 |
        |         200 |                 1 |            0      |           0.0479 |

        We can also visualize the p-values over time:
        ```python
        import seaborn as sns
        import matplotlib.pyplot as plt

        p_value_df = apebench.compute_pvalues_against_best(
            metric_df, ["time_step",], ["net",], "mean_nRMSE", "mean"
        )

        sns.lineplot(
            p_value_df,
            x="time_step",
            y="p_value",
            hue="net",
        )
        plt.hlines(0.05, 0, 200, linestyles="--", colors="black")

        plt.yscale("log")
        ```
        ![pvalue rollout over time](https://github.com/user-attachments/assets/4e99ed9b-735f-499c-8374-a7d9d94d1aca)

        Since the p-value of the ConvNet (the best architecture according to its
        mean performance) against the FNO is constantly below 0.05, we can
        confidently say the ConvNet is significantly better than the FNO.
        However, the UNet's performance's p-value is above 0.05 after ~50 time
        steps and thus, for this temporal horizon, we cannot reject the null
        hypothesis.
    """
    stats_df = (
        df.groupby(grouping_cols + sorting_cols, observed=True, group_keys=True)
        .agg(
            performance_indicator=(value_col, performance_indicator),
            mean=(value_col, "mean"),
            std=(value_col, "std"),
            count=(value_col, "count"),
        )
        .reset_index()
    )

    stats_df = stats_df.groupby(grouping_cols, observed=True, group_keys=False).apply(
        lambda x: x.assign(
            p_value=stats.ttest_ind_from_stats(
                mean1=x["mean"].values[x["performance_indicator"].argmin()],
                std1=x["std"].values[x["performance_indicator"].argmin()],
                nobs1=x["count"].values[x["performance_indicator"].argmin()],
                mean2=x["mean"].values,
                std2=x["std"].values,
                nobs2=x["count"].values,
                alternative=alternative,
                equal_var=equal_var,
            ).pvalue
        )
    )

    if not pivot:
        return stats_df

    p_value_df = stats_df.pivot(
        index=grouping_cols,
        columns=sorting_cols,
        values="p_value",
    )

    return p_value_df

# Main Arguments in APEBench

When running an APEBench scenario, its call signature accepts the following
keyword-based arguments

```python
# For example
scenario = apebench.scenarios.difficulty.Advection()

data, neural_stepper_s = scenario(
    task_config = "predict",
    network_config = "Conv;34;10;relu",
    train_config = "one",
    start_seed = 0,
    num_seeds = 1,
    remove_singleton_axis = True,
)
```

Similarly, if one used the experimental interface, the arguments are similar

```python
data, neural_stepper_s = apebench.run_experiment(
    scenario = "diff_adv",
    task = "predict",
    network = "Conv;34;10;relu",
    train = "one",
    start_seed = 0,
    num_seeds = 1,
)
```

In APEBench, the arguments that are set during the execution of a scenario are
the *dynamic arguments*, wheras the arguments used during the instantiation of a
scenario are the *static arguments*.

```python
scenario = apebench.scenarios.difficulty.Advection(
    # Static argument, used during instantiation/construction
    num_points=256
)
...
```

From the perspective of the experimental interface, however, they appear
similarly

```python
data, neural_stepper_s = apebench.run_experiment(
    scenario = "diff_adv",
    # Like before
    ...,
    # Static argument, used during instantiation/construction
    num_points=256
)
```

Internally, the experimental interface will instantiate the scenario with the
static arguments, and then call the scenario with the dynamic arguments.

## Overview of Arguments

### Scenario Arguments

- [`scenario`](#scenario)

### Dynamics Arguments

- [`task_config` or `task`](#task)
- [`network_config` or `network`](#network)
- [`train_config` or `train`](#train)
- [`start_seed`](#start_seed)
- [`num_seeds`](#num_seeds)
- [`remove_singleton_axis`](#remove_singleton_axis)

### Static Arguments

- Setting up the discretization:
    - [`num_spatial_dims`](#num_spatial_dims)
    - [`num_points`](#num_points)
- Abstract information about the problem:
    - [`num_channels`](#num_channels)
- Settings for both training and testing:
    - [`ic_config`](#ic_config)
    - [`num_warmup_steps`](#num_warmup_steps)
- Setting up the training:
    - [`num_train_samples`](#num_train_samples)
    - [`train_temporal_horizon`](#train_temporal_horizon)
    - [`train_seed`](#train_seed)
- For testing:
    - [`num_test_samples`](#num_test_samples)
    - [`test_temporal_horizon`](#test_temporal_horizon)
    - [`test_seed`](#test_seed)
- For the training configuration:
    - [`optim_config`](#optim_config)
    - [`batch_size`](#batch_size)
- Information for inspection:
    - [`num_trjs_returned`](#num_trjs_returned)
    - [`record_loss_every`](#record_loss_every)
    - [`vlim`](#vlim)
    - [`report_metrics`](#report_metrics)
    - [`callbacks`](#callbacks)

## Static Arguments for only some Scenarios

- For scenarios in physical mode:
    - [`domain_extent`](#domain_extent)
    - [`dt`](#dt)
- For scenarios with nonlinearities:
    - [`num_substeps`](#num_substeps)
    - [`order`](#order)
    - [`dealiasing_fraction`](#dealiasing_fraction)
    - [`num_circle_points`](#num_circle_points)
    - [`circle_radius`](#circle_radius)
- For scenarios with a simple "defective solver" correction scenario:
    - [`coarse_proportion`](#coarse_proportion)

Moreover, each scenario contains its respective constitutive parameters which are listed in their respective documentation.


## Arguments in Detail

### `scenario`

The identifier string of the scenario. Must be one of the keys of
[`apebench.scenarios.scenario_dict`][]. If a custom extended scenario is to be
used, it must be first registered in the `scenario_dict` (adding it to the
dict).

!!! note

    This argument only exists if the interface via
    [`apebench.run_experiment`][], [`apebench.run_study`][], or
    [`apebench.run_study_convenience`][] is used. Otherwise, it is implicitly given by the scenario being instantiated.

### `task`

Describes how the neural stepper is composed. For a full prediction via
`"predict"`, the neural network is supposed to fully replace the numerical
simulator. For a correction setup via `"correct;<XX>"`, there is a coarse solver
component. This depends on the `get_coarse_stepper()` method of the specific
scenario. In the default settings of APEBench, this is a "defective solver" that
is similar to the reference simulator but only performs a fraction of the time
step $\Delta t$. More sophisticated coarse steppers like based on differences in
numerical methods or resolution are possible. The `<XX>` is a placeholder for
the type of correction. Currently supported are:

- `"correct;sequential"`: The network is called on the output of the coarse
  stepper
- `"correct;parallel"`: The network and the coarse stepper are called on the same
  previous state, and the results are added up.
- `correct;sequential_with_bypass"`: Similar to `"correct;sequential"`, but the
  output of the coarse stepper is also bypassed over the network and added to
  its output.

### `network`

Must be a configuration string that matches an entry of the
[`apebench.components.architecture_dict`][]. For example, `"Conv;34;10;relu"`
yields a feedforward convolutional neural netork with `34` hidden channels over
`10` hidden layers, with ReLU activation function.

!!! tip

    With an instantiated scenario, one can access the number of trainable
    parameters and the receptive field via
    ```python
    adv_scenario_1d = apebench.scenarios.difficulty.Advection()

    adv_scenario_1d.get_parameter_count("Conv;23;10;relu")
    # 14652

    adv_scenario_1d.get_receptive_field(
        network_config="Conv;23;10;relu",
        task_config="predict",
    )
    # ((11.0, 11.0),)  # 11 per direction in the one and only dimension
    ```

    Counting parameters is associated with a scenario because it depends
    on the [`num_channels`](#num_channels) and the [`num_points`](#num_points)
    of the scenario.


### `train`

Describes how reference simulator (the stepper one gets with
`scenario.get_ref_stepper()`) interacts with the neural stepper (the trainable
stepper one gets with `scenario.get_neural_stepper(....)`). The easiest
configuration is **one-step supervised** training accessible by setting
`train="one"`. In this case, [`num_train_samples`](#num_train_samples) initial
conditions are drawn from the IC distribution (accessible via
`scenario.get_ic_generator()`) and are then unrolled for
[`train_temporal_horizon`](#train_temporal_horizon) steps (This data can also be
accessed via `scenario.get_train_data()`). For one-step supervised training out
of all trajectories across all time steps, [`batch_size`](#batch_size) windows
of **length two** will be drawn.

This naturally extends to **supervised unrolled** training in which the neural
stepper is autoregressively unrolled for the specified number of steps. This is
achieved via `"sup;T"` with `T` being the number of steps.

Beyond that, APEBench supports diverted chain training, in which the reference
simulator is used differentiably. A simple diverted chain setup with branch
length one is available via `"div;T"`. See the [APEBench
paper](https://arxiv.org/abs/2411.00180) and the [Trainax
library](https://github.com/Ceyron/trainax) for more details.

### `start_seed`

The integer at which the lists of seeds start if the scenario is executed with
parallel training. Ultimately, the seeds investigated will be `range(start_seed, start_seed + num_seeds)`.

### `num_seeds`

How many seeds to investigate in parallel. Oftentimes, trainings in 1D or 2D
with low resolution (low [`num_points`](#num_points)) can be done in parallel to
obtain seed statistics virtually for free. If you are on a modern GPU, a good rule of thumb:

- 1D with <200 points: 5-20 seeds in parallel possible
- 1D in higher resolution: 1-5 seeds in parallel
- 2D with <64 points: 1-3 seeds in parallel
- Anything else: only one seed at a time (run seed statistics sequentially or
  [distribute over multiple GPUs](using_cli.md))

### `remove_singleton_axis`

### `num_spatial_dims`

### `num_points`

### `num_channels`

### `ic_config`

### `num_warmup_steps`

### `num_train_samples`

### `train_temporal_horizon`

### `train_seed`

### `num_test_samples`

### `test_temporal_horizon`

### `test_seed`

### `optim_config`

### `batch_size`

### `num_trjs_returned`

### `record_loss_every`

### `vlim`

### `report_metrics`

### `callbacks`

### `domain_extent`

### `dt`

### `num_substeps`

### `order`

### `dealiasing_fraction`

### `num_circle_points`

### `circle_radius`

### `coarse_proportion`

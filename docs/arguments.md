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

### `task`

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

### `start_seed`

### `num_seeds`

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


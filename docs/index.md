# Getting Started

```bash
pip install apebench
```

Requires Python 3.10+ and JAX 0.4.12+ 👉 [JAX install guide](https://jax.readthedocs.io/en/latest/installation.html).

Quick instruction with fresh Conda environment and JAX CUDA 12 on Linux.

```bash
conda create -n apebench python=3.12 -y
conda activate apebench
pip install -U "jax[cuda12]"
pip install apebench
```

## APEBench Paper

Accepted at Neurips 2024:

- [📄 Paper](https://arxiv.org/abs/2411.00180)
- [🧵 Project Page](https://tum-pbs.github.io/apebench-paper/)

## Quickstart

Train a ConvNet to emulate 1D advection, display train loss, test error metric
rollout, and a sample rollout.

```python
import apebench
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

advection_scenario = apebench.scenarios.difficulty.Advection()

data, trained_nets = advection_scenario(
    task_config="predict",
    network_config="Conv;26;10;relu",
    train_config="one",
    num_seeds=3,
)

data_loss = apebench.melt_loss(data)
data_metrics = apebench.melt_metrics(data)
data_sample_rollout = apebench.melt_sample_rollouts(data)

fig, axs = plt.subplots(1, 3, figsize=(13, 3))

sns.lineplot(data_loss, x="update_step", y="train_loss", ax=axs[0])
axs[0].set_yscale("log")
axs[0].set_title("Training loss")

sns.lineplot(data_metrics, x="time_step", y="mean_nRMSE", ax=axs[1])
axs[1].set_ylim(-0.05, 1.05)
axs[1].set_title("Metric rollout")

axs[2].imshow(
    np.array(data_sample_rollout["sample_rollout"][0])[:, 0, :].T,
    origin="lower",
    aspect="auto",
    vmin=-1,
    vmax=1,
    cmap="RdBu_r",
)
axs[2].set_xlabel("time")
axs[2].set_ylabel("space")
axs[2].set_title("Sample rollout")

plt.show()
```

![](https://github.com/user-attachments/assets/10f968f4-2b30-4972-8753-22b7fad208ed)

## Background

Autoregressive neural emulators can be used to efficiently forecast transient
phenomena, often associated with differential equations. Denote by
$\mathcal{P}_h$ a reference numerical simulator (e.g., the [FTCS
scheme](https://en.wikipedia.org/wiki/FTCS_scheme) for the heat equation). It
advances a state $u_h$ by

$$
u_h^{[t+1]} = \mathcal{P}_h(u_h^{[t]}).
$$

An autoregressive neural emulator $f_\theta$ is trained to mimic $\mathcal{P}_h$, i.e., $f_\theta \approx \mathcal{P}_h$. Doing so requires the following choices:

1. What is the reference simulator $\mathcal{P}_h$?
    1. What is its corresponding continuous transient partial differential
        equation? (advection, diffusion, Burgers, Kuramoto-Sivashinsky,
        Navier-Stokes, etc.)
    2. What consistent numerical scheme is used to discretize the continuous
        transient partial differential equation?
2. What is the architecture of the autoregressive neural emulator $f_\theta$?
3. How do $f_\theta$ and $\mathcal{P}_h$ interact during training (=optimization
    of $\theta$)?
    1. For how many steps are their predictions unrolled and compared?
    2. What is the time-level loss function?
    3. How large is the batch size?
    4. What is the opimizer and its learning rate scheduler?
    5. For how many steps is the training run?
4. Additional training and evaluation related choices:
    1. What is the initial condition distribution?
    2. How long is the time horizon seen during training?
    3. What is the evaluation metric? If it is related to an error rollout, for
        how many steps is the rollout?
    4. How many random seeds are used to draw conclusions?

APEBench is a framework to holistically assess all four ingredients. Component
(1), the discrete reference simulator $\mathcal{P}_h$, is provided by
[`Exponax`](https://github.com/Ceyron/exponax). This is a suite of
[ETDRK](https://www.sciencedirect.com/science/article/abs/pii/S0021999102969950)-based
methods for semi-linear partial differential equations on periodic domains. This
covers a wide range of dynamics. For the most common scenarios, a unique
interface using normalized (non-dimensionalized) coefficients or a
difficulty-based interface (as described in the APEBench paper) can be used. The
second (2) component is given by
[`PDEquinox`](https://github.com/Ceyron/pdequinox). This library uses
[`Equinox`](https://github.com/patrick-kidger/equinox), a JAX-based
deep-learning framework, to implement many commonly found architectures like
convolutional ResNets, U-Nets, and FNOs. The third (3) component is
[`Trainax`](https://github.com/Ceyron/trainax), an abstract implementation of
"trainers" that provide supervised rollout training and many other features. The
fourth (4) component is to wrap up the former three and is given by this
repository.
APEBench encapsulates the entire pipeline of training and evaluating an
autoregressive neural emulator in a scenario. A scenario is a callable
dataclass.

## Citation

This package was developed as part of the [APEBench paper
(arxiv.org/abs/2411.00180)](https://arxiv.org/abs/2411.00180) (accepted at
Neurips 2024). If you find it useful for your research, please consider citing
it:

```bibtex
@article{koehler2024apebench,
  title={{APEBench}: A benchmark for autoregressive neural emulators of pdes},
  author={Koehler, Felix and Niedermayr, Simon and Westermann, R{\"u}diger and Thuerey, Nils},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={120252--120310},
  year={2024}
}
```

(Feel free to also give the project a star on GitHub if you like it.)

## Funding

The main author (Felix Koehler) is a PhD student in the group of [Prof. Thuerey at TUM](https://ge.in.tum.de/) and his research is funded by the [Munich Center for Machine Learning](https://mcml.ai/).

## License

MIT, see [here](https://github.com/Ceyron/apebench/blob/main/LICENSE.txt)

---

> [fkoehler.site](https://fkoehler.site/) &nbsp;&middot;&nbsp;
> GitHub [@ceyron](https://github.com/ceyron) &nbsp;&middot;&nbsp;
> X [@felix_m_koehler](https://twitter.com/felix_m_koehler) &nbsp;&middot;&nbsp;
> LinkedIn [Felix Köhler](https://www.linkedin.com/in/felix-koehler)
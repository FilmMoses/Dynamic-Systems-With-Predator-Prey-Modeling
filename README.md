# Neural Dynamic Systems: Predator-Prey Modeling with Lake Powell Fisheries

A computational framework combining empirical fisheries time-series data with neural ordinary differential equation (Neural ODE) approximations to model non-linear dynamic systems and reconstruct vector field phase portraits.

---

## Project Overview

This project analyzes the non-linear interaction dynamics of an aquatic ecosystem by learning continuous vector fields directly from multi-table historical fisheries data. Using records from the Lake Powell Fisheries dataset (Kaggle), the model integrates empirical observations across fish morphology, diet, and limnological environmental metrics to reconstruct continuous system velocities:
* **Multi-Table Data Harmonization:** Merges and aggregates individual fish specimen tables (`FISH_TABLE.csv`), growth-ring age records (`AGE_GROW_TABLE.csv`), stomach contents (`STOMACH_TABLE.csv`), and limnology metrics (`LAB_COUNTS.csv`) into annual time-series records spanning 1999–2018.
* **Empirical Gradient Estimation:** Calculates discrete temporal velocity gradients ($\dot{X}$) across normalized observables—average fish weight, water temperature, and zooplankton density.
* **Neural Vector Field Approximator:** Implements a PyTorch multi-layer neural network (`EcosystemDynamicsNN`) parameterized with smooth hyperbolic tangent ($\tanh$) activations to map continuous derivative surfaces without divergence.
* **Forward Numerical Integration:** Simulates forward ecological trajectories across a 15-year horizon using iterative Euler steps.
* **Phase Portrait Reconstruction:** Evaluates system equilibria and limit cycles across 2D vector field slices (fish weight vs. plankton density).

---



## Mathematical Formulation

The dynamic state vector $X(t) \in \mathbb{R}^3$ represents normalized ecosystem observables:

$$
X(t) = \begin{bmatrix} x_{\text{WT}}(t) \\ x_{\text{TEMP}}(t) \\ x_{\text{TOT-AVE}}(t) \end{bmatrix}
$$

where $x_{\text{WT}}$ is average specimen weight, $x_{\text{TEMP}}$ is ambient water temperature, and $x_{\text{TOT-AVE}}$ is the mean plankton density.

### 1. Finite-Difference Gradients
Empirical velocity vectors are extracted from the observed annual time series using central difference gradient approximations:

$$
\dot{X}(t) \approx \nabla X(t) = \frac{X(t + \Delta t) - X(t - \Delta t)}{2 \Delta t}
$$

### 2. Neural Vector Field Mapping
The neural network models the non-linear velocity mapping $f_\theta: \mathbb{R}^3 \to \mathbb{R}^3$:

$$
\frac{dX}{dt} = f_\theta(X(t))
$$

The parameters $\theta$ are optimized by minimizing the Mean Squared Error (MSE) loss against numerical gradients over $N$ annual observations:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \left\| f_\theta(X_i) - \dot{X}_i \right\|_2^2
$$

### 3. Iterative Forward Simulation
Future system trajectories are integrated iteratively over $K = 15$ steps ($\Delta t = 1.0$):

$$
X(t + \Delta t) = X(t) + f_\theta(X(t)) \cdot \Delta t
$$

with values clamped to valid biological boundaries $[0.0, 1.2]$.

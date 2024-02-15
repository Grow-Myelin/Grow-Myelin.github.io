# Probabilistic Programming Tools

**[View Project on GitHub](https://github.com/Grow-Myelin/ProbProg/tree/main/ProbProgTools)**

## Technologies Used
- Python
- NumPy
- Jax
- NumPyro

## Project Overview
Designed tools that simulate a random levy process and then uses NumPyro to infer the parameters used to simulate those processes.

## Key Features
- Simulation of processes that resemble stock price movements using random levy processes.
- Parameter inference with NumPyro to understand underlying market dynamics.

## Imports

```python
from scipy.stats import levy_stable
from jax.scipy.linalg import cholesky
from typing import Dict, List, Tuple
import os
from typing import Any, Dict
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpyro
import numpyro.distributions as dist
import seaborn as sns
from numpyro.infer import MCMC, NUTS
```

## Classes

### StockMarketSimulator
```python
class StockMarketSimulator:
    """
    A simulator for stock market prices using Levy processes with JAX for computation and
    Pareto-distributed initial prices using NumPy.

    Attributes:
        n_industries (int): Number of industries.
        n_stocks_per_industry (int): Number of stocks per industry.
        base_stock_price (float): Base stock price for scaling initial prices.
        industries (List[str]): List of industry names.
        stocks (List[str]): List of stock symbols.
        stock_prices (pd.DataFrame): DataFrame to store simulated stock prices.
        seed (int): Int derived through os for true random state
        key (jax.random.PRNGKey): JAX PRNG key for random number generation.
        industry_map (Dict[str, str]): Mapping of stocks to their respective industries.
        alpha_params (Dict[str, float]): Alpha parameter for each industry.
        beta_params (Dict[str, float]): Beta parameter for each industry.
        pareto_shapes (Dict[str, float]): Pareto shape parameter for each industry.
    """
```

## Methods

### __init__()
```python
def __init__(self, n_industries: int = 8, n_stocks_per_industry: int = 10, base_stock_price: float = 100) -> None:
    """
    Initialize the stock market simulator with specified parameters.

    Args:
        n_industries (int): The number of industries to simulate.
        n_stocks_per_industry (int): The number of stocks per industry.
        base_stock_price (float): The base stock price for scaling initial prices.
    """
```

### _initialize_prices()
```python
def _initialize_prices(self) -> None:
    """Initialize the stock prices using Pareto distribution for each industry."""
```

### _simulate_stock_prices()
```python
def simulate_stock_prices(self, n_days: int = 252) -> pd.DataFrame:
    """
    Simulate stock prices over a given number of days.

    Args:
        n_days (int): The number of days to simulate stock prices.

    Returns:
        pd.DataFrame: A DataFrame containing the simulated stock prices.
    """
```

### _apply_correlation
```python
def _apply_correlation(self, increments: np.ndarray) -> np.ndarray:
    """
    Apply a correlation matrix to the increments using Cholesky decomposition.

    Args:
        increments (np.ndarray): An array of increments to apply correlation to.

    Returns:
        np.ndarray: Correlated increments after applying the correlation matrix.
    """
```

## Functions

### plot_posteriors()
```python
def plot_posteriors(posterior_samples: Dict[str, jnp.ndarray], industry: str) -> None:
    """
    Plots the posterior distributions for a given industry.

    Args:
        posterior_samples: Samples from the posterior distribution as a dictionary where keys are parameter names.
        industry (str): The name of the industry for which the posterior distributions are plotted.
    """
```

### run_bayesian_inference()
```python
def run_bayesian_inference(simulator: DI.StockMarketSimulator, n_samples: int = 500, n_warmup: int = 100) -> None:
    """
    Runs Bayesian inference for each industry and plots the posterior distributions.

    Args:
        simulator: An instance of StockMarketSimulator containing stock prices and industry mappings.
        n_samples (int): Number of samples to draw from the posterior distribution.
        n_warmup (int): Number of warmup steps for the sampler.
    """
```

### plot_stock_prices()
```python
def plot_stock_prices(simulator: DI.StockMarketSimulator) -> None:
    """
    Plot the simulated stock prices for all stocks in the simulation.

    Args:
        simulator (DI.StockMarketSimulator): An instance of the StockMarketSimulator class.

    This function generates a line plot for each stock across the simulated days and saves the plot
    to a file named 'simulated_stock_prices.png' in the 'plots' directory.
    """
```

### plot_stock_prices_by_industry()
```python
def plot_stock_prices_by_industry(simulator: DI.StockMarketSimulator) -> None:
    """
    Plot the simulated stock prices for each industry separately.

    Args:
        simulator (DI.StockMarketSimulator): An instance of the StockMarketSimulator class.

    This function generates a line plot for each stock within an industry across the simulated days and saves
    each industry's plot to a separate file in the 'plots' directory, named 'simulated_stock_price_by_[industry].png'.
    """
```

## Example Usage
```python
def main() -> None:
    """
    Main function to initialize the simulator, run the stock price simulation, and plot the results.
    """
    simulator = DI.StockMarketSimulator()
    run_bayesian_inference(simulator)
    simulator.simulate_stock_prices()
    plot_stock_prices(simulator)
    plot_stock_prices_by_industry(simulator)

if __name__ == "__main__":
    main()
```
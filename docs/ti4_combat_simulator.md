# TI4 Combat Simulator

**[View Project on GitHub](https://github.com/Grow-Myelin/ProbProg/tree/main/TI4)**

## Technologies Used
- Python
- Jax

## Project Background
Twilight Imperium (TI4) is a grand strategy board game known for its intricate diplomacy, expansive empire-building, and epic space battles. TI4's combat involves a blend of strategic planning and randomness. While unit statistics and player decisions play a significant role, the combat system incorporates elements of randomness through dice rolls.

## Project Overview
This combat simulator for TI4 utilizes Monte Carlo and probability theory to model the distribution of outcomes for in-game combat scenarios. This tool helps players understand the potential outcomes of their strategic decisions in combat, aiding in planning and execution during gameplay.

## Key Features
- Monte Carlo simulation methods to predict combat outcomes.
- Probability theory to sample from combinations of probabilities.

## ti4_functions

This document describes a set of functions for simulating combat scenarios using JAX.
## Imports
```python
import jax
import jax.numpy as jnp
from jax import random
import os
from typing import Dict, Tuple, Any
```

## Types
```python
KeyType = Tuple[int, float]
SideType = Dict[KeyType, int]
RNGKey = Any
```
## Functions

### apply_hits

```python
def apply_hits(side: SideType, hits_scored: int, rng_key: RNGKey) -> SideType:
    """
    Apply hits to a side with JAX, prioritizing dice with lower hit probabilities 
    and lower health. Incorporates randomness in selecting dice within the same 
    priority level to take hits. Accepts an RNG key for reproducible randomness.

    Parameters:
        side (SideType): Dictionary representing the side's units and their stats.
        hits_scored (int): Number of hits to apply to the side.
        rng_key (RNGKey): JAX random key for generating random numbers.

    Returns:
        SideType: Updated side after applying hits.
    """
```

### simulate_combat_round_jax

```python
def simulate_combat_round_jax(side_a: SideType, side_b: SideType, rng_key: RNGKey) -> Tuple[SideType, SideType]:
    """
    Simulates a single round of combat between two sides using JAX.

    Parameters:
        side_a (SideType): First combatant side.
        side_b (SideType): Second combatant side.
        rng_key (RNGKey): JAX random key for generating random numbers.

    Returns:
        Tuple[SideType, SideType]: Updated states of side_a and side_b after combat.
    """
```

### run_combat_until_elimination

```python
def simulate_combat_round_jax(side_a: SideType, side_b: SideType, rng_key: RNGKey) -> Tuple[SideType, SideType]:
    """
    Simulates a single round of combat between two sides using JAX.

    Parameters:
        side_a (SideType): First combatant side.
        side_b (SideType): Second combatant side.
        rng_key (RNGKey): JAX random key for generating random numbers.

    Returns:
        Tuple[SideType, SideType]: Updated states of side_a and side_b after combat.
    """
```

### monte_carlo_combat_simulation

```python
def monte_carlo_combat_simulation(initial_side_a: SideType, initial_side_b: SideType, num_simulations: int = 1000) -> Dict[str, float]:
    """
    Performs a Monte Carlo simulation of combat between two sides over a specified
    number of simulations to estimate outcome probabilities.

    Parameters:
        initial_side_a (SideType): Initial state of side A.
        initial_side_b (SideType): Initial state of side B.
        num_simulations (int): Number of simulations to run.

    Returns:
        Dict[str, float]: Probabilities of different outcomes.
    """
```

## Example Usage
The following is an example of 2 dreadnaughts, each with 2 health hitting 60% of the time fighting vs 6 fighters, each with 1 health hitting 20% of the time.

```python
initial_side_a = {(2, 0.6): 3}
initial_side_b = {(1, 0.2): 6}
num_simulations = 10000
probabilities = monte_carlo_combat_simulation(initial_side_a, initial_side_b, num_simulations)
print(probabilities)

```

---
[Experience](experience.md) | [Education](education.md) | [Skills](skills.md) | [**Projects**](projects.md) | [Contact](contact.md)

---

[**TI4 Combat Simulator**](ti4_combat_simulator.md) | [Baseball Pitch Predictor](baseball_pitch_predictor.md) | [Probabalistic Programming Tools](prob_prog_tools.md) | [Data Tools](data_tools.md)
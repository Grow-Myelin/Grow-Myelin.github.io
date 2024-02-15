# Baseball Pitch Predictor

**[View Project on GitHub](https://github.com/Grow-Myelin/ProbProg/tree/main/baseball)**

## Technologies Used
- Python
- Jax
- Flax

## Project Overview
Created a predictive model to analyze and forecast baseball pitch types based on historical pitch data.

## Key Features
- Analysis of pitch frequency over varying time periods to predict the next pitch.

## Imports

```python
import pandas as pd
import numpy as np
from jax import numpy as jnp
from typing import Tuple, Dict, List
import time
```

## Functions

### normalize_counts
```python
def normalize_counts(pitch_counts: np.ndarray) -> np.ndarray:
    """
    Normalize the pitch counts to get a distribution proportion.

    Parameters:
    - pitch_counts: Array of pitch counts.

    Returns:
    - np.ndarray: Normalized distribution of pitch counts.
    """
```

### efficient_pitch_distribution
```python
def efficient_pitch_distribution(df: pd.DataFrame, pitch_types: List[str], filter_conditions: Dict[str, str]) -> np.ndarray:
    """
    Calculate and normalize the distribution of pitch types, excluding the current event.
    """
```

## Example Usage

```python
# In progress
```


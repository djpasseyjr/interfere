# Intervention

Apply external manipulations to a dynamic model and observe system responses.

## Intervention Types

| Intervention Class               | Description                                           |
|----------------------------------|-------------------------------------------------------|
| `IdentityIntervention`           | No-op intervention                                     |
| `PerfectIntervention`            | Replace variables with constant values                |
| `SignalIntervention`             | Apply time-varying signals to selected variables      |

## Example

```python
import numpy as np
from interfere.dynamics import Lorenz
from interfere.interventions import PerfectIntervention, SignalIntervention

# Simulation setup
t = np.linspace(0, 10, 500)
x0 = np.array([1.0, 1.0, 1.0])
model = Lorenz(sigma=0.0)

# No intervention
states_orig = model.simulate(t, x0)

# Perfect intervention: set first variable to constant 5.0 after t=0
i = PerfectIntervention(0, 5.0)
states_perfect = model.simulate(t, x0, intervention=i)

# Signal intervention: sinusoidal forcing on second variable
i2 = SignalIntervention(1, np.sin)
states_signal = model.simulate(t, x0, intervention=i2)

# Inspect first five time points of perfect intervention
print(states_perfect[:5])
``` 
"""Global-best PSO with synchronous updates and bounded, damped velocities."""
import time
import numpy as np
from scipy.stats import qmc


def optimize(objective, dimensions=10, population_size=50, max_evaluations=20000,
             seed=0, inertia=0.7298, c1=1.49618, c2=1.49618, **_):
    if dimensions < 1 or population_size < 2 or max_evaluations < population_size:
        raise ValueError("positive dimensions, population >= 2 and budget >= population are required")
    if any(not np.isfinite(x) or x < 0 for x in (inertia, c1, c2)):
        raise ValueError("PSO coefficients must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    population = qmc.LatinHypercube(d=dimensions, seed=int(rng.integers(2**32))).random(population_size)
    velocity = rng.uniform(-0.1, 0.1, population.shape)
    started = time.perf_counter()

    def evaluate(x):
        value = float(objective(x.copy()))
        if not np.isfinite(value):
            raise ValueError("objective returned nonfinite fitness")
        return value

    personal = population.copy()
    values = np.asarray([evaluate(x) for x in population])
    evaluations = population_size
    best = int(np.argmax(values))
    history = [{"evaluations": evaluations, "best": float(values[best])}]
    while evaluations < max_evaluations:
        count = min(population_size, max_evaluations - evaluations)
        global_best = personal[best].copy()
        velocity[:count] = (inertia * velocity[:count]
                            + c1 * rng.random((count, dimensions)) * (personal[:count] - population[:count])
                            + c2 * rng.random((count, dimensions)) * (global_best - population[:count]))
        velocity[:count] = np.clip(velocity[:count], -0.2, 0.2)
        trial = population[:count] + velocity[:count]
        violation = (trial < 0) | (trial > 1)
        velocity[:count][violation] *= -0.5
        population[:count] = np.clip(trial, 0, 1)
        trial_values = np.asarray([evaluate(x) for x in population[:count]])
        improved = trial_values >= values[:count]
        personal[:count][improved] = population[:count][improved]
        values[:count][improved] = trial_values[improved]
        evaluations += count
        best = int(np.argmax(values))
        history.append({"evaluations": evaluations, "best": float(values[best])})
    return dict(best_vector=personal[best].tolist(), best_fitness=float(values[best]),
                evaluations=evaluations, history=history, seconds=time.perf_counter() - started, complete=True)

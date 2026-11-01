"""Uniform random search: the essential equal-evaluation-budget control."""
import time
import numpy as np


def optimize(objective, dimensions=10, max_evaluations=20000, seed=0, record_every=50, **_):
    if dimensions < 1 or max_evaluations < 1 or record_every < 1:
        raise ValueError("dimensions, budget and record interval must be positive")
    rng = np.random.default_rng(seed)
    best, best_vector, history = -np.inf, None, []
    started = time.perf_counter()
    for evaluation in range(1, max_evaluations + 1):
        vector = rng.random(dimensions)
        score = float(objective(vector.copy()))
        if not np.isfinite(score):
            raise ValueError("objective returned nonfinite fitness")
        if score > best:
            best, best_vector = score, vector.tolist()
        if evaluation % record_every == 0 or evaluation == max_evaluations:
            history.append({"evaluations": evaluation, "best": best})
    return dict(best_vector=best_vector, best_fitness=best, evaluations=max_evaluations,
                history=history, seconds=time.perf_counter() - started, complete=True)

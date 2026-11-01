"""Real-coded GA: tournament selection, BLX-alpha, Gaussian mutation, elitism."""
import time
import numpy as np
from scipy.stats import qmc
from .de import bounce_back


def optimize(objective, dimensions=10, population_size=50, max_evaluations=20000,
             seed=0, blend_alpha=0.5, mutation_sigma=0.1, **_):
    if dimensions < 1 or population_size < 4 or max_evaluations < population_size:
        raise ValueError("positive dimensions, population >= 4 and budget >= population are required")
    if blend_alpha < 0 or mutation_sigma < 0:
        raise ValueError("blend_alpha and mutation_sigma must be nonnegative")
    rng = np.random.default_rng(seed)
    population = qmc.LatinHypercube(d=dimensions, seed=int(rng.integers(2**32))).random(population_size)
    started = time.perf_counter()

    def evaluate(x):
        value = float(objective(x.copy()))
        if not np.isfinite(value):
            raise ValueError("objective returned nonfinite fitness")
        return value

    fitness = np.asarray([evaluate(x) for x in population])
    evaluations = population_size
    history = [{"evaluations": evaluations, "best": float(fitness.max())}]
    while evaluations < max_evaluations:
        # Evaluate a full batch of offspring, then preserve the previous elite.
        # This uses NP evaluations per generation just like DE and PSO.
        count = min(population_size, max_evaluations - evaluations)
        children, values = [], []
        for _ in range(count):
            candidates = rng.integers(population_size, size=(2, 3))
            parents = candidates[np.arange(2), np.argmax(fitness[candidates], axis=1)]
            a, b = population[parents]
            coefficient = rng.uniform(-blend_alpha, 1 + blend_alpha, dimensions)
            child = a + coefficient * (b - a)
            mask = rng.random(dimensions) < 1 / dimensions
            child += mask * rng.normal(0, mutation_sigma, dimensions)
            child = bounce_back(child, a, rng)
            children.append(child)
            values.append(evaluate(child))
        elite = int(np.argmax(fitness))
        # For a partial final generation, retain unevaluated old individuals.
        next_population = population.copy()
        next_fitness = fitness.copy()
        next_population[:count] = children
        next_fitness[:count] = values
        worst = int(np.argmin(next_fitness))
        if fitness[elite] > next_fitness[worst]:
            next_population[worst] = population[elite]
            next_fitness[worst] = fitness[elite]
        population, fitness = next_population, next_fitness
        evaluations += count
        history.append({"evaluations": evaluations, "best": float(fitness.max())})
    best = int(np.argmax(fitness))
    return dict(best_vector=population[best].tolist(), best_fitness=float(fitness[best]),
                evaluations=evaluations, history=history, seconds=time.perf_counter() - started, complete=True)

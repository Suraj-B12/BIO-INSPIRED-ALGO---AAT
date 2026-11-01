import numpy as np
import pytest
from bioalgo import de, ga, pso, random_search
from bioalgo.benchmarks import FUNCTIONS, unit_objective


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("dimensions", [2, 5, 10, 30])
def test_known_optima_and_batch_shapes(name, dimensions):
    spec = FUNCTIONS[name]
    x = np.full(dimensions, spec["optimum_coordinate"])
    assert abs(float(spec["function"](x))) < 1e-12
    batched = spec["function"](np.stack((x, x)))
    assert batched.shape == (2,)
    np.testing.assert_allclose(batched, 0, atol=1e-12)


@pytest.mark.parametrize("name", FUNCTIONS)
@pytest.mark.parametrize("seed", range(20))
def test_function_nonnegative_in_domain(name, seed):
    spec = FUNCTIONS[name]
    rng = np.random.default_rng(seed)
    samples = rng.uniform(*spec["bounds"], (100, 10))
    assert np.all(spec["function"](samples) >= -1e-12)


@pytest.mark.parametrize("algorithm", [de, ga, pso, random_search])
@pytest.mark.parametrize("seed", range(20))
def test_exact_budget_bounds_determinism_and_monotonic_best(algorithm, seed):
    seen = []

    def objective(x):
        seen.append(x.copy())
        assert np.all((x >= 0) & (x <= 1))
        return -float(np.sum((x - 0.25) ** 2))

    options = dict(dimensions=3, population_size=8, max_evaluations=83, seed=seed)
    if algorithm is de:
        options["generations"] = 20
    first = algorithm.optimize(objective, **options)
    assert len(seen) == first["evaluations"] == 83
    second = algorithm.optimize(objective, **options)
    assert first["best_vector"] == second["best_vector"]
    assert first["best_fitness"] == second["best_fitness"]
    assert np.all(np.diff([row["best"] for row in first["history"]]) >= 0)

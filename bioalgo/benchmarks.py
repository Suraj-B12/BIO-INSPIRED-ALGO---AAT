"""Standard continuous minimization functions; each optimum has value zero."""
import numpy as np


def sphere(x):
    return np.sum(np.asarray(x, dtype=float) ** 2, axis=-1)


def rosenbrock(x):
    x = np.asarray(x, dtype=float)
    return np.sum(100 * (x[..., 1:] - x[..., :-1] ** 2) ** 2 + (1 - x[..., :-1]) ** 2, axis=-1)


def rastrigin(x):
    x = np.asarray(x, dtype=float)
    return 10 * x.shape[-1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=-1)


def ackley(x):
    x = np.asarray(x, dtype=float)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2, axis=-1))) - np.exp(
        np.mean(np.cos(2 * np.pi * x), axis=-1)) + 20 + np.e


def griewank(x):
    x = np.asarray(x, dtype=float)
    return np.sum(x ** 2, axis=-1) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, x.shape[-1] + 1))), axis=-1) + 1


FUNCTIONS = {
    "Sphere": {"function": sphere, "bounds": (-5.12, 5.12), "optimum_coordinate": 0.0},
    "Rosenbrock": {"function": rosenbrock, "bounds": (-5.0, 10.0), "optimum_coordinate": 1.0},
    "Rastrigin": {"function": rastrigin, "bounds": (-5.12, 5.12), "optimum_coordinate": 0.0},
    "Ackley": {"function": ackley, "bounds": (-32.768, 32.768), "optimum_coordinate": 0.0},
    "Griewank": {"function": griewank, "bounds": (-600.0, 600.0), "optimum_coordinate": 0.0},
}


def unit_objective(name):
    """Adapt a named minimization benchmark to our maximizing unit-cube API."""
    spec = FUNCTIONS[name]
    low, high = spec["bounds"]
    return lambda x: -float(spec["function"](low + np.asarray(x) * (high - low)))

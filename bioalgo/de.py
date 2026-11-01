"""Auditable DE/rand/1/bin, maximizing a scalar objective in the unit cube.

The optimizer owns only its NumPy random generator. CNN objectives must reset
their own training seed. Generation donors are frozen (deferred updating), so
evaluation order cannot change the mathematical population used for mutation.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import time

import numpy as np
from scipy.stats import qmc


def decode(genes):
    """Map seven normalized genes into the preregistered mixed search space."""
    x = np.asarray(genes, dtype=float)
    if x.shape != (7,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("genes must be seven finite values in [0, 1]")
    # Half-up rounding avoids NumPy's surprising tie-to-even integer mapping.
    integer = lambda value: int(np.floor(value + 0.5))
    return {
        "learning_rate": float(10 ** (-4.5 + 2.5 * x[0])),
        "weight_decay": float(10 ** (-6 + 4 * x[1])),
        "dropout": float(0.6 * x[2]),
        "head_units": [0, 128, 256, 512][integer(3 * x[3])],
        "unfreeze_blocks": integer(9 * x[4]),
        "label_smoothing": float(0.2 * x[5]),
        "augmentation": integer(15 * x[6]),
    }


def donor_indices(population_size, target, rng):
    """Three distinct donors, none equal to the target (uniformly sampled)."""
    if population_size < 4 or not 0 <= target < population_size:
        raise ValueError("DE needs at least four individuals and a valid target")
    indices = rng.choice(population_size - 1, size=3, replace=False)
    return indices + (indices >= target)


def mutate(population, target, F, rng):
    if not np.isfinite(F) or not 0 < F <= 2:
        raise ValueError("F must be in (0, 2]")
    r1, r2, r3 = donor_indices(len(population), target, rng)
    return population[r1] + F * (population[r2] - population[r3])


def bounce_back(mutant, parent, rng):
    """Repair violations between the parent and violated bound, not by clipping.

    Random repair preserves diversity and avoids accumulating mass on bounds.
    Already-valid coordinates are copied exactly, without consuming randomness.
    """
    mutant = np.asarray(mutant, dtype=float).copy()
    parent = np.asarray(parent, dtype=float)
    if mutant.shape != parent.shape or not np.isfinite(mutant).all():
        raise ValueError("mutant and parent must be equally shaped finite vectors")
    if not np.isfinite(parent).all() or np.any((parent < 0) | (parent > 1)):
        raise ValueError("parent must lie in the unit cube")
    for mask, bound in ((mutant < 0, 0.0), (mutant > 1, 1.0)):
        mutant[mask] = parent[mask] + rng.random(np.count_nonzero(mask)) * (bound - parent[mask])
    return mutant


def binomial_crossover(parent, mutant, CR, rng, return_mask=False):
    parent, mutant = np.asarray(parent), np.asarray(mutant)
    if parent.ndim != 1 or len(parent) == 0 or parent.shape != mutant.shape:
        raise ValueError("parent and mutant must be equally sized nonempty vectors")
    if not np.isfinite(CR) or not 0 <= CR <= 1:
        raise ValueError("CR must be in [0, 1]")
    mask = rng.random(len(parent)) < CR
    mask[rng.integers(len(parent))] = True  # j_rand guarantees a mutant gene.
    child = np.where(mask, mutant, parent)
    return (child, mask) if return_mask else child


def greedy_select(parent, parent_fitness, child, child_fitness):
    """Accept ties as prescribed; invalid fitness can never replace a parent."""
    accepted = bool(np.isfinite(child_fitness) and child_fitness >= parent_fitness)
    return (np.asarray(child).copy(), float(child_fitness), True) if accepted else (
        np.asarray(parent).copy(), float(parent_fitness), False
    )


def population_diversity(population):
    population = np.asarray(population, dtype=float)
    i, j = np.triu_indices(len(population), k=1)
    return float(np.linalg.norm(population[i] - population[j], axis=1).mean())


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _record(state, elapsed, record_positions):
    values = np.asarray(state["fitness"], dtype=float)
    row = {
        "generation": state["generation"], "evaluations": state["evaluations"],
        "best": float(values.max()), "mean": float(values.mean()),
        "worst": float(values.min()), "std": float(values.std()),
        "diversity": population_diversity(state["population"]),
        "success_rate": state["successes"] / len(values) if state["generation"] else 0.0,
        "seconds": elapsed,
    }
    if record_positions:
        row["population"] = [list(x) for x in state["population"]]
        row["fitness"] = list(state["fitness"])
    state["history"].append(row)


def optimize(objective, dimensions=7, population_size=8, generations=8, seed=42,
             F=0.6, CR=0.9, adaptive=False, state_path=None, resume=True,
             callback=None, max_evaluations=None, record_positions=False,
             dither=None, initial_population=None):
    """Maximize ``objective(genes)``; atomically checkpoint every evaluation.

    Resume reproduces the exact random trajectory after an interrupted objective.
    ``max_evaluations`` is an absolute budget, including the initial population.
    A callback receives generation (0 = initialization), individual, evaluation
    (one based), genes, fitness, accepted, best_fitness and elapsed seconds.
    Objectives must return finite numbers; training failures should explicitly
    return their documented failure score rather than hiding arbitrary errors.
    Checkpointed runs retain ``evaluation_events`` so callers can reconcile an
    external CSV after a crash between the atomic checkpoint and callback.
    """
    if isinstance(dimensions, bool) or int(dimensions) != dimensions or dimensions < 1:
        raise ValueError("dimensions must be a positive integer")
    if isinstance(population_size, bool) or int(population_size) != population_size or population_size < 4:
        raise ValueError("population_size must be an integer >= 4")
    if isinstance(generations, bool) or int(generations) != generations or generations < 0:
        raise ValueError("generations must be a nonnegative integer")
    dimensions, population_size, generations = int(dimensions), int(population_size), int(generations)
    if not np.isfinite(F) or not 0 < F <= 2 or not np.isfinite(CR) or not 0 <= CR <= 1:
        raise ValueError("F must be in (0, 2] and CR in [0, 1]")
    if dither is not None and (len(dither) != 2 or not 0 < dither[0] <= dither[1] <= 2):
        raise ValueError("dither must be two ordered bounds in (0, 2]")
    if dither is not None and adaptive:
        raise ValueError("choose jDE adaptation or generation dithering, not both")
    budget = population_size * (generations + 1) if max_evaluations is None else max_evaluations
    if isinstance(budget, bool) or int(budget) != budget or budget < population_size:
        raise ValueError("evaluation budget must be an integer >= population_size")
    budget = min(int(budget), population_size * (generations + 1))
    signature = dict(dimensions=dimensions, population_size=population_size,
                     seed=int(seed), F=float(F), CR=float(CR), adaptive=bool(adaptive),
                     dither=list(dither) if dither is not None else None,
                     record_positions=bool(record_positions))
    rng = np.random.default_rng(seed)
    if resume and state_path is not None and Path(state_path).exists():
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("parameters") != signature:
            raise ValueError("checkpoint parameters differ; use a fresh state path or resume=False")
        if state["evaluations"] > budget:
            raise ValueError("checkpoint exceeds requested budget")
        rng.bit_generator.state = state["rng_state"]
    else:
        population = qmc.LatinHypercube(d=dimensions, seed=int(rng.integers(2**32))).random(population_size)
        if initial_population is not None:
            initial = np.asarray(initial_population, dtype=float)
            if initial.shape != population.shape or not np.isfinite(initial).all() or np.any((initial < 0) | (initial > 1)):
                raise ValueError("initial_population must match NP x dimensions and lie in [0, 1]")
            population = initial.copy()
        state = dict(schema_version=1, parameters=signature, population=population.tolist(),
                     fitness=[None] * population_size, generation=0, next_individual=0,
                     evaluations=0, history=[], successes=0, source_population=None,
                     scales=[float(F)] * population_size, crossovers=[float(CR)] * population_size,
                     generation_F=float(F), seconds=0.0, complete=False,
                     rng_state=rng.bit_generator.state)
    state.setdefault("evaluation_events", [])
    started = time.perf_counter()
    previous_seconds = state.get("seconds", 0.0)
    source = np.asarray(state["source_population"]) if state["source_population"] is not None else None

    def persist():
        state["rng_state"] = rng.bit_generator.state
        state["seconds"] = previous_seconds + time.perf_counter() - started
        if state_path is not None:
            _atomic_json(state_path, state)

    while state["evaluations"] < budget:
        if state["next_individual"] == population_size:
            state["generation"] += 1
            state["next_individual"] = 0
            state["successes"] = 0
            state["source_population"] = [list(x) for x in state["population"]]
            source = np.asarray(state["source_population"])
            state["generation_F"] = float(rng.uniform(*dither)) if dither is not None else float(F)
        i, generation = state["next_individual"], state["generation"]
        parent = np.asarray(state["population"][i])
        proposed_F, proposed_CR = state["scales"][i], state["crossovers"][i]
        if generation == 0:
            trial = parent.copy()
        else:
            if adaptive:
                # jDE inherits new controls only if the offspring is selected.
                if rng.random() < 0.1:
                    proposed_F = float(0.1 + 0.9 * rng.random())
                if rng.random() < 0.1:
                    proposed_CR = float(rng.random())
            else:
                proposed_F = state["generation_F"]
            mutant = mutate(source, i, proposed_F, rng)
            mutant = bounce_back(mutant, parent, rng)
            trial = binomial_crossover(parent, mutant, proposed_CR, rng)
        score = float(objective(trial.copy()))
        if not math.isfinite(score):
            raise ValueError("objective returned nonfinite fitness; return an explicit finite failure score")
        accepted = generation == 0 or score >= state["fitness"][i]
        if accepted:
            state["population"][i] = trial.tolist()
            state["fitness"][i] = score
            state["scales"][i], state["crossovers"][i] = proposed_F, proposed_CR
            state["successes"] += 1
        state["evaluations"] += 1
        state["next_individual"] += 1
        if state["next_individual"] == population_size:
            _record(state, previous_seconds + time.perf_counter() - started, record_positions)
        state["complete"] = state["evaluations"] == budget
        if "best_fitness" not in state or score > state["best_fitness"]:
            state["best_vector"], state["best_fitness"] = trial.tolist(), score
        if state_path is not None or callback is not None:
            event = dict(generation=generation, individual=i, evaluation=state["evaluations"],
                         genes=trial.tolist(), fitness=score, accepted=accepted,
                         best_fitness=state["best_fitness"],
                         seconds=previous_seconds + time.perf_counter() - started)
            if state_path is not None:
                state["evaluation_events"].append(event)
        if state_path is not None or callback is not None:
            persist()
        if callback is not None:
            callback(event)
    state["complete"] = True
    persist()
    return state


# Descriptive aliases keep classroom demonstrations easy to read.
crossover = binomial_crossover
select_donors = donor_indices

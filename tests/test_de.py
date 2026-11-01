"""Seeded operator properties, failure guards, and exact mid-generation resume."""
import itertools
import json
import numpy as np
import pytest

from bioalgo.de import (decode, donor_indices, mutate, bounce_back,
                        binomial_crossover, greedy_select, optimize)
from bioalgo.benchmarks import unit_objective


@pytest.mark.parametrize("seed", range(300))
def test_donors_distinct_and_exclude_every_target(seed):
    rng = np.random.default_rng(seed)
    count = int(rng.integers(4, 40))
    for target in range(count):
        chosen = donor_indices(count, target, rng)
        assert len(set(chosen)) == 3
        assert target not in chosen
        assert np.all((chosen >= 0) & (chosen < count))


@pytest.mark.parametrize("seed", range(300))
def test_bounce_back_never_escapes_parent_to_bound(seed):
    rng = np.random.default_rng(seed)
    parent = rng.random(100)
    raw = rng.normal(0.5, 10, 100)
    repaired = bounce_back(raw, parent, rng)
    assert np.all((repaired >= 0) & (repaired <= 1))
    assert np.all(repaired[raw < 0] <= parent[raw < 0])
    assert np.all(repaired[raw > 1] >= parent[raw > 1])
    np.testing.assert_array_equal(repaired[(raw >= 0) & (raw <= 1)], raw[(raw >= 0) & (raw <= 1)])


@pytest.mark.parametrize("seed", range(300))
def test_crossover_uses_mutant_and_only_parent_or_mutant(seed):
    rng = np.random.default_rng(seed)
    dimensions = int(rng.integers(1, 200))
    parent, mutant = rng.random((2, dimensions))
    cr = (seed % 11) / 10
    child, mask = binomial_crossover(parent, mutant, cr, rng, return_mask=True)
    assert mask.any()
    np.testing.assert_array_equal(child[mask], mutant[mask])
    np.testing.assert_array_equal(child[~mask], parent[~mask])
    if cr == 0:
        assert mask.sum() == 1
    if cr == 1:
        assert mask.all()


@pytest.mark.parametrize("seed", range(200))
def test_greedy_selection_never_decreases_fitness(seed):
    rng = np.random.default_rng(seed)
    parent, child = rng.random((2, 7))
    old, new = rng.normal(size=2)
    chosen, fitness, accepted = greedy_select(parent, old, child, new)
    assert fitness >= old
    assert accepted == (new >= old)
    np.testing.assert_array_equal(chosen, child if accepted else parent)


@pytest.mark.parametrize("corner", list(itertools.product((0, 1), repeat=7)))
def test_decode_every_corner(corner):
    h = decode(corner)
    assert 10 ** -4.5 <= h["learning_rate"] <= 0.01
    assert 1e-6 <= h["weight_decay"] <= 0.01
    assert h["dropout"] in (0, 0.6)
    assert h["head_units"] in (0, 512)
    assert h["unfreeze_blocks"] in (0, 9)
    assert h["augmentation"] in (0, 15)
    assert h["label_smoothing"] in (0, 0.2)


@pytest.mark.parametrize("seed", range(300))
def test_decode_random_values_and_integer_types(seed):
    rng = np.random.default_rng(seed)
    h = decode(rng.random(7))
    assert 10 ** -4.5 <= h["learning_rate"] <= 0.01
    assert 1e-6 <= h["weight_decay"] <= 0.01
    assert 0 <= h["dropout"] <= 0.6
    assert h["head_units"] in (0, 128, 256, 512)
    assert type(h["unfreeze_blocks"]) is int and 0 <= h["unfreeze_blocks"] <= 9
    assert type(h["augmentation"]) is int and 0 <= h["augmentation"] <= 15
    assert 0 <= h["label_smoothing"] <= 0.2


def test_all_integer_options_reachable():
    samples = [decode(np.full(7, value)) for value in np.linspace(0, 1, 1001)]
    assert {h["head_units"] for h in samples} == {0, 128, 256, 512}
    assert {h["unfreeze_blocks"] for h in samples} == set(range(10))
    assert {h["augmentation"] for h in samples} == set(range(16))


def test_crossover_probability_matches_forced_gene_formula():
    rng = np.random.default_rng(91)
    dimensions, cr, repeats = 30, 0.35, 5000
    fraction = sum(binomial_crossover(np.zeros(dimensions), np.ones(dimensions), cr, rng).sum()
                   for _ in range(repeats)) / (repeats * dimensions)
    expected = cr + (1 - cr) / dimensions
    assert abs(fraction - expected) < 0.006


@pytest.mark.parametrize("seed", range(100))
def test_mutation_matches_selected_difference_vector(seed):
    population = np.random.default_rng(seed + 1).random((12, 7))
    target = seed % len(population)
    reference_rng, actual_rng = np.random.default_rng(seed), np.random.default_rng(seed)
    r1, r2, r3 = donor_indices(len(population), target, reference_rng)
    scale = 0.1 + 1.9 * (seed / 100)
    expected = population[r1] + scale * (population[r2] - population[r3])
    np.testing.assert_array_equal(mutate(population, target, scale, actual_rng), expected)


@pytest.mark.parametrize("seed", range(50))
def test_latin_hypercube_initialization_covers_every_stratum(seed):
    result = optimize(lambda x: 0.0, dimensions=7, population_size=20, generations=0, seed=seed)
    bins = np.floor(np.asarray(result["population"]) * 20).astype(int)
    for dimension in range(7):
        assert sorted(bins[:, dimension]) == list(range(20))


@pytest.mark.parametrize("seed", range(50))
@pytest.mark.parametrize("adaptive", (False, True))
def test_seeded_trajectory_and_greedy_best(seed, adaptive):
    options = dict(dimensions=4, population_size=8, generations=6, seed=seed, adaptive=adaptive)
    first, second = optimize(unit_objective("Sphere"), **options), optimize(unit_objective("Sphere"), **options)
    np.testing.assert_array_equal(first["population"], second["population"])
    assert first["fitness"] == second["fitness"]
    assert np.all(np.diff([row["best"] for row in first["history"]]) >= 0)
    assert first["evaluations"] == 56


@pytest.mark.parametrize("stop", range(1, 50))
def test_resume_at_each_evaluation_matches_uninterrupted(tmp_path, stop):
    options = dict(dimensions=3, population_size=8, generations=6, seed=101, adaptive=True)
    objective = unit_objective("Rastrigin")
    full = optimize(objective, **options)
    state_path = tmp_path / "state.json"
    calls = 0

    def interrupted(x):
        nonlocal calls
        calls += 1
        if calls == stop:
            raise RuntimeError("simulated power interruption")
        return objective(x)

    with pytest.raises(RuntimeError, match="power interruption"):
        optimize(interrupted, state_path=state_path, **options)
    resumed = optimize(objective, state_path=state_path, **options)
    assert resumed["best_vector"] == full["best_vector"]
    assert resumed["fitness"] == full["fitness"]
    assert resumed["population"] == full["population"]
    assert resumed["rng_state"] == full["rng_state"]
    assert resumed["evaluations"] == full["evaluations"]
    assert json.loads(state_path.read_text())["complete"] is True


def test_de_solves_ten_dimensional_sphere():
    result = optimize(unit_objective("Sphere"), dimensions=10, population_size=50,
                      generations=399, max_evaluations=20000, seed=0)
    assert -result["best_fitness"] < 1e-6
    assert result["evaluations"] == 20000


def test_callback_and_partial_generation_budget():
    events = []
    result = optimize(unit_objective("Sphere"), dimensions=2, population_size=4,
                      generations=10, max_evaluations=15, callback=events.append)
    assert result["evaluations"] == len(events) == 15
    assert [row["evaluation"] for row in events] == list(range(1, 16))
    assert events[4]["generation"] == 1 and events[4]["individual"] == 0
    assert all(len(row["genes"]) == 2 for row in events)


@pytest.mark.parametrize("kwargs", [dict(population_size=3), dict(dimensions=0), dict(generations=-1),
                                  dict(F=0), dict(F=3), dict(CR=-1), dict(CR=2),
                                  dict(max_evaluations=2), dict(dither=[0.9, 0.5]),
                                  dict(dither=[0.5, 0.9], adaptive=True), dict(dimensions=2.1)])
def test_invalid_optimizer_parameters(kwargs):
    with pytest.raises(ValueError):
        optimize(lambda x: 0.0, **kwargs)


@pytest.mark.parametrize("genes", [[0] * 6, [-0.1] * 7, [1.1] * 7, [float("nan")] * 7])
def test_invalid_decode(genes):
    with pytest.raises(ValueError):
        decode(genes)


def test_invalid_objective_is_never_silently_accepted():
    with pytest.raises(ValueError, match="nonfinite"):
        optimize(lambda x: np.nan)


def test_checkpoint_rejects_incompatible_settings(tmp_path):
    path = tmp_path / "state.json"
    optimize(lambda x: 0, generations=0, state_path=path)
    with pytest.raises(ValueError, match="parameters differ"):
        optimize(lambda x: 0, generations=0, state_path=path, CR=0.2)


def test_durable_events_survive_callback_failure(tmp_path):
    path = tmp_path / "state.json"

    def broken_callback(event):
        if event["evaluation"] == 5:
            raise OSError("simulated CSV write failure")

    with pytest.raises(OSError, match="CSV write failure"):
        optimize(unit_objective("Sphere"), state_path=path, generations=2, callback=broken_callback)
    persisted = json.loads(path.read_text())
    assert len(persisted["evaluation_events"]) == persisted["evaluations"] == 5
    resumed = optimize(unit_objective("Sphere"), state_path=path, generations=2)
    assert [e["evaluation"] for e in resumed["evaluation_events"]] == list(range(1, 25))

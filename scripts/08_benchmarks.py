"""Run or reproduce the equal-budget optimizer study and its visual evidence."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import ranksums
import yaml

from bioalgo import de, ga, pso, random_search
from bioalgo.benchmarks import FUNCTIONS, unit_objective

ALGORITHMS = ("DE", "jDE", "GA", "PSO", "Random Search")
COLOR_KEYS = {"DE": "de", "jDE": "jde", "GA": "ga", "PSO": "pso", "Random Search": "random_search"}
MARKERS = {"DE": "o", "jDE": "s", "GA": "^", "PSO": "D", "Random Search": "x"}


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def run_one(task):
    name, algorithm, seed, dimensions, budget, population_size = task
    settings = dict(dimensions=dimensions, population_size=population_size,
                    max_evaluations=budget, seed=seed)
    objective = unit_objective(name)
    if algorithm in ("DE", "jDE"):
        result = de.optimize(objective, generations=math.ceil(budget / population_size) - 1,
                             adaptive=algorithm == "jDE", **settings)
    else:
        module = {"GA": ga, "PSO": pso, "Random Search": random_search}[algorithm]
        result = module.optimize(objective, record_every=population_size, **settings)
    low, high = FUNCTIONS[name]["bounds"]
    return dict(function=name, algorithm=algorithm, seed=seed, dimensions=dimensions,
                evaluations=result["evaluations"], population=population_size,
                final_error=max(0.0, -result["best_fitness"]), seconds=result["seconds"],
                best_position=(low + np.asarray(result["best_vector"]) * (high - low)).tolist(),
                convergence={"evaluations": [r["evaluations"] for r in result["history"]],
                             "error": [max(0.0, -r["best"]) for r in result["history"]]})


def aggregate(records, settings, elapsed):
    output = dict(schema_version=1, status="complete", objective_direction="minimize",
                  settings=settings, wall_seconds=elapsed, run_count=len(records),
                  wall_seconds_scope="this invocation; cached trials excluded",
                  total_optimizer_seconds=sum(r["seconds"] for r in records),
                  total_evaluations=sum(r["evaluations"] for r in records), functions={})
    for name, spec in FUNCTIONS.items():
        entry = dict(bounds=list(spec["bounds"]), optimum_value=0.0, algorithms={})
        for algorithm in ALGORITHMS:
            rows = sorted((r for r in records if r["function"] == name and r["algorithm"] == algorithm), key=lambda x: x["seed"])
            errors = np.asarray([r["final_error"] for r in rows])
            curves = np.asarray([r["convergence"]["error"] for r in rows])
            entry["algorithms"][algorithm] = dict(
                mean=float(errors.mean()), std=float(errors.std(ddof=1)) if len(errors) > 1 else 0.0,
                median=float(np.median(errors)), best=float(errors.min()), worst=float(errors.max()),
                mean_seconds=float(np.mean([r["seconds"] for r in rows])),
                convergence=dict(evaluations=rows[0]["convergence"]["evaluations"],
                                 median=np.median(curves, axis=0).tolist(),
                                 q25=np.quantile(curves, 0.25, axis=0).tolist(),
                                 q75=np.quantile(curves, 0.75, axis=0).tolist()), runs=rows)
        de_errors = [r["final_error"] for r in entry["algorithms"]["DE"]["runs"]]
        entry["de_vs_others_rank_sum"] = {}
        for algorithm in ALGORITHMS[1:]:
            other = [r["final_error"] for r in entry["algorithms"][algorithm]["runs"]]
            statistic, pvalue = ranksums(de_errors, other)
            entry["de_vs_others_rank_sum"][algorithm] = dict(statistic=float(statistic), p_value=float(pvalue),
                alternative="two-sided", correction="none; exploratory multiple comparisons")
        output["functions"][name] = entry
    return output


def save_summary(result):
    """Keep the dashboard payload small; full per-seed evidence stays in benchmarks.json."""
    summary = {k: v for k, v in result.items() if k != "functions"}
    summary["functions"] = {}
    for name, function in result["functions"].items():
        summary["functions"][name] = {k: v for k, v in function.items() if k != "algorithms"}
        summary["functions"][name]["algorithms"] = {
            name: {k: v for k, v in data.items() if k != "runs"}
            for name, data in function["algorithms"].items()}
    save_json(ROOT / "results/benchmark_summary.json", summary)


def _plot_style():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data" / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    style = config["style"]
    plt.rcParams.update({"font.family": style.get("font", "DejaVu Sans"), "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.labelcolor": style["foreground"],
                         "text.color": style["foreground"], "axes.titleweight": "bold",
                         "savefig.bbox": "tight", "savefig.dpi": 300})
    return plt, style


def render_figures(result_path=None):
    result_path = Path(result_path or ROOT / "results/benchmarks.json")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    plt, style = _plot_style()
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, (name, entry) in zip(axes.flat, result["functions"].items()):
        for algorithm, data in entry["algorithms"].items():
            curve = data["convergence"]
            color = style[COLOR_KEYS[algorithm]]
            axis.semilogy(curve["evaluations"], np.maximum(curve["median"], 1e-16),
                          label=algorithm, color=color, marker=MARKERS[algorithm], markevery=60,
                          markersize=3, linewidth=1.6)
            axis.fill_between(curve["evaluations"], np.maximum(curve["q25"], 1e-16),
                              np.maximum(curve["q75"], 1e-16), color=color, alpha=0.10)
        axis.set(title=name, xlabel="Function evaluations", ylabel="Best objective error (log scale)")
        ticks = np.linspace(0, result["settings"]["budget"], 5)
        axis.set_xticks(ticks, ["0" if t == 0 else f"{t / 1000:g}k" for t in ticks])
        axis.grid(alpha=0.15)
    axes.flat[5].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[5].legend(handles, labels, loc="center", frameon=False, fontsize=12)
    axes.flat[5].text(0.08, 0.13, "Lines: median · bands: interquartile range\nDisplay floor: 10⁻¹⁶ (raw values retained)",
                      fontsize=9, transform=axes.flat[5].transAxes)
    figure.suptitle(f"Equal budgets, different search trajectories · {result['settings']['runs']} seeds", fontsize=18, y=1.01)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figure_dir / f"fig_benchmark_convergence.{suffix}")
    plt.close(figure)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for axis, (name, entry) in zip(axes.flat, result["functions"].items()):
        values = [np.maximum([r["final_error"] for r in entry["algorithms"][a]["runs"]], 1e-16) for a in ALGORITHMS]
        boxes = axis.boxplot(values, tick_labels=["DE", "jDE", "GA", "PSO", "RS"], patch_artist=True,
                             medianprops={"color": "black", "linewidth": 1.3}, showfliers=True)
        for patch, algorithm in zip(boxes["boxes"], ALGORITHMS):
            patch.set(facecolor=style[COLOR_KEYS[algorithm]], alpha=0.6)
        axis.set(yscale="log", title=name, ylabel="Final objective error (log scale)")
        axis.grid(axis="y", alpha=0.15)
    axes.flat[5].axis("off")
    axes.flat[5].text(0.05, 0.55, f"{result['settings']['dimensions']} dimensions\n{result['settings']['budget']:,} evaluations per run\n{result['settings']['runs']} independent seeds\nLower is better", fontsize=15, linespacing=1.8)
    axes.flat[5].text(0.05, 0.22, "Display floor: 10⁻¹⁶\nRaw values are retained in results.", fontsize=10)
    figure.suptitle("Repeatability across independent optimization runs", fontsize=18, y=1.01)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(figure_dir / f"fig_benchmark_box.{suffix}")
    plt.close(figure)


def make_animation_data():
    trajectories = {}
    for name, spec in FUNCTIONS.items():
        run = de.optimize(unit_objective(name), dimensions=2, population_size=24, generations=80,
                          seed=42, record_positions=True)
        low, high = spec["bounds"]
        trajectories[name] = dict(bounds=list(spec["bounds"]), seed=42, population_size=24,
            F=0.6, CR=0.9, frames=[dict(generation=row["generation"], evaluations=row["evaluations"],
                best_error=max(0, -row["best"]), diversity=row["diversity"], success_rate=row["success_rate"],
                positions=(low + np.asarray(row["population"]) * (high - low)).tolist(),
                errors=(-np.asarray(row["fitness"])).tolist()) for row in run["history"]])
    save_json(ROOT / "results/benchmark_2d.json", trajectories)
    return trajectories


def render_animation(trajectories=None):
    from matplotlib.animation import FuncAnimation, PillowWriter
    trajectories = trajectories or json.loads((ROOT / "results/benchmark_2d.json").read_text())
    run = trajectories["Rastrigin"]
    plt, style = _plot_style()
    low, high = run["bounds"]
    coordinates = np.linspace(low, high, 180)
    x, y = np.meshgrid(coordinates, coordinates)
    z = FUNCTIONS["Rastrigin"]["function"](np.stack((x, y), axis=-1))
    figure, axis = plt.subplots(figsize=(7, 5.5))
    axis.contourf(x, y, z, levels=22, cmap="YlGnBu", alpha=0.8)
    points = axis.scatter([], [], c="#F8DA70", edgecolors="#152C24", s=45)
    axis.scatter([0], [0], marker="*", c="white", edgecolors="black", s=130, label="Global optimum")
    axis.set(xlabel="x₁", ylabel="x₂", xlim=(low, high), ylim=(low, high))
    title = axis.set_title("")

    def update(index):
        frame = run["frames"][index]
        points.set_offsets(frame["positions"])
        title.set_text(f"DE / Rastrigin · generation {frame['generation']} · best {frame['best_error']:.4g}")
        return points, title

    animation = FuncAnimation(figure, update, frames=len(run["frames"]), interval=90, blit=False)
    animation.save(ROOT / "figures/fig_de_rastrigin_2d.gif", writer=PillowWriter(fps=12), dpi=100)
    plt.close(figure)
    figure, axes = plt.subplots(1, 4, figsize=(15, 4))
    for axis, index in zip(axes, (0, 8, 25, 80)):
        frame = run["frames"][index]
        axis.contourf(x, y, z, levels=22, cmap="YlGnBu", alpha=0.8)
        positions = np.asarray(frame["positions"])
        axis.scatter(positions[:, 0], positions[:, 1], c="#F8DA70", edgecolors="#152C24", s=25)
        axis.set(title=f"Generation {index}\nBest error {frame['best_error']:.3g}", xlabel="x₁", ylabel="x₂")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(ROOT / f"figures/fig_de_rastrigin_2d.{suffix}")
    plt.close(figure)


def render_operators():
    """A reproducible real operator draw; its numerical source is saved too."""
    rng = np.random.default_rng(12)
    population = rng.uniform(0.15, 0.85, (8, 2))
    target, scale = 0, 0.6
    donors = de.donor_indices(len(population), target, rng)
    r1, r2, r3 = population[donors]
    mutant = r1 + scale * (r2 - r3)
    repaired = de.bounce_back(mutant, population[target], rng)
    trial, mask = de.binomial_crossover(population[target], repaired, 0, rng, return_mask=True)
    data = dict(seed=12, F=scale, CR=0, parent=population[target].tolist(),
                donor_indices=donors.tolist(), donors=population[donors].tolist(), mutant=mutant.tolist(),
                repaired_mutant=repaired.tolist(), crossover_mask=mask.tolist(), trial=trial.tolist())
    save_json(ROOT / "results/de_operator_example.json", data)
    plt, style = _plot_style()
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axis = axes[0]
    for label, position, color in (("r₁", r1, style["de"]), ("r₂", r2, style["pso"]),
                                    ("r₃", r3, style["pso"]), ("v", mutant, style["random_search"])):
        axis.scatter(*position, c=color, s=90, zorder=4)
        axis.annotate(label, position, xytext=(8, 7), textcoords="offset points", fontsize=13)
    axis.annotate("", xy=r2, xytext=r3, arrowprops=dict(arrowstyle="->", color=style["pso"], lw=2))
    axis.annotate("", xy=mutant, xytext=r1, arrowprops=dict(arrowstyle="->", color=style["de"], lw=2))
    axis.set_title("Mutation · v = r₁ + F(r₂ − r₃)")
    axis = axes[1]
    parent = population[target]
    for label, position, color, marker in (("parent x", parent, style["baseline"], "o"),
                                          ("mutant v", repaired, style["random_search"], "s"),
                                          ("trial u", trial, style["de"], "D")):
        axis.scatter(*position, c=color, marker=marker, s=90, zorder=4, label=label)
    axis.plot([parent[0], trial[0], repaired[0]], [parent[1], trial[1], repaired[1]],
              linestyle="--", color=style["baseline"], linewidth=1.5)
    axis.legend(frameon=False, loc="lower right")
    axis.set_title("Crossover · at least one mutant coordinate")
    for axis in axes:
        axis.set(xlabel="Normalized gene 1", ylabel="Normalized gene 2", xlim=(0, 1), ylim=(0, 1))
        axis.grid(alpha=0.14)
    figure.suptitle("Differential Evolution operators · actual seeded draw", fontsize=17)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(ROOT / f"figures/fig_de_operators_diagram.{suffix}")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", "--smoke", action="store_true", help="Small engineering check; never replaces full results")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args()
    if args.figures_only:
        render_figures()
        render_animation()
        render_operators()
        save_summary(json.loads((ROOT / "results/benchmarks.json").read_text(encoding="utf-8")))
        return
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["benchmarks"]
    settings = dict(dimensions=config["dimensions"], budget=1000 if args.quick else config["evaluations"],
                    runs=3 if args.quick else config["runs"], population=config["population"],
                    F=0.6, CR=0.9, GA="tournament-3 / BLX-0.5 / Gaussian sigma=0.1, p=1/D / elitism",
                    PSO="inertia=0.7298, c1=c2=1.49618, velocity limit=0.2, damped bounds", seed_range="0..runs-1")
    cache = ROOT / "results" / ("benchmark_runs_quick" if args.quick else "benchmark_runs")
    cache.mkdir(parents=True, exist_ok=True)
    tasks, records = [], []
    for name in FUNCTIONS:
        for algorithm in ALGORITHMS:
            for seed in range(settings["runs"]):
                file = cache / f"{name}_{algorithm.replace(' ', '')}_{seed:02d}.json"
                if file.exists():
                    record = json.loads(file.read_text())
                    if (record["evaluations"], record["dimensions"], record["population"]) == (
                            settings["budget"], settings["dimensions"], settings["population"]):
                        records.append(record)
                        continue
                tasks.append((name, algorithm, seed, settings["dimensions"], settings["budget"], settings["population"]))
    started = time.perf_counter()
    total_tasks = len(records) + len(tasks)
    print(f"Benchmark study: {len(records)} cached; running {len(tasks)} trials at {settings['budget']:,} evaluations", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            save_json(cache / f"{record['function']}_{record['algorithm'].replace(' ', '')}_{record['seed']:02d}.json", record)
            if len(records) % 25 == 0:
                print(f"{len(records)}/{total_tasks} complete; latest {record['function']} {record['algorithm']}: {record['final_error']:.5g}", flush=True)
    result = aggregate(records, settings, time.perf_counter() - started)
    output = ROOT / "results" / ("benchmarks_quick.json" if args.quick else "benchmarks.json")
    save_json(output, result)
    print(f"Saved {len(records)} real trials / {result['total_evaluations']:,} function evaluations to {output.name}", flush=True)
    if not args.quick:
        save_summary(result)
        render_figures(output)
        render_animation(make_animation_data())
        render_operators()
        print("Benchmark figures and 2-D animation saved.", flush=True)


if __name__ == "__main__":
    main()

"""Shared task-based MAML implementation for algorithm configuration."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import learn2learn as l2l
import numpy as np
import torch
from torch import nn


SEQUENCE_COLUMNS = ("r_j", "t_k", "p_j", "L_k", "S_k", "CP_k")


@dataclass(frozen=True)
class Observation:
    configuration: Any
    cost: float
    feature_vector: tuple[float, ...] = ()


@dataclass(frozen=True)
class TaskData:
    task_id: str
    observations: tuple[Observation, ...]
    problem_size: tuple[int, int] | None = None


class CostPredictor(nn.Module):
    """The common MLP base learner used by both predictors."""

    def __init__(self, input_dim: int, hidden_units: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 1),
        )

    def forward(self, inputs):
        return self.network(inputs)


def safe_sequence(value) -> tuple[float, ...]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"Expected a nonempty sequence, received {value!r}.")
    return tuple(float(item) for item in value)


def instance_feature_vector(row) -> np.ndarray:
    """Extract five scalars and four summary statistics for six sequences."""
    batch_bound = row["B"] if "B" in row and row["B"] is not None else row["N"]
    features = [
        float(row["N"]),
        float(row["T"]),
        float(batch_bound),
        float(row["seta"]),
        float(row["c"]),
    ]
    for name in SEQUENCE_COLUMNS:
        values = np.asarray(safe_sequence(row[name]), dtype=np.float64)
        features.extend(
            [float(values.mean()), float(values.std()), float(values.max()), float(values.min())]
        )
    return np.asarray(features, dtype=np.float32)


def instance_task_id(row) -> str:
    batch_bound = row["B"] if "B" in row and row["B"] is not None else row["N"]
    payload = {
        "scalars": [
            float(row["N"]),
            float(row["T"]),
            float(batch_bound),
            float(row["seta"]),
            float(row["c"]),
        ],
        "sequences": {
            name: list(safe_sequence(row[name])) for name in SEQUENCE_COLUMNS
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def split_tasks(
    tasks: Sequence[TaskData],
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    *,
    stratify_by_problem_size: bool = True,
    expected_task_count: int | None = None,
    expected_tasks_per_problem_size: int | None = None,
):
    """Split complete instance-level tasks without leaking configurations."""
    tasks = tuple(tasks)
    if not tasks:
        raise ValueError("No instance-level tasks were supplied.")
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("Invalid task split fractions.")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("The train, validation, and test fractions must sum to 1.")
    if expected_task_count is not None and len(tasks) != expected_task_count:
        raise ValueError(
            f"The training dataset contains {len(tasks)} tasks; "
            f"expected {expected_task_count}."
        )

    count = len(tasks)
    if count < 3:
        return tasks, tuple(), tuple()

    train_end = max(1, int(round(count * train_fraction)))
    validation_count = max(1, int(round(count * validation_fraction)))
    train_end = min(train_end, count - 2)
    validation_end = min(train_end + validation_count, count - 1)
    targets = (train_end, validation_end - train_end, count - validation_end)

    if not stratify_by_problem_size:
        shuffled = sorted(tasks, key=lambda task: task.task_id)
        random.Random(seed).shuffle(shuffled)
        return (
            tuple(shuffled[:train_end]),
            tuple(shuffled[train_end:validation_end]),
            tuple(shuffled[validation_end:]),
        )

    if any(task.problem_size is None for task in tasks):
        raise ValueError(
            "Every task must define problem_size=(N, T) for stratified splitting."
        )

    groups = defaultdict(list)
    for task in sorted(tasks, key=lambda item: item.task_id):
        groups[tuple(task.problem_size)].append(task)
    if expected_tasks_per_problem_size is not None:
        invalid = {
            size: len(group)
            for size, group in groups.items()
            if len(group) != expected_tasks_per_problem_size
        }
        if invalid:
            details = ", ".join(
                f"{size}: {size_count}" for size, size_count in sorted(invalid.items())
            )
            raise ValueError(
                f"Each problem size expected {expected_tasks_per_problem_size} "
                f"tasks; found {details}."
            )

    allocation = {}
    remaining_by_group = {}
    assigned_totals = [0, 0, 0]
    for size, group in groups.items():
        counts = [int(math.floor(len(group) * fraction)) for fraction in fractions]
        if any(value == 0 for value in counts):
            raise ValueError(
                f"Problem size {size} has too few tasks ({len(group)}) to place "
                "at least one task in every split."
            )
        allocation[size] = counts
        remaining_by_group[size] = len(group) - sum(counts)
        for index, value in enumerate(counts):
            assigned_totals[index] += value

    deficits = [target - assigned for target, assigned in zip(targets, assigned_totals)]
    if any(value < 0 for value in deficits):
        raise ValueError("The requested global split is incompatible with the strata.")

    rng = random.Random(seed)
    allocation_order = sorted(groups)
    rng.shuffle(allocation_order)
    for size in allocation_order:
        for _ in range(remaining_by_group[size]):
            candidates = [index for index, value in enumerate(deficits) if value > 0]
            if not candidates:
                raise ValueError("Unable to complete the requested stratified split.")
            largest_deficit = max(deficits[index] for index in candidates)
            tied = [
                index for index in candidates if deficits[index] == largest_deficit
            ]
            selected = rng.choice(tied)
            allocation[size][selected] += 1
            deficits[selected] -= 1
    if any(deficits):
        raise ValueError("Unable to match the requested global split sizes.")

    split_lists = [[], [], []]
    for size in sorted(groups):
        group = list(groups[size])
        rng.shuffle(group)
        start = 0
        for split_index, split_count in enumerate(allocation[size]):
            split_lists[split_index].extend(group[start : start + split_count])
            start += split_count
    return tuple(tuple(items) for items in split_lists)


def summarize_task_split(train_tasks, validation_tasks, test_tasks):
    """Return a serializable summary of task counts by problem size."""
    summary = {}
    for name, tasks in (
        ("train", train_tasks),
        ("validation", validation_tasks),
        ("test", test_tasks),
    ):
        by_size = Counter(task.problem_size for task in tasks)
        summary[name] = {
            "task_count": len(tasks),
            "by_problem_size": {
                f"N={N},T={T}": count
                for (N, T), count in sorted(by_size.items())
            },
        }
    return summary


def split_support_query(
    task: TaskData, support_size: int | None, seed: int
) -> tuple[tuple[Observation, ...], tuple[Observation, ...]]:
    observations = list(task.observations)
    if len(observations) < 2:
        raise ValueError(f"Task {task.task_id} needs at least two observations.")
    random.Random(seed).shuffle(observations)
    if support_size is None:
        support_size = max(1, len(observations) // 2)
    support_size = max(1, min(int(support_size), len(observations) - 1))
    return tuple(observations[:support_size]), tuple(observations[support_size:])


def _all_observations(tasks: Iterable[TaskData]):
    return [observation for task in tasks for observation in task.observations]


def fit_standardizers(tasks: Sequence[TaskData]):
    observations = _all_observations(tasks)
    if not observations:
        raise ValueError("The training split contains no observations.")
    inputs = np.asarray([item.feature_vector for item in observations], dtype=np.float32)
    targets = np.asarray([item.cost for item in observations], dtype=np.float32)
    x_mean = inputs.mean(axis=0)
    x_scale = inputs.std(axis=0)
    x_scale[x_scale < 1e-8] = 1.0
    y_mean = float(targets.mean())
    y_scale = float(targets.std())
    if y_scale < 1e-8:
        y_scale = 1.0
    return x_mean, x_scale, y_mean, y_scale


def _tensors(observations, x_mean, x_scale, y_mean, y_scale):
    inputs = np.asarray([item.feature_vector for item in observations], dtype=np.float32)
    targets = np.asarray([item.cost for item in observations], dtype=np.float32)
    inputs = (inputs - x_mean) / x_scale
    targets = (targets - y_mean) / y_scale
    return torch.tensor(inputs), torch.tensor(targets).reshape(-1, 1)


def _adapt_and_query_loss(
    maml,
    task,
    x_mean,
    x_scale,
    y_mean,
    y_scale,
    inner_steps,
    support_size,
    seed,
):
    support, query = split_support_query(task, support_size, seed)
    support_x, support_y = _tensors(support, x_mean, x_scale, y_mean, y_scale)
    query_x, query_y = _tensors(query, x_mean, x_scale, y_mean, y_scale)
    learner = maml.clone()
    for _ in range(inner_steps):
        learner.adapt(nn.functional.mse_loss(learner(support_x), support_y))
    return nn.functional.mse_loss(learner(query_x), query_y)


def train_task_maml(
    tasks: Sequence[TaskData],
    model_path: str | Path,
    metadata: dict[str, Any],
    *,
    hidden_units: int = 128,
    epochs: int = 300,
    meta_batch_size: int = 8,
    inner_lr: float = 0.01,
    outer_lr: float = 0.001,
    inner_steps: int = 3,
    patience: int = 30,
    seed: int = 42,
    support_size: int | None = None,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    stratify_by_problem_size: bool = True,
    expected_task_count: int | None = None,
    expected_tasks_per_problem_size: int | None = None,
):
    """Train exact MAML episodes whose support and query sets share one task."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_tasks, validation_tasks, test_tasks = split_tasks(
        tasks,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        stratify_by_problem_size=stratify_by_problem_size,
        expected_task_count=expected_task_count,
        expected_tasks_per_problem_size=expected_tasks_per_problem_size,
    )
    split_summary = summarize_task_split(
        train_tasks, validation_tasks, test_tasks
    )
    print(
        "Task split: "
        f"train={len(train_tasks)}, validation={len(validation_tasks)}, "
        f"test={len(test_tasks)} "
        f"({'stratified by (N, T)' if stratify_by_problem_size else 'random'})"
    )
    x_mean, x_scale, y_mean, y_scale = fit_standardizers(train_tasks)
    model = CostPredictor(len(x_mean), hidden_units)
    maml = l2l.algorithms.MAML(model, lr=inner_lr, first_order=False)
    optimizer = torch.optim.Adam(maml.parameters(), lr=outer_lr)

    best_state = copy.deepcopy(model.state_dict())
    best_validation = math.inf
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_tasks = list(train_tasks)
        random.Random(seed + epoch).shuffle(epoch_tasks)
        train_losses = []

        for start in range(0, len(epoch_tasks), meta_batch_size):
            meta_batch = epoch_tasks[start : start + meta_batch_size]
            query_losses = [
                _adapt_and_query_loss(
                    maml,
                    task,
                    x_mean,
                    x_scale,
                    y_mean,
                    y_scale,
                    inner_steps,
                    support_size,
                    seed + epoch * 100_000 + start + offset,
                )
                for offset, task in enumerate(meta_batch)
            ]
            meta_loss = torch.stack(query_losses).mean()
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()
            train_losses.append(float(meta_loss.detach()))

        validation_source = validation_tasks or train_tasks
        validation_losses = []
        model.eval()
        with torch.enable_grad():
            for offset, task in enumerate(validation_source):
                loss = _adapt_and_query_loss(
                    maml,
                    task,
                    x_mean,
                    x_scale,
                    y_mean,
                    y_scale,
                    inner_steps,
                    support_size,
                    seed + 9_000_000 + offset,
                )
                validation_losses.append(float(loss.detach()))

        train_rmse = math.sqrt(float(np.mean(train_losses))) * y_scale
        validation_rmse = math.sqrt(float(np.mean(validation_losses))) * y_scale
        history.append(
            {"epoch": epoch, "train_rmse": train_rmse, "validation_rmse": validation_rmse}
        )

        if validation_rmse < best_validation - 1e-9:
            best_validation = validation_rmse
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:3d} | train RMSE {train_rmse:.3f} | "
                f"validation RMSE {validation_rmse:.3f}"
            )
        if stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    test_rmse = evaluate_task_maml(
        model,
        test_tasks,
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        inner_lr,
        inner_steps,
        support_size,
        seed + 20_000_000,
    )

    bundle = {
        "format_version": 1,
        "model_state": model.state_dict(),
        "input_dim": len(x_mean),
        "hidden_units": hidden_units,
        "x_mean": x_mean.tolist(),
        "x_scale": x_scale.tolist(),
        "y_mean": y_mean,
        "y_scale": y_scale,
        "inner_lr": inner_lr,
        "inner_steps": inner_steps,
        "metadata": metadata,
        "history": history,
        "split_strategy": (
            "stratified_by_problem_size"
            if stratify_by_problem_size
            else "random_at_task_level"
        ),
        "split_seed": seed,
        "split_fractions": {
            "train": train_fraction,
            "validation": validation_fraction,
            "test": test_fraction,
        },
        "split_summary": split_summary,
        "split_task_ids": {
            "train": [task.task_id for task in train_tasks],
            "validation": [task.task_id for task in validation_tasks],
            "test": [task.task_id for task in test_tasks],
        },
        "test_rmse": test_rmse,
    }
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, model_path)
    return bundle


def evaluate_task_maml(
    model,
    tasks,
    x_mean,
    x_scale,
    y_mean,
    y_scale,
    inner_lr,
    inner_steps,
    support_size,
    seed,
):
    if not tasks:
        return None
    losses = []
    maml = l2l.algorithms.MAML(model, lr=inner_lr, first_order=False)
    with torch.enable_grad():
        for offset, task in enumerate(tasks):
            loss = _adapt_and_query_loss(
                maml,
                task,
                x_mean,
                x_scale,
                y_mean,
                y_scale,
                inner_steps,
                support_size,
                seed + offset,
            )
            losses.append(float(loss.detach()))
    return math.sqrt(float(np.mean(losses))) * y_scale


def load_bundle(model_path: str | Path):
    bundle = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    model = CostPredictor(bundle["input_dim"], bundle["hidden_units"])
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return bundle, model


def predict_observation_costs(
    bundle,
    model,
    observations: Sequence[Observation],
):
    """Predict candidate costs from the learned shared initialization."""
    x_mean = np.asarray(bundle["x_mean"], dtype=np.float32)
    x_scale = np.asarray(bundle["x_scale"], dtype=np.float32)
    y_mean = float(bundle["y_mean"])
    y_scale = float(bundle["y_scale"])

    candidate_x, _ = _tensors(
        [Observation(item.configuration, y_mean, item.feature_vector) for item in observations],
        x_mean,
        x_scale,
        y_mean,
        y_scale,
    )
    model.eval()
    with torch.no_grad():
        normalized = model(candidate_x).reshape(-1).numpy()
    return normalized * y_scale + y_mean

"""Task-based MAML predictor for CGDH algorithm configurations."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd

import maml_config as settings
from maml_core import (
    Observation,
    TaskData,
    instance_feature_vector,
    instance_task_id,
    load_bundle,
    predict_observation_costs,
    train_task_maml,
)


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "result_CGDH.csv"
MODEL_PATH = PROJECT_DIR / "models" / "maml_cgdh_task_model.pt"
FEATURE_SCHEMA_VERSION = 2
POLICY_COST_COLUMNS = OrderedDict(
    (("used", "HU_cost"), ("cost", "HC_cost"), ("maxcols", "HM_cost"))
)


def _raw_instance_key(row):
    batch_bound = row["B"] if "B" in row and row["B"] is not None else row["N"]
    return (str(row["N"]), str(row["T"]), str(batch_bound)) + tuple(
        str(row[name])
        for name in ("seta", "c", "r_j", "t_k", "p_j", "L_k", "S_k", "CP_k")
    )


def _configuration_features(
    row, configuration, base_features=None
) -> tuple[float, ...]:
    _max_iter, max_new_cols, policy = configuration
    policy = str(policy).strip().lower()
    if policy not in settings.CGDH_POLICIES:
        raise ValueError(f"Unknown CGDH recovery policy: {policy!r}.")
    one_hot_policy = [float(policy == name) for name in POLICY_COST_COLUMNS]
    if base_features is None:
        base_features = instance_feature_vector(row)
    features = np.concatenate(
        (
            base_features,
            np.asarray([max_new_cols, *one_hot_policy], dtype=np.float32),
        )
    )
    return tuple(float(value) for value in features)


def build_cgdh_tasks(frame: pd.DataFrame) -> tuple[TaskData, ...]:
    """Build one MAML task per instance without cross-joining configurations."""
    required = {
        "N",
        "T",
        "seta",
        "c",
        "r_j",
        "t_k",
        "p_j",
        "L_k",
        "S_k",
        "CP_k",
        "MAX_ITER",
        "MAX_NEW_COLS",
        *POLICY_COST_COLUMNS.values(),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CGDH data are missing columns: {sorted(missing)}")

    data = frame.copy()
    numeric_columns = ["MAX_ITER", "MAX_NEW_COLS", *POLICY_COST_COLUMNS.values()]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    fixed_max_iter = int(settings.CGDH_MAX_ITER)
    data = data[data["MAX_ITER"] == fixed_max_iter].copy()
    if data.empty:
        raise ValueError(
            f"CGDH data contain no observations with MAX_ITER={fixed_max_iter}."
        )

    grouped: dict[str, dict[tuple[int, int, str], Observation]] = {}
    instance_cache = {}
    task_sizes = {}
    for _, row in data.iterrows():
        if pd.isna(row["MAX_ITER"]) or pd.isna(row["MAX_NEW_COLS"]):
            continue
        raw_key = _raw_instance_key(row)
        if raw_key not in instance_cache:
            instance_cache[raw_key] = (
                instance_task_id(row),
                instance_feature_vector(row),
                (int(row["N"]), int(row["T"])),
            )
        task_id, base_features, problem_size = instance_cache[raw_key]
        task_sizes[task_id] = problem_size
        observations = grouped.setdefault(task_id, {})
        for policy, cost_column in POLICY_COST_COLUMNS.items():
            if pd.isna(row[cost_column]):
                continue
            configuration = (
                int(row["MAX_ITER"]),
                int(row["MAX_NEW_COLS"]),
                policy,
            )
            observations[configuration] = Observation(
                configuration=configuration,
                cost=float(row[cost_column]),
                feature_vector=_configuration_features(
                    row, configuration, base_features=base_features
                ),
            )

    tasks = tuple(
        TaskData(
            task_id,
            tuple(observations.values()),
            problem_size=task_sizes[task_id],
        )
        for task_id, observations in sorted(grouped.items())
        if len(observations) >= 2
    )
    if not tasks:
        raise ValueError("No valid CGDH tasks could be constructed.")
    return tasks


def _metadata_from_tasks(tasks):
    configurations = sorted(
        {
            observation.configuration
            for task in tasks
            for observation in task.observations
        }
    )
    return {
        "predictor": "CGDH",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "fixed_max_iter": int(settings.CGDH_MAX_ITER),
        "configurations": configurations,
    }


def train_cgdh_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
):
    frame = pd.read_csv(data_path)
    tasks = build_cgdh_tasks(frame)
    print(
        f"CGDH data: {len(tasks)} instance-level tasks and "
        f"{sum(len(task.observations) for task in tasks)} observations."
    )
    bundle = train_task_maml(
        tasks,
        model_path=model_path,
        metadata=_metadata_from_tasks(tasks),
        hidden_units=settings.HIDDEN_UNITS,
        epochs=settings.TRAINING_EPOCHS,
        meta_batch_size=settings.META_BATCH_SIZE,
        inner_lr=settings.INNER_LEARNING_RATE,
        outer_lr=settings.OUTER_LEARNING_RATE,
        inner_steps=settings.INNER_STEPS,
        patience=settings.EARLY_STOPPING_PATIENCE,
        seed=settings.RANDOM_SEED,
        support_size=settings.CGDH_TRAIN_SUPPORT_SIZE,
        train_fraction=settings.TRAIN_TASK_FRACTION,
        validation_fraction=settings.VALIDATION_TASK_FRACTION,
        test_fraction=settings.TEST_TASK_FRACTION,
        stratify_by_problem_size=settings.STRATIFY_TASKS_BY_PROBLEM_SIZE,
        expected_task_count=settings.EXPECTED_MAML_TASK_COUNT,
        expected_tasks_per_problem_size=settings.EXPECTED_TASKS_PER_PROBLEM_SIZE,
    )
    print(f"CGDH-MAML model saved to {Path(model_path).resolve()}")
    return bundle


def _load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Train the CGDH-MAML model before prediction."
        )
    bundle, model = load_bundle(model_path)
    version = bundle.get("metadata", {}).get("feature_schema_version")
    if version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            "The saved CGDH-MAML model does not match the published feature schema."
        )
    return bundle, model


def _candidate_configurations(metadata):
    candidates = tuple(
        (settings.CGDH_MAX_ITER, max_new_cols, policy)
        for max_new_cols in settings.CGDH_NEW_COLUMN_LIMITS
        for policy in settings.CGDH_POLICIES
    )
    trained = {tuple(item) for item in metadata["configurations"]}
    missing = [candidate for candidate in candidates if candidate not in trained]
    if missing:
        raise ValueError(
            "The saved CGDH-MAML model is missing published candidates: "
            f"{missing}."
        )
    return candidates


def select_cgdh_configuration(
    instance,
    model_path: str | Path = MODEL_PATH,
):
    """Select the candidate configuration with the lowest predicted cost."""
    bundle, model = _load_model(model_path)
    candidates = _candidate_configurations(bundle["metadata"])
    candidate_observations = [
        Observation(
            configuration,
            0.0,
            _configuration_features(instance, configuration),
        )
        for configuration in candidates
    ]
    predictions = predict_observation_costs(bundle, model, candidate_observations)
    estimated_costs = {
        observation.configuration: float(prediction)
        for observation, prediction in zip(candidate_observations, predictions)
    }
    configuration = min(estimated_costs, key=estimated_costs.get)
    return {
        "configuration": configuration,
        "predicted_cost": estimated_costs[configuration],
    }


def main():
    import config

    required_fields = (
        "N", "T", "B", "r_j", "seta", "c",
        "t_k", "p_j", "L_k", "S_k", "CP_k",
    )
    missing = [name for name in required_fields if not hasattr(config, name)]
    if missing:
        raise AttributeError(
            "config.py is incomplete. Missing fields: " + ", ".join(missing)
        )
    instance = {name: getattr(config, name) for name in required_fields}
    print("Scheduling instance loaded from config.py")
    result = select_cgdh_configuration(instance)
    print("Selected CGDH configuration:", result["configuration"])
    print(f"Estimated production cost: {result['predicted_cost']:.3f}")


if __name__ == "__main__":
    main()

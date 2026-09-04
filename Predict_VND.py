"""Task-based MAML predictor for VND neighborhood sequences."""

from __future__ import annotations

import ast
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
DATA_PATH = PROJECT_DIR / "data" / "result_VND.csv"
MODEL_PATH = PROJECT_DIR / "models" / "maml_vnd_task_model.pt"


def canonical_sequence(value) -> str:
    if isinstance(value, (tuple, list, np.ndarray)):
        values = value
    else:
        text = str(value).strip()
        try:
            parsed = ast.literal_eval(text)
            values = parsed if isinstance(parsed, (tuple, list)) else text.split(",")
        except (ValueError, SyntaxError):
            values = text.replace("[", "").replace("]", "").split(",")
    return ",".join(str(int(item)) for item in values)


def sequence_tuple(value) -> tuple[int, ...]:
    return tuple(int(item) for item in canonical_sequence(value).split(","))


def _raw_instance_key(row):
    batch_bound = row["B"] if "B" in row and row["B"] is not None else row["N"]
    return (str(row["N"]), str(row["T"]), str(batch_bound)) + tuple(
        str(row[name])
        for name in ("seta", "c", "r_j", "t_k", "p_j", "L_k", "S_k", "CP_k")
    )


def _configuration_features(
    row, sequence, sequence_names, base_features=None
) -> tuple[float, ...]:
    sequence = canonical_sequence(sequence)
    one_hot = [float(sequence == name) for name in sequence_names]
    if base_features is None:
        base_features = instance_feature_vector(row)
    return tuple(
        float(value)
        for value in np.concatenate(
            (base_features, np.asarray(one_hot, dtype=np.float32))
        )
    )


def build_vnd_tasks(frame: pd.DataFrame):
    """Build one MAML task per instance and preserve each measured sequence."""
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
        "NEIGHBORS",
        "Target",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"VND data are missing columns: {sorted(missing)}")

    data = frame.copy()
    data["Target"] = pd.to_numeric(data["Target"], errors="coerce")
    data["_sequence"] = data["NEIGHBORS"].map(canonical_sequence)
    sequence_names = tuple(sorted(data["_sequence"].dropna().unique()))

    grouped: dict[str, dict[str, Observation]] = {}
    instance_cache = {}
    task_sizes = {}
    for _, row in data.iterrows():
        if pd.isna(row["Target"]):
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
        sequence = row["_sequence"]
        grouped.setdefault(task_id, {})[sequence] = Observation(
            configuration=sequence,
            cost=float(row["Target"]),
            feature_vector=_configuration_features(
                row, sequence, sequence_names, base_features=base_features
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
        raise ValueError("No valid VND tasks could be constructed.")
    return tasks, sequence_names


def train_vnd_model(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = MODEL_PATH,
):
    tasks, sequence_names = build_vnd_tasks(pd.read_csv(data_path))
    print(
        f"VND data: {len(tasks)} instance-level tasks and "
        f"{sum(len(task.observations) for task in tasks)} observations."
    )
    bundle = train_task_maml(
        tasks,
        model_path=model_path,
        metadata={"predictor": "VND", "sequences": sequence_names},
        hidden_units=settings.HIDDEN_UNITS,
        epochs=settings.TRAINING_EPOCHS,
        meta_batch_size=settings.META_BATCH_SIZE,
        inner_lr=settings.INNER_LEARNING_RATE,
        outer_lr=settings.OUTER_LEARNING_RATE,
        inner_steps=settings.INNER_STEPS,
        patience=settings.EARLY_STOPPING_PATIENCE,
        seed=settings.RANDOM_SEED,
        support_size=settings.VND_TRAIN_SUPPORT_SIZE,
        train_fraction=settings.TRAIN_TASK_FRACTION,
        validation_fraction=settings.VALIDATION_TASK_FRACTION,
        test_fraction=settings.TEST_TASK_FRACTION,
        stratify_by_problem_size=settings.STRATIFY_TASKS_BY_PROBLEM_SIZE,
        expected_task_count=settings.EXPECTED_MAML_TASK_COUNT,
        expected_tasks_per_problem_size=settings.EXPECTED_TASKS_PER_PROBLEM_SIZE,
    )
    print(f"VND-MAML model saved to {Path(model_path).resolve()}")
    return bundle


def _load_model(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} does not exist. Train the VND-MAML model before prediction."
        )
    return load_bundle(model_path)


def select_vnd_configuration(
    instance,
    model_path: str | Path = MODEL_PATH,
):
    """Select the candidate neighborhood sequence with the lowest predicted cost."""
    bundle, model = _load_model(model_path)
    sequence_names = tuple(bundle["metadata"]["sequences"])
    candidate_observations = [
        Observation(
            name,
            0.0,
            _configuration_features(instance, name, sequence_names),
        )
        for name in sequence_names
    ]

    predictions = predict_observation_costs(bundle, model, candidate_observations)
    estimated_costs = {
        observation.configuration: float(prediction)
        for observation, prediction in zip(candidate_observations, predictions)
    }
    predicted_name = min(estimated_costs, key=estimated_costs.get)
    return {
        "configuration": sequence_tuple(predicted_name),
        "predicted_cost": estimated_costs[predicted_name],
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
    result = select_vnd_configuration(instance)
    print("Selected VND neighborhood sequence:", result["configuration"])
    print(f"Estimated production cost: {result['predicted_cost']:.3f}")


if __name__ == "__main__":
    main()

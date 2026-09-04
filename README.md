# Serial-batch scheduling in cloud manufacturing

This repository contains the implementation accompanying the manuscript
*Meta-learning-enhanced column generation and variable neighborhood descent
frameworks*.

## Main programs

- `CPLEX.py`: the MILP formulation solved with IBM ILOG CPLEX.
- `CGDH_random.py`: CGDH with random configuration selection.
- `CGDH_MAML.py`: CGDH with offline MAML-based configuration selection.
- `VND_random.py`: VND with random neighborhood-sequence selection.
- `VND_MAML.py`: VND with offline MAML-based configuration selection.
- `Predict_CGDH.py` and `Predict_VND.py`: MAML training and prediction code.
- `maml_core.py`: shared task construction, meta-training, and inference logic.
- `maml_config.py`: the parameters reported in the manuscript.

The published MAML variants load the learned shared initialization, predict the
cost of every candidate configuration, and select the candidate with the lowest
predicted cost.

## Running an instance

1. Copy an instance definition into `config.py`.
2. Open the required main program in VS Code.
3. Select the appropriate Python environment and click **Run Python File**.

Before running a MAML-based algorithm, train the corresponding predictor as
described below. The programs print the run-level results and summary statistics
in the terminal.

## Retraining the predictors

Training data and trained model weights are not distributed with this
repository. Place `result_CGDH.csv` and `result_VND.csv` in `data/`, then run
`train_cgdh_model()` in `Predict_CGDH.py` and `train_vnd_model()` in
`Predict_VND.py`, respectively. These functions reproduce the two
meta-training workflows and save the resulting weights under the local
`models/` directory. Both `data/` and `models/` are ignored by Git. Training is
never started automatically when a model file is missing.

## Solver requirements

CGDH requires a licensed Gurobi installation. Install `requirements.txt` in the
main algorithm environment. `CPLEX.py` requires a licensed IBM ILOG CPLEX
installation and the packages listed in `requirements-cplex.txt`. The CPLEX
program is independent of the PyTorch/MAML environment and can be run from the
compatible Python environment supplied for the solver installation.
Set `CPLEX_TIME_LIMIT` at the top of `CPLEX.py` to `3600` for small- and
medium-scale instances or `9000` for large-scale instances.

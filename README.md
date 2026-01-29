# Methods for the Serial-Batch Scheduling Problem in Cloud Manufacturing 

This repository provides implementations of solution methods for the serial-batch scheduling problem in cloud manufacturing environments. It includes:

- a column generation-driven heuristic (**CGDH**) integrated with model-agnostic meta-learning (**MAML**)
- a model-agnostic meta-learning-boosted variable neighborhood descent (**VND**) algorithm
- mixed-integer linear programming (**MILP**) model solved via CPLEX

# Requirements

- Python 3.7 or higher (When using CPLEX 12.10, please note potential compatibility issues and ensure proper setup)

# Dependencies
- numpy>=1.19.0
- pandas>=1.1.0
- scikit-learn>=0.23.0
- torch>=1.7.0
- learn2learn>=0.1.0
- matplotlib>=3.2.0
- joblib>=0.14.0
- gurobipy>=9.1.0(academic licence free)
- docplex>=2.15.0
- CPLEX>=12.10.0  # IBM CPLEX Studio must be installed first

# Running
1. Clone the repository:
```
git https://github.com/ZXR2001/Serial-batch-scheduling-in-cloud-manufacturing.git
```
2. Install dependencies and run an example:
```
python CGDH_MAML.py
```

# Project Structure
- ```config.py```:  Configuration module that defines all tunable parameters and settings. Includes a sample instance for quick algorithm testing and benchmarking.
- ```CPLEX.py```:  Implements an **exact solver** for the problem using the **CPLEX** optimizer (formulated as a mixed-integer linear program — MILP).
- ```Predict_GC.py```:  Deploys a **MAML** model to predict key parameters for the **CGDH**, including:  
  maximum number of iterations, number of new columns per iteration, and scheduling policies.
- ```GC_random.py```:  Baseline **CGDH** that randomly samples the key parameters from the candidate ranges.
- ```GCDH_MAML.py```:  Enhanced **CGDH** where **MAML** is used to intelligently select the key parameters.
- ```Predict_nei.py```:  Deploys a **MAML** model to predict key parameters for **VND** algorithm, including:
  neighborhood structures and their order.
- ```VND_random.py```:  Baseline **VND** that randomly samples the neighborhood structures and orders from predefined candidates.
- ```VND_MAML.py```:   Enhanced **VND** assisted by **MAML**-predicted neighborhood selection and ordering.

# Example Output
After running the ```CPLEX.py``` script, you will see the detailed Gurobi optimization log in the console, including presolve statistics, root relaxation, branch-and-bound progress (nodes explored, incumbent solutions, best bound, gap), and the final results — all printed progressively within the time deadline.

After running - ```GC_random.py```, ```GCDH_MAML.py```, ```VND_random.py```, or ```VND_MAML.py```, you will see the solution process details (including initial solution and cost), along with a summary of 10 independent runs — best cost, average cost, and average runtime — as well as the cost, solution, and time for each individual run.

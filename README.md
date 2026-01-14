# Methods for the Serial-Batch Scheduling Problem in Cloud Manufacturing 

This project implements meta-learning-enhanced column generation  heuristic, variable neighborhood descent and mixed-integer linear programming model to solve a serial-batch scheduling  problem in cloud manufacturing.

# Requirements

- Python 3.10 (newer 3.9–3.11 work, but CI is pinned to 3.10)
- Gurobi ≥ 10.0 (academic licence free)

# Dependencies
- gurobipy>=10.0
- numpy>=1.23
- pandas>=1.5
- scikit-learn>=1.2
- torch>=2.0
- learn2learn>=0.1.7
- matplotlib>=3.6
- joblib>=1.2

Install all required packages with one command:

```
pip install gurobipy>=10.0 numpy>=1.23 pandas>=1.5 scikit-learn>=1.2 torch>=2.0 learn2learn>=0.1.7 matplotlib>=3.6 joblib>=1.2
```

# Running
1. Clone the repository:
```
git https://github.com/ZXR2001/Serial-batch-scheduling-in-cloud-manufacturing-Meta-learning-enhanced-column-generation-and-variable.git
```
2. After installing required dependencies run the Python files directly, for example: 
```
python CGDH_MAML.py
```

# Project Structure
- ```config.py```: Configuration file containing parameters for the project. We provide a sample instance for testing algorithms.
- ```gurobi.py```: Implements an exact solver for the problem using the Gurobi optimizer.
- ```Predict_GC.py```: Deploys a **Model-Agnostic Meta-Learning (MAML)** model to predict key parameters for the column generation heuristic (CGDH), including:  
  maximum number of iterations, number of new columns per iteration, and heuristic strategy
- ```GC_random.py```:   Baseline column generation heuristic (**CGDH**) that randomly samples the key parameters from the candidate ranges.
- ```GCDH_MAML.py```:  Enhanced column generation heuristic (**CGDH**) where **MAML** is used to intelligently select the key parameters.
- ```Predict_nei.py```:  Deploys a **MAML** model to predict key parameters for variable neighborhood descent (**VND**), including:
  neighborhood structures and their order.
- ```VND_random.py```:  Baseline variable neighborhood descent (**VND**) that randomly samples the neighborhood structures and orders from predefined candidates.
- ```VND_MAML.py```:   Enhanced variable neighborhood descent (**VND**) assisted by **MAML**-predicted neighborhood selection and ordering.

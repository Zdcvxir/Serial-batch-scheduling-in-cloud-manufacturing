import gurobipy as gp
from gurobipy import GRB
import time
import random,os,sys
from collections import defaultdict, deque
import numpy as np
import pandas as pd

best_obj = float('inf') 
best_solution = None
protected_cols = set()
Heuristic_solution_list = []

def reload_config():
    try:
        import importlib
        import config
        importlib.reload(config)
        return (config.N, config.T, config.B,
                config.r_j, config.seta, config.c,
                config.t_k, config.p_j,
                config.L_k, config.S_k, config.CP_k)
    except (ImportError, AttributeError):
        print('config.py not found')


def initialize_model():
    global master_model, columns, W_vars, dual_zeta, dual_sigma
    global N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k, column_fingerprints
    global T_dict, J_dict

    N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k = reload_config()

    T_dict = {a: (L_k[a-1], t_k[a-1], CP_k[a-1]) for a in range(1, T+1)}
    J_dict = {a: (p_j[a-1], r_j[a-1]) for a in range(1, N+1)}
    column_fingerprints = []
    
    master_model = gp.Model("ColumnGeneration")
    master_model.setParam("OutputFlag", 0)
    
    columns = []
    W_vars = {}
    
    dual_zeta = [0.0] * N
    dual_sigma = [0.0] * (T+1)

    _initialize_columns()
    _build_master_problem()
    
def _initialize_columns():
    sorted_T = sorted(T_dict.items(), key= lambda item: item[1][1] + item[1][2] / item[1][0])
    sorted_J = sorted(J_dict.items(), key=lambda item: item[1][1])
    j = 0
    k = 0
    Match = {i[0]: [] for i in sorted_T}
    time_slots = list(Match.keys())
    Total_Target = 0
    while j < N:
        
        next_schedule = [t[0] for t in Match[time_slots[k]]] + [sorted_J[j][0]]
        _, test_time = _calculate_target(next_schedule, sorted_T[k][0])

        if test_time <= sorted_T[k][1][0]:
            Match[time_slots[k]].append(sorted_J[j])
            j += 1
        else:
            k += 1

    Match = {key: value for key, value in Match.items() if value}

    for key, value_list in Match.items():
        Schedule = []
        Vector = [0 for _ in range(N)]
        Target = 0
        Timeslot = key

        for tup in value_list:
            Schedule.append(tup[0])
            if 0 <= tup[0]-1 < N:
                Vector[tup[0]-1] = 1
        Target = _calculate_target(Schedule, Timeslot)[0]
        _add_column(Schedule, Target, Vector, Timeslot)
        Total_Target += Target

    print(f"Initial solution value: {Total_Target}")
    _snapshot_feasible_solution(Match)

def _snapshot_feasible_solution(match_dict):
    ts_tasks = {}
    for ts, tasks in match_dict.items():
        if tasks and isinstance(tasks[0], tuple):
            ts_tasks[ts] = [t[0] for t in tasks]
        else:
            ts_tasks[ts] = tasks
    total_cost = sum(_calculate_target(ts_tasks[ts], ts)[0] for ts in match_dict)
    Heuristic_solution_list.append((total_cost, ts_tasks))
    return total_cost, ts_tasks

def _add_column(Schedule, Target, Vector, Timeslot):
    fingerprint = (tuple(Schedule), Timeslot)
    
    if fingerprint in column_fingerprints:
        return False
    
    column_fingerprints.append(fingerprint)
    
    s = len(columns)
    var = master_model.addVar(vtype=GRB.CONTINUOUS, obj=Target, name=f"W_{s}")
    W_vars[s] = var
    columns.append((Schedule, Target, Vector, Timeslot))
    return True
        
def _build_master_problem():
    if master_model.getAttr("NumConstrs") > 0:
        master_model.remove(list(master_model.getConstrs()))
    
    for j in range(N):
        expr = gp.quicksum(W_vars[s] * columns[s][2][j] for s in W_vars)
        master_model.addConstr(expr >= 1, name=f"cover_{j}")

    for k in range(1, T + 1):
        expr = gp.quicksum(W_vars[s] for s in W_vars if columns[s][3] == k)
        master_model.addConstr(expr <= 1, name=f"timeslot_{k}")

    if best_obj < float('inf'):
        master_model.setParam("BestObjStop", best_obj)
    
    master_model.update()

def _calculate_target(Schedule, Timeslot):
    if not Schedule:
        return 0.0
    Processing_time = 0
    sorted_J = []
    
    for key in Schedule:
        if key in J_dict:
            sorted_J.append((key, J_dict[key]))
    
    def DP(X, Y):
        F_k = {0: 0}
        m = []
        R_j = [i[1][1] for i in sorted_J]
        P_j = [i[1][0] for i in sorted_J]
        for i in range(1, X + 1):
            left = max(i - c, 0)
            right = i
            for l in range(left, right):
                r_j_max = max(R_j[l: right])
                a = max(F_k[l], r_j_max, S_k[Y-1])
                p_j_total = sum(P_j[l: right])
                m.append(a + seta + p_j_total)
            n = min(m)
            F_k[i] = n
            m = []
        F_k[X] = F_k[X] - S_k[Y-1]
        return F_k[X]
    
    Processing_time = DP(len(Schedule), Timeslot)

    if Processing_time > L_k[Timeslot-1]:
       Processing_time = 9999999 

    total = Processing_time * t_k[Timeslot-1] + CP_k[Timeslot-1]

    return total, Processing_time

def solve_master_problem():
    global dual_zeta, dual_sigma, best_obj, best_solution
    
    master_model.optimize()
    if master_model.status != GRB.OPTIMAL:
        raise Exception(f"Primary query resolution failed, status code: {master_model.status}")

    for j in range(N):
        dual_zeta[j] = master_model.getConstrByName(f"cover_{j}").Pi
    for k in range(1, T + 1):
        dual_sigma[k] = master_model.getConstrByName(f"timeslot_{k}").Pi

    if master_model.status == GRB.INFEASIBLE:
            print("The main issue is not feasible; analysis is underway...")
            master_model.computeIIS()
            for c in master_model.getConstrs():
                if c.IISConstr:
                    print(f"Conflict Constraints: {c.constrName}")
            raise Exception("The main problem is not feasible. Please check the input data.")
    
    if master_model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise Exception(f"Primary query resolution failed, status code: {master_model.status}")
  
    obj_val = master_model.objVal 

    # if obj_val < best_obj:
    #     best_obj = obj_val
    #     best_solution = _extract_current_solution()

    return obj_val

def _extract_current_solution():
    selected = [s for s, w in W_vars.items() if w.X > 0.1]
    ts_tasks = {}
    for s in selected:
        ts = columns[s][3]
        ts_tasks.setdefault(ts, []).extend(columns[s][0])
    return {ts: tasks for ts, tasks in ts_tasks.items()}

def solve_pricing_subproblem1(Iterations, max_new_cols):
    new_columns = []  
    attempts = 0

    while attempts < Iterations and len(new_columns) < max_new_cols:
        Vector = [0] * N
        Schedule = []
        Timeslot = None
        Target = 0

        for _ in range(100):
            M = random.randint(1, N)
            random_indices = random.sample(range(1, N + 1), M)
            Ts = random.randint(1, T)
            total, Processing_time = _calculate_target(random_indices, Ts)
            if Processing_time <= L_k[Ts - 1]:
                Schedule = random_indices
                Timeslot = Ts
                Target = total
                for index in random_indices:
                    Vector[index - 1] = 1
                break

        if not Schedule:
            continue  

        total_dual_zeta = sum(dual_zeta[j - 1] for j in Schedule)
        # print(total_dual_zeta)
        reduced_cost = Target - total_dual_zeta - dual_sigma[Timeslot]

        if reduced_cost < 100:
            new_columns.append((Schedule, Target, Vector, Timeslot))

        attempts += 1

    for schedule, target, vector, timeslot in new_columns:
        _add_column(schedule, target, vector, timeslot)

    _build_master_problem()

    min_reduced_cost = min([t[1] - sum(dual_zeta[j - 1] for j in t[0]) - dual_sigma[t[3]] for t in new_columns]) if new_columns else float('-inf')
    return min_reduced_cost

def remain_usage_based():
    global columns, W_vars, column_fingerprints, master_model

    master_model.optimize()
    if master_model.status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        return
    
    Usage_list = []
    for s, var in W_vars.items():
        x_val = abs(var.X) if var.X is not None else 0.0
        sched, tgt, vec, ts = columns[s]
        if x_val > 0:
            Usage_list.append((round(x_val, 3), sched, ts, int(round(tgt)))) 
    Usage_list.sort(key=lambda t: t[0], reverse=True)

    return Usage_list

def column_generation_loop(max_iter, tol, max_new_cols):
    global W_vars
    for iter in range(max_iter):
        try:
            obj_val = solve_master_problem()
            min_reduced_cost = solve_pricing_subproblem1(float('inf'), max_new_cols)
            print(f"Iter {iter + 1}: Primary Objective={obj_val:.2f}, Minimum reduced cost={min_reduced_cost:.4f}")

            if min_reduced_cost > tol:
                print("Column generation converged")
                break

        except Exception as e:
            print(f"Iteration failed: {str(e)}")
            break

def convert_to_integer():
    for var in W_vars.values():
        var.vtype = GRB.BINARY

    if best_solution: 

        for var in W_vars.values():
            var.Start = 0.0

        ts_target = {}
        ts_target = {ts: set(tasks) for ts, tasks in best_solution.items()}

        for s, (tasks, _, _, ts) in enumerate(columns):
            if ts not in ts_target:
                continue
            overlap = len(set(tasks) & ts_target[ts])
            if overlap == len(ts_target[ts]):
                W_vars[s].Start = 1.0
                break

    master_model.setParam("MIPGap", 0.01)
    master_model.setParam("TimeLimit", 7200)

def get_optimal_solution():
    global best_solution

    if master_model.status == GRB.LOADED and Heuristic_solution_list:
        best_cost, best_snap = min(Heuristic_solution_list, key=lambda x: x[0])
        print(f"[Heuristic] Select the best solution from {len(Heuristic_solution_list)} candidate solutions"
              f"Cost={best_cost:.2f}")
        print(best_snap)
        return best_snap, best_cost
    
    if master_model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        if master_model.status == GRB.SUBOPTIMAL:
            print("Warning: Suboptimal solution found; did not meet optimality tolerance.")
        return _extract_current_solution()

    if best_solution:
        print("Warning: Using the optimal solution generated during the column generation phase; no better solution was found by the integer programming algorithm.")
        return best_solution

    return {}

def print_columns():
    print("\nDetails for all columns: ")
    for idx, (Schedule, Target, Vector, Timeslot) in enumerate(columns):
        print(f"Column {idx}:")
        print(f"  Sequence: {Schedule}")
        print(f"  Single-column cost: {Target:.2f}")
        print(f"  Coverage Vector: {Vector}")
        print(f"  production time slot: {Timeslot}")
        print()

def print_solution(solution: dict):
    print("\nBest scheme:")

    ts_cost = {ts: (_calculate_target(tasks, ts)[0], tasks)
               for ts, tasks in solution.items()}

    total_target = 0

    for ts in sorted(ts_cost.keys()):
        cost, tasks = ts_cost[ts]
        print(f"Time slot {ts}: job sequence {tasks}, cost {cost:.1f}")
        total_target += cost

    print(f"\nTotal production cost: {total_target:.1f}")

def Heuristic_solution(usage_list, strategy):

    used_jobs = set()
    ts_jobs   = {}
    ts_cap    = {k: L_k[k-1] for k in range(1, T+1)}
    selected_whole_cols = [] 

    if strategy == 'used':
        for use, sched, ts, tgt in usage_list:
            if not sched:
                continue
            if used_jobs.intersection(set(sched)):
                continue
            curr_dur = _calculate_target(ts_jobs.get(ts, []), ts)[1] if ts in ts_jobs else 0
            _, add_dur = _calculate_target(sched, ts)
            if curr_dur + add_dur > ts_cap[ts]:
                continue
            ts_jobs.setdefault(ts, []).extend(sched)
            used_jobs.update(sched)
            selected_whole_cols.append((ts, sched.copy()))

    elif strategy == 'cost':
        costed_cols = []
        for use, sched, ts, tgt in usage_list:
            if not sched:
                continue
            unit_cost = tgt / len(sched)
            costed_cols.append((unit_cost, sched, ts, tgt))

        costed_cols.sort(key=lambda x: x[0])

        for _, sched, ts, tgt in costed_cols:
            if used_jobs.intersection(set(sched)):
                continue
            curr_dur = _calculate_target(ts_jobs.get(ts, []), ts)[1] if ts in ts_jobs else 0
            _, add_dur = _calculate_target(sched, ts)
            if curr_dur + add_dur > ts_cap[ts]:
                continue
            ts_jobs.setdefault(ts, []).extend(sched)
            used_jobs.update(sched)
            selected_whole_cols.append((ts, sched.copy()))
            
    elif strategy == 'maxcols':
        len_cols = []
        for use, sched, ts, tgt in usage_list:
            if not sched:
                continue
            len_cols.append((len(sched), sched, ts, tgt))  

        len_cols.sort(key=lambda x: x[0])

        for _, sched, ts, tgt in len_cols:
            if used_jobs.intersection(set(sched)):
                continue
            curr_dur = _calculate_target(ts_jobs.get(ts, []), ts)[1] if ts in ts_jobs else 0
            _, add_dur = _calculate_target(sched, ts)
            if curr_dur + add_dur > ts_cap[ts]:
                continue
            ts_jobs.setdefault(ts, []).extend(sched)
            used_jobs.update(sched)
            selected_whole_cols.append((ts, sched.copy()))
    
    used_jobs = set(job for _, jobs in selected_whole_cols for job in jobs)

    J_rem = {a: (p_j[a-1], r_j[a-1]) for a in range(1, N+1) if a not in used_jobs}
    T_rem = {ts: (ts, ts_cap[ts], random.random()) for ts in range(1, T + 1) if ts not in ts_jobs}

    max_trials = 50000
    best_Match = {}
    best_Total = float('inf')

    for _ in range(max_trials):
        sorted_T = sorted(T_rem.items(), key=lambda x: random.random())
        sorted_J = sorted(J_rem.items(), key=lambda x: random.random())

        j = k = 0
        Match = {i[0]: [] for i in sorted_T}
        time_slots = list(Match.keys())

        while j < len(sorted_J):
    
            next_schedule = [t[0] for t in Match[time_slots[k]]] + [sorted_J[j][0]]
            _, test_time = _calculate_target(next_schedule, sorted_T[k][0])
            
            if test_time <= sorted_T[k][1][1]:
                Match[time_slots[k]].append(sorted_J[j])
                j += 1
            else:
                k += 1

        Match = {key: val for key, val in Match.items() if val}
        Total = 0
        for key, val in Match.items():
            Total += _calculate_target([t[0] for t in val], key)[0]

        if Total < best_Total:
            best_Total = Total
            best_Match = Match

    full_jobs = {ts: jobs.copy() for ts, jobs in selected_whole_cols}
    for ts, job_tuples in best_Match.items():
        full_jobs.setdefault(ts, []).extend(t[0] for t in job_tuples)
    _snapshot_feasible_solution(full_jobs)

    print("\n---- Columns generated by filtering and fully retained ----")
    for rank, (ts, jobs) in enumerate(selected_whole_cols, 1):
        print(f"Sel{rank}: 区间={ts} 工件={jobs}")

    for ts, jobs in best_Match.items():
        Schedule = [t[0] for t in jobs]
        Vector = [0] * N
        for job in Schedule:
            Vector[job - 1] = 1
        Target, _ = _calculate_target(Schedule, ts)
        _add_column(Schedule, Target, Vector, ts)
        
    print("\n---- New Heuristic Solution ----")
    total_target = 0.0
    ts_target = {}
    for ts, jobs in best_Match.items():
        Schedule = [t[0] for t in jobs]
        single_target, _ = _calculate_target(Schedule, ts)
        ts_target[ts] = single_target
        total_target += single_target
        print(f"Time slot {ts}: job {Schedule}  -> cost {ts_target[ts]:.3f}")
        Vector = [0] * N
        for job in Schedule:
            Vector[job - 1] = 1
        _add_column(Schedule, single_target, Vector, ts)
    
    for ts, jobs in (selected_whole_cols):
        single_target, _ = _calculate_target(jobs, ts)
        ts_target[ts] = single_target
        total_target += single_target
        print(f"Time slot {ts}: job {jobs}  -> cost {ts_target[ts]:.3f}")

    _build_master_problem()

    print(f"New Heuristic Solution for Total Cost {total_target:.3f}")

    return best_Match, total_target


def run_once():

    start = time.perf_counter()

    initialize_model()

    # Based on the results from Predict_GC.py, set max-iter, tol, and Heuristic_solution
    column_generation_loop(max_iter=80, tol=150, max_new_cols=10)
    usage_list = remain_usage_based()

    Heuristic_solution(usage_list, 'used')
    # Heuristic_solution(usage_list, 'cost')
    # Heuristic_solution(usage_list, 'maxcols')

    # convert_to_integer()
    # master_model.optimize()
    solution, cost = get_optimal_solution()

    end = time.perf_counter()
    elapsed = end - start
    if solution is None:
        cost = 1e8
    return elapsed, cost



if __name__ == "__main__":
    run_times, objectives = [], []
    for run_id in range(10): 
        t, obj = run_once()
        run_times.append(t)
        objectives.append(obj)

    print("\n===== Statistics from 10 Random Experiments =====")
    print(f"Average Cost: {np.mean(objectives):.2f}")
    print(f"Best cost: {min(objectives):.2f}")
    print(f"Average time: {np.mean(run_times):.2f}s")

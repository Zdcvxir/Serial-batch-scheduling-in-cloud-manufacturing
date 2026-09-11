import gurobipy as gp
from gurobipy import GRB
import time
import random
import numpy as np
import maml_config as maml_settings

Heuristic_solution_list = []

def reload_config():
    """Dynamically load the complete scheduling instance from config.py."""
    import importlib
    import config

    importlib.reload(config)
    required_fields = (
        "N", "T", "B", "r_j", "seta", "c",
        "t_k", "p_j", "L_k", "S_k", "CP_k",
    )
    missing = [name for name in required_fields if not hasattr(config, name)]
    if missing:
        raise AttributeError(
            "config.py is incomplete. Missing fields: " + ", ".join(missing)
        )
    return tuple(getattr(config, name) for name in required_fields)


def configure_instance(instance=None):
    """Load one explicit instance, or fall back to config.py for direct runs."""
    global N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k
    global T_dict, J_dict

    if instance is None:
        values = reload_config()
    else:
        values = tuple(
            instance[name]
            for name in (
                "N", "T", "B", "r_j", "seta", "c",
                "t_k", "p_j", "L_k", "S_k", "CP_k",
            )
        )
    N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k = values
    r_j, t_k, p_j = list(r_j), list(t_k), list(p_j)
    L_k, S_k, CP_k = list(L_k), list(S_k), list(CP_k)
    T_dict = {a: (L_k[a - 1], t_k[a - 1], CP_k[a - 1]) for a in range(1, T + 1)}
    J_dict = {a: (p_j[a - 1], r_j[a - 1]) for a in range(1, N + 1)}


def initialize_model(instance=None):
    """Initialize the restricted master problem and its initial columns."""
    global master_model, columns, W_vars, dual_zeta, dual_sigma
    global N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k, column_fingerprints
    global T_dict, J_dict

    configure_instance(instance)
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
    """Generate initial columns with the three construction strategies."""
    global _initial_feasible_solution
    _initial_feasible_solution = None
    snapshot_start = len(Heuristic_solution_list)
    sorted_T = sorted(T_dict.items(), key= lambda item: item[1][1] + item[1][2] / item[1][0])
    sorted_J = sorted(J_dict.items(), key=lambda item: item[1][1])
    j = 0
    k = 0
    Match = {i[0]: [] for i in sorted_T}
    time_slots = list(Match.keys())
    Total_Target = 0
    while j < N and k < len(time_slots):

        next_schedule = [t[0] for t in Match[time_slots[k]]] + [sorted_J[j][0]]
        _, test_time = _calculate_target(next_schedule, sorted_T[k][0])

        if test_time <= sorted_T[k][1][0]:
            Match[time_slots[k]].append(sorted_J[j])
            j += 1
        else:
            k += 1

    Match = {key: value for key, value in Match.items() if value} if j == N else {}

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

    if j == N:
        print(f"Initial solution cost: {Total_Target}")
        _snapshot_feasible_solution(Match)
    else:
        print("Deterministic construction failed; trying randomized initialization.")

    _add_heuristic_columns('rj')
    _add_heuristic_columns('rt')
    _add_heuristic_columns('rtj')
    initial_candidates = Heuristic_solution_list[snapshot_start:]
    if not initial_candidates:
        raise RuntimeError("No feasible initial solution found within the construction budget.")
    cost, schedule = min(initial_candidates, key=lambda entry: entry[0])
    _initial_feasible_solution = (cost, {slot: jobs.copy() for slot, jobs in schedule.items()})

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
    """Add a nonduplicate schedule column to the restricted master problem."""
    fingerprint = (tuple(Schedule), Timeslot)

    if fingerprint in column_fingerprints:
        return False

    column_fingerprints.append(fingerprint)

    s = len(columns)
    var = master_model.addVar(vtype=GRB.CONTINUOUS, obj=Target, name=f"W_{s}")
    W_vars[s] = var
    columns.append((Schedule, Target, Vector, Timeslot))
    return True

def _add_heuristic_columns(strategy):
    """Generate the best column found by repeated randomized construction.

    'rt' randomizes time slots, 'rj' randomizes jobs, and 'rtj'
    randomizes both.
    """
    max_trials = 3000
    best_Match = {}
    best_Total = float('inf')

    for _ in range(max_trials):
        if strategy in ('rt', 'rtj'):
            sorted_T = sorted(T_dict.items(), key=lambda x: random.random())
        else:
            sorted_T = sorted(T_dict.items(),
                              key=lambda x: x[1][2])

        if strategy in ('rj', 'rtj'):
            sorted_J = sorted(J_dict.items(), key=lambda x: random.random())
        else:
            sorted_J = sorted(J_dict.items(), key=lambda x: x[1][1])

        j = k = 0
        Match = {i[0]: [] for i in sorted_T}
        time_slots = list(Match.keys())
        while j < N and k < len(time_slots):
            next_schedule = [t[0] for t in Match[time_slots[k]]] + [sorted_J[j][0]]
            _, test_time = _calculate_target(next_schedule, sorted_T[k][0])
            if test_time <= sorted_T[k][1][0]:
                Match[time_slots[k]].append(sorted_J[j])
                j += 1
            else:
                k += 1

        if j < N:
            continue
        Match = {key: val for key, val in Match.items() if val}
        Total = 0
        for key, val in Match.items():
            Total += _calculate_target([t[0] for t in val], key)[0]

        if Total < best_Total:
            best_Total = Total
            best_Match = Match

    if best_Total == float('inf'):
        return
    for key, value_list in best_Match.items():
        Schedule = [tup[0] for tup in value_list]
        Vector = [0] * N
        for tup in value_list:
            if 0 <= tup[0] - 1 < N:
                Vector[tup[0] - 1] = 1
        Target = _calculate_target(Schedule, key)[0]
        _add_column(Schedule, Target, Vector, key)
    print("Heuristic column cost:", best_Total)

    _snapshot_feasible_solution(best_Match)

def _build_master_problem():
    """Build the restricted master problem."""

    if master_model.getAttr("NumConstrs") > 0:
        master_model.remove(list(master_model.getConstrs()))

    for j in range(N):
        expr = gp.quicksum(W_vars[s] * columns[s][2][j] for s in W_vars)
        master_model.addConstr(expr >= 1, name=f"cover_{j}")

    for k in range(1, T + 1):
        expr = gp.quicksum(W_vars[s] for s in W_vars if columns[s][3] == k)
        master_model.addConstr(expr <= 1, name=f"timeslot_{k}")

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
        slot_start = S_k[Y - 1]
        F_k = [0] * (X + 1)
        F_k[0] = slot_start
        R_j = [i[1][1] for i in sorted_J]
        P_j = [i[1][0] for i in sorted_J]

        # Reuse rolling arrays to preprocess the candidate batches of each state.
        batch_max_ready = [0] * (c + 1)
        batch_processing_time = [0] * (c + 1)
        for beta in range(1, X + 1):
            max_batch_size = min(c, beta)
            max_ready = float("-inf")
            total_processing = 0
            for batch_size in range(1, max_batch_size + 1):
                l = beta - batch_size
                max_ready = max(max_ready, R_j[l])
                total_processing += P_j[l]
                batch_max_ready[batch_size] = max_ready
                batch_processing_time[batch_size] = total_processing

            best_completion = float("inf")
            for batch_size in range(1, max_batch_size + 1):
                l = beta - batch_size
                batch_start = max(F_k[l], batch_max_ready[batch_size])
                completion = (
                    batch_start + seta + batch_processing_time[batch_size]
                )
                if completion < best_completion:
                    best_completion = completion
            F_k[beta] = best_completion

        return F_k[X] - slot_start

    Processing_time = DP(len(Schedule), Timeslot)

    if Processing_time > L_k[Timeslot-1]:
       Processing_time = 9999999

    total = Processing_time * t_k[Timeslot-1] + CP_k[Timeslot-1]

    return total, Processing_time

def solve_master_problem():
    """Solve the restricted master problem and obtain dual values."""
    global dual_zeta, dual_sigma

    master_model.optimize()
    if master_model.status != GRB.OPTIMAL:
        raise RuntimeError(f"Master problem failed with status {master_model.status}.")

    for j in range(N):
        dual_zeta[j] = master_model.getConstrByName(f"cover_{j}").Pi
    for k in range(1, T + 1):
        dual_sigma[k] = master_model.getConstrByName(f"timeslot_{k}").Pi

    if master_model.status == GRB.INFEASIBLE:
            print("The master problem is infeasible; computing an IIS.")
            master_model.computeIIS()
            for c in master_model.getConstrs():
                if c.IISConstr:
                    print(f"Conflicting constraint: {c.constrName}")
            raise RuntimeError("The master problem is infeasible; check the input data.")

    if master_model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL]:
        raise RuntimeError(f"Master problem failed with status {master_model.status}.")

    obj_val = master_model.objVal

    return obj_val

def _extract_current_solution():
    selected = [s for s, w in W_vars.items() if w.X > 0.1]
    ts_tasks = {}
    for s in selected:
        ts = columns[s][3]
        ts_tasks.setdefault(ts, []).extend(columns[s][0])
    return {ts: tasks for ts, tasks in ts_tasks.items()}

def solve_pricing_subproblem1(
    max_pricing_trials,
    max_new_cols,
    reduced_cost_cutoff=maml_settings.CGDH_REDUCED_COST_CUTOFF,
):
    """Sample at most ``max_pricing_trials`` candidates and add qualifying columns."""
    new_columns = []
    new_fingerprints = set()
    existing_fingerprints = set(column_fingerprints)
    trials = 0
    feasible_trials = 0

    while trials < max_pricing_trials and len(new_columns) < max_new_cols:
        trials += 1
        M = random.randint(1, N)
        schedule = random.sample(range(1, N + 1), M)
        timeslot = random.randint(1, T)
        target, processing_time = _calculate_target(schedule, timeslot)
        if processing_time > L_k[timeslot - 1]:
            continue

        feasible_trials += 1
        reduced_cost = (
            target
            - sum(dual_zeta[j - 1] for j in schedule)
            - dual_sigma[timeslot]
        )
        fingerprint = (tuple(schedule), timeslot)
        if (
            reduced_cost < reduced_cost_cutoff
            and fingerprint not in existing_fingerprints
            and fingerprint not in new_fingerprints
        ):
            vector = [0] * N
            for job in schedule:
                vector[job - 1] = 1
            new_columns.append(
                (schedule, target, vector, timeslot, reduced_cost)
            )
            new_fingerprints.add(fingerprint)

    columns_added = 0
    for schedule, target, vector, timeslot, _ in new_columns:
        if _add_column(schedule, target, vector, timeslot):
            columns_added += 1
    if columns_added:
        _build_master_problem()

    reduced_costs = [column[4] for column in new_columns]
    return {
        "min_reduced_cost": min(reduced_costs) if reduced_costs else None,
        "trials": trials,
        "feasible_trials": feasible_trials,
        "columns_added": columns_added,
    }

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

def column_generation_loop(
    max_iter,
    max_new_cols,
    max_pricing_trials=maml_settings.CGDH_MAX_PRICING_TRIALS,
    reduced_cost_cutoff=maml_settings.CGDH_REDUCED_COST_CUTOFF,
):
    """Run heuristic column generation and return diagnostics."""
    global W_vars
    stats = {
        "iterations_completed": 0,
        "pricing_trials": 0,
        "feasible_pricing_trials": 0,
        "columns_added": 0,
        "stop_reason": "max_iterations",
        "last_min_reduced_cost": None,
    }
    for iteration in range(max_iter):
        try:
            obj_val = solve_master_problem()
            pricing = solve_pricing_subproblem1(
                max_pricing_trials=max_pricing_trials,
                max_new_cols=max_new_cols,
                reduced_cost_cutoff=reduced_cost_cutoff,
            )
            min_reduced_cost = pricing["min_reduced_cost"]
            stats["iterations_completed"] = iteration + 1
            stats["pricing_trials"] += pricing["trials"]
            stats["feasible_pricing_trials"] += pricing["feasible_trials"]
            stats["columns_added"] += pricing["columns_added"]
            stats["last_min_reduced_cost"] = min_reduced_cost
            reduced_cost_text = (
                f"{min_reduced_cost:.4f}" if min_reduced_cost is not None else "None"
            )
            print(
                f"Iter {iteration + 1}: master objective={obj_val:.2f}, "
                f"minimum sampled reduced cost={reduced_cost_text}, "
                f"columns added={pricing['columns_added']}"
            )

            if pricing["columns_added"] == 0:
                stats["stop_reason"] = "no_qualifying_sampled_column"
                print("Heuristic pricing found no eligible column within the sampling budget.")
                break

        except Exception as e:
            print(f"Column-generation iteration failed: {e}")
            raise
    return stats

def get_optimal_solution():
    """Return the best solution as a mapping from time slots to job lists."""
    if master_model.status == GRB.LOADED and Heuristic_solution_list:
        best_cost, best_snap = min(Heuristic_solution_list, key=lambda x: x[0])
        print(
            f"[Heuristic] Best of {len(Heuristic_solution_list)} candidates: "
            f"cost={best_cost:.2f}"
        )
        print(best_snap)
        return best_snap, best_cost

    if master_model.status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        if master_model.status == GRB.SUBOPTIMAL:
            print("Warning: the solver returned a suboptimal incumbent.")
        return _extract_current_solution()

    return {}

def print_columns():
    """Print the schedule, cost, coverage vector, and time slot of each column."""
    print("\nColumn details:")
    for idx, (Schedule, Target, Vector, Timeslot) in enumerate(columns):
        print(f"Column {idx}:")
        print(f"  Job sequence: {Schedule}")
        print(f"  Column cost: {Target:.2f}")
        print(f"  Coverage vector: {Vector}")
        print(f"  Production time slot: {Timeslot}")
        print()

def print_solution(solution: dict):
    print("\nBest time-slot assignment:")

    ts_cost = {ts: (_calculate_target(tasks, ts)[0], tasks)
               for ts, tasks in solution.items()}

    total_target = 0

    for ts in sorted(ts_cost.keys()):
        cost, tasks = ts_cost[ts]
        print(f"Time slot {ts}: job sequence {tasks}, cost {cost:.1f}")
        total_target += cost

    print(f"\nTotal production cost: {total_target:.1f}")

def _recovery_failure():
    """Retain a known feasible solution when all completion attempts fail."""
    candidates = list(Heuristic_solution_list)
    initial = globals().get('_initial_feasible_solution')
    if initial is not None:
        candidates.append(initial)
    if not candidates:
        raise RuntimeError("Recovery failed and no known feasible initial solution is available.")
    cost, schedule = min(candidates, key=lambda entry: entry[0])
    fallback = {slot: jobs.copy() for slot, jobs in schedule.items()}
    if not Heuristic_solution_list or cost < min(entry[0] for entry in Heuristic_solution_list):
        Heuristic_solution_list.append((cost, fallback))
    _build_master_problem()
    print("Recovery failed; retaining the best known feasible solution.")
    return {slot: [(job, J_dict[job]) for job in jobs]
            for slot, jobs in fallback.items()}, cost


def Heuristic_solution(usage_list, strategy):

    used_jobs = set()
    used_slots = set()
    ts_jobs   = {}
    ts_cap    = {k: L_k[k-1] for k in range(1, T+1)}
    selected_whole_cols = []

    if strategy == 'used':
        for use, sched, ts, tgt in usage_list:
            if not sched:
                continue
            if used_jobs.intersection(set(sched)):
                continue
            if ts in used_slots:
                continue
            curr_dur = _calculate_target(ts_jobs.get(ts, []), ts)[1] if ts in ts_jobs else 0
            _, add_dur = _calculate_target(sched, ts)
            if curr_dur + add_dur > ts_cap[ts]:
                continue
            ts_jobs.setdefault(ts, []).extend(sched)
            used_jobs.update(sched)
            used_slots.add(ts)
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
            if ts in used_slots:
                continue
            curr_dur = _calculate_target(ts_jobs.get(ts, []), ts)[1] if ts in ts_jobs else 0
            _, add_dur = _calculate_target(sched, ts)
            if curr_dur + add_dur > ts_cap[ts]:
                continue
            ts_jobs.setdefault(ts, []).extend(sched)
            used_jobs.update(sched)
            used_slots.add(ts)
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
            if ts in used_slots:
                continue
            curr_dur = _calculate_target(ts_jobs.get(ts, []), ts)[1] if ts in ts_jobs else 0
            _, add_dur = _calculate_target(sched, ts)
            if curr_dur + add_dur > ts_cap[ts]:
                continue
            ts_jobs.setdefault(ts, []).extend(sched)
            used_jobs.update(sched)
            used_slots.add(ts)
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

        while j < len(sorted_J) and k < len(time_slots):

            next_schedule = [t[0] for t in Match[time_slots[k]]] + [sorted_J[j][0]]
            _, test_time = _calculate_target(next_schedule, sorted_T[k][0])

            if test_time <= sorted_T[k][1][1]:
                Match[time_slots[k]].append(sorted_J[j])
                j += 1
            else:
                k += 1

        if j < len(sorted_J):
            continue
        Match = {key: val for key, val in Match.items() if val}
        Total = 0
        for key, val in Match.items():
            Total += _calculate_target([t[0] for t in val], key)[0]

        if Total < best_Total:
            best_Total = Total
            best_Match = Match

    if best_Total == float('inf'):
        return _recovery_failure()
    full_jobs = {ts: jobs.copy() for ts, jobs in selected_whole_cols}
    for ts, job_tuples in best_Match.items():
        full_jobs.setdefault(ts, []).extend(t[0] for t in job_tuples)
    _snapshot_feasible_solution(full_jobs)

    print("\n---- Complete columns retained after filtering ----")
    for rank, (ts, jobs) in enumerate(selected_whole_cols, 1):
        print(f"Selection {rank}: time slot={ts}, jobs={jobs}")

    for ts, jobs in best_Match.items():
        Schedule = [t[0] for t in jobs]
        Vector = [0] * N
        for job in Schedule:
            Vector[job - 1] = 1
        Target, _ = _calculate_target(Schedule, ts)
        _add_column(Schedule, Target, Vector, ts)

    print("\n---- Repaired heuristic solution ----")
    total_target = 0.0
    ts_target = {}
    for ts, jobs in best_Match.items():
        Schedule = [t[0] for t in jobs]
        single_target, _ = _calculate_target(Schedule, ts)
        ts_target[ts] = single_target
        total_target += single_target
        print(f"Time slot {ts}: jobs {Schedule} -> cost {ts_target[ts]:.3f}")
        Vector = [0] * N
        for job in Schedule:
            Vector[job - 1] = 1
        _add_column(Schedule, single_target, Vector, ts)

    for ts, jobs in (selected_whole_cols):
        single_target, _ = _calculate_target(jobs, ts)
        ts_target[ts] = single_target
        total_target += single_target
        print(f"Time slot {ts}: jobs {jobs} -> cost {ts_target[ts]:.3f}")

    _build_master_problem()

    print(f"Repaired heuristic solution cost: {total_target:.3f}")

    return best_Match, total_target


def run_once(
    seed: int,
    instance=None,
    pricing_cutoff=maml_settings.CGDH_REDUCED_COST_CUTOFF,
    max_pricing_trials=maml_settings.CGDH_MAX_PRICING_TRIALS,
    return_details=False,
):
    """Run CGDH once with a randomly selected candidate configuration."""
    global Heuristic_solution_list
    Heuristic_solution_list = []
    start = time.perf_counter()
    random.seed(seed)
    np.random.seed(seed)
    gp.setParam('Seed', seed)

    max_iter = maml_settings.CGDH_MAX_ITER
    max_new_cols = random.choice(maml_settings.CGDH_NEW_COLUMN_LIMITS)
    policy = random.choice(maml_settings.CGDH_POLICIES)
    print(
        "Random CGDH configuration: "
        f"max_iter={max_iter}, max_new_cols={max_new_cols}, policy={policy}"
    )

    initialize_model(instance)
    cg_stats = column_generation_loop(
        max_iter=max_iter,
        max_new_cols=max_new_cols,
        max_pricing_trials=max_pricing_trials,
        reduced_cost_cutoff=pricing_cutoff,
    )
    usage_list = remain_usage_based()

    Heuristic_solution(usage_list, policy)

    solution, cost = get_optimal_solution()
    end = time.perf_counter()
    elapsed = end - start
    if solution is None:
        cost = 1e8
    if return_details:
        return {
            "runtime_sec": elapsed,
            "cost": cost,
            "max_iter": max_iter,
            "max_new_cols": max_new_cols,
            "policy": policy,
            **cg_stats,
        }
    return elapsed, cost


if __name__ == "__main__":
    run_times, objectives = [], []
    for run_id in range(maml_settings.NUMBER_OF_RUNS):
        t, obj = run_once(run_id + 1)
        run_times.append(t)
        objectives.append(obj)
        print(f"Run {run_id + 1}: time={t:.2f}s, cost={obj:.2f}")

    avg_time = np.mean(run_times)
    avg_obj  = np.mean(objectives)
    best_obj = min(objectives)

    print(f"\n========== Summary of {maml_settings.NUMBER_OF_RUNS} runs ==========")
    print(f"Mean runtime: {avg_time:.2f} s")
    print(f"Mean cost: {avg_obj:.2f}")
    print(f"Best cost: {best_obj:.2f}")

import time
import random
import copy
import numpy as np

Test = 4000

def reload_config():
    """Reload the complete scheduling instance from config.py."""
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


configure_instance()

def _initialize_columns():
    """Return the better deterministic or randomized initial solution."""
    sorted_T_orig = sorted(T_dict.items(),
                           key=lambda item: item[1][1] + item[1][2] / item[1][0])
    sorted_J_orig = sorted(J_dict.items(), key=lambda item: item[1][1])

    match_A = _build_match(sorted_T_orig, sorted_J_orig)
    cost_A = sum(_calculate_target(jobs, slot)[0]
                 for slot, jobs in match_A.items()) if match_A is not None else float('inf')

    best_B_match, best_B_cost = None, float('inf')
    for _ in range(3000):
        sorted_T_rand = sorted(T_dict.items(), key=lambda x: random.random())
        sorted_J_rand = sorted(J_dict.items(), key=lambda x: random.random())
        cand_match = _build_match(sorted_T_rand, sorted_J_rand)
        if cand_match is None:
            continue
        cand_cost = sum(_calculate_target(jobs, slot)[0]
                        for slot, jobs in cand_match.items())
        if cand_cost < best_B_cost:
            best_B_cost = cand_cost
            best_B_match = cand_match

    if best_B_cost < cost_A:
        return best_B_match
    elif match_A is None:
        raise RuntimeError("No feasible initial solution found within the construction budget.")
    else:
        return match_A

def _build_match(sorted_T, sorted_J):
    j = 0
    k = 0
    match = {t[0]: [] for t in sorted_T}
    time_slots = list(match.keys())

    while j < N:
        if k >= len(time_slots):
            return None
        next_schedule = match[time_slots[k]] + [sorted_J[j][0]]
        _, test_time = _calculate_target(next_schedule, time_slots[k])
        if test_time <= sorted_T[k][1][0]:
            match[time_slots[k]].append(sorted_J[j][0])
            j += 1
        else:
            k += 1

    return {key: val for key, val in match.items() if val}


def neighborhood_q1(ini_solution):

    for _ in range(Test):
        list_timeslot = [0 for _ in range(T)]
        one_indices_timeslot = [i for i, x in ini_solution.items()]
        for idx in one_indices_timeslot:
            list_timeslot[idx-1] = 1

        ones = [i for i, val in enumerate(list_timeslot) if val == 1]
        zeros = [i for i, val in enumerate(list_timeslot) if val == 0]

        job_list = [x for x in ini_solution.values()]
        if job_list and zeros:
            src_idx = random.randrange(len(job_list))
            src_jobs = job_list[src_idx]
            if len(src_jobs) >= 2:
                k = random.randint(1, len(src_jobs)-1)
                tail_jobs = src_jobs[k:]
                job_list[src_idx] = src_jobs[:k]

                tgt_slot = random.choice(zeros)
                zeros.remove(tgt_slot)
                list_timeslot[tgt_slot] = 1
                job_list.append(tail_jobs)

        original_order = list(ini_solution.keys())
        new_ones = {i + 1 for i, val in enumerate(list_timeslot) if val == 1}
        added_slots = new_ones - set(original_order)
        new_order = original_order + sorted(added_slots)
        new_solution = {k: job_list[i] for i, k in enumerate(new_order)}
        new_solution = {k: v for k, v in new_solution.items() if v}


        new_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in new_solution.items())
        ini_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in ini_solution.items())

        if new_target < ini_target:
            ini_solution = new_solution

    return ini_solution

def neighborhood_q2(ini_solution):
    for _ in range(Test):
        backup_solution = copy.deepcopy(ini_solution)
        list_timeslot = [0] * T
        for idx in ini_solution.keys():
            list_timeslot[idx - 1] = 1

        ones = [i for i, v in enumerate(list_timeslot) if v]
        zeros = [i for i, v in enumerate(list_timeslot) if not v]

        if ones and zeros:
            flip_out = random.choice(ones)
            flip_in = random.choice(zeros)
            out_slot = flip_out + 1
            job_list = [v for v in ini_solution.values()]
            keys = list(ini_solution.keys())
            if out_slot in ini_solution:
                ini_solution[flip_in + 1] = ini_solution.pop(out_slot)

        job_list = list(ini_solution.values())
        if len(job_list) >= 2:
            a, b = random.sample(range(len(job_list)), 2)
            if job_list[a] and job_list[b]:
                elem = random.choice(job_list[a])
                job_list[a].remove(elem)
                job_list[b].insert(random.randint(0, len(job_list[b])), elem)

        new_solution = {k: v for k, v in ini_solution.items() if v}
        new_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in new_solution.items())
        old_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in backup_solution.items())

        if new_target < old_target:
            ini_solution = new_solution
        else:
            ini_solution = backup_solution

    return ini_solution

def neighborhood_q3(ini_solution):
    for _ in range(Test):
        backup_solution = copy.deepcopy(ini_solution)

        list_timeslot = [0] * T
        for idx in ini_solution.keys():
            list_timeslot[idx - 1] = 1
        ones = [i for i, v in enumerate(list_timeslot) if v]
        zeros = [i for i, v in enumerate(list_timeslot) if not v]

        if len(ones) > 1 and len(zeros) > 1:
            flip_out1, flip_out2 = random.sample(ones, 2)
            flip_in1, flip_in2 = random.sample(zeros, 2)

            slot_out1, slot_out2 = flip_out1 + 1, flip_out2 + 1
            slot_in1, slot_in2 = flip_in1 + 1, flip_in2 + 1

            job_in1 = ini_solution.pop(slot_out1, [])
            job_in2 = ini_solution.pop(slot_out2, [])
            if job_in1:
                ini_solution[slot_in1] = job_in1
            if job_in2:
                ini_solution[slot_in2] = job_in2

        job_list = list(ini_solution.values())
        if len(job_list) >= 2:
            a, b = random.sample(range(len(job_list)), 2)
            if job_list[a] and job_list[b]:
                elem = random.choice(job_list[a])
                job_list[a].remove(elem)
                job_list[b].insert(random.randint(0, len(job_list[b])), elem)

        new_solution = {k: v for k, v in ini_solution.items() if v}
        new_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in new_solution.items())
        old_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in backup_solution.items())

        if new_target < old_target:
            ini_solution = new_solution
        else:
            ini_solution = backup_solution

    return ini_solution

def neighborhood_q4(ini_solution):
    for _ in range(Test):
        backup_solution = copy.deepcopy(ini_solution)

        nonempty_slots = [slot for slot, jobs in ini_solution.items() if jobs]
        if len(nonempty_slots) < 2:
            continue

        # Swap one randomly selected job between two different job lists.
        slot_a, slot_b = random.sample(nonempty_slots, 2)
        jobs_a, jobs_b = ini_solution[slot_a], ini_solution[slot_b]
        pos_a = random.randrange(len(jobs_a))
        pos_b = random.randrange(len(jobs_b))
        jobs_a[pos_a], jobs_b[pos_b] = jobs_b[pos_b], jobs_a[pos_a]

        # Insert one job from a randomly selected source list into another list.
        source_slot, target_slot = random.sample(nonempty_slots, 2)
        source_jobs = ini_solution[source_slot]
        target_jobs = ini_solution[target_slot]
        moved_job = source_jobs.pop(random.randrange(len(source_jobs)))
        target_jobs.insert(random.randint(0, len(target_jobs)), moved_job)

        new_solution = {k: v for k, v in ini_solution.items() if v}
        new_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in new_solution.items())
        old_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in backup_solution.items())

        if new_target < old_target:
            ini_solution = new_solution
        else:
            ini_solution = backup_solution

    return ini_solution

def neighborhood_q5(ini_solution):
    for _ in range(Test):
        backup = copy.deepcopy(ini_solution)

        nonempty_slots = [slot for slot, jobs in ini_solution.items() if jobs]
        if len(nonempty_slots) < 2:
            continue

        # Swap one randomly selected job between two different job lists.
        slot_a, slot_b = random.sample(nonempty_slots, 2)
        jobs_a, jobs_b = ini_solution[slot_a], ini_solution[slot_b]
        pos_a = random.randrange(len(jobs_a))
        pos_b = random.randrange(len(jobs_b))
        jobs_a[pos_a], jobs_b[pos_b] = jobs_b[pos_b], jobs_a[pos_a]

        new_sol = {k: v for k, v in ini_solution.items() if v}
        new_target = sum(_calculate_target(jobs, slot)[0] for slot, jobs in new_sol.items())
        old_target = sum(_calculate_target(jobs, slot)[0] for slot, jobs in backup.items())

        if new_target < old_target:
            ini_solution = new_sol
        else:
            ini_solution = backup
    return ini_solution

def neighborhood_q6(ini_solution):
    for _ in range(Test):
        backup = copy.deepcopy(ini_solution)

        job_list = list(ini_solution.values())
        if len(job_list) < 2:
            continue

        idx_a, idx_b = random.sample(range(len(job_list)), 2)
        if job_list[idx_a] and job_list[idx_b]:
            elem = random.choice(job_list[idx_a])
            job_list[idx_a].remove(elem)
            job_list[idx_b].insert(random.randint(0, len(job_list[idx_b])), elem)

        new_sol = {k: v for k, v in ini_solution.items() if v}
        new_target = sum(_calculate_target(jobs, slot)[0] for slot, jobs in new_sol.items())
        old_target = sum(_calculate_target(jobs, slot)[0] for slot, jobs in backup.items())

        if new_target < old_target:
            ini_solution = new_sol
        else:
            ini_solution = backup

    return ini_solution


NEIGHBORS = {1: neighborhood_q1, 2: neighborhood_q2, 3: neighborhood_q3,
             4: neighborhood_q4, 5: neighborhood_q5, 6: neighborhood_q6}

def VND(Match, T_max):
    start_time = time.perf_counter()
    test = 0
    ini_solution = copy.deepcopy(Match)
    ini_target = 0
    for slot, jobs in ini_solution.items():
        ini_target += _calculate_target(jobs, slot)[0]
    print(f"Initial solution: {ini_solution}; cost={ini_target}")
    target_list = []
    target_list.append((ini_target, copy.deepcopy(ini_solution), 0.0))

    order = NEIGHBOR_ORDER
    nei_ptr = 0

    def try_move(nei_id):
        nonlocal ini_solution, ini_target, test, nei_ptr, target_list
        target_list = [(ini_target, copy.deepcopy(ini_solution), 0.0)]
        backup = copy.deepcopy(ini_solution)
        new_sol = NEIGHBORS[nei_id](copy.deepcopy(ini_solution))
        new_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in new_sol.items())
        if new_target < ini_target:
            target_list.append((new_target, copy.deepcopy(new_sol), time.perf_counter() - start_time))
            ini_solution, ini_target = new_sol, new_target
            test = 0
            nei_ptr = 0
        else:
            ini_solution = backup
            test += 1
            nei_ptr = nei_ptr + 1

    while nei_ptr < len(NEIGHBOR_ORDER):
        if time.perf_counter() - start_time > T_max:
            print("The time limit has been reached; terminating the search.")
            break
        print(nei_ptr)
        q_id = order[nei_ptr]
        try_move(q_id)

    print(target_list)
    best_target, best_solution = min(target_list, key=lambda x: x[0])[:2]
    target_list.append((best_target, best_solution, T_max))
    print(f"Final solution: {best_solution}; cost={best_target}")
    return best_target, best_solution, target_list

def _calculate_target(Schedule, Timeslot):
    if not Schedule:
        return 0.0, 0.0
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
       Processing_time = 999999

    total = Processing_time * t_k[Timeslot-1] + CP_k[Timeslot-1]

    return total, Processing_time

def run_once(seed: int, time_limit=60, instance=None):
    """Run one VND experiment with a random neighborhood sequence."""
    if instance is not None:
        configure_instance(instance)
    random.seed(seed)
    neighborhood = random.choice([
        "3,1,5", "3,2,4", "4,3,2", "6,4,1", "5,2,1", "4,1,2", "1,5,4",
        "3,5,1", "4,2,5", "4,3,5", "1,4,2,3", "5,3,4,6", "6,4,2,5",
        "2,4,1,6", "2,6,1,4", "3,6,2,5", "1,2,3,4", "6,4,5,2", "2,6,1,5",
        "4,3,6,1", "3,4,6,1,2", "3,4,6,5,2", "5,2,4,3,6", "6,5,2,4,3",
        "6,4,5,1,2", "1,4,3,6,2", "2,6,4,5,3", "4,6,3,2,5", "3,5,2,4,6",
        "6,4,5,3,2", "2,4,1,6,5,3", "3,1,4,2,6,5", "6,5,2,1,3,4",
        "6,2,3,1,4,5", "3,2,1,4,5,6", "2,6,3,4,5,1", "4,6,5,2,3,1",
        "1,5,2,4,3,6", "5,4,6,2,3,1", "2,6,5,1,3,4"
    ])
    global NEIGHBOR_ORDER
    NEIGHBOR_ORDER = list(map(int, neighborhood.split(",")))

    start = time.perf_counter()
    Match = _initialize_columns()
    best_target, _, _ = VND(Match, T_max=time_limit)
    elapsed = time.perf_counter() - start
    return elapsed, best_target, neighborhood


if __name__ == "__main__":
    times, objs = [], []
    for k in range(10):
        t, obj, neighborhood = run_once(seed=k + 1)
        times.append(t)
        objs.append(obj)
        print(
            f"Run {k + 1}: time={t:.2f}s, cost={obj:.2f}, "
            f"neighborhoods={neighborhood}"
        )

    avg_time = np.mean(times)
    avg_obj  = np.mean(objs)
    best_obj = min(objs)

    print({"best_cost": best_obj, "mean_cost": avg_obj, "mean_time": avg_time})

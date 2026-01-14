import time
import random
import copy
import matplotlib.pyplot as plt
import numpy as np
import os,json
import pandas as pd

NEIGHBOR_ORDER = list(map(int, os.getenv("NEIGHBORS", "6,5,2,4,3").split(",")))
Test = 4000

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

N, T, B, r_j, seta, c, t_k, p_j, L_k, S_k, CP_k = reload_config()

F_k = {0: 0}
T_dict = {a: (L_k[a-1], t_k[a-1], CP_k[a-1]) for a in range(1, T+1)}
J_dict = {a: (p_j[a-1], r_j[a-1]) for a in range(1, N+1)}

def _initialize_columns():
    sorted_T_orig = sorted(T_dict.items(),
                           key=lambda item: item[1][1] + item[1][2] / item[1][0])
    sorted_J_orig = sorted(J_dict.items(), key=lambda item: item[1][1])

    match = _build_match(sorted_T_orig, sorted_J_orig)

    return match

def _build_match(sorted_T, sorted_J):
    j = 0
    k = 0
    match = {t[0]: [] for t in sorted_T}
    time_slots = list(match.keys())

    while j < N:
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

        keys = sorted(ini_solution.keys())
        for i in range(len(keys) - 1):
            slot_a, slot_b = keys[i], keys[i + 1]
            chain_a, chain_b = ini_solution[slot_a], ini_solution[slot_b]
            if chain_a and chain_b: 
                elem_a = random.choice(chain_a)
                elem_b = random.choice(chain_b)
                chain_a.remove(elem_a)
                chain_b.remove(elem_b)
                chain_a.append(elem_b)
                chain_b.append(elem_a)
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

def neighborhood_q5(ini_solution):
    for _ in range(Test):
        backup = copy.deepcopy(ini_solution)

        keys = sorted(ini_solution.keys())
        for i in range(len(keys)-1):
            a, b = keys[i], keys[i+1]
            chain_a, chain_b = ini_solution[a], ini_solution[b]
            if chain_a and chain_b:
                elem_a = random.choice(chain_a)
                elem_b = random.choice(chain_b)
                chain_a.remove(elem_a)
                chain_b.remove(elem_b)
                chain_a.append(elem_b)
                chain_b.append(elem_a)

        job_list = list(ini_solution.values())
        if len(job_list) >= 2:
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
    print(f"Initial solution{ini_solution}, Initial cost value{ini_target}")
    target_list = []
    target_list.append((ini_target, copy.deepcopy(ini_solution), 0.0))

    order = NEIGHBOR_ORDER
    nei_ptr = 0

    def try_move(nei_id):
        nonlocal ini_solution, ini_target, test, nei_ptr, target_list
        target_list = [(ini_target, ini_solution, 0.0)]
        backup = copy.deepcopy(ini_solution)
        new_sol = NEIGHBORS[nei_id](ini_solution)
        new_target = sum(_calculate_target(jobs, slot)[0]
                         for slot, jobs in new_sol.items())
        if new_target < ini_target:
            target_list.append((new_target, new_sol, time.perf_counter() - start_time))
            ini_solution, ini_target = new_sol, new_target
            test = 0      
            nei_ptr = 0  
        else:
            ini_solution = backup
            test += 1 
            nei_ptr = nei_ptr + 1

    while nei_ptr < len(NEIGHBOR_ORDER):
        if time.perf_counter() - start_time > T_max:
            print("Reach runtime, terminate loop")
            break
        print(nei_ptr)
        q_id = order[nei_ptr] 
        try_move(q_id)

    print(target_list)
    best_target, best_solution = min(target_list, key=lambda x: x[0])[:2]
    target_list.append((best_target, best_solution, T_max))
    print(f"Final solution{best_solution}, Final cost value{best_target}")
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
       Processing_time = 999999 

    total = Processing_time * t_k[Timeslot-1] + CP_k[Timeslot-1]

    return total, Processing_time        

def plot_convergence_time(target_list, save_path=None):
    elapsed_times = [t[2] for t in target_list]
    best_so_far = []
    cur_best = float('inf')
    for cost, _, _ in target_list:
        cur_best = min(cur_best, cost)
        best_so_far.append(cur_best)

    plt.figure(figsize=(8, 4))
    plt.plot(elapsed_times, best_so_far, color='tab:blue', linewidth=1)
    plt.scatter(elapsed_times, best_so_far, color='tab:blue', s=8, zorder=3)
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Best Cost So Far')
    plt.title('VND Convergence Curve (Time-based)')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to {save_path}")
    else:
        plt.show()
        
def run_once():
    start = time.perf_counter()
    Match = _initialize_columns()
    best_target, _, _ = VND(Match, T_max=60)
    elapsed = time.perf_counter() - start
    return elapsed, best_target


if __name__ == "__main__":
    run_times, objectives = [], []
    results = []

    for run_id in range(10):
        t, obj = run_once()
        run_times.append(t)
        objectives.append(obj)
        print(f"The {run_id+1}th: Time={t:.2f}s, Cost={obj:.2f}")

        results.append({
            'Run ID': run_id + 1,
            'Cost': obj,
            'Time': t
        })

    avg_time = np.mean(run_times)
    avg_obj  = np.mean(objectives)
    best_obj = min(objectives)

    print("\n========== Statistics from 10 Random Experiments ==========")
    print(f"Average run time: {avg_time:.2f} s")
    print(f"Average Cost: {avg_obj:.2f}")
    print(f"Best Cost: {best_obj:.2f}")
import time
import random
import copy
import matplotlib.pyplot as plt
import numpy as np

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

def _initialize():
    sorted_T_orig = sorted(T_dict.items(),
                           key=lambda item: item[1][1] + item[1][2] / item[1][0])
    sorted_J_orig = sorted(J_dict.items(), key=lambda item: item[1][1])

    match_A = _build_match(sorted_T_orig, sorted_J_orig)
    cost_A = sum(_calculate_target(jobs, slot)[0]
                 for slot, jobs in match_A.items())

    best_B_match, best_B_cost = None, float('inf')
    for _ in range(3000):
        sorted_T_rand = sorted(T_dict.items(), key=lambda x: random.random())
        sorted_J_rand = sorted(J_dict.items(), key=lambda x: random.random())
        cand_match = _build_match(sorted_T_rand, sorted_J_rand)
        cand_cost = sum(_calculate_target(jobs, slot)[0]
                        for slot, jobs in cand_match.items())
        if cand_cost < best_B_cost:
            best_B_cost = cand_cost
            best_B_match = cand_match

    if best_B_cost < cost_A:
        return best_B_match
    else:
        return match_A

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


def neighborhood_q1(ini_solution, start_time, T_max):
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

def neighborhood_q2(ini_solution, start_time, T_max):
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

def neighborhood_q3(ini_solution, start_time, T_max):
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

def neighborhood_q4(ini_solution, start_time, T_max):
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

def neighborhood_q5(ini_solution, start_time, T_max):
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

def neighborhood_q6(ini_solution, start_time, T_max):
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
        new_sol = NEIGHBORS[nei_id](ini_solution, start_time, T_max)
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
        
def run_once():
    neighborhood = random.choice([
        "3,1,5", "3,2,4", "4,3,2", "6,4,1", "5,2,1", "4,1,2", "1,5,4",
        "3,5,1", "4,2,5", "4,3,5", "1,4,2,3", "5,3,4,6", "6,4,2,5",
        "2,4,1,6", "2,6,1,4", "3,6,2,5", "5,3,4,6", "6,4,5,2", "2,6,1,5",
        "4,3,6,1", "3,4,6,1,2", "3,4,6,5,2", "5,2,4,3,6", "6,5,2,4,3",
        "6,4,5,1,2", "1,4,3,6,2", "2,6,4,5,3", "4,6,3,2,5", "3,5,2,4,6",
        "6,4,5,3,2", "2,4,1,6,5,3", "3,1,4,2,6,5", "6,5,2,1,3,4",
        "6,2,3,1,4,5", "3,2,1,4,5,6", "2,6,3,4,5,1", "4,6,5,2,3,1",
        "1,5,2,4,3,6", "5,4,6,2,3,1", "2,6,5,1,3,4"
    ])
    global NEIGHBOR_ORDER
    NEIGHBOR_ORDER = list(map(int, neighborhood.split(",")))

    start = time.perf_counter()
    Match = _initialize()
    best_target, _, _ = VND(Match, T_max=60)
    elapsed = time.perf_counter() - start
    return elapsed, best_target, neighborhood


if __name__ == "__main__":
    
    times, objs = [], []
    for k in range(10):
        t, obj, neighborhood = run_once()
        times.append(t)
        objs.append(obj)
        print(f"The {k+1}th: Time={t:.2f}s, Objective={obj:.2f}, neighborhood={neighborhood}")

    avg_time = np.mean(times)
    avg_obj  = np.mean(objs)
    best_obj = min(objs)

    print({"Best cost": best_obj,
                      "Average cost" : avg_obj,

                      "Average time": avg_time})

from episode import read_problem
from factory_status import FactoryStatus
import sys
from Simulator import Sim
from copy import deepcopy as cp
import datetime
import random
from patd3_agent import build_agent
import torch
import numpy as np


def load_test_problem(problem_index, params):
    drx = read_problem(params['test_group'], problem_index, params)
    fs = FactoryStatus(drx.pi)
    fs.initialize_all()
    sim = Sim(drx.pi, fs)

    return sim

def possible_lot_machine(waiting_lot_list, idle_machine_list, sim):
    machine_dict = {}
    lot_dict = {}
    temp_list = []
    for machine in idle_machine_list:
        cur_machine = sim.pi.machine_map[machine]
        if cur_machine.last_work_job_operation == 0:
            setup = sim.pi.INITIAL_SETUP[machine]
        else:
            setup = cur_machine.last_work_job_operation
        f = cur_machine.factory_id
        mt = cur_machine.machine_type
        key = f'{f}-{mt}-{setup}'
        if key not in machine_dict:
            machine_dict[key] = machine
    omitted_idle_machine_list = list(machine_dict.values())
    omitted_idle_machine_list = sorted(omitted_idle_machine_list, key=lambda x: int(x.split('_')[1]))
    for lot in waiting_lot_list:
        LOT_id = lot.split('-')[0]
        cur_lot = sim.pi.lot_map[LOT_id]
        job_ope = cur_lot.job_type + ('_') + str(cur_lot.current_operation)
        f = cur_lot.factory_id
        key = f'{f}-{job_ope}'
        if key not in lot_dict:
            lot_dict[f'{f}-{job_ope}'] = LOT_id
    omitted_waiting_lot_list = list(lot_dict.values())
    omitted_waiting_lot_list = sorted(omitted_waiting_lot_list, key=lambda x: int(x.split('_')[1].split('-')[0]))

    for LOT1 in omitted_waiting_lot_list:
        cur_lot = sim.pi.lot_map[LOT1]
        job_ope = cur_lot.job_type + ('_') + str(cur_lot.current_operation)
        for machine1 in omitted_idle_machine_list:
            cur_machine = sim.pi.machine_map[machine1]
            if job_ope in cur_machine.processing_time_map:
                temp_list.append((LOT1, machine1))

    return temp_list


def apply_random_branching(candidates, branching_value, n_total=None):
    # Second branching phase (Algorithm 2): sample beta_t = max(int(N_t * p_b), 1) candidates.
    if not candidates:
        return candidates
    p = float(branching_value)
    if p >= 1.0:
        return candidates
    k = max(int((len(candidates) if n_total is None else n_total) * p), 1)
    if k >= len(candidates):
        return candidates
    return random.sample(candidates, k)


def Rollout(params, load_path):
    dataset_sum_real_makespan = 0
    dataset_sum_duration = datetime.timedelta(days=0, hours=0, minutes=0, seconds=0)

    validation_num = params['train_validation_test'][0] + params['train_validation_test'][1]
    test_num = params['train_validation_test'][0] + params['train_validation_test'][1] + \
               params['train_validation_test'][2]

    agent = build_agent(params)
    agent.load_model(load_path)

    for j in range(validation_num + 1, test_num + 1):
        dataset_num = params['dataset']
        problem_index = f'{dataset_num}-{j}'
        sim = load_test_problem(problem_index, params)

        is_break = False
        real_min_makespan = sys.maxsize
        start_T = datetime.datetime.now()

        cached_policy_makespan = None
        committed_moves, best_move_seq = [], None
        k = 0
        while not sim.fs.is_done:
            min_makespan = sys.maxsize
            sim.fs.waiting_lot_list = sorted(sim.fs.waiting_lot_list, key=lambda x: int(x.split('_')[1].split('-')[0]))
            sim.fs.idle_machine_list = sorted(sim.fs.idle_machine_list, key=lambda x: int(x.split('_')[1]))
            temp_list = possible_lot_machine(sim.fs.waiting_lot_list, sim.fs.idle_machine_list, sim)
            temp_list = sorted(temp_list, key=lambda x: (int(x[0].split('_')[1]), int(x[1].split('_')[1])))
            n_candidates = len(temp_list)
            p_lot, p_machine = None, None
            p_b = float(params.get('branching_value', 1.0))
            # The actor's own candidate is left out of the random sample; it is added back at the
            # first decision, or when it is the only candidate.
            if temp_list and bool(params.get('rollout_random_branching', True)) and p_b < 1.0:
                pi_copy, fs_copy = cp(sim.pi), cp(sim.fs)
                policy_state = sim.get_state(fs_copy.waiting_lot_list, fs_copy.idle_machine_list, pi_copy, fs_copy)
                policy_state_tensor = torch.as_tensor(torch.from_numpy(np.asarray(policy_state)), dtype=torch.float32)
                policy_action = agent.select_action(policy_state_tensor)
                p_lot, p_machine = sim.cal_action(fs_copy.waiting_lot_list, fs_copy.idle_machine_list, policy_action, params, pi_copy, fs_copy)
                pool = [x for x in temp_list if x != (p_lot, p_machine)]
                temp_list = apply_random_branching(pool, p_b, n_candidates)
                if p_lot and not temp_list:
                    temp_list.append((p_lot, p_machine))
                if k == 0 and p_lot and (p_lot, p_machine) not in temp_list:
                    temp_list.insert(0, (p_lot, p_machine))
            if temp_list:
                k += 1
                for lot_machine in temp_list:
                    LOT, MACHINE = lot_machine[0], lot_machine[1]
                    is_policy = (p_lot is not None and LOT == p_lot and MACHINE == p_machine)
                    if is_policy and cached_policy_makespan is not None:
                        cached = cached_policy_makespan
                        if cached < min_makespan:
                            min_makespan = cached
                            min_lot, min_machine = LOT, MACHINE
                        if cached < real_min_makespan:
                            real_min_makespan = cached
                        now_T = datetime.datetime.now()
                        duration = now_T - start_T
                        continue
                    pi, fs = cp(sim.pi), cp(sim.fs)
                    sim.step(LOT, MACHINE, pi, fs)
                    policy_moves = []
                    while not fs.is_done:
                        s = sim.get_state(fs.waiting_lot_list, fs.idle_machine_list, pi, fs)
                        state_tensor = torch.as_tensor(torch.from_numpy(np.asarray(s)), dtype=torch.float32)
                        a = agent.select_action(state_tensor)
                        lot_id, machine_id = sim.cal_action(fs.waiting_lot_list, fs.idle_machine_list, a, params, pi, fs)
                        if lot_id:
                            policy_moves.append((lot_id, machine_id))
                        sim.step(lot_id, machine_id, pi, fs)

                    if (fs.makespan < min_makespan):
                        min_makespan = fs.makespan
                        min_lot, min_machine = LOT, MACHINE
                    if (fs.makespan < real_min_makespan):
                        real_min_makespan = fs.makespan
                        best_move_seq = list(committed_moves) + [(LOT, MACHINE)] + policy_moves
                    if is_policy:
                        cached_policy_makespan = fs.makespan
                    now_T = datetime.datetime.now()
                    duration = now_T - start_T
                    if duration.total_seconds() > params['total_seconds']:
                        is_break = True
                        break

                # Carried over so that a step whose only candidate is the actor's skips the look-ahead.
                cached_policy_makespan = min_makespan

                # Advance along the best complete schedule found so far (U* in Algorithm 3).
                if best_move_seq is not None and len(committed_moves) < len(best_move_seq):
                    min_lot, min_machine = best_move_seq[len(committed_moves)]
                committed_moves.append((min_lot, min_machine))
                sim.step(min_lot, min_machine, sim.pi, sim.fs)

            else:
                sim.step('', '', sim.pi, sim.fs)
                continue
            if (is_break == True):
                break

        if real_min_makespan in (0, sys.maxsize):
            raise RuntimeError(
                f'{problem_index}: the rollout produced no complete schedule '
                f'(total_seconds={params["total_seconds"]}s). '
                f'Increase the time limit or lower branching_value.')

        print(problem_index, real_min_makespan)
        dataset_sum_real_makespan += real_min_makespan
        dataset_sum_duration += duration

    average_real_makespan = dataset_sum_real_makespan / int(params['train_validation_test'][2])
    average_duration = dataset_sum_duration / float(params['train_validation_test'][2])

    print(average_real_makespan)

    return average_real_makespan, average_duration

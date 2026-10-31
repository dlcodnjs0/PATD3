import sys
import torch
import itertools
from collections import defaultdict


class Sim():
    def __init__(self, pi, fs):
        self.pi = pi
        self.fs = fs
        self.T = 0
        self.score = 0
        self.action_count = 0
        self.build_tables(pi)

    def build_tables(self, pi):
        table = {}
        for row in pi.SETUP_TIME:
            key = (int(row[0][0]), bool(row[0][1]), bool(row[0][2]))
            assert key not in table, f'duplicate SETUP_TIME key: {key}'
            table[key] = int(row[1])
        assert len(table) == len(pi.SETUP_TIME)
        self.setup_time_table = table
        self.machines_by_operation = {}
        st = list(itertools.chain.from_iterable(pi.setup_time_dict.values()))
        self.max_setup_time, self.min_setup_time = max(st), min(st)
        pt = pi.PROCESSING_TIME_all.values()
        self.max_op_processing_time, self.min_op_processing_time = max(pt), min(pt)
        self.max_operation_num = max(pi.operation_num_dict.values())
        self.lot_ids = ['LOT_' + str(i + 1) for i in range(pi.num_of_lot)]
        self.machine_indices = [(m, int(m.split('_')[1]) - 1) for m in pi.machine_list]
        self.jobtype_divisors = {jt: (pi.operation_num_dict[jt], pi.PRODUCTION_REQUIREMENT[jt])
                                 for jt in pi.PRODUCTION_REQUIREMENT}
        self.id_numbers, self.base_lot_ids, self.operation_parts = {}, {}, {}

    def id_number_key(self, lst):
        d = self.id_numbers
        for x in lst:
            if x not in d:
                d[x] = int(x.split('_')[1].split('-')[0])
        return d.__getitem__

    def available_machines(self, pi, job_ope):
        lst = self.machines_by_operation.get(job_ope)
        if lst is None:
            lst = [ma for ma in pi.machine_list if job_ope in pi.machine_map[ma].processing_time_map]
            self.machines_by_operation[job_ope] = lst
        return lst

    def setup(self, lotid, machineid, pi):
        current_lot = pi.lot_map[lotid]
        current_machine = pi.machine_map[machineid]

        if current_machine.last_work_job_operation == 0:
            setup = pi.INITIAL_SETUP[machineid]
        else:
            setup = current_machine.last_work_job_operation

        job_op = self.operation_parts.get(setup)
        if job_op is None:
            job, op = setup.split('_')
            job_op = self.operation_parts[setup] = (job, int(op))
        current_setup_lst = []
        current_setup_lst.append(current_machine.machine_type)
        if job_op[0] == current_lot.job_type:
            current_setup_lst.append(True)
        else:
            current_setup_lst.append(False)
        if job_op[1] == current_lot.current_operation:
            current_setup_lst.append(True)
        else:
            current_setup_lst.append(False)

        return self.setup_time_table[(int(current_setup_lst[0]), bool(current_setup_lst[1]), bool(current_setup_lst[2]))]

    def setup_signature(self, machine, machineid, pi):
        if machine.last_work_job_operation == 0:
            setup = pi.INITIAL_SETUP[machineid]
        else:
            setup = machine.last_work_job_operation
        job_op = self.operation_parts.get(setup)
        if job_op is None:
            job, op = setup.split('_')
            job_op = self.operation_parts[setup] = (job, int(op))
        return int(machine.machine_type), job_op[0], job_op[1]

    def cal_action(self, waiting_lot_list, idle_machine_list, actor_action, params, pi, fs):
        action_list = []
        min_lot_id = ''
        min_machine_id = ''
        machine_dict = {}
        lot_dict = {}
        waiting_lot_list = sorted(waiting_lot_list, key=self.id_number_key(waiting_lot_list))
        idle_machine_list = sorted(idle_machine_list, key=self.id_number_key(idle_machine_list))

        # Candidates: one machine per (factory, machine type, setup) and one lot per
        # (factory, operation type); identical pairs would have identical action vectors.
        for machine in idle_machine_list:
            cur_machine = pi.machine_map[machine]
            if cur_machine.last_work_job_operation == 0:
                setup = pi.INITIAL_SETUP[machine]
            else:
                setup = cur_machine.last_work_job_operation
            f = cur_machine.factory_id
            mt = cur_machine.machine_type
            key = f'{f}-{mt}-{setup}'
            if key not in machine_dict:
                machine_dict[key] = machine
        omitted_idle_machine_list = list(machine_dict.values())
        omitted_idle_machine_list = sorted(omitted_idle_machine_list, key=self.id_number_key(omitted_idle_machine_list))
        base_lot_ids = self.base_lot_ids
        for lot in waiting_lot_list:
            LOT_id = base_lot_ids.get(lot)
            if LOT_id is None:
                LOT_id = base_lot_ids[lot] = lot.split('-')[0]
            cur_lot = pi.lot_map[LOT_id]
            job_ope = cur_lot.job_type + ('_') + str(cur_lot.current_operation)
            f = cur_lot.factory_id
            key = f'{f}-{job_ope}'
            if key not in lot_dict:
                lot_dict[f'{f}-{job_ope}'] = LOT_id
        omitted_waiting_lot_list = list(lot_dict.values())
        omitted_waiting_lot_list = sorted(omitted_waiting_lot_list, key=self.id_number_key(omitted_waiting_lot_list))

        setup_table = self.setup_time_table
        machine_map = pi.machine_map
        num_of_lot = pi.num_of_lot
        st_max_value, st_min_value = self.max_setup_time, self.min_setup_time
        st_span = st_max_value - st_min_value
        pt_min, pt_span = self.min_op_processing_time, self.max_op_processing_time - self.min_op_processing_time
        feature_index = [i for i, tok in enumerate(('ST', 'PT', 'RO', 'RL', 'RP', 'AS', 'MS'))
                         if tok in params['action_feature']]
        aver_setup_by_ope = {}

        machine_signature = {ma: self.setup_signature(machine_map[ma], ma, pi) for ma in pi.machine_list}
        machine_info = {}
        for machine in omitted_idle_machine_list:
            cur_machine = machine_map[machine]
            if cur_machine.last_work_job_operation == 0:
                cur_setup = pi.INITIAL_SETUP[machine]
            else:
                cur_setup = cur_machine.last_work_job_operation
            machine_setup_op_num = fs.waiting_optype_dict[cur_setup]
            machine_info[machine] = (cur_machine.processing_time_map, machine_signature[machine],
                                     2 * (machine_setup_op_num / num_of_lot) - 1)

        index, lot_machine_dict = 0, {}
        for LOT_id in omitted_waiting_lot_list:
            cur_lot = pi.lot_map[LOT_id]
            lot_job, lot_op = cur_lot.job_type, cur_lot.current_operation
            job_ope = lot_job + ('_') + str(lot_op)
            remnant = pi.operation_num_dict[lot_job] - lot_op
            remnant_lot_num = fs.waiting_optype_dict[job_ope]
            remnant_total_processing_time = pi.op_remnant_total_processing_time[job_ope]

            aver_all_setup_time = aver_setup_by_ope.get(job_ope)
            if aver_all_setup_time is None:
                avail_machines = self.available_machines(pi, job_ope)
                total_setup_on_avail_ma = 0
                for ma in avail_machines:
                    mt, sj, so = machine_signature[ma]
                    total_setup_on_avail_ma += setup_table[(mt, sj == lot_job, so == lot_op)]
                aver_all_setup_time = aver_setup_by_ope[job_ope] = total_setup_on_avail_ma / len(avail_machines)

            remnant_lot_num_a = 2 * ((remnant_lot_num - 1) / (num_of_lot - 1)) - 1
            aver_all_setup_time_a = 2*((aver_all_setup_time - st_min_value) / st_span)-1
            remnant_a = 2 * (remnant / self.max_operation_num) - 1
            remnant_total_processing_time_a = 2 * (remnant_total_processing_time / pi.max_processing_time) - 1

            for machine in omitted_idle_machine_list:
                ptmap, (mt, sj, so), machine_setup_op_num_a = machine_info[machine]
                processing_time = ptmap.get(job_ope)
                if processing_time is None:
                    continue
                setup_time = setup_table[(mt, sj == lot_job, so == lot_op)]

                setup_time_a = 2 * ((setup_time - st_min_value) / st_span) - 1
                processing_time_a = 2*((processing_time - pt_min) / pt_span)-1

                all_action = [setup_time_a, processing_time_a, remnant_a, remnant_lot_num_a, remnant_total_processing_time_a, aver_all_setup_time_a, machine_setup_op_num_a]
                action = [all_action[i] for i in feature_index]
                action_list.append(action)
                lot_machine_dict[index] = (LOT_id, machine)

                index += 1

        if action_list:
            tensor1 = torch.tensor(action_list)
            tensor2 = torch.as_tensor(actor_action[0])

            # eq. (10): each candidate is scored by its inner product with the proxy action;
            # the highest score wins (ties go to the first candidate).
            score_list = torch.sum(tensor1 * tensor2, dim=1)
            neg_score_list = (-score_list).tolist()
            min_index = neg_score_list.index(min(neg_score_list))

            min_lot_id, min_machine_id = lot_machine_dict[min_index]

        return min_lot_id, min_machine_id

    def step(self, min_lot_id, min_machine_id, pi, fs):
        """Advance the simulation by one step.

        With a (lot, machine) pair, the lot's current operation is scheduled on the machine
        (setup, then processing). With an empty lot id, no pair is feasible: time moves to the
        earliest machine completion, and the machines and lots released there become available.
        The episode ends when every lot has finished its last operation.
        """
        if min_lot_id != '':
            min_lot_id = min_lot_id.split('-')[0]
            min_machine = pi.machine_map[min_machine_id]
            min_lot = pi.lot_map[min_lot_id]

            max_value = max(min_lot.last_work_finish_time, min_machine.last_work_finish_time)
            job_oper = min_lot.job_type + '_' + str(min_lot.current_operation)
            current_calculated_time = max_value
            current_calculated_time += self.setup(min_lot_id, min_machine_id, pi)
            current_calculated_time += min_machine.processing_time_map[job_oper]

            min_job_type = min_lot.job_type
            min_lot_job = min_lot_id + ('-') + min_job_type
            min_lot_job_operation = min_lot_id + ('-') + job_oper

            pi.lot_map[min_lot_id].factory_id = min_machine.factory_id
            fs.Schedule[min_machine_id].append(min_lot_job_operation)
            fs.waiting_lot_list.remove(min_lot_job)
            fs.waiting_optype_dict[job_oper] -= 1
            fs.idle_machine_list.remove(min_machine_id)
            min_lot.last_work_finish_time = current_calculated_time
            min_machine.last_work_finish_time = current_calculated_time
            fs.machine_completion_time_list[int(min_machine_id.split('_')[1]) - 1] = current_calculated_time
            min_lot.current_operation += 1
            min_machine.last_work_job_operation = job_oper
            min_machine.last_work_lot = min_lot_id+ ('-') + min_job_type

        else:
            fastest_completion_time = sys.maxsize
            fastest_completion_lots = []
            fastest_completion_machines = []

            for mac in pi.machine_list:
                if mac not in fs.idle_machine_list:
                    cur_machine = pi.machine_map[mac]
                    if cur_machine.last_work_finish_time < fastest_completion_time:
                        fastest_completion_time = cur_machine.last_work_finish_time
                        fastest_completion_lots = [cur_machine.last_work_lot]
                        fastest_completion_machines = [mac]
                    elif cur_machine.last_work_finish_time == fastest_completion_time:
                        fastest_completion_lots.append(cur_machine.last_work_lot)
                        fastest_completion_machines.append(mac)
            self.T = fastest_completion_time
            for f in range(len(fastest_completion_lots)):
                fastest_completion_lot = fastest_completion_lots[f]
                fastest_completion_machine = fastest_completion_machines[f]
                lot = fastest_completion_lot.split('-')[0]
                current_lot = pi.lot_map[lot]
                if current_lot.current_operation - 1 != pi.lot_ope_num_lst[int(lot.split('_')[1])-1]:
                    fs.waiting_lot_list.append(fastest_completion_lot)
                    fs.waiting_optype_dict[current_lot.job_type + '_' + str(current_lot.current_operation)] += 1
                fs.idle_machine_list.append(fastest_completion_machine)

        fs.makespan = max(fs.machine_completion_time_list) / 60
        cor_ope_num = 0

        for i in range(len(pi.lot_ope_num_lst)):
            Lot_id = 'LOT_' + str(i + 1)
            current_lot = pi.lot_map[Lot_id]

            if current_lot.current_operation - 1 == pi.lot_ope_num_lst[i]:
                cor_ope_num += 1
        if cor_ope_num == len(pi.lot_ope_num_lst):
            fs.is_done = True

    def get_state(self, waiting_lot_list, idle_machine_list, pi, fs):
        lot_map = pi.lot_map
        jobtype_divisors = self.jobtype_divisors
        jobtype_progress_dict = defaultdict(int)
        for lot_id in self.lot_ids:
            cur_lot = lot_map[lot_id]
            jt = cur_lot.job_type
            op_num, prod_req = jobtype_divisors[jt]
            jobtype_progress_dict[jt] += (((cur_lot.current_operation - 1) / op_num) / prod_req)
        JTPR_ave = sum(value for value in jobtype_progress_dict.values()) / len(jobtype_progress_dict)

        waiting_operation_type_dict = defaultdict(int)
        idle_setup_type_dict = defaultdict(int)
        idle_setup_type_num = 0
        if idle_machine_list:
            for machine in idle_machine_list:
                cur_machine = pi.machine_map[machine]
                if cur_machine.last_work_job_operation == 0:
                    setup = pi.INITIAL_SETUP[machine]
                else:
                    setup = cur_machine.last_work_job_operation
                idle_setup_type_dict[setup] += 1
            idle_setup_type_num = len(idle_setup_type_dict.keys())
        waiting_optype_num = 0
        if waiting_lot_list:
            base_lot_ids = self.base_lot_ids
            for lot in waiting_lot_list:
                lot_id = base_lot_ids.get(lot)
                if lot_id is None:
                    lot_id = base_lot_ids[lot] = lot.split('-')[0]
                cur_lot = pi.lot_map[lot_id]
                job_ope = cur_lot.job_type + ('_') + str(cur_lot.current_operation)
                waiting_operation_type_dict[job_ope] += 1
            waiting_optype_num = len(waiting_operation_type_dict.keys())

        machine_map = pi.machine_map
        initial_setup = pi.INITIAL_SETUP
        completion_times = fs.machine_completion_time_list
        setuptype_makespan_dict = defaultdict(int)
        setuptype_count_dict = defaultdict(int)
        for machine, m in self.machine_indices:
            cur_machine = machine_map[machine]
            if cur_machine.last_work_job_operation == 0:
                setup = initial_setup[machine]
            else:
                setup = cur_machine.last_work_job_operation
            setuptype_makespan_dict[setup] += completion_times[m]
            setuptype_count_dict[setup] += 1

        MTMS_ave2_result_dict = {k: (setuptype_makespan_dict[k] / setuptype_count_dict[k]) for k in setuptype_makespan_dict}
        MTMS_ave2 = sum(value for value in MTMS_ave2_result_dict.values()) / len(MTMS_ave2_result_dict)

        if waiting_lot_list:
            waiting_optype_num = waiting_optype_num / len(waiting_lot_list)
        else:
            waiting_optype_num = 0
        if idle_machine_list:
            idle_setup_type_num = idle_setup_type_num / len(idle_machine_list)
        else:
            idle_setup_type_num = 0

        # [s^P, s^C, s^W, s^I] (Section IV-A-1)
        state = [JTPR_ave, MTMS_ave2, waiting_optype_num, idle_setup_type_num]

        return state

    def get_reward(self, lot_id, machine_id, pi):
        reward = - self.setup(lot_id, machine_id, pi)
        return reward

import csv
import os
from functools import lru_cache

from problem import ProblemInstance, Lot, Machine


@lru_cache(maxsize=None)
def read_table(path):
    with open(path, newline='', encoding='utf-8') as f:
        return tuple(csv.DictReader(f, delimiter='\t'))


class DatasetReader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.pi = ProblemInstance()

    def table(self, name):
        return read_table(os.path.join(self.data_dir, name + '.txt'))

    def read_problem(self, dataset_index, problem_index):
        row = next(r for r in self.table('problem')
                   if r['dataset_index'] == dataset_index and int(r['problem_index']) == problem_index)
        pi = self.pi

        demand = {}
        for item in row['ProductionRequirements'].split(','):
            job, qty = item.rsplit('_', 1)
            demand[job] = int(qty)
        for job in sorted(demand):
            pi.PRODUCTION_REQUIREMENT[job] = demand[job]

        for part in row['MachineNumbers'].split(';'):
            machine_type, machines = part.split(':')
            first, last = (int(m.split('_')[1]) for m in machines.split('-'))
            pi.machine_type[int(machine_type)] = ['M_' + str(i) for i in range(first, last + 1)]
        for machine_lst in pi.machine_type.values():
            pi.machine_list += machine_lst

        for r in self.table('jobTypes'):
            if r['jobTypeName'] in pi.PRODUCTION_REQUIREMENT:
                pi.operation_num_dict[r['jobTypeName']] = int(r['operationTypeSequence'].split(',')[-1])
        for job, num in pi.operation_num_dict.items():
            for op in range(1, num + 1):
                pi.operation_type_list.append(job + '_' + str(op))
        pi.operation_type_list = sorted(pi.operation_type_list)

        for setup in row['MachineSettings'].split(','):
            machine, operation = setup.split('-')
            pi.INITIAL_SETUP[machine] = operation

        for r in self.table('setupTime'):
            pi.SETUP_TIME.append([[int(r['MachineTypeId']), r['isJobTypeSame'] == 'TRUE',
                                   r['isOperationTypeSame'] == 'TRUE'], int(r['SetupTime'])])
        for machine_type in pi.machine_type:
            pi.setup_time_dict[machine_type] = []
        for (machine_type, _, _), setup_time in pi.SETUP_TIME:
            pi.setup_time_dict[machine_type].append(setup_time)

        pi.num_of_lot = sum(pi.PRODUCTION_REQUIREMENT.values())
        pi.num_of_machine = len(pi.INITIAL_SETUP)

        listed = {}
        for item in row['ProcessingTimes'].split(','):
            operation, pt = item.split(':')
            listed[operation] = int(pt)
        # Operation types are taken from the first 3 x (number of operation types) rows of operationTypes.
        window = self.table('operationTypes')[:sum(pi.operation_num_dict.values()) * 3]
        time_by_machine_type = {}
        for machine_type in pi.machine_type:
            pi.operation_type[machine_type] = [r['operationTypeName'] for r in window
                                               if int(r['machineTypeId']) == machine_type]
            time_by_machine_type[machine_type] = {op: listed[op] for op in pi.operation_type[machine_type]
                                                  if op in listed}
            pi.PROCESSING_TIME_all.update(time_by_machine_type[machine_type])

        processing_time = {op: pt for op, pt in pi.PROCESSING_TIME_all.items() if op in pi.operation_type_list}
        jobtype_total_pt = {job: 0 for job in pi.operation_num_dict}
        for op, pt in processing_time.items():
            jobtype_total_pt[op.split('_')[0]] += pt
        pi.max_processing_time = max(jobtype_total_pt.values())

        for job, num in pi.operation_num_dict.items():
            pi.lot_ope_num_lst += [num] * pi.PRODUCTION_REQUIREMENT[job]

        for optype in pi.operation_type_list:
            op_num, job = int(optype.split('_')[1]), optype.split('_')[0]
            pi.op_remnant_total_processing_time[optype] = sum(
                processing_time[job + '_' + str(op)] for op in range(op_num + 1, pi.operation_num_dict[job] + 1))

        for machine_type, machines in pi.machine_type.items():
            for machine_id in machines:
                pi.machine_map[machine_id] = Machine(machine_id, machine_type, 0, 0,
                                                     time_by_machine_type[machine_type], 1, '')
        pi.machine_map = {f'M_{i}': pi.machine_map[f'M_{i}'] for i in range(1, pi.num_of_machine + 1)}

    def set_lot(self):
        lot_num = 0
        for job, qty in self.pi.PRODUCTION_REQUIREMENT.items():
            for _ in range(qty):
                lot_num += 1
                lot_id = 'LOT_' + str(lot_num)
                self.pi.lot_map[lot_id] = Lot(lot_id, job, 1, 0, 1)

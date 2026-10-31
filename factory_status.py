class FactoryStatus(object):
    def __init__(self, pi):
        self.pi = pi
        self.waiting_lot_list = []
        self.idle_machine_list = []
        self.waiting_optype_dict = {}
        self.machine_completion_time_list = []
        self.Schedule = {}
        self.makespan = 0
        self.is_done = False

    def initialize_all(self):
        for lot_id in self.pi.lot_map.keys():
            self.waiting_lot_list.append(lot_id + '-' + self.pi.lot_map[lot_id].job_type)

        self.idle_machine_list = list(self.pi.machine_map.keys())

        for optype in self.pi.operation_type_list:
            self.waiting_optype_dict[optype] = 0

        for lot_id, value in self.pi.lot_map.items():
            optype = value.job_type + ('_') + str(value.current_operation)
            self.waiting_optype_dict[optype] += 1

        for i in range(self.pi.num_of_machine):
            self.machine_completion_time_list.append(0)

        for key in self.pi.machine_map.keys():
            self.Schedule[key] = []

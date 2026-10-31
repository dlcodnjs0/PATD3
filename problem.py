class ProblemInstance(object):
    def __init__(self):
        self.PRODUCTION_REQUIREMENT = {}
        self.operation_num_dict = {}
        self.PROCESSING_TIME_all = {}
        self.INITIAL_SETUP = {}
        self.SETUP_TIME = []
        self.lot_map = {}
        self.machine_map = {}
        self.num_of_lot = 0
        self.num_of_machine = 0
        self.machine_type = {}
        self.operation_type = {}
        self.operation_type_list = []
        self.machine_list = []
        self.setup_time_dict = {}
        self.max_processing_time = 0
        self.lot_ope_num_lst = []
        self.op_remnant_total_processing_time = {}


class Lot(object):
    def __init__(self, lot_id, job_type, current_operation, last_work_finish_time, factory_id):
        self.lot_id = lot_id
        self.job_type = job_type
        self.current_operation = current_operation
        self.last_work_finish_time = last_work_finish_time
        self.factory_id = factory_id


class Machine(object):
    def __init__(self, machine_id, machine_type, last_work_job_operation, last_work_finish_time, processing_time_map, factory_id, last_work_lot):
        self.machine_id = machine_id
        self.machine_type = machine_type
        self.last_work_job_operation = last_work_job_operation
        self.last_work_finish_time = last_work_finish_time
        self.processing_time_map = processing_time_map
        self.factory_id = factory_id
        self.last_work_lot = last_work_lot

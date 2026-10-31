from Simulator import Sim
import random
from copy import deepcopy as cp
import read_dataset
from factory_status import FactoryStatus
import torch
import numpy as np


def read_problem(group, problem_index, params):
    dataset, problem = problem_index.split('-')
    drx = read_dataset.DatasetReader(params['data_dir'])
    drx.read_problem(group + dataset, int(problem))
    drx.set_lot()
    return drx


class EpisodeRunner():
    def load_problem(self, problem_index, params):
        return read_problem(params['train_group'], problem_index, params)

    def load_test_problem(self, problem_index, params):
        return read_problem(params['test_group'], problem_index, params)

    def resample_initial_setup(self, drx):
        for machine in drx.pi.machine_list:
            current_machine = drx.pi.machine_map[machine]
            if current_machine.last_work_job_operation == 0:
                for type, m_lst in drx.pi.machine_type.items():
                    if machine in m_lst:
                        break
                filtered_initial_setups = [setup for setup in drx.pi.operation_type[type] if
                                           setup.split('_')[0] in drx.pi.PRODUCTION_REQUIREMENT]
                cur_INITIAL_SETUP = random.choice(filtered_initial_setups)
                drx.pi.INITIAL_SETUP[machine] = cur_INITIAL_SETUP
        return drx

    def run_episode(self, params, drx, is_store, agent):
        pi = cp(drx.pi)
        fs = FactoryStatus(pi)
        fs.initialize_all()
        sim = Sim(pi, fs)
        last_T = 0
        last_s, last_a, last_r = None, None, 0

        while not sim.fs.is_done:
            s = sim.get_state(sim.fs.waiting_lot_list, sim.fs.idle_machine_list, sim.pi, sim.fs)

            state_tensor = torch.as_tensor(torch.from_numpy(np.asarray(s)), dtype=torch.float32)

            # Proxy action (Algorithm 1, line 7): uniform random during warm-up,
            # then the actor output plus Gaussian exploration noise.
            if (is_store == True) and (len(agent.memory) < params['TRAINING_START']):
                a = [np.random.uniform(-agent.max_action, agent.max_action,
                                       size=(params['action_size'],))]
            elif (is_store == True) and (len(agent.memory) >= params['TRAINING_START']):
                a = ((agent.select_action(state_tensor)
                      + np.random.normal(0, agent.expl_noise * agent.max_action,
                                         size=agent.action_dim)).clip(-agent.max_action, agent.max_action))
            else:
                a = (agent.select_action(state_tensor))

            lot_id, machine_id = sim.cal_action(sim.fs.waiting_lot_list, sim.fs.idle_machine_list, a,
                                                params, sim.pi, sim.fs)

            if lot_id == '':
                sim.step(lot_id, machine_id, sim.pi, sim.fs)
            else:
                sim.action_count += 1
                # Machine idle time since the previous decision is charged to the previous reward.
                if (last_s is not None) and (sim.T != last_T):
                    for machine in sim.pi.machine_list:
                        current_machine = sim.pi.machine_map[machine]
                        if sim.T > current_machine.last_work_finish_time:
                            machine_idle = min(sim.T - current_machine.last_work_finish_time, sim.T-last_T)
                            last_r -= machine_idle
                if last_s is not None:
                    if is_store == True:
                        agent.append_sample(last_s, last_a, [last_r], s, [1.0])

                        sim.score += last_r
                        if len(agent.memory) > params['TRAINING_START']:
                            agent.train_model(params['batch_size'], params['gamma'])

                r = sim.get_reward(lot_id, machine_id, sim.pi)
                last_s, last_a, last_r = s, a[0], r
                last_T = sim.T
                sim.step(lot_id, machine_id, sim.pi, sim.fs)
        if is_store == True:
            sim.T = sim.fs.makespan * 60
            s = sim.get_state(sim.fs.waiting_lot_list, sim.fs.idle_machine_list, sim.pi, sim.fs)
            if (sim.T != last_T):
                for machine in sim.pi.machine_list:
                    current_machine = sim.pi.machine_map[machine]
                    if sim.T > current_machine.last_work_finish_time:
                        machine_idle = min(sim.T - current_machine.last_work_finish_time, sim.T - last_T)
                        last_r -= machine_idle
            agent.append_sample(last_s, last_a, [last_r], s, [0.0])

            sim.score += last_r
            if len(agent.memory) > params['TRAINING_START']:
                agent.train_model(params['batch_size'], params['gamma'])

        return agent, sim

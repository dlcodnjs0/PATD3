from patd3_agent import build_agent
import random
import sys
import numpy as np
import os
from episode import EpisodeRunner
import datetime
import time
import torch
from Rollout import Rollout


model_save_path = './saved_models'

dataset = [1]
test_dataset = [1, 2, 3, 4, 5]
num_of_repetitions = 5

params = {
    'run_rollout': True,
    'rollout_random_branching': True,
    'branching_value': 1.0,            # p_b (Algorithm 2)
    'total_seconds': 600,              # rollout time limit per problem

    'NUM_EPISODE': 1000,               # N_E
    'TRAINING_START': 25000,           # warm-up steps before the first update
    'MEMORY_SIZE': 100000,             # |B|
    'batch_size': 256,                 # N_TR
    'gamma': 0.99,
    'save_interval': 10,               # checkpoint interval (validation candidates)
    'print_interval': 10,
    'seed': 42,                        # repetition k uses seed + k

    'dataset': 1,
    'train_validation_test': [1, 9, 20],  # train / validation / test problems per dataset
    'data_dir': 'data',
    'train_group': 'SD',               # dataset group trained on: SD, MD or LD (Table I)
    'test_group': 'SD',                # dataset group tested on

    'state_size': 4,                   # (s^P, s^C, s^W, s^I), Section IV-A-1
    'action_feature': ['ST', 'RO', 'RL', 'AS', 'MS'],  # (a^S, a^R, a^J, a^A, a^M), Section IV-A-2
    'action_size': 5,

    'actor_lr': 3e-4, 'critic_lr': 3e-4,
    'tau': 0.005,                      # rho
    'policy_freq': 2,                  # psi
    'max_action': 1.0,
    'policy_noise': 0.2,
    'noise_clip': 0.5,
    'expl_noise': 0.1,

    'is_initial_setup_changes': True,  # resample the initial machine setups every episode
}


params['action_size'] = len(params['action_feature'])


def Train(date_time, params, model_save_path):
    is_store = True

    date_time_path = f"{model_save_path}/{date_time}"
    if not os.path.exists(date_time_path):
        os.makedirs(date_time_path)

    agent = build_agent(params)

    train_num = params['train_validation_test'][0]
    dataset_num = params['dataset']

    for n_epi in range(params['NUM_EPISODE']):
        util = EpisodeRunner()
        i = random.randint(1, train_num)
        problem_index = f'{dataset_num}-{i}'
        drx = util.load_problem(problem_index, params)

        if params['is_initial_setup_changes'] == True:
            drx = util.resample_initial_setup(drx)
        file_name = f'{date_time}_{problem_index}_{n_epi}'
        model_path = date_time_path + f'/{file_name}'

        start = time.time()

        agent, sim = util.run_episode(params, drx, is_store, agent)

        end = time.time()
        calculation_time = end - start

        if n_epi % params['print_interval'] == 0:
            print(problem_index, n_epi, sim.fs.makespan, sim.score, calculation_time)

        if (n_epi + 1) % params['save_interval'] == 0:
            agent.save_model(model_path)

    return date_time_path


def Validation(params, load_path):
    is_store = False

    best_average_makespan = sys.maxsize
    best_model_path = None
    train_num = params['train_validation_test'][0]
    validation_num = params['train_validation_test'][0] + params['train_validation_test'][1]
    filenames = os.listdir(f'{load_path}/')
    if not filenames:
        raise RuntimeError(
            f'No checkpoint in {load_path}. Checkpoints are saved every save_interval '
            f'({params["save_interval"]}) episodes, so none is written when '
            f'NUM_EPISODE ({params["NUM_EPISODE"]}) is smaller.')
    filenames.sort(key=lambda x: int(x.split('-')[1].split('_')[1]))

    for filename in filenames:
        model_sum_makespan = 0
        full_filename = load_path + f'/{filename}'

        agent = build_agent(params)
        agent.load_model(full_filename)

        for j in range(train_num + 1, validation_num + 1):
            dataset_num = params['dataset']
            problem_index = f'{dataset_num}-{j}'
            util = EpisodeRunner()
            drx = util.load_problem(problem_index, params)

            agent, sim = util.run_episode(params, drx, is_store, agent)

            model_sum_makespan += sim.fs.makespan

        model_average_makespan = model_sum_makespan / int(params['train_validation_test'][1])
        print(model_average_makespan, filename, int(params['train_validation_test'][1]))

        if model_average_makespan < best_average_makespan:
            best_average_makespan = model_average_makespan
            best_model_path = full_filename
    return best_model_path

def Test(params, load_path):
    is_store = False

    validation_num = params['train_validation_test'][0] + params['train_validation_test'][1]
    test_num = params['train_validation_test'][0] + params['train_validation_test'][1] + params['train_validation_test'][2]
    model_sum_makespan = 0
    model_sum_time = 0

    agent = build_agent(params)
    agent.load_model(load_path)

    for i in range(validation_num + 1, test_num + 1):
        dataset_num = params['dataset']
        problem_index = f'{dataset_num}-{i}'
        util = EpisodeRunner()
        drx = util.load_test_problem(problem_index, params)

        start = time.time()

        agent, sim = util.run_episode(params, drx, is_store, agent)
        end = time.time()
        calculation_time = end - start

        model_sum_makespan += sim.fs.makespan
        model_sum_time += calculation_time
    model_average_makespan = model_sum_makespan / int(params['train_validation_test'][2])
    model_average_time = model_sum_time / float(params['train_validation_test'][2])
    print('PATD3', model_average_makespan, load_path, int(params['train_validation_test'][2]))

    return model_average_makespan, model_average_time


def main():
    for k in range(num_of_repetitions):
        if params['seed'] is not None:
            random.seed(params['seed'] + k)
            np.random.seed(params['seed'] + k)
            torch.manual_seed(params['seed'] + k)

        for i in dataset:
            date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            params['dataset'] = i

            load_path = Train(date_time, params, model_save_path)
            best_model_path = Validation(params, load_path)

            for t in test_dataset:
                params['dataset'] = t
                network_dataset = i
                print(f"[CrossTest] network_dataset={network_dataset} eval_dataset={t}")

                Test(params, best_model_path)

                if params['run_rollout']:
                    Rollout(params, best_model_path)


if __name__ == '__main__':
    main()

import sys
import os
from setup_args_pkl import setup_args

graph_size = int(sys.argv[1])
setup_args(
    graph_size=graph_size,
    n_steps=1,
    rand_init_steps=0,
    perturb_nodes=10,
    epochs=-1,
    n_rollout=100,
)

import arguments
import pickle
from tqdm import tqdm
import torch
import numpy as np
import timeit
from functools import reduce
from pprint import pprint


from lib.egate_model import Model
from lib.utils_train_from_loaded_model import (
    create_batch_env,
    random_init,
    roll_out,
    IterativeMean,
)


def evaluate_perturbated_solutions(
    model,
    paths,
    batches_coords,
    batch_size,
    n_jobs,
    pre_steps,
    n_rollout,
    rollout_steps,
    eval_sampled_solutions_path,
    res_path,
):
    start = timeit.default_timer()
    model.eval()
    assert len(paths[0]) % batch_size == 0
    batches = len(paths)
    epoch_best_cost_avg, epoch_cost_avg = IterativeMean(), IterativeMean()
    best_tours = []

    mean_costs_tt = IterativeMean()
    mean_best_costs_tt = IterativeMean()
    for batch_i in tqdm(range(batches)):
        curr_paths = paths[batch_i]
        curr_batches_coords = batches_coords[batch_i]
        envs = create_batch_env(
            curr_paths, curr_batches_coords, n_jobs, batch_size, save_best_tours=True
        )

        states, prev_mean_cost = random_init(envs, pre_steps, batch_size, curr_paths)
        mean_costs = [[prev_mean_cost]]
        mean_best_costs = [[prev_mean_cost]]

        envs.reset_temperature()
        for i in range(n_rollout):
            _, states, curr_mean_cost, curr_mean_best_costs = roll_out(
                model, envs, states, rollout_steps, batch_size, n_jobs, is_last=False
            )
            mean_costs.append(curr_mean_cost)
            mean_best_costs.append(curr_mean_best_costs)

        mean_costs_tt.update_state(np.concatenate(mean_costs, axis=0))
        mean_best_costs_tt.update_state(np.concatenate(mean_best_costs, axis=0))

        cost_val = np.mean([env.cost for env in envs.envs])
        best_cost_val = np.mean([env.best for env in envs.envs])
        print("cost_val:", cost_val)
        print("best_cost_val:", best_cost_val)
        epoch_cost_avg.update_state(cost_val)
        epoch_best_cost_avg.update_state(best_cost_val)
        best_tours.append([env.best_tours for env in envs.envs])

        if res_path is not None:
            with open(f"{res_path}/costs_{batch_i}.pkl", "wb") as handle:
                pickle.dump(
                    {
                        "mean_costs_tt": mean_costs_tt.result(),
                        "mean_best_costs_tt": mean_best_costs_tt.result(),
                    },
                    handle,
                )

    print(f"Average_cost: {epoch_cost_avg.result()}")
    print(f"Best_average_cost: {epoch_best_cost_avg.result()}")
    end = timeit.default_timer()
    print("tiempo:", end - start)
    best_tours = reduce(lambda x, y: x + y, best_tours)
    with open(f"{eval_sampled_solutions_path}/cdcp_paths.pkl", "wb") as handle:
        pickle.dump(best_tours, handle)


def get_pilist_data(eval_sampled_solutions_path):
    paths = np.load(f"{eval_sampled_solutions_path}/adm_paths.npy", allow_pickle=True)
    batches_coords = np.load(
        f"{eval_sampled_solutions_path}/batches_coords.npy", allow_pickle=True
    )

    return paths, batches_coords


def perturbation(
    paths,
    batches_coords,
    model_path,
    eval_sampled_solutions_path,
    save_results_periodically=False,
):
    args = arguments.args()
    print("perturbation args:")
    pprint(vars(args))
    print("perturbation args:\n")

    device = torch.device(args.device)
    model = Model(4, 64, 2, 16)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    batch_size = int(args.BATCH)
    n_jobs = int(args.N_JOBS)
    pre_steps = int(args.RAND_INIT_STEPS)
    n_rollout = int(args.N_ROLLOUT)
    rollout_steps = int(args.ROLLOUT_STEPS)

    assert len(paths) == len(batches_coords) and len(paths) % batch_size == 0
    paths = list(
        map(lambda x: x.tolist(), np.split(paths, len(paths) // batch_size, axis=0))
    )
    batches_coords = list(
        map(
            lambda x: x.tolist(),
            np.split(batches_coords, len(batches_coords) // batch_size, axis=0),
        )
    )

    res_path = None
    if save_results_periodically:
        res_path = model_path[: model_path.rfind("/")] + "/results_tt/"
        if not os.path.isdir(res_path):
            os.makedirs(res_path)

    evaluate_perturbated_solutions(
        model,
        paths,
        batches_coords,
        batch_size,
        n_jobs,
        pre_steps,
        n_rollout,
        rollout_steps,
        eval_sampled_solutions_path,
        res_path,
    )


def main():
    model_path = sys.argv[2]
    eval_sampled_solutions_path = sys.argv[3]

    # Make sure the arguments in test_models.py (this file) and arguments.py are coordinated
    print("--- getting pilist and data ----")
    pi_list, data = get_pilist_data(eval_sampled_solutions_path)
    print("--- finish getting pilist and data ----\n")

    print("--- running perturbation model ----")
    perturbation(pi_list, data, model_path, eval_sampled_solutions_path)
    print("--- finish running perturbation model ----")


if __name__ == "__main__":
    main()

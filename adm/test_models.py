from tqdm import tqdm
import torch
import numpy as np
import warnings
from functools import reduce
import timeit
import sys

import sys

sys.path.append("../adm/")

from utils import generate_data_onfly, FastTensorDataLoader, CAPACITIES

from attention_dynamic_model import set_decode_type
from reinforce_baseline import load_pt_model


def adm_solve(data, batch_size, model):
    model.eval()
    train_batches = FastTensorDataLoader(
        data[0], data[1], data[2], batch_size=batch_size, shuffle=False
    )
    with torch.no_grad():
        epoch_costs = []
        pi_list = []
        for num_batch, x_batch in tqdm(enumerate(train_batches)):
            cost, _, pi = model(x_batch, return_pi=True)
            pi_list.append(pi)
            epoch_costs.append(cost.view(-1).cpu())
        epoch_cost_avg = torch.cat(epoch_costs, dim=0).mean().item()
    print(f"Average_cost: {epoch_cost_avg}")
    return pi_list, data, epoch_costs


def remove_repeated_cs(pi):
    new_pi = []
    px = -1
    for x in pi:
        if x != px:
            new_pi.append(x)
        px = x
    assert new_pi[-1] == 0
    return new_pi[:-1]


def constructive(model_path, graph_size, instances, batch_size, torch_data, dec_type):
    MODEL_PATH = model_path
    GRAPH_SIZE = graph_size
    INSTANCES = instances
    BATCH = batch_size
    DEMAND = CAPACITIES[GRAPH_SIZE]
    EMBEDDING_DIM = 128

    # Initialize model
    model = load_pt_model(
        MODEL_PATH, embedding_dim=EMBEDDING_DIM, graph_size=GRAPH_SIZE, device="cuda"
    )
    warnings.warn(f"Model is in {dec_type} mode")
    set_decode_type(model, dec_type)

    pi_list, _, costs = adm_solve(torch_data, BATCH, model)
    costs = torch.cat(costs, -1)
    pi_list = reduce(
        lambda x, y: x + y,
        [
            [[0] + remove_repeated_cs(pi.tolist()) + [0] for pi in pi_b]
            for pi_b in pi_list
        ],
    )

    depo, graphs, demand = torch_data

    demand = demand * DEMAND
    demand = torch.cat([torch.zeros(demand.shape[0], 1), demand], -1)
    coords = torch.cat([depo.unsqueeze(-2), graphs], -2)
    coords_demand = torch.cat([coords, demand.unsqueeze(-1)], -1)
    coords_demand = coords_demand.numpy()
    pi_list = np.array(pi_list, dtype=object)

    return pi_list, coords_demand, costs


def constructive_g_or_s(model_path, graph_size, instances, batch_size, samples=1):
    if samples == 1:
        return constructive(
            model_path, graph_size, instances, batch_size, torch_data, "greedy"
        )
    torch_data = generate_data_onfly(instances, graph_size)
    pi_list, coord_demand, costs = constructive(
        model_path, graph_size, instances, batch_size, torch_data, "sampling"
    )
    costs_tt = [costs.mean()]
    for i in range(samples - 1):
        pis, cds, curr_costs = constructive(
            model_path, graph_size, instances, batch_size, torch_data, "sampling"
        )
        for i, change in enumerate(curr_costs < costs):
            if change:
                pi_list[i] = pis[i]
                coord_demand[i] = cds[i]
                costs[i] = curr_costs[i]
        costs_tt.append(costs.mean())
    return pi_list, coord_demand, costs_tt


def main():
    start_t = timeit.default_timer()

    graph_size = int(sys.argv[1])
    model_path = sys.argv[2]
    eval_samples_path = sys.argv[3]
    instances = int(sys.argv[4])
    batch_size = int(sys.argv[5])
    samples = int(sys.argv[6]) if len(sys.argv) == 7 else 1
    print("Doing {samples} samples")

    assert instances % batch_size == 0

    pi_list, coords_demand, _ = constructive_g_or_s(
        model_path, graph_size, instances, batch_size, samples
    )

    np.save(f"{eval_samples_path}/adm_paths.npy", pi_list)
    np.save(f"{eval_samples_path}/batches_coords.npy", coords_demand)

    end_t = timeit.default_timer()
    print("tiempo:", end_t - start_t)


if __name__ == "__main__":
    main()

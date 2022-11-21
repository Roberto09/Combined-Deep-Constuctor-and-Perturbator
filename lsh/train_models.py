import sys
from setup_args_pkl import setup_args

graph_size = int(sys.argv[1])
epochs = int(sys.argv[2])
n_steps = int(sys.argv[3])
rand_init_steps = int(sys.argv[4])
perturb_nodes = int(sys.argv[5])
batch = int(sys.argv[6])

setup_args(
    graph_size=graph_size,
    n_steps=n_steps,
    rand_init_steps=rand_init_steps,
    perturb_nodes=perturb_nodes,
    epochs=epochs,
    batch=batch,
)

# Perturbation model utils
import numpy as np
import torch
from arguments import args
from lib.utils_train_from_loaded_model import train
from lib.egate_model import Model
from pprint import pprint
import time


def get_paths_batches_coords(paths_batches_coords_dir):
    # call load_data with allow_pickle implicitly set to true
    paths = np.load(f"{paths_batches_coords_dir}/adm_paths.npy", allow_pickle=True)
    batches_coords = np.load(
        f"{paths_batches_coords_dir}/batches_coords.npy", allow_pickle=True
    )
    return paths, batches_coords


if __name__ == "__main__":
    paths_batches_coords_dir = sys.argv[7]
    mod_dir = sys.argv[8]
    paths, batches_coords = get_paths_batches_coords(paths_batches_coords_dir)

    arg = args()
    device = torch.device(arg.device)

    model = Model(4, 64, 2, 16)
    model = model.to(device)

    start = time.time()
    print("-------------args-----------")
    pprint(vars(arg))
    print("----------------------------\n")

    train(model, paths, batches_coords, mod_dir)

    end = time.time()
    print("time:", end - start)

import pickle
import arguments
import copy
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def create_basic_args():
    args = arguments.default_args()
    args.SHOULD_EVAL_RAND = False
    args.SHOULD_SAVE = True
    return args


def save_args(args):
    with open(f"{DIR_PATH}/args.pkl", "wb") as handle:
        pickle.dump(args, handle)


def setup_args(
    graph_size=None,
    n_steps=None,
    rand_init_steps=None,
    perturb_nodes=None,
    epochs=None,
    batch=None,
    n_rollout=None,
):
    def mod_sa_arg(args, val):
        args.N_STEPS = val
        return args

    def mod_rand_init_arg(args, val):
        args.RAND_INIT_STEPS = val
        return args

    def mod_perturb_nodes(args, val):
        args.PERTURB_NODES = val
        return args

    def mod_n_jobs(args, val):
        args.N_JOBS = val
        return args

    def mod_epochs(args, val):
        args.EPOCHS = val
        return args

    def mod_batch_size(args, val):
        args.BATCH = val
        return args

    def mod_n_rollout(args, val):
        args.N_ROLLOUT = val
        return args

    func_and_override = [
        (mod_n_jobs, graph_size),
        (mod_sa_arg, n_steps),
        (mod_rand_init_arg, rand_init_steps),
        (mod_perturb_nodes, perturb_nodes),
        (mod_epochs, epochs),
        (mod_batch_size, batch),
        (mod_n_rollout, n_rollout),
    ]

    args = copy.deepcopy(create_basic_args())
    func_and_override = filter(lambda fo: fo[1] is not None, func_and_override)
    for func, override in func_and_override:
        args = func(args, override)
    save_args(args)

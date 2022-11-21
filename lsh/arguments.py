import pickle
import os

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

CAPACITIES = {10: 20.0, 20: 30.0, 50: 40.0, 100: 50.0}


class def_arg:
    def __init__(self):
        self.EPOCHS = 400
        self.TRAIN_STEPS = 4
        self.N_JOBS = 20  # 100 #99
        self.BATCH_SAMPLING = 128
        self.BATCH = 128
        self.MAX_COORD = 1  # 100
        self.MAX_DIST = 2**0.5  # 100*2**0.5
        self.LR = 3e-4
        self.DEPOT_END = 300
        self.SERVICE_TIME = 10
        self.TW_WIDTH = 30
        self.N_ROLLOUT = 10
        self.ROLLOUT_STEPS = 10
        self.N_STEPS = self.N_ROLLOUT * self.ROLLOUT_STEPS
        self.RAND_INIT_N_STEPS_SA = self.N_ROLLOUT * self.ROLLOUT_STEPS
        self.init_T = 100.0
        self.final_T = 1.0
        self.device = "cuda"  # "cuda:0"
        self.PERTURB_NODES = 10
        self.PERTURB_NODES_RAND_INIT = 10
        self.RAND_INIT_STEPS = 0
        self.SHOULD_EVAL_RAND = True
        self.SHOULD_SAVE = True
        self.THREADS = 1

    @property
    def CAP(self):
        return CAPACITIES[self.N_JOBS]


def default_args():
    args = def_arg()
    return args


def args():
    with open(f"{DIR_PATH}/args.pkl", "rb") as handle:
        args = pickle.load(handle)
    return args

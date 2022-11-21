import sys
from time import gmtime, strftime
import subprocess
import os
from uuid import uuid4


def run_adm(graph_size, mod_dir, instances, batch_size):
    model_path = f"{mod_dir}/adm.pt"
    eval_samples_path = f"{mod_dir}/sampled_vrps_and_sols/for_training"
    if not os.path.isdir(eval_samples_path):
        os.makedirs(eval_samples_path)

    python = sys.executable
    res = subprocess.run(
        [
            python,
            "./adm/test_models.py",
            str(graph_size),
            model_path,
            eval_samples_path,
            str(instances),
            str(batch_size),
        ]
    )
    return res.returncode


def train_adm(graph_size, dirpath, mem_efficient):
    all_settings = {
        20: {
            "samples": 512 * 2500,
            "batch": 512,
            "end_epoch": 150,
        },
        50: {
            "samples": 512 * 2500,
            "batch": 512,
            "end_epoch": 70,
        },
        100: {
            "samples": 256 * 2500,
            "batch": 256,
            "end_epoch": 110,
        },
    }

    curr_settings = all_settings[graph_size]

    python = sys.executable
    res = subprocess.run(
        [
            python,
            "./adm/train_models.py",
            str(graph_size),
            str(curr_settings["samples"]),
            str(curr_settings["batch"]),
            str(curr_settings["end_epoch"]),
            mem_efficient,
            dirpath,
        ]
    )
    return res.returncode


def train_lsh(
    graph_size, mod_dir, epochs, n_steps, rand_init_steps, perturb_nodes, batch_size
):
    eval_samples_path = f"{mod_dir}/sampled_vrps_and_sols/for_training"

    python = sys.executable
    res = subprocess.run(
        [
            python,
            "-u",
            "./lsh/train_models.py",
            str(graph_size),
            str(epochs),
            str(n_steps),
            str(rand_init_steps),
            str(perturb_nodes),
            str(batch_size),
            eval_samples_path,
            mod_dir,
        ]
    )
    return res.returncode


def main():
    VRP_SIZE = int(sys.argv[1])
    ADM_MEM_EFFICIENT = sys.argv[2]
    TRAIN_STEPS = sys.argv[3]

    dt = strftime("%Y-%m-%d", gmtime())
    mod_dir = f"{mod_dir}/{dt}_{str(uuid4())[:4]}"
    assert not os.path.exists(mod_dir)
    os.makedirs(mod_dir)
    MOD_DIR = mod_dir  # sys.argv[4]

    # Constants found to be optimal in experiments. Feel free to change or to make them argvs.
    N_STEPS = 1  # sys.argv[5]
    RAND_INIT_STEPS = 0  # sys.argv[6]
    PERTURB_NODES = 10  # sys.argv[7]
    BATCH_SIZE = 64 if VRP_SIZE >= 100 else 128  # sys.argv[8]

    os.makedirs(MOD_DIR)

    print("============ TRAINING ADM ===============")
    ret_code_training_adm = train_adm(VRP_SIZE, MOD_DIR, ADM_MEM_EFFICIENT)
    print("============ FINISH TRAINING ADM ===============")
    if ret_code_training_adm:
        exit(ret_code_training_adm)

    print("============ RUNNING ADM ===============")
    print("We must run the adm to generate the training paths for LSH")
    ret_code_running_adm = run_adm(VRP_SIZE, MOD_DIR, 128 * 102, 128 * 6)
    print("============ FINISH RUNNING ADM ===============")
    if ret_code_running_adm:
        exit(ret_code_running_adm)

    print("============ TRAINING LSH ===============")
    ret_code_lsh = train_lsh(
        VRP_SIZE,
        MOD_DIR,
        TRAIN_STEPS,
        N_STEPS,
        RAND_INIT_STEPS,
        PERTURB_NODES,
        BATCH_SIZE,
    )
    print("============ FINISH TRAINING LSH ===============")
    if ret_code_lsh:
        exit(ret_code_lsh)


if __name__ == "__main__":
    main()

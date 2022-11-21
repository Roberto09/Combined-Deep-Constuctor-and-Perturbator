import sys
import subprocess
import os


def run_adm(graph_size, mod_dir, instances, batch_size, samples):
    model_path = f"{mod_dir}/adm.pt"
    eval_samples_path = f"{mod_dir}/sampled_vrps_and_sols/for_evaluation"
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
            str(samples),
        ]
    )
    return res.returncode


def run_lsh(graph_size, mod_dir, eval_samples_path):
    model_path = f"{mod_dir}/lsh.pt"

    python = sys.executable
    res = subprocess.run(
        [
            python,
            "./lsh/test_models.py",
            str(graph_size),
            model_path,
            eval_samples_path,
        ]
    )
    return res.returncode


def main():
    VRP_SIZE = int(sys.argv[1])
    INSTANCES = int(sys.argv[2])
    BATCH_SIZE = int(sys.argv[3])
    SAMPLES = int(sys.argv[4])  # 1 defaults to greedy model

    if len(sys.argv) >= 6:
        MOD_DIR = sys.argv[5]
        print(f"Using custom models found in {MOD_DIR}")
    else:
        MOD_DIR = f"./models/model_{VRP_SIZE}_paper"
        print(f"Using paper models found in {MOD_DIR}")

    assert os.path.isdir(MOD_DIR)
    print("============ RUNNING ADM ===============")
    ret_code_adm = run_adm(VRP_SIZE, MOD_DIR, INSTANCES, BATCH_SIZE, SAMPLES)
    print("============ FINISH RUNNING ADM ===============")
    if ret_code_adm:
        exit(ret_code_adm)

    print("============ RUNNING LSH ===============")
    eval_samples_path = f"{MOD_DIR}/sampled_vrps_and_sols/for_evaluation"
    ret_code_lsh = run_lsh(VRP_SIZE, MOD_DIR, eval_samples_path)
    print("============ FINISH RUNNING LSH ===============")
    if ret_code_lsh:
        exit(ret_code_lsh)


if __name__ == "__main__":
    main()

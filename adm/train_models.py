import torch
from attention_dynamic_model import AttentionDynamicModel, set_decode_type
from reinforce_baseline import RolloutBaseline
from train import train_model

from utils import generate_data_onfly, get_cur_time
import sys


def main():
    # Params of model
    GRAPH_SIZE = int(sys.argv[1])
    SAMPLES = int(sys.argv[2])
    BATCH = int(sys.argv[3])
    END_EPOCH = int(sys.argv[4])
    MEM_EFFICIENT = sys.argv[5] == "True"
    MOD_DIR = sys.argv[6] if len(sys.argv) >= 7 else None
    START_EPOCH = 0
    embedding_dim = 128
    LEARNING_RATE = 0.0003
    ROLLOUT_SAMPLES = 10000
    NUMBER_OF_WP_EPOCHS = 1
    GRAD_NORM_CLIPPING = 1.0
    BATCH_VERBOSE = 2500
    VAL_BATCH_SIZE = 256
    VALIDATE_SET_SIZE = 256 * 4

    # Initialize model
    model_pt = AttentionDynamicModel(embedding_dim).cuda()
    set_decode_type(model_pt, "sampling")
    print(get_cur_time(), "model initialized")

    # Create and save validation dataset
    validation_dataset = generate_data_onfly(VALIDATE_SET_SIZE, GRAPH_SIZE)

    # Initialize optimizer
    optimizer = torch.optim.Adam(params=model_pt.parameters(), lr=LEARNING_RATE)

    # Initialize baseline
    baseline = RolloutBaseline(
        model_pt,
        wp_n_epochs=NUMBER_OF_WP_EPOCHS,
        epoch=0,
        num_samples=ROLLOUT_SAMPLES,
        embedding_dim=embedding_dim,
        graph_size=GRAPH_SIZE,
    )
    print(get_cur_time(), "baseline initialized")

    train_model(
        optimizer,
        model_pt,
        baseline,
        validation_dataset,
        samples=SAMPLES,
        batch=BATCH,
        val_batch_size=VAL_BATCH_SIZE,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
        grad_norm_clipping=GRAD_NORM_CLIPPING,
        batch_verbose=BATCH_VERBOSE,
        graph_size=GRAPH_SIZE,
        mod_dir=MOD_DIR,
        mem_efficient=MEM_EFFICIENT,
    )


if __name__ == "__main__":
    main()

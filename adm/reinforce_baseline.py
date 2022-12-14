import torch
from scipy.stats import ttest_rel
from tqdm import tqdm
import numpy as np

from attention_dynamic_model import AttentionDynamicModel
from attention_dynamic_model import set_decode_type
from utils import generate_data_onfly, FastTensorDataLoader, get_dev_of_mod, CAPACITIES


def copy_of_pt_model(model, embedding_dim=128, graph_size=20):
    """Copy model weights to new model"""

    data_random = [
        torch.rand(
            (
                2,
                2,
            ),
            dtype=torch.float32,
        ),
        torch.rand((2, graph_size, 2), dtype=torch.float32),
        torch.randint(low=1, high=10, size=(2, graph_size), dtype=torch.float32)
        / CAPACITIES[graph_size],
    ]

    new_model = AttentionDynamicModel(embedding_dim).to(get_dev_of_mod(model))
    set_decode_type(new_model, "sampling")
    new_model.eval()
    with torch.no_grad():
        _, _ = new_model(data_random)
    model_dict = model.state_dict()
    new_model.load_state_dict(model_dict)

    new_model.eval()

    return new_model


def get_costs_rollout(model, train_batches, disable_tqdm):
    costs_list = []
    for batch in tqdm(
        train_batches, disable=disable_tqdm, desc="Rollout greedy execution"
    ):
        cost, _ = model(batch)
        costs_list.append(cost)
        # torch.cuda.empty_cache()
    return costs_list


def rollout(model, dataset, batch_size=2048, disable_tqdm=False):
    # Evaluate model in greedy mode
    set_decode_type(model, "greedy")

    train_batches = FastTensorDataLoader(
        dataset[0], dataset[1], dataset[2], batch_size=batch_size, shuffle=False
    )

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        costs_list = get_costs_rollout(model, train_batches, disable_tqdm)

    if model_was_training:
        model.train()  # restore original model training state

    return torch.cat(costs_list, dim=0)


def validate(dataset, model, batch_size=1000):
    """Validates model on given dataset in greedy mode"""
    # rollout will set the model to eval mode and turn it back to it's original mode after it finishes
    val_costs = rollout(model, dataset, batch_size=batch_size)
    set_decode_type(model, "sampling")
    mean_cost = torch.mean(val_costs)
    print(f"Validation score: {np.round(mean_cost, 4)}")
    return mean_cost


class RolloutBaseline:
    def __init__(
        self,
        model,
        path_to_checkpoint=None,
        wp_n_epochs=1,
        epoch=0,
        num_samples=10000,
        warmup_exp_beta=0.8,
        embedding_dim=128,
        graph_size=20,
    ):
        """
        Args:
            model: current model
            path_to_checkpoint: path to baseline model weights
            wp_n_epochs: number of warm-up epochs
            epoch: current epoch number
            num_samples: number of samples to be generated for baseline dataset
            warmup_exp_beta: warmup mixing parameter (exp. moving average parameter)

        """

        self.num_samples = num_samples
        self.cur_epoch = epoch
        self.wp_n_epochs = wp_n_epochs
        self.beta = warmup_exp_beta

        # controls the amount of warmup
        self.alpha = 0.0

        self.running_average_cost = None

        # Checkpoint params
        self.path_to_checkpoint = path_to_checkpoint

        # Problem params
        self.embedding_dim = embedding_dim
        self.graph_size = graph_size

        # create and evaluate initial baseline
        self._update_baseline(model, epoch)

    def _update_baseline(self, model, epoch):

        self.model = copy_of_pt_model(
            model, embedding_dim=self.embedding_dim, graph_size=self.graph_size
        )
        self.model.eval()

        # We generate a new dataset for baseline model on each baseline update to prevent possible overfitting
        self.dataset = generate_data_onfly(
            num_samples=self.num_samples, graph_size=self.graph_size
        )

        print(f"Evaluating baseline model on baseline dataset (epoch = {epoch})")
        self.bl_vals = rollout(self.model, self.dataset)
        self.mean = torch.mean(self.bl_vals)
        self.cur_epoch = epoch

    def ema_eval(self, cost):
        """This is running average of cost through previous batches (only for warm-up epochs)"""

        if self.running_average_cost is None:
            self.running_average_cost = torch.mean(cost)
        else:
            self.running_average_cost = self.beta * self.running_average_cost + (
                1.0 - self.beta
            ) * torch.mean(cost)

        return self.running_average_cost

    def eval(self, batch, cost):
        """Evaluates current baseline model on single training batch"""

        if self.alpha == 0:
            return self.ema_eval(cost)

        if self.alpha < 1:
            v_ema = self.ema_eval(cost)
        else:
            v_ema = torch.tensor(0.0)

        with torch.no_grad():
            v_b, _ = self.model(batch)

        # Combination of baseline cost and exp. moving average cost
        return self.alpha * v_b.detach() + (1 - self.alpha) * v_ema.detach()

    def eval_all(self, dataset):
        """Evaluates current baseline model on the whole dataset only for non warm-up epochs"""

        if self.alpha < 1:
            return None

        val_costs = rollout(self.model, dataset)

        return val_costs

    def epoch_callback(self, model, epoch):
        """Compares current baseline model with the training model and updates baseline if it is improved"""

        self.cur_epoch = epoch

        print(
            f"Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})"
        )
        candidate_vals = rollout(
            model, self.dataset
        )  # costs for training model on baseline dataset
        candidate_mean = torch.mean(candidate_vals)

        diff = candidate_mean - self.mean

        print(
            f"Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline epoch {self.cur_epoch} mean {self.mean}, difference {diff}"
        )

        if diff < 0:
            # statistic + p-value
            t, p = ttest_rel(candidate_vals, self.bl_vals)

            p_val = p / 2
            print(f"p-value: {p_val}")

            if p_val < 0.05:
                print("Update baseline")
                self._update_baseline(model, self.cur_epoch)

        # alpha controls the amount of warmup
        if self.alpha < 1.0:
            self.alpha = (self.cur_epoch + 1) / float(self.wp_n_epochs)
            print(f"alpha was updated to {self.alpha}")


def load_pt_model(
    path, embedding_dim=128, graph_size=20, n_encode_layers=2, device="cpu"
):
    """Load model weights from hd5 file"""

    data_random = [
        torch.rand(
            (
                2,
                2,
            ),
            dtype=torch.float32,
        ),
        torch.rand((2, graph_size, 2), dtype=torch.float32),
        torch.randint(low=1, high=10, size=(2, graph_size), dtype=torch.float32)
        / CAPACITIES[graph_size],
    ]

    model_loaded = AttentionDynamicModel(
        embedding_dim, n_encode_layers=n_encode_layers
    ).to(device)

    set_decode_type(model_loaded, "greedy")
    _, _ = model_loaded(data_random)

    model_loaded.load_state_dict(torch.load(path))

    return model_loaded

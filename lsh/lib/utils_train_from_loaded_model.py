import numpy as np
import os
import random
import torch
from torch_geometric.data import Data, DataLoader
from lib.rms import RunningMeanStd
from arguments import args
import timeit
from time import gmtime, strftime
from uuid import uuid4

from vrp_env import create_batch_env

args = args()

DEVICE = str(args.device)
N_JOBS = int(args.N_JOBS)
CAP = int(args.CAP)
BATCH_SIZE_SAMPLING = int(args.BATCH_SAMPLING)
BATCH_SIZE = int(args.BATCH)
MAX_COORD = int(args.MAX_COORD)
MAX_DIST = float(args.MAX_DIST)
LR = float(args.LR)

EPOCHS = int(args.EPOCHS)
TRAIN_STEPS = int(args.TRAIN_STEPS)
N_ROLLOUT = int(args.N_ROLLOUT)
ROLLOUT_STEPS = int(args.ROLLOUT_STEPS)
N_STEPS = int(args.N_STEPS)
PERTURB_NODES = int(args.PERTURB_NODES)
PERTURB_NODES_RAND_INIT = int(args.PERTURB_NODES_RAND_INIT)
RAND_INIT_STEPS = int(args.RAND_INIT_STEPS)

INIT_T = float(args.init_T)
FINAL_T = float(args.final_T)

SHOULD_EVAL_RAND = bool(args.SHOULD_EVAL_RAND)
SHOULD_SAVE = bool(args.SHOULD_SAVE)
THREADS = int(args.THREADS)

reward_norm = RunningMeanStd()


class IterativeMean:
    def __init__(self):
        self.sum = 0
        self.n = 0

    def update_state(self, val):
        self.sum += val
        self.n += 1

    def result(self):
        return self.sum / self.n


def create_replay_buffer(n_jobs):
    class Buffer(object):
        def __init__(self, n_jobs):
            super(Buffer, self).__init__()
            self.buf_nodes = []
            self.buf_edges = []
            self.buf_actions = []
            self.buf_rewards = []
            self.buf_values = []  # ?
            self.buf_log_probs = []  # ?
            self.n_jobs = n_jobs

            edges = []
            for i in range(n_jobs + 1):
                for j in range(n_jobs + 1):
                    edges.append([i, j])

            self.edge_index = torch.LongTensor(
                edges
            ).T  # Tensor of shape:(2,(n_jobs+1)**2) with the edges indices.

        def obs(self, nodes, edges, actions, rewards, log_probs, values):
            self.buf_nodes.append(nodes)
            self.buf_edges.append(edges)
            self.buf_actions.append(actions)
            self.buf_rewards.append(rewards)
            self.buf_values.append(values)
            self.buf_log_probs.append(log_probs)

        def compute_values(self, last_v=0, _lambda=1.0):
            rewards = np.array(self.buf_rewards)
            pred_vs = np.array(self.buf_values)

            target_vs = np.zeros_like(rewards)
            advs = np.zeros_like(rewards)

            v = last_v
            for i in reversed(range(rewards.shape[0])):
                v = rewards[i] + _lambda * v
                target_vs[i] = v
                adv = v - pred_vs[i]
                advs[i] = adv

            return target_vs, advs

        def gen_datas(self, batch_size, last_v=0, _lambda=1.0):
            target_vs, advs = self.compute_values(last_v, _lambda)
            advs = (advs - advs.mean()) / advs.std()
            l, w = target_vs.shape

            datas = []
            for i in range(l):
                for j in range(w):
                    nodes = self.buf_nodes[i][j]
                    edges = self.buf_edges[i][j]
                    action = self.buf_actions[i][j]
                    v = target_vs[i][j]
                    adv = advs[i][j]
                    log_prob = self.buf_log_probs[i][j]
                    data = Data(
                        x=torch.from_numpy(nodes).float(),
                        edge_index=self.edge_index,
                        edge_attr=torch.from_numpy(edges).float(),
                        v=torch.tensor([v]).float(),
                        action=torch.tensor(action).long(),
                        log_prob=torch.tensor([log_prob]).float(),
                        adv=torch.tensor([adv]).float(),
                    )
                    datas.append(data)

            return datas

        def create_data(self, _nodes, _edges):
            datas = []
            l = len(_nodes)  # Batch_size
            for i in range(l):  # For each instance.
                nodes = _nodes[i]
                edges = _edges[i]
                data = Data(
                    x=torch.from_numpy(nodes).float(),
                    edge_index=self.edge_index,
                    edge_attr=torch.from_numpy(edges).float(),
                )
                datas.append(data)
            # Data loader which merges data objects from a torch_geometric.data.dataset to a mini-batch.
            dl = DataLoader(datas, batch_size=l)

            return list(dl)[0]  # ?

    return Buffer(n_jobs)


def train_once(model, opt, dl, epoch, step, batch_size, batch_size_sampling, alpha=1.0):
    model.train()

    losses = []
    loss_vs = []
    loss_ps = []
    _entropy = []

    def should_step(i):
        return ((i + 1) % (batch_size_sampling // batch_size)) == 0

    opt.zero_grad()
    for i, batch in enumerate(dl):
        torch.cuda.empty_cache()
        batch = batch.to(DEVICE)
        actions = batch.action.reshape((batch_size, -1))
        log_p, v, entropy = model.evaluate(batch, actions)
        _entropy.append(entropy.mean().item())

        target_vs = batch.v.squeeze(-1)
        old_log_p = batch.log_prob.squeeze(-1)
        adv = batch.adv.squeeze(-1)

        loss_v = ((v - target_vs) ** 2).mean()

        ratio = torch.exp(log_p - old_log_p)
        obj = ratio * adv
        obj_clipped = ratio.clamp(1.0 - 0.2, 1.0 + 0.2) * adv
        loss_p = -torch.min(obj, obj_clipped).mean()
        loss = loss_p + alpha * loss_v

        losses.append(loss.item())
        loss_vs.append(loss_v.item())
        loss_ps.append(loss_p.item())

        loss.backward()
        if should_step(i):
            opt.step()
            opt.zero_grad()

    print(
        "epoch:",
        epoch,
        "step:",
        step,
        "loss_v:",
        np.mean(loss_vs),
        "loss_p:",
        np.mean(loss_ps),
        "loss:",
        np.mean(losses),
        "entropy:",
        np.mean(_entropy),
    )


def roll_out(
    model,
    envs,
    states,
    n_steps,
    batch_size,
    n_jobs,
    is_last=False,
    greedy=False,
    _lambda=0.99,
):
    mean_costs = []
    mean_best_costs = []

    buffer = create_replay_buffer(n_jobs)
    with torch.no_grad():
        model.eval()
        nodes, edges = states
        _sum = 0
        _entropy = []

        for i in range(n_steps):
            data = buffer.create_data(nodes, edges)
            data = data.to(DEVICE)
            actions, log_p, values, entropy = model(
                data, PERTURB_NODES, greedy
            )  # forward() from egate_model.
            new_nodes, new_edges, rewards = envs.step(actions.cpu().numpy())

            cost_val = np.mean([env.cost for env in envs.envs])
            best_cost_val = np.mean([env.best for env in envs.envs])
            mean_costs.append(cost_val)
            mean_best_costs.append(best_cost_val)

            rewards = np.array(rewards)
            _sum = _sum + rewards
            rewards = reward_norm(
                rewards
            )  # reward_norm() is __call__() from RunningMeanStandard from lib/rms
            _entropy.append(entropy.mean().cpu().numpy())

            buffer.obs(
                nodes,
                edges,
                actions.cpu().numpy(),
                rewards,
                log_p.cpu().numpy(),
                values.cpu().numpy(),
            )  # Save in buffer.
            nodes, edges = new_nodes, new_edges

        mean_value = _sum.mean()

        if not is_last:
            data = buffer.create_data(nodes, edges)
            data = data.to(DEVICE)
            actions, log_p, values, entropy = model(data, PERTURB_NODES, greedy)
            values = values.cpu().numpy()
        else:
            values = 0

        dl = buffer.gen_datas(batch_size, values, _lambda=_lambda)
        return dl, (nodes, edges), mean_costs, mean_best_costs


def eval_random(epochs, batch_size, envs, n_steps, paths):
    def eval_once(n_instance=batch_size, n_steps=n_steps):
        nodes, edges = envs.reset(paths)
        _sum = np.zeros(n_instance)
        for i in range(n_steps):
            actions = [
                random.sample(range(0, N_JOBS), PERTURB_NODES)
                for i in range(n_instance)
            ]
            actions = np.array(actions)
            new_nodes, new_edges, rewards = envs.step(actions)
            _sum += rewards

        return np.mean([env.cost for env in envs.envs])

    print(
        "<<<<<<<<<<===================== random mean cost:",
        np.mean([eval_once() for i in range(epochs)]),
    )


def random_init(envs, n_steps, batch_size, paths):
    tiempo = timeit.default_timer()
    nodes, edges = envs.reset(paths)
    tiempo = timeit.default_timer() - tiempo
    print("tiempo random_init(),envs.reset():", tiempo)

    for i in range(n_steps):
        # For each instance, actions is a vector with 10% random nodes from all the nodes (excluding depot).
        actions = [
            random.sample(range(0, N_JOBS), PERTURB_NODES_RAND_INIT)
            for i in range(batch_size)
        ]
        actions = np.array(actions)

        tiempo = timeit.default_timer()
        nodes, edges, rewards = envs.step(actions)  # rewards is not used.
        tiempo = timeit.default_timer() - tiempo
        print("tiempo random_init(),envs.step():", tiempo)

    return (nodes, edges), np.mean([env.cost for env in envs.envs])


def train(model, paths, batches_coords, mod_dir, create_dir=False):

    if create_dir:
        dt = strftime("%Y-%m-%d", gmtime())
        mod_dir = f"{mod_dir}/{dt}_{str(uuid4())[:4]}"
        if not os.path.exists(mod_dir):
            os.makedirs(mod_dir)

    epochs = EPOCHS
    batch_size = BATCH_SIZE
    batch_size_sampling = BATCH_SIZE_SAMPLING
    train_steps = TRAIN_STEPS
    n_rollout = N_ROLLOUT
    rollout_steps = ROLLOUT_STEPS
    n_jobs = N_JOBS

    tiempo = timeit.default_timer()
    opt = torch.optim.Adam(model.parameters(), LR)
    tiempo = timeit.default_timer() - tiempo
    print("tiempo opt:", tiempo)

    assert len(paths) == len(batches_coords) and len(paths) % batch_size_sampling == 0
    paths = list(
        map(
            lambda x: x.tolist(),
            np.split(paths, len(paths) // batch_size_sampling, axis=0),
        )
    )
    batches_coords = list(
        map(
            lambda x: x.tolist(),
            np.split(
                batches_coords, len(batches_coords) // batch_size_sampling, axis=0
            ),
        )
    )

    pre_steps = RAND_INIT_STEPS
    min_mean_best_cost = 1e9
    for epoch in range(epochs + 1):
        curr_paths = paths[epoch % len(paths)]
        curr_batches_coords = batches_coords[epoch % len(paths)]
        envs = create_batch_env(
            curr_paths, curr_batches_coords, n_jobs, batch_size_sampling
        )

        tiempo = timeit.default_timer()
        states, mean_cost = random_init(
            envs, pre_steps, batch_size_sampling, curr_paths
        )
        envs.reset_temperature()
        tiempo = timeit.default_timer() - tiempo
        print("tiempo random_init():", tiempo)

        print("=================>>>>>>>> before mean cost:", mean_cost)

        all_datas = []
        for i in range(n_rollout):
            tiempo = timeit.default_timer()

            datas, states, _, _ = roll_out(
                model,
                envs,
                states,
                rollout_steps,
                batch_size_sampling,
                n_jobs,
                is_last=False,
            )
            tiempo = timeit.default_timer() - tiempo
            print("tiempo roll_out():", tiempo)

            all_datas.extend(datas)

        assert (len(all_datas) % batch_size) == 0
        dl = DataLoader(all_datas, batch_size=batch_size, shuffle=True)

        for j in range(train_steps):
            tiempo = timeit.default_timer()
            train_once(model, opt, dl, epoch, 0, batch_size, batch_size_sampling)
            tiempo = timeit.default_timer() - tiempo
            print("tiempo train_once():", tiempo)

        mean_best_cost = np.mean([env.best for env in envs.envs])
        print(
            "=================>>>>>>>> mean cost:",
            np.mean([env.cost for env in envs.envs]),
        )
        print("=================>>>>>>>> mean best cost:", mean_best_cost)
        if SHOULD_EVAL_RAND and epoch % 100 == 0:
            tiempo = timeit.default_timer()
            eval_random(3, batch_size_sampling, envs, N_STEPS + pre_steps, curr_paths)
            tiempo = timeit.default_timer() - tiempo
            print("tiempo eval_random():", tiempo)

        if SHOULD_SAVE and epoch % 25 == 0:
            # torch.save(model.state_dict(), f"{mod_dir}/lsh_{epoch}.pt")
            if mean_best_cost < min_mean_best_cost:
                print(
                    f"Saving model on train step {epoch}. Found better"
                    f"mean best cost: {mean_best_cost} < {min_mean_best_cost}"
                )
                min_mean_best_cost = mean_best_cost
                torch.save(model.state_dict(), f"{mod_dir}/lsh.pt")

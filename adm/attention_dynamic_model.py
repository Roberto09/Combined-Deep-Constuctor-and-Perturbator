import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import math
import numpy as np

from attention_graph_encoder import GraphAttentionEncoder
from layers import scaled_attention
from environment import AgentVRP
from utils import get_dev_of_mod


def set_decode_type(model, decode_type):
    model.set_decode_type(decode_type)


class AttentionDynamicModel(nn.Module):
    def __init__(self, embedding_dim, n_encode_layers=2, n_heads=8, tanh_clipping=10.0):

        super().__init__()

        # attributes for MHA
        self.embedding_dim = embedding_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None

        # attributes for VRP problem
        self.problem = AgentVRP
        self.n_heads = n_heads

        # Encoder part
        self.embedder = GraphAttentionEncoder(
            input_dim=self.embedding_dim,
            num_heads=self.n_heads,
            num_layers=self.n_encode_layers,
        )

        # Decoder part

        self.output_dim = self.embedding_dim
        self.num_heads = n_heads

        self.head_depth = self.output_dim // self.num_heads
        self.dk_mha_decoder = float(self.head_depth)  # for decoding in mha_decoder
        self.dk_get_loc_p = float(self.output_dim)  # for decoding in mha_decoder

        if self.output_dim % self.num_heads != 0:
            raise ValueError("number of heads must divide d_model=output_dim")

        self.tanh_clipping = tanh_clipping

        # we split projection matrix Wq into 2 matrices: Wq*[h_c, h_N, D] = Wq_context*h_c + Wq_step_context[h_N, D]
        self.wq_context = nn.Linear(
            self.embedding_dim, self.output_dim
        )  # (d_q_context, output_dim)
        self.wq_step_context = nn.Linear(
            self.embedding_dim + 1, self.output_dim, bias=False
        )  # (d_q_step_context, output_dim)

        # we need two Wk projections since there is MHA followed by 1-head attention - they have different keys K
        self.wk = nn.Linear(
            self.embedding_dim, self.output_dim, bias=False
        )  # (d_k, output_dim)
        self.wk_tanh = nn.Linear(
            self.embedding_dim, self.output_dim, bias=False
        )  # (d_k_tanh, output_dim)

        # we dont need Wv projection for 1-head attention: only need attention weights as outputs
        self.wv = nn.Linear(
            self.embedding_dim, self.output_dim, bias=False
        )  # (d_v, output_dim)

        # we dont need wq for 1-head tanh attention, since we can absorb it into w_out
        self.w_out = nn.Linear(
            self.embedding_dim, self.output_dim, bias=False
        )  # (d_model, d_model)

        self.dev = None

    def set_decode_type(self, decode_type):
        self.decode_type = decode_type

    def split_heads(self, tensor, batch_size):
        """Function for computing attention on several heads simultaneously
        Splits tensor to be multi headed.
        """
        # (batch_size, seq_len, output_dim) -> (batch_size, seq_len, n_heads, head_depth)
        splitted_tensor = tensor.view(batch_size, -1, self.n_heads, self.head_depth)
        return splitted_tensor.transpose(
            1, 2
        )  # (batch_size, n_heads, seq_len, head_depth)

    def _select_node(self, logits):
        """Select next node based on decoding type."""

        # assert tf.reduce_all(logits) == logits, "Probs should not contain any nans"

        if self.decode_type == "greedy":
            selected = torch.argmax(logits, dim=-1)  # (batch_size, 1)

        elif self.decode_type == "sampling":
            # logits has a shape of (batch_size, 1, n_nodes), we have to squeeze it
            # to (batch_size, n_nodes) since tf.random.categorical requires matrix
            cat_dist = Categorical(
                logits=logits[:, 0, :]
            )  # creates categorical distribution from tensor (batch_size)
            selected = cat_dist.sample()  # takes a single sample from distribution
        else:
            assert False, "Unknown decode type"

        return torch.squeeze(selected, -1)  # (batch_size,)

    def get_step_context(self, state, embeddings):
        """Takes a state and graph embeddings,
        Returns a part [h_N, D] of context vector [h_c, h_N, D],
        that is related to RL Agent last step.
        """
        # index of previous node
        prev_node = state.prev_a.to(self.dev)  # (batch_size, 1)

        # from embeddings=(batch_size, n_nodes, input_dim) select embeddings of previous nodes
        cur_embedded_node = embeddings.gather(
            1,
            prev_node.view(prev_node.shape[0], -1, 1).repeat_interleave(
                embeddings.shape[-1], -1
            ),
        )  # (batch_size, 1, input_dim)

        # add remaining capacity
        step_context = torch.cat(
            [
                cur_embedded_node,
                (self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, None]).to(
                    self.dev
                ),
            ],
            dim=-1,
        )

        return step_context  # (batch_size, 1, input_dim + 1)

    def decoder_mha(self, Q, K, V, mask=None):
        """Computes Multi-Head Attention part of decoder
        Args:
            mask: a mask for visited nodes,
                has shape (batch_size, seq_len_q, seq_len_k), seq_len_q = 1 for context vector attention in decoder
            Q: query (context vector for decoder)
                    has shape (batch_size, n_heads, seq_len_q, head_depth) with seq_len_q = 1 for context_vector attention in decoder
            K, V: key, value (projections of nodes embeddings)
                have shape (batch_size, n_heads, seq_len_k, head_depth), (batch_size, n_heads, seq_len_v, head_depth),
                                                                with seq_len_k = seq_len_v = n_nodes for decoder
        """

        # Add dimension to mask so that it can be broadcasted across heads
        # (batch_size, seq_len_q, seq_len_k) --> (batch_size, 1, seq_len_q, seq_len_k)
        if mask is not None:
            mask = mask.unsqueeze(1)

        attention = scaled_attention(
            Q, K, V, mask
        )  # (batch_size, n_heads, seq_len_q, head_depth)
        # transpose attention to (batch_size, seq_len_q, n_heads, head_depth)
        attention = attention.transpose(1, 2).contiguous()
        # concatenate results of all heads (batch_size, seq_len_q, self.output_dim)
        attention = attention.view(self.batch_size, -1, self.output_dim)

        output = self.w_out(attention)

        return output

    def get_log_p(self, Q, K, mask=None):
        """Single-Head attention sublayer in decoder,
        computes log-probabilities for node selection.

        Args:
            mask: mask for nodes
            Q: query (output of mha layer)
                    has shape (batch_size, seq_len_q, output_dim), seq_len_q = 1 for context attention in decoder
            K: key (projection of node embeddings)
                    has shape  (batch_size, seq_len_k, output_dim), seq_len_k = n_nodes for decoder
        """

        compatibility = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
        compatibility = torch.tanh(compatibility) * self.tanh_clipping
        if mask is not None:
            compatibility = compatibility.masked_fill(mask == 1, -1e9)

        log_p = F.log_softmax(
            compatibility, dim=-1
        )  # (batch_size, seq_len_q, seq_len_k)

        return log_p

    def get_likelihood_selection(self, _log_p, a):

        # Get log_p corresponding to selected actions for every batch
        indices = a.view(a.shape[0], -1)
        select = _log_p.gather(-1, indices)
        return select.view(-1)

    def get_projections(self, embeddings, context_vectors):

        # we compute some projections (common for each policy step) before decoding loop for efficiency
        K = self.wk(embeddings)  # (batch_size, n_nodes, output_dim)
        K_tanh = self.wk_tanh(embeddings)  # (batch_size, n_nodes, output_dim)
        V = self.wv(embeddings)  # (batch_size, n_nodes, output_dim)
        Q_context = self.wq_context(
            context_vectors[:, None, :]
        )  # (batch_size, 1, output_dim)

        # we dont need to split K_tanh since there is only 1 head; Q will be split in decoding loop
        K = self.split_heads(
            K, self.batch_size
        )  # (batch_size, num_heads, n_nodes, head_depth)
        V = self.split_heads(
            V, self.batch_size
        )  # (batch_size, num_heads, n_nodes, head_depth)

        return K_tanh, Q_context, K, V

    def fwd_rein_loss(self, inputs, baseline, bl_vals, num_batch, return_pi=False):
        """
        Forward and calculate loss for REINFORCE algorithm in a memory efficient way.
        This sacrifices a bit of performance but is way better in memory terms and works
        by reordering the terms in the gradient formula such that we don't store gradients
        for all the seguence for a long time which hence produces a lot of memory consumption.
        """

        on_training = self.training
        self.eval()
        with torch.no_grad():
            cost, log_likelihood, seq = self(inputs, True)
            bl_val = (
                bl_vals[num_batch]
                if bl_vals is not None
                else baseline.eval(inputs, cost)
            )
            pre_cost = cost - bl_val.detach()
            detached_loss = torch.mean((pre_cost) * log_likelihood)

        if on_training:
            self.train()
        return detached_loss, self(inputs, return_pi, seq, pre_cost)

    def forward(self, inputs, return_pi=False, pre_selects=None, pre_cost=None):
        """
        Forward method. Works as expected except and as described on the paper, however
        if pre_selects is None which hence implies that pre_cost should be none it's because
        fwd_rein_loss is calling it; check that method for a description of why this is useful.
        """

        self.batch_size = inputs[0].shape[0]

        state = self.problem(inputs)  # use CPU inputs for state
        inputs = self.set_input_device(
            inputs
        )  # sent inputs to GPU for training if it's being used
        sequences = []
        ll = torch.zeros(self.batch_size)

        if pre_selects is not None:
            pre_selects = pre_selects.transpose(0, 1)
        # Perform decoding steps
        pre_select_idx = 0
        while not state.all_finished():

            state.i = torch.zeros(1, dtype=torch.int64)
            att_mask, cur_num_nodes = state.get_att_mask()
            att_mask, cur_num_nodes = att_mask.to(self.dev), cur_num_nodes.to(self.dev)
            embeddings, context_vectors = self.embedder(inputs, att_mask, cur_num_nodes)
            K_tanh, Q_context, K, V = self.get_projections(embeddings, context_vectors)

            while not state.partial_finished():

                step_context = self.get_step_context(
                    state, embeddings
                )  # (batch_size, 1, input_dim + 1)
                Q_step_context = self.wq_step_context(
                    step_context
                )  # (batch_size, 1, output_dim)
                Q = Q_context + Q_step_context

                # split heads for Q
                Q = self.split_heads(
                    Q, self.batch_size
                )  # (batch_size, num_heads, 1, head_depth)

                # get current mask
                mask = state.get_mask().to(
                    self.dev
                )  # (batch_size, 1, n_nodes) True -> mask, i.e. agent can NOT go

                # compute MHA decoder vectors for current mask
                mha = self.decoder_mha(Q, K, V, mask)  # (batch_size, 1, output_dim)

                # compute probabilities
                log_p = self.get_log_p(mha, K_tanh, mask)  # (batch_size, 1, n_nodes)

                # next step is to select node
                if pre_selects is None:
                    selected = self._select_node(log_p.detach())  # (batch_size,)
                else:
                    selected = pre_selects[pre_select_idx]

                state.step(selected.detach().cpu())

                curr_ll = self.get_likelihood_selection(
                    log_p[:, 0, :].cpu(), selected.detach().cpu()
                )
                if pre_selects is not None:
                    curr_loss = (curr_ll * pre_cost).sum() / self.batch_size
                    curr_loss.backward(retain_graph=True)
                    curr_ll = curr_ll.detach()
                ll += curr_ll

                sequences.append(selected.detach().cpu())
                pre_select_idx += 1
                # torch.cuda.empty_cache()
            # torch.cuda.empty_cache()

        pi = torch.stack(sequences, dim=1)  # (batch_size, len(outputs))
        cost = self.problem.get_costs(
            (
                inputs[0].detach().cpu(),
                inputs[1].detach().cpu(),
                inputs[2].detach().cpu(),
            ),
            pi,
        )

        ret = [cost, ll]
        if return_pi:
            ret.append(pi)
        return ret

    def set_input_device(self, inp_tens):
        if self.dev is None:
            self.dev = get_dev_of_mod(self)
        return (
            inp_tens[0].to(self.dev),
            inp_tens[1].to(self.dev),
            inp_tens[2].to(self.dev),
        )

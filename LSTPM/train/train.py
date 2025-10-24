import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch
import argparse
import json
from collections import defaultdict
import gc
import os
from math import radians, cos, sin, asin, sqrt
from collections import deque,Counter
import time

torch.autograd.set_detect_anomaly(True)
# auto-select device: MPS (for M1/M2) > CUDA > CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)
    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')
    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)
    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)
    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, **kwargs):
    batch_size = kwargs.get('batch_size', 128)
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

def pad_batch_of_lists_masks(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded_mask = [[1.0]*(len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l)) + [0.0] * (max_len - len(l)) for l in batch_of_lists]
    return padded, padded_mask, padde_mask_non_local

def pad_batch_of_lists_masks_test(batch_of_lists, max_len):
    padded = [l + [0] * (max_len - len(l)) for l in batch_of_lists]
    padded2 = [l[:-1] + [0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padded_mask = [[0.0]*(len(l) - 2) + [1.0] + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    padde_mask_non_local = [[1.0] * (len(l) - 1) + [0.0] * (max_len - len(l) + 1) for l in batch_of_lists]
    return padded, padded2, padded_mask, padde_mask_non_local

def pad_batch_of_lists(batch_of_lists, max_len, pad_value=0):
    """Pad a list of python lists to max_len with the provided pad_value."""
    if max_len <= 0:
        return [[] for _ in batch_of_lists]
    return [lst + [pad_value] * (max_len - len(lst)) for lst in batch_of_lists]

def to_tid48(t):
    """Convert various time encodings to 0-47 integer time-slot id.
    Accepts: int/float already in 0..47; str 'YYYY-mm-dd HH:MM:SS';
    list/tuple [timestamp, tid] as used in some datasets.
    """
    # Pair format [timestamp, tid]
    if isinstance(t, (list, tuple)):
        if len(t) >= 2 and isinstance(t[1], (int, np.integer)):
            return int(t[1])
        t = t[0]
    # ISO time string
    if isinstance(t, str):
        try:
            tm = time.strptime(t, "%Y-%m-%d %H:%M:%S")
            # weekdays 0-4 -> 0..23, weekends 5-6 -> 24..47
            return tm.tm_hour if tm.tm_wday < 5 else tm.tm_hour + 24
        except Exception:
            # numeric string fallback
            return int(float(t))
    # numeric
    if isinstance(t, (np.integer, int)):
        return int(t)
    if isinstance(t, (np.floating, float)):
        return int(t)
    # last resort
    try:
        return int(t)
    except Exception:
        return 0

class Model(nn.Module):
    def __init__(self, n_users, n_items, emb_size=500, hidden_units=500, dropout=0.8, user_dropout=0.5, data_neural = None, tim_sim_matrix = None):
        super(self.__class__, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.hidden_units = hidden_units
        if emb_size == None:
            emb_size = hidden_units
        self.emb_size = emb_size
        ## todo why embeding?
        self.item_emb = nn.Embedding(n_items, emb_size)
        self.emb_tim = nn.Embedding(48, 10)
        self.lstmcell = nn.LSTM(input_size=emb_size, hidden_size=hidden_units)
        self.lstmcell_history = nn.LSTM(input_size=emb_size, hidden_size=hidden_units)
        self.linear = nn.Linear(hidden_units * 2, n_items)
        self.dropout = nn.Dropout(0.0)
        self.user_dropout = nn.Dropout(user_dropout)
        self.data_neural = data_neural
        self.tim_sim_matrix = tim_sim_matrix
        self.dilated_rnn = nn.LSTMCell(input_size=emb_size, hidden_size=hidden_units)  # could be the same as self.lstmcell
        self.linear1 = nn.Linear(hidden_units, hidden_units)

        # caches for vectorized forward
        self._distance_cache = None
        self._distance_cache_id = None
        self._distance_cache_device = None
        self._poi_vocab_size = n_items
        self._register_empty_session_cache()

        if tim_sim_matrix is not None and len(tim_sim_matrix) > 0:
            tim_tensor = torch.as_tensor(tim_sim_matrix, dtype=torch.float32)
        else:
            tim_tensor = torch.zeros(1, 1, dtype=torch.float32)
        self.register_buffer("_tim_sim_matrix", tim_tensor, persistent=False)

        if data_neural is not None and len(data_neural) > 0:
            self._build_session_cache(data_neural)
        else:
            self._cache_ready = False

        self.init_weights()

    def init_weights(self):
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def _register_empty_session_cache(self):
        self.max_sessions = 0
        self.max_session_len = 0
        self.max_session_key = 0
        self.register_buffer('_user_lookup', torch.full((1,), -1, dtype=torch.long), persistent=False)
        self.register_buffer('_session_lookup', torch.full((1, 1), -1, dtype=torch.long), persistent=False)
        self.register_buffer('_session_items', torch.zeros(1, 1, 1, dtype=torch.long), persistent=False)
        self.register_buffer('_session_times', torch.zeros(1, 1, 1, dtype=torch.long), persistent=False)
        self.register_buffer('_session_mask', torch.zeros(1, 1, 1, dtype=torch.bool), persistent=False)
        self.register_buffer('_session_lengths', torch.zeros(1, 1, dtype=torch.long), persistent=False)
        self._cache_ready = False

    def _build_session_cache(self, data_neural):
        user_ids = sorted(data_neural.keys())
        if not user_ids:
            self._cache_ready = False
            return

        self.max_sessions = max(len(data_neural[uid]['sessions']) for uid in user_ids)
        self.max_sessions = max(self.max_sessions, 1)
        session_key_candidates = [max(data_neural[uid]['sessions'].keys(), default=-1) for uid in user_ids]
        self.max_session_key = max(session_key_candidates + [0])
        session_lengths_all = [len(sess) for uid in user_ids for sess in data_neural[uid]['sessions'].values()]
        self.max_session_len = max(session_lengths_all + [1])

        user_lookup_size = max(user_ids) + 1
        user_lookup = torch.full((user_lookup_size,), -1, dtype=torch.long)
        for idx, uid in enumerate(user_ids):
            user_lookup[uid] = idx

        session_lookup = torch.full((len(user_ids), self.max_session_key + 1), -1, dtype=torch.long)
        session_items = torch.zeros(len(user_ids), self.max_sessions, self.max_session_len, dtype=torch.long)
        session_times = torch.zeros_like(session_items)
        session_mask = torch.zeros(len(user_ids), self.max_sessions, self.max_session_len, dtype=torch.bool)
        session_lengths = torch.zeros(len(user_ids), self.max_sessions, dtype=torch.long)

        for u_pos, uid in enumerate(user_ids):
            sessions = data_neural[uid]['sessions']
            sorted_keys = sorted(sessions.keys())
            for s_pos, s_key in enumerate(sorted_keys):
                if s_pos >= self.max_sessions:
                    break
                if s_key <= self.max_session_key:
                    session_lookup[u_pos, s_key] = s_pos
                seq = sessions[s_key]
                if not seq:
                    continue
                items = torch.tensor([step[0] for step in seq], dtype=torch.long)
                times = torch.tensor([to_tid48(step[1]) for step in seq], dtype=torch.long)
                length = items.numel()
                length = min(length, self.max_session_len)
                session_lengths[u_pos, s_pos] = length
                session_items[u_pos, s_pos, :length] = items[:length]
                session_times[u_pos, s_pos, :length] = times[:length]
                session_mask[u_pos, s_pos, :length] = True

        self._buffers['_user_lookup'] = user_lookup
        self._buffers['_session_lookup'] = session_lookup
        self._buffers['_session_items'] = session_items
        self._buffers['_session_times'] = session_times
        self._buffers['_session_mask'] = session_mask
        self._buffers['_session_lengths'] = session_lengths
        self._cache_ready = True

    def _distance_tensor(self, poi_distance_matrix, device):
        if isinstance(poi_distance_matrix, torch.Tensor):
            if poi_distance_matrix.device != device:
                return poi_distance_matrix.to(device)
            return poi_distance_matrix

        cache_invalid = (
            self._distance_cache is None
            or self._distance_cache_id != id(poi_distance_matrix)
            or self._distance_cache_device != device
        )
        if cache_invalid:
            tensor = torch.as_tensor(poi_distance_matrix, dtype=torch.float32, device=device)
            self._distance_cache = tensor
            self._distance_cache_id = id(poi_distance_matrix)
            self._distance_cache_device = device
        return self._distance_cache

    @staticmethod
    def _masked_softmax(logits, mask, dim):
        mask_float = mask.float()
        # Avoid -inf propagation when mask is all zeros
        max_logits = torch.where(
            mask,
            logits,
            torch.full_like(logits, float('-inf')),
        ).max(dim=dim, keepdim=True).values
        max_logits = torch.where(torch.isfinite(max_logits), max_logits, torch.zeros_like(max_logits))
        shifted = logits - max_logits
        exp_logits = torch.exp(shifted) * mask_float
        denom = exp_logits.sum(dim=dim, keepdim=True).clamp_min(1e-9)
        return exp_logits / denom

    def _compute_dilated_states(self, embeddings, dilated_indices, mask):
        # embeddings: (B, T, E); dilated_indices: (B, T) with -1 meaning “no parent”
        B, T, _ = embeddings.shape
        H = self.hidden_units
        device = embeddings.device
        mask_f = mask.float()

        # bank[0] is zeros; bank[t+1] will be h_t
        bank_h = [embeddings.new_zeros(B, H)]
        bank_c = [embeddings.new_zeros(B, H)]
        out_h = []

        for t in range(T):
            # parent index per batch (shift by +1 so -1 -> 0)
            parents = dilated_indices[:, t]
            # safety: parents must refer to strictly prior steps; clamp into [-1, t-1]
            if t == 0:
                parents = torch.full_like(parents, -1)
            else:
                parents = parents.clamp(min=-1, max=t-1)
            # optional runtime check (helps debug any remaining bad indices)
            if torch.any(parents >= t):
                raise RuntimeError(f"Invalid dilated parent index at step t={t}: max={int(parents.max().item())}")
            idx = (parents + 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, H)

            # stack previous states once per step; shapes (B, t+1, H)
            prev_h_stack = torch.stack(bank_h, dim=1)
            prev_c_stack = torch.stack(bank_c, dim=1)

            prev_h = torch.gather(prev_h_stack, 1, idx).squeeze(1)
            prev_c = torch.gather(prev_c_stack, 1, idx).squeeze(1)

            inp_t = embeddings[:, t, :]
            h_t, c_t = self.dilated_rnn(inp_t, (prev_h, prev_c))

            active = mask_f[:, t].unsqueeze(1)
            h_t = h_t * active
            c_t = c_t * active

            bank_h.append(h_t)
            bank_c.append(c_t)
            out_h.append(h_t)

        # (B, T, H)
        hidden_store = torch.stack(out_h, dim=1)
        return hidden_store

    def _run_history_lstm(self, history_embeddings, history_mask, history_session_active):
        batch_size, num_sessions, max_len, _ = history_embeddings.shape
        outputs = []
        h = history_embeddings.new_zeros(1, batch_size, self.hidden_units)
        c = history_embeddings.new_zeros(1, batch_size, self.hidden_units)
        for s in range(num_sessions):
            session_active = history_session_active[:, s].view(1, batch_size, 1)
            seq_mask = history_mask[:, s, :].unsqueeze(2).float()
            seq_emb = history_embeddings[:, s, :, :] * seq_mask
            seq_emb_t = seq_emb.transpose(0, 1)
            h_prev = h
            c_prev = c
            out, (h, c) = self.lstmcell_history(seq_emb_t, (h, c))
            inactive = (~history_session_active[:, s]).view(1, batch_size, 1)
            h = torch.where(inactive, h_prev, h)
            c = torch.where(inactive, c_prev, c)
            out = out.transpose(0, 1) * seq_mask
            outputs.append(out)
        if not outputs:
            return history_embeddings.new_zeros(batch_size, num_sessions, max_len, self.hidden_units)
        return torch.stack(outputs, dim=1)

    def _compute_history_context(self, user_vectors, session_ids, sequence_tim_batch, item_vectors, mask, base_out, is_train, poi_distance_matrix):
        batch_size, seq_len = item_vectors.shape
        device = item_vectors.device
        if seq_len <= 1 or not self._cache_ready:
            return base_out.new_zeros(batch_size, seq_len, self.hidden_units)

        cache_indices = self._user_lookup[user_vectors]
        if (cache_indices < 0).any():
            return base_out.new_zeros(batch_size, seq_len, self.hidden_units)

        session_lookup_rows = self._session_lookup[cache_indices]
        session_ids_clamped = session_ids.clamp(min=0, max=session_lookup_rows.size(1) - 1)
        session_pos = torch.gather(session_lookup_rows, 1, session_ids_clamped.unsqueeze(1)).squeeze(1)
        session_pos = torch.where(session_pos >= 0, session_pos, torch.zeros_like(session_pos))
        history_counts = session_pos

        max_sessions = self.max_sessions
        session_indices = torch.arange(max_sessions, device=device).view(1, max_sessions)
        history_session_mask = session_indices < history_counts.unsqueeze(1)

        session_items = self._session_items[cache_indices]
        session_times = self._session_times[cache_indices]
        session_mask = self._session_mask[cache_indices]
        session_lengths = self._session_lengths[cache_indices]

        history_mask = session_mask & history_session_mask.unsqueeze(2)
        valid_sessions = history_mask.any(dim=2)
        history_session_active = valid_sessions & history_session_mask

        history_embeddings = self.item_emb(session_items)
        history_embeddings = history_embeddings * history_mask.unsqueeze(3).float()
        history_outputs = self._run_history_lstm(history_embeddings, history_mask, history_session_active)

        current_mask = mask[:, :-1]
        if current_mask.size(1) <= 0:
            return base_out.new_zeros(batch_size, seq_len, self.hidden_units)
        current_mask_float = current_mask.unsqueeze(2).float()
        current_session_embed = base_out[:, :-1, :]

        tim_matrix = self._tim_sim_matrix.to(device)
        current_tim = sequence_tim_batch[:, :-1]
        time_features = tim_matrix[current_tim]
        time_features = time_features.unsqueeze(2).expand(-1, -1, max_sessions, -1)
        time_indices = session_times.unsqueeze(1)
        jaccard_raw = torch.gather(time_features, -1, time_indices)
        hist_mask_exp = history_mask.unsqueeze(1)
        attn_weights = self._masked_softmax(jaccard_raw, hist_mask_exp, dim=-1)
        sessions_represent = torch.einsum('btsl,bsld->btsd', attn_weights, history_outputs)
        sessions_represent = sessions_represent * history_session_active.unsqueeze(1).unsqueeze(3).float()
        sessions_represent = sessions_represent * current_mask_float.unsqueeze(2)

        if is_train:
            total_embed = (current_session_embed * current_mask_float).sum(dim=1, keepdim=True)
            denom = current_mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
            current_repr = total_embed / denom
            current_repr = current_repr.repeat(1, current_session_embed.size(1), 1)
        else:
            prefix_sum = (current_session_embed * current_mask_float).cumsum(dim=1)
            prefix_count = current_mask.cumsum(dim=1).unsqueeze(2).clamp_min(1.0)
            current_repr = prefix_sum / prefix_count
        current_repr = current_repr * current_mask_float

        session_mask_exp = history_session_active.unsqueeze(1).expand(-1, current_repr.size(1), -1)
        sim_logits = torch.einsum('btsd,btd->bts', sessions_represent, current_repr)
        sim_weights = self._masked_softmax(sim_logits, session_mask_exp, dim=-1)
        sim_weights = sim_weights * session_mask_exp.float()
        out_y_current = torch.selu(self.linear1(torch.einsum('btsd,bts->btd', sessions_represent, sim_weights)))

        layer_2_current = 0.5 * out_y_current + 0.5 * current_session_embed

        distance_tensor = self._distance_tensor(poi_distance_matrix, device)
        current_items = item_vectors[:, :-1]
        distance_rows = distance_tensor[current_items]
        distance_rows = distance_rows.unsqueeze(2).expand(-1, -1, max_sessions, -1)
        distance_indices = session_items.unsqueeze(1)
        distance_values = torch.gather(distance_rows, -1, distance_indices)
        distance_values = distance_values * hist_mask_exp.float()
        distance_sum = distance_values.sum(dim=-1)
        distance_count = hist_mask_exp.float().sum(dim=-1)
        avg_distance = torch.where(
            distance_count > 0,
            distance_sum / distance_count.clamp_min(1e-9),
            torch.ones_like(distance_sum)
        )
        avg_distance = avg_distance * session_mask_exp.float()

        layer2_logits = torch.einsum('btsd,btd->bts', sessions_represent, layer_2_current)
        layer2_logits = layer2_logits / avg_distance.clamp_min(1e-6)
        layer2_weights = self._masked_softmax(layer2_logits, session_mask_exp, dim=-1)
        layer2_weights = layer2_weights * session_mask_exp.float()
        out_layer_2 = torch.einsum('btsd,bts->btd', sessions_represent, layer2_weights)
        out_layer_2 = out_layer_2 * current_mask_float

        history_context = base_out.new_zeros(batch_size, seq_len, self.hidden_units)
        history_context[:, :-1, :] = out_layer_2
        return history_context

    def forward(self, user_vectors, item_vectors, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, is_train, poi_distance_matrix, sequence_dilated_rnn_index_batch):
        device = item_vectors.device
        user_vectors = user_vectors.to(device=device, dtype=torch.long)
        mask_tensor = mask_batch_ix_non_local.to(device=device)

        if not torch.is_tensor(session_id_batch):
            session_ids = torch.as_tensor(session_id_batch, device=device, dtype=torch.long)
        else:
            session_ids = session_id_batch.to(device=device, dtype=torch.long)

        if not torch.is_tensor(sequence_tim_batch):
            sequence_tim_tensor = torch.as_tensor(sequence_tim_batch, device=device, dtype=torch.long)
        else:
            sequence_tim_tensor = sequence_tim_batch.to(device=device, dtype=torch.long)

        if not torch.is_tensor(sequence_dilated_rnn_index_batch):
            dilated_indices = torch.as_tensor(sequence_dilated_rnn_index_batch, device=device, dtype=torch.long)
        else:
            dilated_indices = sequence_dilated_rnn_index_batch.to(device=device, dtype=torch.long)

        items = self.item_emb(item_vectors)
        x = items.transpose(0, 1)
        batch_size = item_vectors.size(0)
        h1 = torch.zeros(1, batch_size, self.hidden_units, device=device)
        c1 = torch.zeros(1, batch_size, self.hidden_units, device=device)
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        base_out = out.transpose(0, 1)

        mask_bool = mask_tensor > 0.0
        dilated_states = self._compute_dilated_states(items, dilated_indices, mask_bool)
        history_context = self._compute_history_context(
            user_vectors,
            session_ids,
            sequence_tim_tensor,
            item_vectors,
            mask_bool,
            base_out,
            is_train,
            poi_distance_matrix,
        )

        out_hie = F.selu(dilated_states)
        out_base = F.selu(base_out)
        combined = (out_base + out_hie) * 0.5
        history_act = F.selu(history_context)
        logits_input = torch.cat([history_act, combined], dim=2)
        logits = self.linear(logits_input)
        return F.log_softmax(logits, dim=-1)

    def _forward_legacy(self, user_vectors, item_vectors, mask_batch_ix_non_local, session_id_batch, sequence_tim_batch, is_train, poi_distance_matrix, sequence_dilated_rnn_index_batch):
        batch_size = item_vectors.size()[0]
        sequence_size = item_vectors.size()[1]
        items = self.item_emb(item_vectors)
        item_vectors = item_vectors.cpu()
        x = items
        x = x.transpose(0, 1)
        h1 = torch.zeros(1, batch_size, self.hidden_units).to(device)
        c1 = torch.zeros(1, batch_size, self.hidden_units).to(device)
        out, (h1, c1) = self.lstmcell(x, (h1, c1))
        out = out.transpose(0, 1)#batch_size * sequence_length * embedding_dim
        x1 = items
        # ###########################################################
        user_batch = np.array(user_vectors.cpu())
        y_list = []
        out_hie = []
        for ii in range(batch_size):
            ##########################################
            current_session_input_dilated_rnn_index = sequence_dilated_rnn_index_batch[ii]
            hiddens_current = x1[ii]
            dilated_lstm_outs_h = []
            dilated_lstm_outs_c = []
            for index_dilated in range(len(current_session_input_dilated_rnn_index)):
                index_dilated_explicit = current_session_input_dilated_rnn_index[index_dilated]
                hidden_current = hiddens_current[index_dilated].unsqueeze(0)
                if index_dilated == 0:
                    h = torch.zeros(1, self.hidden_units).to(device)
                    c = torch.zeros(1, self.hidden_units).to(device)
                    (h, c) = self.dilated_rnn(hidden_current, (h, c))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
                else:
                    (h, c) = self.dilated_rnn(hidden_current, (dilated_lstm_outs_h[index_dilated_explicit], dilated_lstm_outs_c[index_dilated_explicit]))
                    dilated_lstm_outs_h.append(h)
                    dilated_lstm_outs_c.append(c)
            dilated_lstm_outs_h.append(hiddens_current[len(current_session_input_dilated_rnn_index):])
            dilated_out = torch.cat(dilated_lstm_outs_h, dim = 0).unsqueeze(0)
            out_hie.append(dilated_out)
            user_id_current = user_batch[ii]
            current_session_timid = sequence_tim_batch[ii][:-1]
            current_session_poiid = item_vectors[ii][:len(current_session_timid)]
            session_id_current = session_id_batch[ii]
            current_session_embed = out[ii]
            current_session_mask = mask_batch_ix_non_local[ii].unsqueeze(1)
            sequence_length = int(sum(np.array(current_session_mask.cpu()))[0])
            current_session_represent_list = []
            if is_train:
                for iii in range(sequence_length-1):
                    current_session_represent = torch.sum(current_session_embed * current_session_mask, dim=0).unsqueeze(0)/sum(current_session_mask)
                    current_session_represent_list.append(current_session_represent)
            else:
                for iii in range(sequence_length-1):
                    current_session_represent_rep_item = current_session_embed[0:iii+1]
                    current_session_represent_rep_item = torch.sum(current_session_represent_rep_item, dim = 0).unsqueeze(0)/(iii + 1)
                    current_session_represent_list.append(current_session_represent_rep_item)

            current_session_represent = torch.cat(current_session_represent_list, dim = 0)
            list_for_sessions = []
            list_for_avg_distance = []
            h2 = torch.zeros(1, 1, self.hidden_units).to(device)###whole sequence
            c2 = torch.zeros(1, 1, self.hidden_units).to(device)
            for jj in range(session_id_current):
                sequence = [s[0] for s in self.data_neural[user_id_current]['sessions'][jj]]
                sequence = torch.LongTensor(np.array(sequence)).to(device)
                sequence_emb = self.item_emb(sequence).unsqueeze(1)
                sequence = sequence.cpu()
                sequence_emb, (h2, c2) = self.lstmcell_history(sequence_emb, (h2, c2))
                sequence_tim_id = [to_tid48(s[1]) for s in self.data_neural[user_id_current]['sessions'][jj]]
                jaccard_sim_row = torch.FloatTensor(self.tim_sim_matrix[current_session_timid]).to(device)
                jaccard_sim_expicit = jaccard_sim_row[:,sequence_tim_id]
                distance_row = poi_distance_matrix[current_session_poiid]
                # distance_row[:, sequence] may reduce to 1-D if only one candidate; ensure 2-D
                distance_row_expicit = distance_row[:, sequence]
                if isinstance(distance_row_expicit, np.ndarray):
                    dr = torch.as_tensor(distance_row_expicit, dtype=torch.float32, device=device)
                    if dr.dim() == 1:
                        dr = dr.unsqueeze(1)
                else:
                    dr = torch.FloatTensor(distance_row_expicit).to(device)
                    if dr.dim() == 1:
                        dr = dr.unsqueeze(1)
                distance_row_expicit_avg = torch.mean(dr, dim = 1)
                jaccard_sim_expicit_last = F.softmax(jaccard_sim_expicit, dim=1)
                hidden_sequence_for_current1 = torch.mm(jaccard_sim_expicit_last, sequence_emb.squeeze(1))
                hidden_sequence_for_current =  hidden_sequence_for_current1
                list_for_sessions.append(hidden_sequence_for_current.unsqueeze(0))
                list_for_avg_distance.append(distance_row_expicit_avg.unsqueeze(0))
            if len(list_for_sessions) == 0:
                # No history; pad zeros with correct shapes so bmm works
                current_items = max(1, len(current_session_timid))
                # sessions_represent should be (current_items, hist_len, H) after cat+transpose,
                # so before those ops we use [1, current_items, H]
                list_for_sessions = [torch.zeros(1, current_items, self.hidden_units, device=device)]
                # avg_distance becomes (current_items,) per item; start with ones to avoid div-by-zero
                list_for_avg_distance = [torch.ones(current_items, device=device).unsqueeze(0)]
            avg_distance = torch.cat(list_for_avg_distance, dim = 0).transpose(0,1)
            sessions_represent = torch.cat(list_for_sessions, dim=0).transpose(0,1) ##current_items * history_session_length * embedding_size
            current_session_represent = current_session_represent.unsqueeze(2) ### current_items * embedding_size * 1
            sims = F.softmax(sessions_represent.bmm(current_session_represent).squeeze(2), dim = 1).unsqueeze(1) ##==> current_items * 1 * history_session_length
            #out_y_current = sims.bmm(sessions_represent).squeeze(1)
            out_y_current =torch.selu(self.linear1(sims.bmm(sessions_represent).squeeze(1)))
            ##############layer_2
            #layer_2_current = (lambda*out_y_current + (1-lambda)*current_session_embed[:sequence_length-1]).unsqueeze(2) #lambda from [0.1-0.9] better performance
            # layer_2_current = (out_y_current + current_session_embed[:sequence_length-1]).unsqueeze(2)##==>current_items * embedding_size * 1
            layer_2_current = (0.5 *out_y_current + 0.5 * current_session_embed[:sequence_length - 1]).unsqueeze(2)
            layer_2_sims =  F.softmax(sessions_represent.bmm(layer_2_current).squeeze(2) * 1.0/avg_distance, dim = 1).unsqueeze(1)##==>>current_items * 1 * history_session_length
            out_layer_2 = layer_2_sims.bmm(sessions_represent).squeeze(1)
            out_y_current_padd = torch.zeros(sequence_size - sequence_length + 1, self.emb_size, device=device)
            out_layer_2_list = []
            out_layer_2_list.append(out_layer_2)
            out_layer_2_list.append(out_y_current_padd)
            out_layer_2 = torch.cat(out_layer_2_list,dim = 0).unsqueeze(0)
            y_list.append(out_layer_2)
        y = torch.selu(torch.cat(y_list,dim=0))
        out_hie = F.selu(torch.cat(out_hie, dim = 0))
        out = F.selu(out)
        out = (out + out_hie) * 0.5
        out_put_emb_v1 = torch.cat([y, out], dim=2)
        output_ln = self.linear(out_put_emb_v1)
        output = F.log_softmax(output_ln, dim=-1)
        return output




def calculate_time_sim(data_neural):
    time_checkin_set = defaultdict(set)
    for uid in data_neural:
        uid_sessions = data_neural[uid]
        for sid in uid_sessions['sessions']:
            session_current = uid_sessions['sessions'][sid]
            for checkin in session_current:
                timid = to_tid48(checkin[1])
                locid = checkin[0]
                if timid not in time_checkin_set:
                    time_checkin_set[timid] = set()
                time_checkin_set[timid].add(locid)
    sim_matrix = np.zeros((48,48))
    for i in range(48):
        for j in range(48):
            set_i = time_checkin_set[i]
            set_j = time_checkin_set[j]
            union = set_i | set_j
            jacc = 0.0 if len(union) == 0 else len(set_i & set_j) / len(union)
            sim_matrix[i][j] = jacc
    return sim_matrix

def calculate_poi_distance(poi_coors):
    print("distance matrix")
    sim_matrix = np.zeros((len(poi_coors) + 1, len(poi_coors) + 1))
    for i in range(len(poi_coors)):
        for j in range(i , len(poi_coors)):
            poi_current = i + 1
            poi_target = j + 1
            poi_current_coor = poi_coors[poi_current]
            poi_target_coor = poi_coors[poi_target]
            distance_between = geodistance(poi_current_coor[1], poi_current_coor[0], poi_target_coor[1], poi_target_coor[0])
            if distance_between<1:
                distance_between = 1
            sim_matrix[poi_current][poi_target] = distance_between
            sim_matrix[poi_target][poi_current] = distance_between
    pickle.dump(sim_matrix, open('distance.pkl', 'wb'))
    return sim_matrix

def generate_input_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            trace = {}
            loc_np = np.reshape(np.array([s[0] for s in session[:-1]], dtype=np.int64), (len(session[:-1]), 1))
            tim_np = np.reshape(np.array([to_tid48(s[1]) for s in session[:-1]], dtype=np.int64), (len(session[:-1]), 1))
            target = np.array([s[0] for s in session[1:]], dtype=np.int64)
            trace['loc'] = torch.LongTensor(loc_np)
            trace['target'] = torch.LongTensor(target)
            trace['tim'] = torch.LongTensor(tim_np)
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history = sorted(history, key=lambda x: to_tid48(x[1]), reverse=False)
            history_loc = np.reshape(np.array([s[0] for s in history], dtype=np.int64), (len(history), 1))
            history_tim = np.reshape(np.array([to_tid48(s[1]) for s in history], dtype=np.int64), (len(history), 1))
            trace['history_loc'] = torch.LongTensor(history_loc)
            trace['history_tim'] = torch.LongTensor(history_tim)
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_input_long_history(data_neural, mode, candidate=None):
    data_train = {}
    train_idx = {}
    if candidate is None:
        candidate = data_neural.keys()
    for u in candidate:
        sessions = data_neural[u]['sessions']
        train_id = data_neural[u][mode]
        data_train[u] = {}
        for c, i in enumerate(train_id):
            trace = {}
            if mode == 'train' and c == 0:
                continue
            session = sessions[i]
            # print(len(session), end=' ')
            target = np.array([s[0] for s in session[1:]], dtype=np.int64)
            history = []
            if mode == 'test':
                test_id = data_neural[u]['train']
                for tt in test_id:
                    history.extend([(s[0], s[1]) for s in sessions[tt]])
            for j in range(c):
                history.extend([(s[0], s[1]) for s in sessions[train_id[j]]])
            history_tim = [to_tid48(t[1]) for t in history]
            history_count = [1]
            last_t = history_tim[0]
            count = 1
            for t in history_tim[1:]:
                if t == last_t:
                    count += 1
                else:
                    history_count[-1] = count
                    history_count.append(1)
                    last_t = t
                    count = 1
            history_loc = np.reshape(np.array([s[0] for s in history], dtype=np.int64), (len(history), 1))
            history_tim = np.reshape(np.array(history_tim, dtype=np.int64), (len(history), 1))
            trace['history_loc'] = torch.LongTensor(history_loc)
            trace['history_tim'] = torch.LongTensor(history_tim)
            trace['history_count'] = history_count
            # build combined loc/tim for current session prefix appended to history
            locs_hist = [s[0] for s in history]
            tims_hist = [to_tid48(s[1]) for s in history]
            locs_sess = [s[0] for s in session[:-1]]
            tims_sess = [to_tid48(s[1]) for s in session[:-1]]
            loc_all = locs_hist + locs_sess
            tim_all = tims_hist + tims_sess
            loc_np = np.reshape(np.array(loc_all, dtype=np.int64), (len(loc_all), 1))
            tim_np = np.reshape(np.array(tim_all, dtype=np.int64), (len(tim_all), 1))
            trace['loc'] = torch.LongTensor(loc_np)
            trace['tim'] = torch.LongTensor(tim_np)
            trace['target'] = torch.LongTensor(target)
            data_train[u][i] = trace
        train_idx[u] = train_id
    return data_train, train_idx

def generate_queue(train_idx, mode, mode2):
    user = list(train_idx.keys())
    train_queue = list()
    if mode == 'random':
        initial_queue = {}
        for u in user:
            if mode2 == 'train':
                initial_queue[u] = deque(train_idx[u][1:])
            else:
                initial_queue[u] = deque(train_idx[u])
        queue_left = 1
        while queue_left > 0:
            for j, u in enumerate(user):
                if len(initial_queue[u]) > 0:
                    train_queue.append((u, initial_queue[u].popleft()))
            queue_left = sum([1 for x in initial_queue if len(initial_queue[x]) > 0])
    elif mode == 'normal':
        for u in user:
            for i in train_idx[u]:
                train_queue.append((u, i))
    return train_queue


def create_dilated_rnn_input(session_sequence_current, poi_distance_matrix):
    sequence_length = len(session_sequence_current)
    session_sequence_current.reverse()
    # -1 means “no parent”, safe for step t=0
    session_dilated_rnn_input_index = [-1] * sequence_length
    for i in range(sequence_length - 1):
        current_poi = [session_sequence_current[i]]
        poi_before = session_sequence_current[i + 1 :]
        distance_row = poi_distance_matrix[current_poi]
        distance_row_explicit = distance_row[:, poi_before][0]
        index_closet = np.argmin(distance_row_explicit)
        parent_idx = sequence_length - 2 - index_closet - i  # points to a prior step
        # store, actual safety clamp happens in forward
        session_dilated_rnn_input_index[sequence_length - i - 1] = int(parent_idx)
    session_sequence_current.reverse()
    return session_dilated_rnn_input_index



def generate_detailed_batch_data(one_train_batch):
    session_id_batch = []
    user_id_batch = []
    sequence_batch = []
    sequences_lens_batch = []
    sequences_tim_batch = []
    sequences_dilated_input_batch = []
    for sample in one_train_batch:
        user_id_batch.append(sample[0])
        session_id_batch.append(sample[1])
        session_sequence_current = [s[0] for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_tim_current = [to_tid48(s[1]) for s in data_neural[sample[0]]['sessions'][sample[1]]]
        session_sequence_dilated_input = create_dilated_rnn_input(session_sequence_current, poi_distance_matrix)
        sequence_batch.append(session_sequence_current)
        sequences_lens_batch.append(len(session_sequence_current))
        sequences_tim_batch.append(session_sequence_tim_current)
        sequences_dilated_input_batch.append(session_sequence_dilated_input)
    return user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequences_tim_batch, sequences_dilated_input_batch


def train_network(network, num_epoch=40 ,batch_size = 32,criterion = None, save_dir: str = None, checkpoint_dir: str = "checkpoint", final_model_name: str = "res.m", patience: int = 5):
    candidate = data_neural.keys()
    data_train, train_idx = generate_input_history(data_neural, 'train', candidate=candidate)
    # checkpointing setup
    if save_dir is None:
        save_dir = os.getcwd()
    tmp_path = os.path.join(save_dir, checkpoint_dir)
    os.makedirs(tmp_path, exist_ok=True)
    best_acc = -1.0
    best_epoch = -1
    metrics_accuracy = []
    
    # Early stopping variables
    best_valid_loss = float('inf')
    patience_counter = 0
    for epoch in range(num_epoch):
        network.train(True)
        i = 0
        run_queue = generate_queue(train_idx, 'random', 'train')
        for one_train_batch in minibatch(run_queue, batch_size = batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(one_train_batch)
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = torch.LongTensor(np.array(padded_sequence_batch)).to(device)
            mask_batch_ix = torch.FloatTensor(np.array(mask_batch_ix)).to(device)
            mask_batch_ix_non_local = torch.FloatTensor(np.array(mask_batch_ix_non_local)).to(device)
            user_id_batch = torch.LongTensor(np.array(user_id_batch)).to(device)
            padded_tim_batch = torch.LongTensor(np.array(pad_batch_of_lists(sequence_tim_batch, max_len, pad_value=0))).to(device)
            padded_dilated_batch = torch.LongTensor(np.array(pad_batch_of_lists(sequence_dilated_rnn_index_batch, max_len, pad_value=-1))).to(device)
            session_id_tensor = torch.LongTensor(np.array(session_id_batch)).to(device)
            logp_seq = network(
                user_id_batch,
                padded_sequence_batch,
                mask_batch_ix_non_local,
                session_id_tensor,
                padded_tim_batch,
                True,
                poi_distance_matrix,
                padded_dilated_batch,
            )
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum() / mask_batch_ix[:, :-1].sum()
            # train with backprop
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 5.0)
            opt.step()
            if (i + 1) % 20 == 0:
                print("epoch" + str(epoch) + ": loss: " + str(loss))
            i += 1
        results = evaluate(network, 1)
        print("Scores: ", results)
        
        # Calculate validation loss for early stopping
        valid_loss = evaluate_loss(network, 1)
        print(f"Validation loss: {valid_loss:.4f}")
        
        # Save checkpoint for this epoch
        save_name_tmp = f'ep_{epoch}.m'
        torch.save(network.state_dict(), os.path.join(tmp_path, save_name_tmp))
        
        # Track best by top-1 accuracy (results[0])
        try:
            acc_at1 = float(results[0])
        except Exception:
            acc_at1 = -1.0
        metrics_accuracy.append(acc_at1)
        if acc_at1 > best_acc:
            best_acc = acc_at1
            best_epoch = epoch
            print(f"New best @1={best_acc:.4f} at epoch {best_epoch}")
        
        # Early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            print(f'==>New best validation loss: {best_valid_loss:.4f}')
        else:
            patience_counter += 1
            print(f'==>Validation loss did not improve. Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'==>Early stopping triggered after {patience} epochs without improvement')
            break

    # reload best and save final
    if best_epoch >= 0:
        load_name_tmp = f'ep_{best_epoch}.m'
        best_path = os.path.join(tmp_path, load_name_tmp)
        network.load_state_dict(torch.load(best_path, map_location=device))
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, final_model_name)
        torch.save(network.state_dict(), final_path)
        # optional cleanup of tmp checkpoints to mirror DeepMove behavior
        '''
        try:
            for name in os.listdir(tmp_path):
                remove_path = os.path.join(tmp_path, name)
                if os.path.isfile(remove_path):
                    os.remove(remove_path)
            os.rmdir(tmp_path)
        except Exception:
            pass
        '''


def get_acc(target, scores):
    target = target.data.cpu().numpy()
    val, idxx = scores.data.topk(10, 1)
    predx = idxx.cpu().numpy()
    acc = np.zeros((3, 1))
    ndcg = np.zeros((3, 1))
    for i, p in enumerate(predx):
        t = target[i]
        if t != 0:
            if t in p[:10] and t > 0:
                acc[0] += 1
                rank_list = list(p[:10])
                rank_index = rank_list.index(t)
                ndcg[0] += 1.0 / np.log2(rank_index + 2)
            if t in p[:5] and t > 0:
                acc[1] += 1
                rank_list = list(p[:5])
                rank_index = rank_list.index(t)
                ndcg[1] += 1.0 / np.log2(rank_index + 2)
            if t == p[0] and t > 0:
                acc[2] += 1
                rank_list = list(p[:1])
                rank_index = rank_list.index(t)
                ndcg[2] += 1.0 / np.log2(rank_index + 2)
        else:
            break
    return acc.tolist(), ndcg.tolist()

def evaluate_loss(network, batch_size=2):
    """Calculate validation loss for early stopping"""
    network.train(False)
    candidate = data_neural.keys()
    data_test, test_idx = generate_input_long_history(data_neural, 'test', candidate=candidate)

    # Fallback: if all test lists are empty, use the train split for evaluation
    if all(len(v) == 0 for v in test_idx.values()):
        data_test, test_idx = generate_input_long_history(data_neural, 'train', candidate=candidate)

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        run_queue = generate_queue(test_idx, 'normal', 'test')
        for one_test_batch in minibatch(run_queue, batch_size=batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(
                one_test_batch)
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = torch.LongTensor(np.array(padded_sequence_batch)).to(device)
            mask_batch_ix = torch.FloatTensor(np.array(mask_batch_ix)).to(device)
            mask_batch_ix_non_local = torch.FloatTensor(np.array(mask_batch_ix_non_local)).to(device)
            user_id_batch = torch.LongTensor(np.array(user_id_batch)).to(device)
            padded_tim_batch = torch.LongTensor(np.array(pad_batch_of_lists(sequence_tim_batch, max_len, pad_value=0))).to(device)
            padded_dilated_batch = torch.LongTensor(np.array(pad_batch_of_lists(sequence_dilated_rnn_index_batch, max_len, pad_value=-1))).to(device)
            session_id_tensor = torch.LongTensor(np.array(session_id_batch)).to(device)
            logp_seq = network(
                user_id_batch,
                padded_sequence_batch,
                mask_batch_ix_non_local,
                session_id_tensor,
                padded_tim_batch,
                False,
                poi_distance_matrix,
                padded_dilated_batch,
            )
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, None])
            loss = -logp_next.sum()
            total_loss += loss.item()
            total_tokens += mask_batch_ix[:, :-1].sum().item()
    
    if total_tokens == 0:
        return float('inf')
    return total_loss / total_tokens

def evaluate(network, batch_size = 2):
    network.train(False)
    candidate = data_neural.keys()
    data_test, test_idx = generate_input_long_history(data_neural, 'test', candidate=candidate)

    # Fallback: if all test lists are empty, use the train split for evaluation
    if all(len(v) == 0 for v in test_idx.values()):
        data_test, test_idx = generate_input_long_history(data_neural, 'train', candidate=candidate)

    users_acc = {}
    with torch.no_grad():
        run_queue = generate_queue(test_idx, 'normal', 'test')
        for one_test_batch in minibatch(run_queue, batch_size=batch_size):
            user_id_batch, session_id_batch, sequence_batch, sequences_lens_batch, sequence_tim_batch, sequence_dilated_rnn_index_batch = generate_detailed_batch_data(
                one_test_batch)
            user_id_batch_test = user_id_batch
            max_len = max(sequences_lens_batch)
            padded_sequence_batch, mask_batch_ix, mask_batch_ix_non_local = pad_batch_of_lists_masks(sequence_batch,
                                                                                                     max_len)
            padded_sequence_batch = torch.LongTensor(np.array(padded_sequence_batch)).to(device)
            mask_batch_ix = torch.FloatTensor(np.array(mask_batch_ix)).to(device)
            mask_batch_ix_non_local = torch.FloatTensor(np.array(mask_batch_ix_non_local)).to(device)
            user_id_batch_tensor = torch.LongTensor(np.array(user_id_batch)).to(device)
            padded_tim_batch = torch.LongTensor(np.array(pad_batch_of_lists(sequence_tim_batch, max_len, pad_value=0))).to(device)
            padded_dilated_batch = torch.LongTensor(np.array(pad_batch_of_lists(sequence_dilated_rnn_index_batch, max_len, pad_value=-1))).to(device)
            session_id_tensor = torch.LongTensor(np.array(session_id_batch)).to(device)
            logp_seq = network(
                user_id_batch_tensor,
                padded_sequence_batch,
                mask_batch_ix_non_local,
                session_id_tensor,
                padded_tim_batch,
                False,
                poi_distance_matrix,
                padded_dilated_batch,
            )
            predictions_logp = logp_seq[:, :-1] * mask_batch_ix[:, :-1, None]
            actual_next_tokens = padded_sequence_batch[:, 1:]
            batch_n = predictions_logp.shape[0]
            for ii in range(batch_n):
                u_current = user_id_batch_test[ii]
                if u_current not in users_acc:
                    users_acc[u_current] = [0, 0, 0, 0, 0, 0, 0]
                # skip if sequence length < 2 (no next token)
                if actual_next_tokens[ii].numel() == 0 or predictions_logp[ii].numel() == 0:
                    continue
                acc, ndcg = get_acc(actual_next_tokens[ii], predictions_logp[ii])
                users_acc[u_current][1] += acc[2][0]#@1
                users_acc[u_current][2] += acc[1][0]#@5
                users_acc[u_current][3] += acc[0][0]#@10
                ###ndcg
                users_acc[u_current][4] += ndcg[2][0]  # @1
                users_acc[u_current][5] += ndcg[1][0]  # @5
                users_acc[u_current][6] += ndcg[0][0]  # @10
                users_acc[u_current][0] += (sequences_lens_batch[ii]-1)
        tmp_acc = [0.0,0.0,0.0, 0.0, 0.0, 0.0]##last 3 ndcg
        sum_test_samples = 0.0
        for u in users_acc:
            tmp_acc[0] = users_acc[u][1] + tmp_acc[0]
            tmp_acc[1] = users_acc[u][2] + tmp_acc[1]
            tmp_acc[2] = users_acc[u][3] + tmp_acc[2]

            tmp_acc[3] = users_acc[u][4] + tmp_acc[3]
            tmp_acc[4] = users_acc[u][5] + tmp_acc[4]
            tmp_acc[5] = users_acc[u][6] + tmp_acc[5]
            sum_test_samples = sum_test_samples + users_acc[u][0]
        if sum_test_samples == 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        avg_acc = (np.array(tmp_acc)/sum_test_samples).tolist()
        return avg_acc

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance/1000,3)
    return distance


def cli_main():
    parser = argparse.ArgumentParser(description="Train LSTPM model with dataset, metadata, and distance matrix")
    parser.add_argument("--data_pk", required=True, help="Path to preprocessed dataset .pk file")
    parser.add_argument("--metadata_json", required=False, help="Path to metadata.json (optional, for logging/consistency checks)")
    parser.add_argument("--distance", required=True, help="Path to distance.pkl file matching the dataset vocabulary")
    parser.add_argument("--save_dir", required=True, help="Directory to save checkpoints and final model")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--early_stopping", type=int, default=5, help="number of epochs to wait for validation loss improvement before early stopping")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay (L2)")
    args = parser.parse_args()

    np.random.seed(1)
    torch.manual_seed(1)
    print(f"Using device: {device}")

    # Load dataset
    if not os.path.exists(args.data_pk):
        raise FileNotFoundError(f"Dataset .pk not found at {args.data_pk}")
    data = pickle.load(open(args.data_pk, 'rb'), encoding='iso-8859-1')
    vid_list = data['vid_list']
    uid_list = data['uid_list']
    # wire globals used by generators/evaluate
    global data_neural
    data_neural = data['data_neural']

    # Metadata (optional consistency/info)
    if args.metadata_json:
        if not os.path.exists(args.metadata_json):
            print(f"Warning: metadata file not found at {args.metadata_json}; continuing without it")
        else:
            try:
                meta = json.load(open(args.metadata_json, 'r'))
                print(f"Loaded metadata with {len(meta.get('pid_mapping', {}))} POIs and {len(meta.get('users', []))} users")
            except Exception as e:
                print(f"Warning: failed to parse metadata json: {e}")

    # Time similarity matrix from data
    time_sim_matrix = calculate_time_sim(data_neural)

    # Distance matrix must be provided
    if not os.path.exists(args.distance):
        raise FileNotFoundError(f"distance.pkl not found at {args.distance}")
    with open(args.distance, 'rb') as fh:
        dist = pickle.load(fh, encoding='iso-8859-1')
    # wire global used by model forward/evaluate
    global poi_distance_matrix
    poi_distance_matrix = np.asarray(dist, dtype=np.float32)
    # check for zeros off the diagonal\

    if np.any((poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)):
        print("Warning: distance matrix has zero(s) off the diagonal; this may cause instability")
        # replace them with small value to avoid div-by-zero
        poi_distance_matrix[(poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)] = 1e-16
        # also replace diagonal with small value
        np.fill_diagonal(poi_distance_matrix, 1e-16)

    gc.collect()
    n_users = len(uid_list)
    n_items = len(vid_list)

    # Build and train model
    network = Model(n_users=n_users, n_items=n_items, data_neural=data_neural, tim_sim_matrix=time_sim_matrix).to(device)
    global opt
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.NLLLoss().to(device)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Training on users={n_users}, items={n_items}; saving to {args.save_dir}")
    train_network(network, num_epoch=args.epochs, batch_size=args.batch_size, criterion=criterion, save_dir=args.save_dir, checkpoint_dir="checkpoint", final_model_name="res.m", patience=args.early_stopping)


if __name__ == '__main__':
    cli_main()

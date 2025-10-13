# coding: utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F
# auto-select MPS (for M1/M2), then CUDA, else CPU
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
from torch.autograd import Variable


# ############# simple rnn model ####################### #
class TrajPreSimple(nn.Module):
    """baseline rnn model"""

    def __init__(self, parameters):
        super(TrajPreSimple, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)
        self.init_weights()

        self.fc = nn.Linear(self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform_(t)
        for t in hh:
            nn.init.orthogonal_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def forward(self, loc, tim):
        # ensure tensors have shape (seq_len, batch)
        if loc.dim() == 3 and loc.size(-1) == 1:
            loc = loc.squeeze(-1)
        if tim.dim() == 3 and tim.size(-1) == 1:
            tim = tim.squeeze(-1)

        if loc.dim() != 2 or tim.dim() != 2:
            raise ValueError("Expected loc and tim tensors with shape (seq_len, batch)")

        seq_len, batch_size = loc.size(0), loc.size(1)

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        # initialize hidden states on correct device
        h1 = x.new_zeros(1, batch_size, self.hidden_size)
        if self.rnn_type == 'LSTM':
            c1 = x.new_zeros(1, batch_size, self.hidden_size)

        if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
            out, h1 = self.rnn(x, h1)
        elif self.rnn_type == 'LSTM':
            out, (h1, c1) = self.rnn(x, (h1, c1))

        out = F.selu(out)
        out = self.dropout(out)

        out_flat = out.contiguous().view(seq_len * batch_size, self.hidden_size)
        y = self.fc(out_flat)
        score = F.log_softmax(y, dim=-1)  # calculate loss by NLLoss
        score = score.view(seq_len, batch_size, self.loc_size)

        if batch_size == 1:
            score = score.squeeze(1)
        return score


# ############# rnn model with attention ####################### #
class Attn(nn.Module):
    """Attention Module. Heavily borrowed from Practical Pytorch
    https://github.com/spro/practical-pytorch/tree/master/seq2seq-translation"""

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history, history_mask=None):
        """Compute attention weights.

        Args:
            out_state: Tensor of shape (tgt_len, batch, hidden).
            history:   Tensor of shape (hist_len, batch, hidden).
            history_mask: Optional bool/binary Tensor of shape (batch, hist_len)
                          indicating valid history positions.
        Returns:
            Attention weights of shape (batch, tgt_len, hist_len).
        """
        if out_state.dim() != 3 or history.dim() != 3:
            raise ValueError("Attn expects tensors with shape (len, batch, hidden)")

        tgt_len, batch, hidden = out_state.size()
        hist_len = history.size(0)

        out_state_b = out_state.permute(1, 0, 2)  # (batch, tgt_len, hidden)
        history_b = history.permute(1, 0, 2)      # (batch, hist_len, hidden)

        if self.method == 'dot':
            energies = torch.bmm(out_state_b, history_b.transpose(1, 2))
        elif self.method == 'general':
            proj = self.attn(history_b.reshape(-1, hidden))
            proj = proj.view(batch, hist_len, hidden)
            energies = torch.bmm(out_state_b, proj.transpose(1, 2))
        elif self.method == 'concat':
            out_exp = out_state_b.unsqueeze(2).expand(batch, tgt_len, hist_len, hidden)
            hist_exp = history_b.unsqueeze(1).expand(batch, tgt_len, hist_len, hidden)
            energy = self.attn(torch.cat((out_exp, hist_exp), dim=-1))
            energy = torch.tanh(energy)
            energies = torch.matmul(energy, self.other)
            energies = energies.squeeze(-1)
        else:
            raise ValueError(f"Unknown attention method: {self.method}")

        if history_mask is not None:
            # mask shape (batch, hist_len) -> expand to (batch, 1, hist_len)
            mask = history_mask.unsqueeze(1)
            energies = energies.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(energies, dim=-1)
        return attn_weights


# ##############long###########################
class TrajPreAttnAvgLongUser(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreAttnAvgLongUser, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type
        self.use_cuda = parameters.use_cuda

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size + self.uid_emb_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, history_loc, history_tim, history_count, uid, target_len):
        def _prepare_seq(t):
            if isinstance(t, torch.Tensor):
                if t.dim() == 3 and t.size(-1) == 1:
                    t = t.squeeze(-1)
                if t.dim() == 1:
                    t = t.unsqueeze(1)
            return t

        loc = _prepare_seq(loc)
        tim = _prepare_seq(tim)
        history_loc = _prepare_seq(history_loc)
        history_tim = _prepare_seq(history_tim)

        if loc.dim() != 2 or tim.dim() != 2:
            raise ValueError("loc and tim must have shape (seq_len, batch)")

        seq_len, batch_size = loc.size()

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        # initialize hidden states
        h1 = x.new_zeros(1, batch_size, self.hidden_size)
        if self.rnn_type == 'LSTM':
            c1 = x.new_zeros(1, batch_size, self.hidden_size)

        if self.rnn_type in ['GRU', 'RNN']:
            out_state, h1 = self.rnn(x, h1)
        else:
            out_state, (h1, c1) = self.rnn(x, (h1, c1))

        # prepare history embeddings
        loc_emb_history = self.emb_loc(history_loc)
        tim_emb_history = self.emb_tim(history_tim)

        counts_per_batch = []
        if isinstance(history_count, torch.Tensor):
            if history_count.dim() == 1:
                counts_per_batch = [history_count.tolist()]
            else:
                counts_per_batch = [row[row > 0].tolist() for row in history_count]
        elif isinstance(history_count, list):
            if batch_size == 1 or (history_count and isinstance(history_count[0], int)):
                counts_per_batch = [list(history_count)]
            else:
                counts_per_batch = [list(c) for c in history_count]
        else:
            raise ValueError("history_count must be list or tensor")

        if len(counts_per_batch) == 1 and batch_size > 1:
            counts_per_batch = counts_per_batch * batch_size

        history_loc_agg = []
        history_tim_agg = []
        max_groups = 0

        for b in range(batch_size):
            counts = counts_per_batch[b] if b < len(counts_per_batch) else counts_per_batch[0]

            # flatten possible nested lists (e.g., [[3, 2], [1]]) and drop non-positive entries
            flat_counts = []
            stack = list(counts if isinstance(counts, (list, tuple)) else [counts])
            while stack:
                item = stack.pop()
                if isinstance(item, (list, tuple)):
                    stack.extend(item)
                else:
                    val = int(item)
                    if val > 0:
                        flat_counts.append(val)
            counts = flat_counts

            total_events = sum(counts)
            loc_seq = loc_emb_history[:total_events, b, :] if total_events > 0 else loc_emb_history.new_zeros((0, loc_emb_history.size(-1)))
            tim_seq = tim_emb_history[:total_events, b, :] if total_events > 0 else tim_emb_history.new_zeros((0, tim_emb_history.size(-1)))

            agg_loc = []
            agg_tim = []
            idx = 0
            for c in counts:
                chunk_loc = loc_seq[idx:idx + c]
                chunk_tim = tim_seq[idx]
                if chunk_loc.size(0) == 0:
                    break
                agg_loc.append(chunk_loc.mean(dim=0))
                agg_tim.append(chunk_tim)
                idx += c

            if agg_loc:
                loc_tensor = torch.stack(agg_loc, dim=0)
                tim_tensor = torch.stack(agg_tim, dim=0)
            else:
                loc_tensor = loc_emb_history.new_zeros((0, loc_emb_history.size(-1)))
                tim_tensor = tim_emb_history.new_zeros((0, tim_emb_history.size(-1)))

            history_loc_agg.append(loc_tensor)
            history_tim_agg.append(tim_tensor)
            max_groups = max(max_groups, loc_tensor.size(0))

        if max_groups == 0:
            history_proj = x.new_zeros(1, batch_size, self.hidden_size)
            history_mask_tensor = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        else:
            loc_pad = loc_emb_history.new_zeros((max_groups, batch_size, self.loc_emb_size))
            tim_pad = tim_emb_history.new_zeros((max_groups, batch_size, self.tim_emb_size))
            mask_pad = torch.zeros(batch_size, max_groups, dtype=torch.bool, device=x.device)

            for b in range(batch_size):
                loc_tensor = history_loc_agg[b]
                tim_tensor = history_tim_agg[b]
                length = loc_tensor.size(0)
                if length == 0:
                    continue
                loc_pad[:length, b, :] = loc_tensor
                tim_pad[:length, b, :] = tim_tensor
                mask_pad[b, :length] = True

            history_concat = torch.cat((loc_pad, tim_pad), dim=-1)
            history_flat = history_concat.view(-1, history_concat.size(-1))
            history_proj = F.tanh(self.fc_attn(history_flat)).view(max_groups, batch_size, self.hidden_size)
            history_mask_tensor = mask_pad

        # determine target lengths
        if isinstance(target_len, int):
            target_lengths = [target_len] * batch_size
        elif isinstance(target_len, torch.Tensor):
            target_lengths = target_len.tolist()
        else:
            target_lengths = [int(t) for t in target_len]

        outputs = []
        max_target = max(target_lengths) if target_lengths else 0

        uid_tensor = uid
        if not isinstance(uid_tensor, torch.Tensor):
            uid_tensor = torch.tensor(uid, device=x.device)
        uid_tensor = uid_tensor.view(-1)
        if uid_tensor.size(0) == 1 and batch_size > 1:
            uid_tensor = uid_tensor.expand(batch_size)

        uid_emb_all = self.emb_uid(uid_tensor)

        for b in range(batch_size):
            tgt_len_b = target_lengths[b]
            if tgt_len_b == 0:
                outputs.append(out_state.new_zeros((0, self.loc_size)))
                continue

            out_state_b = out_state[-tgt_len_b:, b, :].unsqueeze(1)

            hist_mask_b = history_mask_tensor[b]
            hist_len_b = int(hist_mask_b.sum().item())

            if hist_len_b > 0:
                history_b = history_proj[:hist_len_b, b, :].unsqueeze(1)
                mask_b = hist_mask_b[:hist_len_b].unsqueeze(0)
                attn_weights = self.attn(out_state_b, history_b, history_mask=mask_b)
                context = torch.bmm(attn_weights, history_b.transpose(0, 1)).squeeze(0)
            else:
                history_b = history_proj.new_zeros((1, 1, self.hidden_size))
                context = out_state_b.new_zeros((tgt_len_b, self.hidden_size))

            out_b = out_state_b.squeeze(1)
            combined = torch.cat((out_b, context), dim=1)
            uid_emb = uid_emb_all[b].unsqueeze(0).expand(tgt_len_b, -1)
            combined = torch.cat((combined, uid_emb), dim=1)
            combined = self.dropout(combined)

            logits = self.fc_final(combined)
            scores = F.log_softmax(logits, dim=-1)
            outputs.append(scores)

        if max_target == 0:
            return out_state.new_zeros((0, batch_size, self.loc_size))

        out_pad = out_state.new_full((max_target, batch_size, self.loc_size), float('-inf'))
        for b in range(batch_size):
            scores = outputs[b]
            length = scores.size(0)
            if length == 0:
                continue
            out_pad[-length:, b, :] = scores

        if batch_size == 1:
            return out_pad.squeeze(1)
        return out_pad


class TrajPreLocalAttnLong(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(TrajPreLocalAttnLong, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.use_cuda = parameters.use_cuda
        self.rnn_type = parameters.rnn_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)

        input_size = self.loc_emb_size + self.tim_emb_size
        self.attn = Attn(self.attn_type, self.hidden_size)
        self.fc_attn = nn.Linear(input_size, self.hidden_size)

        if self.rnn_type == 'GRU':
            self.rnn_encoder = nn.GRU(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.GRU(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'LSTM':
            self.rnn_encoder = nn.LSTM(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)
        elif self.rnn_type == 'RNN':
            self.rnn_encoder = nn.RNN(input_size, self.hidden_size, 1)
            self.rnn_decoder = nn.LSTM(input_size, self.hidden_size, 1)

        self.fc_final = nn.Linear(2 * self.hidden_size, self.loc_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, target_len):
        def _prepare_seq(t):
            if isinstance(t, torch.Tensor) and t.dim() == 3 and t.size(-1) == 1:
                return t.squeeze(-1)
            return t

        loc = _prepare_seq(loc)
        tim = _prepare_seq(tim)

        if loc.dim() != 2 or tim.dim() != 2:
            raise ValueError("loc and tim must have shape (seq_len, batch)")

        seq_len, batch_size = loc.size()

        if isinstance(target_len, torch.Tensor):
            if target_len.dim() == 0:
                target_len = int(target_len.item())
            else:
                vals = target_len.tolist()
                if len(set(vals)) != 1:
                    raise ValueError("All target lengths must be equal for batched local attention")
                target_len = int(vals[0])
        elif isinstance(target_len, (list, tuple)):
            if len(set(target_len)) != 1:
                raise ValueError("All target lengths must be equal for batched local attention")
            target_len = int(target_len[0])
        else:
            target_len = int(target_len)

        if target_len <= 0 or target_len > seq_len:
            raise ValueError("target_len must be within (0, seq_len]")

        loc_emb = self.emb_loc(loc)
        tim_emb = self.emb_tim(tim)
        x = torch.cat((loc_emb, tim_emb), 2)
        x = self.dropout(x)

        hist_part = x[:-target_len]
        tgt_part = x[-target_len:]

        h1 = x.new_zeros(1, batch_size, self.hidden_size)
        h2 = x.new_zeros(1, batch_size, self.hidden_size)
        if self.rnn_type == 'LSTM':
            c1 = x.new_zeros(1, batch_size, self.hidden_size)
            c2 = x.new_zeros(1, batch_size, self.hidden_size)

        if self.rnn_type in ['GRU', 'RNN']:
            hidden_history, h1 = self.rnn_encoder(hist_part, h1)
            hidden_state, h2 = self.rnn_decoder(tgt_part, h2)
        else:
            hidden_history, (h1, c1) = self.rnn_encoder(hist_part, (h1, c1))
            hidden_state, (h2, c2) = self.rnn_decoder(tgt_part, (h2, c2))

        attn_weights = self.attn(hidden_state, hidden_history)
        history_b = hidden_history.permute(1, 0, 2)  # (batch, hist_len, hidden)
        context = torch.bmm(attn_weights, history_b)  # (batch, tgt_len, hidden)
        context = context.permute(1, 0, 2)           # (tgt_len, batch, hidden)

        out = torch.cat((hidden_state, context), dim=2)
        out = self.dropout(out)

        logits = self.fc_final(out.view(-1, out.size(-1)))
        score = F.log_softmax(logits, dim=-1).view(target_len, batch_size, self.loc_size)

        if batch_size == 1:
            score = score.squeeze(1)
        return score

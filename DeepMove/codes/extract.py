import os
import argparse
import pickle
import json
from typing import List

import torch
import torch.nn as nn

from tqdm import tqdm

from train import RnnParameterData
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def load_model(parameters, model_mode, ckpt_path):
    if model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters)
    elif model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser(parameters)
    else:
        model = TrajPreLocalAttnLong(parameters)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    return model.to(device).eval()


def pad_to_batch(sequences: List[List[int]]):
    max_len = max(len(seq) for seq in sequences)
    batch = torch.zeros(max_len, len(sequences), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        batch[:len(seq), idx] = torch.tensor(seq, dtype=torch.long)
    return batch


def compute_step_scores(model, model_mode, loc_seq, tim_seq, uid_idx, history_loc, history_tim, history_count, target_len):
    loc = torch.tensor(loc_seq, dtype=torch.long, device=device).unsqueeze(1)
    tim = torch.tensor(tim_seq, dtype=torch.long, device=device).unsqueeze(1)

    with torch.no_grad():
        if model_mode in ['simple', 'simple_long']:
            scores = model(loc, tim)
        elif model_mode == 'attn_avg_long_user':
            history_loc_tensor = torch.tensor(history_loc, dtype=torch.long, device=device).unsqueeze(1)
            history_tim_tensor = torch.tensor(history_tim, dtype=torch.long, device=device).unsqueeze(1)
            history_count_tensor = torch.tensor(history_count, dtype=torch.long, device=device)
            uid_tensor = torch.tensor([uid_idx], dtype=torch.long, device=device)
            scores = model(
                loc,
                tim,
                history_loc_tensor,
                history_tim_tensor,
                history_count_tensor,
                uid_tensor,
                target_len,
            )
        else:
            scores = model(loc, tim, target_len)

    if scores.dim() == 2:
        scores = scores.unsqueeze(1)
    if scores.dim() == 1:
        scores = scores.unsqueeze(0).unsqueeze(0)
    return scores.squeeze(1)  # (seq_len, vocab)


def rank_ground_truth(scores, targets):
    ranks = []
    for t, score in zip(targets, scores):
        sorted_values, sorted_indices = torch.sort(score, descending=True)
        rank = (sorted_indices == t).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    return ranks


def trajectory_rank_proxy(ranks: List[int], vocab_size: int, max_len: int) -> int:
    total = 0
    n = len(ranks)
    for i, r in enumerate(ranks, start=1):
        exponent = max_len - i
        total += (r - 1) * (vocab_size ** exponent)
    return total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract trajectory rank proxies using DeepMove models")
    parser.add_argument('--data_pk', type=str, required=True, help="path to processed .pk file")
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model")
    parser.add_argument('--model_mode', type=str, required=True,
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--metadata_json', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--steps', type=int, default=None,
                        help="maximum number of steps to consider per trajectory")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    params = RnnParameterData(metadata=args.metadata_json)
    params.model_mode = args.model_mode
    params.use_cuda = (device.type != 'cpu')
    params.rnn_type = getattr(params, 'rnn_type', 'LSTM')
    params.attn_type = getattr(params, 'attn_type', 'dot')
    params.dropout_p = getattr(params, 'dropout_p', 0.3)
    params.tim_size = getattr(params, 'tim_size', 48)

    with open(args.metadata_json, 'r') as f:
        meta = json.load(f)
    params.loc_size = len(meta.get('pid_mapping', {})) + 1
    params.uid_size = len(meta.get('users', []))
    user_to_idx = {str(u): i for i, u in enumerate(meta.get('users', []))}

    model = load_model(params, args.model_mode, args.model_path)

    data = pickle.load(open(args.data_pk, 'rb'))
    sessions_all = data['data_neural']
    uid_list = data['uid_list']
    idx_to_user = {v[0]: k for k, v in uid_list.items()}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as out_f:
        out_f.write("tid,rank_proxy\n")

        for uid_emb, udata in tqdm(sessions_all.items()):
            label = idx_to_user.get(uid_emb, None)
            uid_idx = user_to_idx.get(str(label), None)
            if uid_idx is None:
                continue

            sess_ids = sorted(udata['sessions'].keys())
            merged = []
            for sid in sess_ids:
                merged.extend(udata['sessions'][sid])
            if len(merged) < 2:
                continue

            loc_seq = [p[0] for p in merged][:-1]
            tim_seq = [p[1] for p in merged][:-1]
            targets = [p[0] for p in merged][1:]

            if args.steps is not None:
                loc_seq = loc_seq[:args.steps]
                tim_seq = tim_seq[:args.steps]
                targets = targets[:args.steps]

            scores = compute_step_scores(
                model,
                args.model_mode,
                loc_seq,
                tim_seq,
                uid_idx,
                loc_seq,
                tim_seq,
                [1] * len(loc_seq),
                len(targets),
            )

            ranks = rank_ground_truth(scores, torch.tensor(targets, dtype=torch.long, device=device))
            max_len = len(targets)
            proxy = trajectory_rank_proxy(ranks, params.loc_size, max_len)
            # compute # of digits in result
            num_digits = len(str(proxy))
            out_f.write(f"{label},{num_digits},{proxy}\n")

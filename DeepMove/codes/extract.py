import os
import argparse
import pickle
import json
from typing import List, Optional

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


def batch_scores_simple(model, batch_entries):
    if not batch_entries:
        return []

    lengths = [len(entry['loc_seq']) for entry in batch_entries]
    if not any(lengths):
        return [torch.empty(0, model.loc_size if hasattr(model, 'loc_size') else 0)] * len(batch_entries)

    max_len = max(lengths)
    batch_size = len(batch_entries)

    loc_tensor = torch.zeros(max_len, batch_size, dtype=torch.long, device=device)
    tim_tensor = torch.zeros(max_len, batch_size, dtype=torch.long, device=device)

    for idx, entry in enumerate(batch_entries):
        length = lengths[idx]
        if length == 0:
            continue
        loc_tensor[:length, idx] = torch.tensor(entry['loc_seq'], dtype=torch.long, device=device)
        tim_tensor[:length, idx] = torch.tensor(entry['tim_seq'], dtype=torch.long, device=device)

    with torch.no_grad():
        scores = model(loc_tensor, tim_tensor)

    if scores.dim() == 2:
        scores = scores.unsqueeze(1)

    return [scores[:lengths[idx], idx, :].detach() for idx in range(batch_size)]


def batch_step_scores(model, model_mode, batch_entries):
    if model_mode in ['simple', 'simple_long']:
        return batch_scores_simple(model, batch_entries)

    # fallback to sequential processing for attention models
    outputs = []
    for entry in batch_entries:
        scores = compute_step_scores(
            model,
            model_mode,
            entry['loc_seq'],
            entry['tim_seq'],
            entry['uid_idx'],
            entry['loc_seq'],
            entry['tim_seq'],
            [1] * len(entry['loc_seq']),
            len(entry['targets']),
        )
        outputs.append(scores.detach())
    return outputs


def rank_ground_truth(scores, targets):
    if targets.numel() == 0:
        return []

    target_scores = scores.gather(1, targets.unsqueeze(1))
    ranks = (scores > target_scores).sum(dim=1) + 1
    return ranks.tolist()


def trajectory_rank_proxy(ranks: List[int], vocab_size: int, max_len: int) -> int:
    total = 0
    n = len(ranks)
    for i, r in enumerate(ranks, start=1):
        exponent = max_len - i
        total += (r - 1) * (vocab_size ** exponent)
    return total


def prepare_trajectories(data, user_to_idx, model_mode, steps: Optional[int]):
    sessions_all = data['data_neural']
    uid_list = data['uid_list']
    idx_to_user = {v[0]: k for k, v in uid_list.items()}

    trajectories = []
    step_limit = None if steps is None else max(steps, 0)

    for uid_emb, udata in sessions_all.items():
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

        full_loc_seq = [p[0] for p in merged]
        full_tim_seq = [p[1] for p in merged]
        targets = full_loc_seq[1:]

        if model_mode == 'attn_local_long':
            loc_seq = full_loc_seq
            tim_seq = full_tim_seq
        else:
            loc_seq = full_loc_seq[:-1]
            tim_seq = full_tim_seq[:-1]

        if step_limit is not None:
            targets = targets[:step_limit]

        if not targets:
            continue

        if model_mode == 'attn_local_long':
            required_len = len(targets) + 1
            loc_seq = loc_seq[:required_len]
            tim_seq = tim_seq[:required_len]
            if len(loc_seq) <= len(targets):
                continue
        else:
            loc_seq = loc_seq[:len(targets)]
            tim_seq = tim_seq[:len(targets)]
            if len(loc_seq) != len(targets):
                cutoff = min(len(loc_seq), len(targets))
                loc_seq = loc_seq[:cutoff]
                tim_seq = tim_seq[:cutoff]
                targets = targets[:cutoff]

        if not targets or not loc_seq:
            continue

        trajectories.append({
            'label': label,
            'uid_idx': uid_idx,
            'loc_seq': loc_seq,
            'tim_seq': tim_seq,
            'targets': targets,
        })

    return trajectories


def ensure_directory(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def process_file(pk_path, output_path, model, model_mode, user_to_idx, params, steps, batch_size):
    data = pickle.load(open(pk_path, 'rb'))
    trajectories = prepare_trajectories(data, user_to_idx, model_mode, steps)

    ensure_directory(output_path)

    with open(output_path, 'w') as out_f:
        out_f.write("tid,digits,rank_proxy\n")

        if not trajectories:
            return 0

        max_seq_len = max(len(entry['targets']) for entry in trajectories)
        total_batches = (len(trajectories) + batch_size - 1) // batch_size

        for start in tqdm(range(0, len(trajectories), batch_size), total=total_batches,
                          desc=os.path.basename(pk_path)):
            batch = trajectories[start:start + batch_size]
            scores_list = batch_step_scores(model, model_mode, batch)

            for entry, scores in zip(batch, scores_list):
                targets = torch.tensor(entry['targets'], dtype=torch.long, device=device)
                if targets.numel() == 0:
                    continue
                ranks = rank_ground_truth(scores, targets)
                proxy = trajectory_rank_proxy(ranks, params.loc_size, max_seq_len)
                num_digits = len(str(proxy))
                out_f.write(f"{entry['label']},{num_digits},{proxy}\n")

    return len(trajectories)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract trajectory rank proxies using DeepMove models")
    parser.add_argument('--data_pk', type=str, default=None, help="path to processed .pk file")
    parser.add_argument('--data_dir', type=str, default=None, help="path to a directory of processed .pk files")
    parser.add_argument('--model_path', type=str, required=True, help="path to trained model")
    parser.add_argument('--model_mode', type=str, required=True,
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--metadata_json', type=str, required=True)
    parser.add_argument('--output', type=str, default=None,
                        help="output CSV file (for --data_pk) or directory (for --data_dir)")
    parser.add_argument('--steps', type=int, default=None,
                        help="maximum number of steps to consider per trajectory")
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')
    if args.data_pk and not args.output:
        parser.error('--output is required when using --data_pk')

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

    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")

        pk_files = sorted(
            os.path.join(args.data_dir, fname)
            for fname in os.listdir(args.data_dir)
            if fname.endswith('.pk')
        )
        if not pk_files:
            raise FileNotFoundError(f"No .pk files found in {args.data_dir}")

        output_dir = args.output or 'extraction'
        os.makedirs(output_dir, exist_ok=True)

        for pk_path in pk_files:
            base = os.path.splitext(os.path.basename(pk_path))[0]
            out_path = os.path.join(output_dir, f"{base}_extraction_proxy.csv")
            print(f"Processing {pk_path} -> {out_path}")
            processed = process_file(
                pk_path,
                out_path,
                model,
                args.model_mode,
                user_to_idx,
                params,
                args.steps,
                args.batch_size,
            )
            if processed == 0:
                print(f"Warning: No valid trajectories found in {pk_path}")
    else:
        processed = process_file(
            args.data_pk,
            args.output,
            model,
            args.model_mode,
            user_to_idx,
            params,
            args.steps,
            args.batch_size,
        )
        if processed == 0:
            raise SystemExit("No valid trajectories found for extraction.")

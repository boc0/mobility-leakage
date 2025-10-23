import os
import argparse
import pickle
from typing import List
from collections import defaultdict

import numpy as np
import torch

from tqdm import tqdm

import train as lstpm_train


def auto_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


def load_data(pk_path):
    data = pickle.load(open(pk_path, 'rb'), encoding='iso-8859-1')
    vid_list = data['vid_list']
    uid_list = data['uid_list']
    data_neural = data['data_neural']
    poi_coordinate = data.get('vid_lookup', {})
    return vid_list, uid_list, data_neural, poi_coordinate


def build_distance_matrix(vid_lookup, n_items):
    fallback = np.full((n_items + 1, n_items + 1), 1e6, dtype=np.float32)
    np.fill_diagonal(fallback, 0.0)
    if not isinstance(vid_lookup, dict) or len(vid_lookup) == 0:
        return fallback
    coerced = {}
    try:
        for k, v in vid_lookup.items():
            coerced[int(k)] = v
    except Exception:
        return fallback
    ok_keys = [k for k in coerced.keys() if 1 <= k <= n_items]
    if not ok_keys:
        return fallback
    mat = np.array(fallback, copy=True)
    for i in ok_keys:
        if i not in coerced:
            continue
        lon_i, lat_i = coerced[i][0], coerced[i][1]
        for j in ok_keys:
            if j < i or j not in coerced:
                continue
            lon_j, lat_j = coerced[j][0], coerced[j][1]
            dist = lstpm_train.geodistance(lon_i, lat_i, lon_j, lat_j)
            if dist < 1:
                dist = 1
            mat[i, j] = dist
            mat[j, i] = dist
    return mat


def trajectory_rank_proxy(ranks: List[int], vocab_size: int, max_len: int, prefix_len: int = 0) -> int:
    total = 0
    for i, r in enumerate(ranks, start=1):
        exponent = max(max_len - prefix_len - i, 0)
        total += (r - 1) * (vocab_size ** exponent)
    return total


def normalize_prefix_lengths(prefix_lengths: List[int]) -> List[int]:
    normalized = []
    seen = set()
    for value in prefix_lengths:
        val = max(int(value), 0)
        if val in seen:
            continue
        seen.add(val)
        normalized.append(val)
    if not normalized:
        raise ValueError("prefix_lengths must contain at least one non-negative integer.")
    return normalized


def ensure_directory(path: str):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def process_file(pk_path, output_path, state_dict, n_items_ckpt, n_users_ckpt, device,
                 prefix_lengths, steps, batch_size, distance_override):
    vid_list, uid_list, data_neural, poi_coordinate = load_data(pk_path)
    n_users = len(uid_list)

    time_sim_matrix = np.asarray(lstpm_train.calculate_time_sim(data_neural), dtype=np.float32)

    if distance_override and os.path.exists(distance_override):
        poi_distance_matrix = pickle.load(open(distance_override, 'rb'))
    else:
        auto_path = os.path.join(os.path.dirname(pk_path), 'distance.pkl')
        if os.path.exists(auto_path):
            poi_distance_matrix = pickle.load(open(auto_path, 'rb'))
        else:
            poi_distance_matrix = build_distance_matrix(poi_coordinate, len(vid_list))
    poi_distance_matrix = np.asarray(poi_distance_matrix, dtype=np.float32)

    if np.any((poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)):
        poi_distance_matrix[(poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)] = 1e-9
        np.fill_diagonal(poi_distance_matrix, 1e-9)

    lstpm_train.data_neural = data_neural
    lstpm_train.poi_distance_matrix = poi_distance_matrix

    max_item_id = max(
        (s[0] for u in data_neural for sid in data_neural[u]['sessions'] for s in data_neural[u]['sessions'][sid]),
        default=-1,
    )
    if max_item_id >= n_items_ckpt:
        raise ValueError(f"Data contains item id {max_item_id} >= checkpoint n_items {n_items_ckpt}. Use a compatible dataset.")

    eff_n_users = n_users_ckpt if n_users_ckpt is not None else n_users

    model = lstpm_train.Model(
        n_users=eff_n_users,
        n_items=n_items_ckpt,
        data_neural=data_neural,
        tim_sim_matrix=time_sim_matrix,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    idx_to_user = {v[0]: k for k, v in uid_list.items()}

    step_limit = None if steps is None else max(int(steps), 0)
    prefix_lengths = normalize_prefix_lengths(prefix_lengths)
    header = ",".join(f"prefix-{p}" for p in prefix_lengths)

    ensure_directory(output_path)
    batch_size = max(int(batch_size), 1)

    samples = []
    user_order = []
    for u_idx in sorted(data_neural.keys()):
        label = idx_to_user.get(u_idx, str(u_idx))
        if label not in user_order:
            user_order.append(label)
        sessions = data_neural[u_idx]['sessions']
        for sid in sorted(sessions.keys()):
            seq = [p[0] for p in sessions[sid]]
            if len(seq) < 2:
                continue
            tim = [lstpm_train.to_tid48(p[1]) for p in sessions[sid]]
            seq_dil = lstpm_train.create_dilated_rnn_input(list(seq), poi_distance_matrix)
            samples.append({
                'user_idx': u_idx,
                'label': label,
                'session_id': sid,
                'seq': seq,
                'tim': tim,
                'seq_dil': seq_dil,
            })

    if not samples:
        with open(output_path, 'w') as out_f:
            out_f.write(f"tid,{header}\n")
        return 0

    user_ranks = defaultdict(list)

    total_batches = (len(samples) + batch_size - 1) // batch_size
    with torch.no_grad():
        for start in tqdm(range(0, len(samples), batch_size), total=total_batches,
                          desc=os.path.basename(pk_path)):
            batch = samples[start:start + batch_size]
            seqs = [s['seq'] for s in batch]
            lengths = [len(seq) for seq in seqs]
            max_len_batch = max(lengths)

            padded_seq, _, mask_non_local = lstpm_train.pad_batch_of_lists_masks(seqs, max_len_batch)
            padded_seq_t = torch.LongTensor(padded_seq).to(device)
            mask_non_local_t = torch.FloatTensor(mask_non_local).to(device)
            user_tensor = torch.LongTensor([s['user_idx'] for s in batch]).to(device)

            session_ids = [s['session_id'] for s in batch]
            tim_lists = [s['tim'] for s in batch]
            seq_dil_batch = [s['seq_dil'] for s in batch]

            padded_tim_t = torch.LongTensor(lstpm_train.pad_batch_of_lists(tim_lists, max_len_batch, pad_value=0)).to(device)
            padded_dilated_t = torch.LongTensor(lstpm_train.pad_batch_of_lists(seq_dil_batch, max_len_batch, pad_value=-1)).to(device)
            session_tensor = torch.LongTensor(session_ids).to(device)

            logp_seq = model(
                user_tensor,
                padded_seq_t,
                mask_non_local_t,
                session_tensor,
                padded_tim_t,
                False,
                poi_distance_matrix,
                padded_dilated_t,
            )
            if logp_seq.dim() == 2:
                logp_seq = logp_seq.unsqueeze(0)
            predictions_logp = logp_seq[:, :-1, :]
            targets_full = padded_seq_t[:, 1:]

            for idx, sample in enumerate(batch):
                tgt_len = lengths[idx] - 1
                if tgt_len <= 0:
                    continue
                preds = predictions_logp[idx, :tgt_len, :]
                tgt = targets_full[idx, :tgt_len]
                target_scores = preds.gather(1, tgt.unsqueeze(1))
                ranks = (preds > target_scores).sum(dim=1) + 1
                ranks_list = ranks.detach().cpu().tolist()

                if step_limit is not None:
                    remaining = step_limit - len(user_ranks[sample['label']])
                    if remaining <= 0:
                        continue
                    ranks_list = ranks_list[:remaining]

                if not ranks_list:
                    continue

                user_ranks[sample['label']].extend(ranks_list)

    valid_entries = [(label, user_ranks[label]) for label in user_order if user_ranks[label]]

    if not valid_entries:
        with open(output_path, 'w') as out_f:
            out_f.write(f"tid,{header}\n")
        return 0

    max_len = max(len(ranks) for _, ranks in valid_entries)

    with open(output_path, 'w') as out_f:
        out_f.write(f"tid,{header}\n")
        for label, ranks in valid_entries:
            proxies = [
                str(trajectory_rank_proxy(ranks, n_items_ckpt, max_len, prefix_len=p))
                for p in prefix_lengths
            ]
            out_f.write(f"{label},{','.join(proxies)}\n")

    return len(valid_entries)


def main():
    parser = argparse.ArgumentParser(description="Extract trajectory rank proxies using LSTPM models")
    parser.add_argument('--data_pk', type=str, default=None, help='Path to a processed .pk file')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to a directory of processed .pk files')
    parser.add_argument('--model_m', type=str, required=True, help='Path to model .m checkpoint')
    parser.add_argument('--output', type=str, default=None, help='Output CSV (for --data_pk) or directory (for --data_dir)')
    parser.add_argument('--distance', type=str, default=None, help='Optional path to distance.pkl')
    parser.add_argument('--steps', type=int, default=None, help='Maximum number of ranks to keep per trajectory')
    parser.add_argument('--prefix_lengths', '--prefix_lens', type=int, nargs='+', default=[0],
                        help='List of prefix lengths for rank proxy computation (non-negative integers)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for parallel processing of sessions')
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')
    if args.data_pk and not args.output:
        parser.error('--output is required when using --data_pk')

    device = auto_device()
    state = torch.load(args.model_m, map_location=device)
    n_items_ckpt = state['item_emb.weight'].shape[0]
    n_users_ckpt = state.get('user_emb.weight', None)
    if isinstance(n_users_ckpt, torch.Tensor):
        n_users_ckpt = n_users_ckpt.shape[0]
    else:
        n_users_ckpt = None

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
                state,
                n_items_ckpt,
                n_users_ckpt,
                device,
                args.prefix_lengths,
                args.steps,
                args.batch_size,
                args.distance,
            )
            if processed == 0:
                print(f"Warning: No valid trajectories found in {pk_path}")
    else:
        processed = process_file(
            args.data_pk,
            args.output,
            state,
            n_items_ckpt,
            n_users_ckpt,
            device,
            args.prefix_lengths,
            args.steps,
            args.batch_size,
            args.distance,
        )
        if processed == 0:
            raise SystemExit("No valid trajectories found for extraction.")


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    main()

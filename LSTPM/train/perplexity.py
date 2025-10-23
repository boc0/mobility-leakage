import argparse
import os
import pickle
import numpy as np
import torch

# Reuse training utilities and Model
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
    """Return a (n_items+1, n_items+1) distance matrix.
    If vid_lookup is missing or not 1..n contiguous, return a fallback matrix
    with large distances off-diagonal and 0 on diagonal to avoid crashes.
    """
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
    if len(ok_keys) == 0:
        return fallback
    mat = np.array(fallback, copy=True)
    for i in ok_keys:
        lon_i, lat_i = coerced[i][0], coerced[i][1]
        for j in ok_keys:
            if j < i:
                continue
            lon_j, lat_j = coerced[j][0], coerced[j][1]
            d = lstpm_train.geodistance(lon_i, lat_i, lon_j, lat_j)
            if d < 1:
                d = 1
            mat[i, j] = d
            mat[j, i] = d
    return mat


def main():
    parser = argparse.ArgumentParser(description="Compute per-trajectory perplexity for an LSTPM model on processed .pk data.")
    parser.add_argument('--data_pk', type=str, default=None, help='Path to a single dataset .pk file')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to a directory containing .pk files')
    parser.add_argument('--model_m', required=True, type=str, help='Path to model .m checkpoint')
    parser.add_argument('--output', type=str, default=None, help='For single file: CSV path. For directory: output directory path.')
    parser.add_argument('--distance', type=str, default=None, help='Optional path to distance.pkl; falls back to file-local or building from vid_lookup')
    parser.add_argument('--verbose', action='store_true', help='If set, print perplexities to console instead of writing to file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for batched session evaluation')
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')

    device = auto_device()
    print(f"Using device: {device}")

    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")
        pk_files = sorted(
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith('.pk')
        )
        if not pk_files:
            raise FileNotFoundError(f"No .pk files found in directory: {args.data_dir}")
    else:
        if not os.path.exists(args.data_pk):
            raise FileNotFoundError(f".pk file not found: {args.data_pk}")
        pk_files = [args.data_pk]

    state = torch.load(args.model_m, map_location=device)
    n_items_ckpt = state['item_emb.weight'].shape[0]
    n_users_ckpt = state['user_emb.weight'].shape[0] if 'user_emb.weight' in state else None

    output_dir = None
    single_out_f = None
    if args.data_dir:
        output_dir = args.output if args.output else 'perplexity'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output for directory processing will be in: {output_dir}")
    else:
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            single_out_f = open(args.output, 'w')
            single_out_f.write("tid,perplexity\n")
        elif not args.verbose:
            print("No output path provided; results will be printed to stdout.")

    for pk_path in pk_files:
        if args.data_dir:
            base = os.path.splitext(os.path.basename(pk_path))[0]
            out_path = os.path.join(output_dir, f"{base}_perplexity.csv")
            if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
                print(f"⏭️ Skipping {pk_path} — already processed at {out_path}")
                continue

        print(f"Processing {pk_path}...")

        vid_list, uid_list, data_neural, poi_coordinate = load_data(pk_path)
        n_items = len(vid_list)
        n_users = len(uid_list)

        time_sim_matrix = np.asarray(lstpm_train.calculate_time_sim(data_neural), dtype=np.float32)

        if args.distance and os.path.exists(args.distance):
            poi_distance_matrix = pickle.load(open(args.distance, 'rb'))
        else:
            auto_path = os.path.join(os.path.dirname(pk_path), 'distance.pkl')
            if os.path.exists(auto_path):
                poi_distance_matrix = pickle.load(open(auto_path, 'rb'))
            else:
                poi_distance_matrix = build_distance_matrix(poi_coordinate, n_items)
        poi_distance_matrix = np.asarray(poi_distance_matrix, dtype=np.float32)

        if np.any((poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)):
            # print("Warning: distance matrix has zero(s) off the diagonal; this may cause instability")
            poi_distance_matrix[(poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)] = 1e-9
            np.fill_diagonal(poi_distance_matrix, 1e-9)

        lstpm_train.data_neural = data_neural
        lstpm_train.poi_distance_matrix = poi_distance_matrix

        max_item_id = max(
            (
                entry[0]
                for user_data in data_neural.values()
                for session in user_data['sessions'].values()
                for entry in session
            ),
            default=-1,
        )
        if max_item_id >= n_items_ckpt:
            raise ValueError(
                f"Data contains item id {max_item_id} >= checkpoint n_items {n_items_ckpt}. Use a compatible dataset."
            )

        eff_n_users = n_users_ckpt if n_users_ckpt is not None else n_users

        model = lstpm_train.Model(
            n_users=eff_n_users,
            n_items=n_items_ckpt,
            data_neural=data_neural,
            tim_sim_matrix=time_sim_matrix,
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        idx_to_user = {v[0]: k for k, v in uid_list.items()}

        out_f = single_out_f
        opened_here = False
        if output_dir is not None:
            base = os.path.splitext(os.path.basename(pk_path))[0]
            out_path = os.path.join(output_dir, f"{base}_perplexity.csv")
            out_f = open(out_path, 'w')
            out_f.write("tid,perplexity\n")
            opened_here = True

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
                samples.append(
                    {
                        'user_idx': u_idx,
                        'label': label,
                        'session_id': sid,
                        'seq': seq,
                        'tim': tim,
                        'seq_dil': seq_dil,
                    }
                )

        if not samples:
            if opened_here and out_f:
                out_f.close()
            continue

        user_logps = {label: [] for label in user_order}
        batch_size = max(int(args.batch_size), 1)

        with torch.no_grad():
            for start in range(0, len(samples), batch_size):
                batch = samples[start:start + batch_size]
                lengths = [len(s['seq']) for s in batch]
                max_len_batch = max(lengths)

                padded_seq, _, mask_non_local = lstpm_train.pad_batch_of_lists_masks(
                    [s['seq'] for s in batch],
                    max_len_batch,
                )
                padded_seq_t = torch.LongTensor(padded_seq).to(device)
                mask_non_local_t = torch.FloatTensor(mask_non_local).to(device)
                user_tensor = torch.LongTensor([s['user_idx'] for s in batch]).to(device)
                padded_tim_t = torch.LongTensor(
                    lstpm_train.pad_batch_of_lists([s['tim'] for s in batch], max_len_batch, pad_value=0)
                ).to(device)
                padded_dilated_t = torch.LongTensor(
                    lstpm_train.pad_batch_of_lists([s['seq_dil'] for s in batch], max_len_batch, pad_value=-1)
                ).to(device)
                session_tensor = torch.LongTensor([s['session_id'] for s in batch]).to(device)

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
                predictions_logp = logp_seq[:, :-1, :]
                targets_full = padded_seq_t[:, 1:]

                for idx, sample in enumerate(batch):
                    tgt_len = lengths[idx] - 1
                    if tgt_len <= 0:
                        continue
                    preds = predictions_logp[idx, :tgt_len, :]
                    tgt = targets_full[idx, :tgt_len]
                    logp_values = preds.gather(1, tgt.unsqueeze(1)).squeeze(1)
                    if logp_values.numel() == 0:
                        continue
                    user_logps[sample['label']].extend(logp_values.detach().cpu().tolist())

        for label in user_order:
            values = user_logps.get(label, [])
            if not values:
                continue
            nll_sum = float(-sum(values))
            tok_count = float(len(values))
            ppl = nll_sum / max(tok_count, 1.0)
            line = f"{label},{ppl:.3f}"
            if out_f:
                out_f.write(line + "\n")
            if args.verbose or not out_f:
                print(line)

        if opened_here and out_f:
            out_f.close()

    if single_out_f:
        single_out_f.close()


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    main()

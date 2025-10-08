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
    # Fallback: large distances, 0 diagonal
    fallback = np.full((n_items + 1, n_items + 1), 1e6, dtype=np.float32)
    np.fill_diagonal(fallback, 0.0)
    if not isinstance(vid_lookup, dict) or len(vid_lookup) == 0:
        return fallback
    # Try to coerce keys to int and ensure 1..n_items coverage when possible
    coerced = {}
    try:
        for k, v in vid_lookup.items():
            ki = int(k)
            coerced[ki] = v
    except Exception:
        return fallback
    ok_keys = [k for k in coerced.keys() if 1 <= k <= n_items]
    if len(ok_keys) == 0:
        return fallback
    mat = np.array(fallback, copy=True)
    for i in ok_keys:
        if i not in coerced:
            continue
        lon_i, lat_i = coerced[i][0], coerced[i][1]
        for j in ok_keys:
            if j < i:
                continue
            if j not in coerced:
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
    parser.add_argument('--merge_sessions', action='store_true', help='Merge all sessions per user before computing perplexity', default=True)
    parser.add_argument('--no-merge', dest='merge_sessions', action='store_false', help='Disable session merging')
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')

    device = auto_device()
    print(f"Using device: {device}")

    # Collect files
    pk_files = []
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")
        pk_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pk')]
        if not pk_files:
            raise FileNotFoundError(f"No .pk files found in directory: {args.data_dir}")
    else:
        if not os.path.exists(args.data_pk):
            raise FileNotFoundError(f".pk file not found: {args.data_pk}")
        pk_files = [args.data_pk]

    # Determine output mode
    output_dir = None
    single_out_f = None
    if args.data_dir:
        output_dir = args.output if args.output else 'perplexities'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output for directory processing will be in: {output_dir}")
    else:
        if args.output:
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            single_out_f = open(args.output, 'w')
            if args.merge_sessions:
                single_out_f.write("tid,perplexity\n")
            else:
                single_out_f.write("tid,session,perplexity\n")

    # Load checkpoint once
    state = torch.load(args.model_m, map_location=device)
    n_items_ckpt = state['item_emb.weight'].shape[0]
    n_users_ckpt = state.get('user_emb.weight', torch.empty(1, 1)).shape[0] if 'user_emb.weight' in state else None

    for pk in pk_files:
        print(f"Processing {pk}...")

        vid_list, uid_list, data_neural, poi_coordinate = load_data(pk)
        n_items = len(vid_list)
        n_users = len(uid_list)

        # Prepare similarity/distance matrices for this file
        time_sim_matrix = lstpm_train.calculate_time_sim(data_neural)
        time_sim_matrix = np.asarray(time_sim_matrix, dtype=np.float32)

        poi_distance_matrix = None
        if args.distance and os.path.exists(args.distance):
            poi_distance_matrix = pickle.load(open(args.distance, 'rb'))
        else:
            auto_path = os.path.join(os.path.dirname(pk), 'distance.pkl')
            if os.path.exists(auto_path):
                poi_distance_matrix = pickle.load(open(auto_path, 'rb'))
            else:
                poi_distance_matrix = build_distance_matrix(poi_coordinate, n_items)
        poi_distance_matrix = np.asarray(poi_distance_matrix, dtype=np.float32)

        if np.any((poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)):
            print("Warning: distance matrix has zero(s) off the diagonal; this may cause instability")
            poi_distance_matrix[(poi_distance_matrix == 0) & ~np.eye(poi_distance_matrix.shape[0], dtype=bool)] = 1e-9
            np.fill_diagonal(poi_distance_matrix, 1e-9)

        # Wire globals expected by forward/evaluate()
        lstpm_train.data_neural = data_neural
        lstpm_train.poi_distance_matrix = poi_distance_matrix

        # Safety: ensure item ids in data are < n_items_ckpt
        max_item_id = max((s[0] for u in data_neural for sid in data_neural[u]['sessions'] for s in data_neural[u]['sessions'][sid]), default=-1)
        if max_item_id >= n_items_ckpt:
            raise ValueError(f"Data contains item id {max_item_id} >= checkpoint n_items {n_items_ckpt}. Use a compatible dataset.")

        # If checkpoint lacks user_emb, fall back to dataset users
        eff_n_users = n_users_ckpt if n_users_ckpt is not None else n_users

        model = lstpm_train.Model(
            n_users=eff_n_users,
            n_items=n_items_ckpt,
            data_neural=data_neural,
            tim_sim_matrix=time_sim_matrix
        ).to(device)
        model.load_state_dict(state)
        model.eval()

        # Build inverse mapping: embedded_idx -> original user label
        idx_to_user = {v[0]: k for k, v in uid_list.items()}

        # Determine output file for this pk
        out_f = single_out_f
        if output_dir is not None:
            base = os.path.splitext(os.path.basename(pk))[0]
            out_path = os.path.join(output_dir, f"{base}_perplexity.csv")
            out_f = open(out_path, 'w')
            if args.merge_sessions:
                out_f.write("tid,perplexity\n")
            else:
                out_f.write("tid,session,perplexity\n")

        with torch.no_grad():
            for u_idx in sorted(data_neural.keys()):
                u_label = idx_to_user.get(u_idx, str(u_idx))
                sessions = data_neural[u_idx]['sessions']

                if args.merge_sessions:
                    all_logps = []
                    for sid in sorted(sessions.keys()):
                        seq = [p[0] for p in sessions[sid]]
                        tim = [lstpm_train.to_tid48(p[1]) for p in sessions[sid]]
                        if len(seq) < 2:
                            continue
                        seq_dil = lstpm_train.create_dilated_rnn_input(list(seq), poi_distance_matrix)
                        max_len = len(seq)
                        padded_seq, mask_batch_ix, mask_batch_non_local = lstpm_train.pad_batch_of_lists_masks([seq], max_len)
                        padded_seq_t = torch.LongTensor(np.array(padded_seq)).to(device)
                        mask_ix_t = torch.FloatTensor(np.array(mask_batch_ix)).to(device)
                        mask_non_local_t = torch.FloatTensor(np.array(mask_batch_non_local)).to(device)
                        user_t = torch.LongTensor(np.array([u_idx])).to(device)

                        logp_seq = model(user_t, padded_seq_t, mask_non_local_t, [sid], [tim], False, poi_distance_matrix, [seq_dil])
                        predictions_logp = logp_seq[:, :-1]
                        targets = padded_seq_t[:, 1:]
                        if predictions_logp.numel() == 0 or targets.numel() == 0:
                            continue
                        logp_next = torch.gather(predictions_logp, dim=2, index=targets[:, :, None]).squeeze(-1)
                        valid_mask = (mask_ix_t[:, :-1] > 0)
                        if valid_mask.any():
                            all_logps.append(logp_next[valid_mask])

                    if len(all_logps) == 0:
                        continue
                    concat_logps = torch.cat(all_logps).float()
                    nll_sum = float((-concat_logps).sum().item())
                    tok_count = float(concat_logps.numel())
                    ppl = nll_sum / max(tok_count, 1.0)
                    line = f"{u_label},{ppl:.3f}"
                    if out_f:
                        out_f.write(line + "\n")
                    else:
                        print(line)
                else:
                    for sid in sorted(sessions.keys()):
                        seq = [p[0] for p in sessions[sid]]
                        tim = [lstpm_train.to_tid48(p[1]) for p in sessions[sid]]
                        if len(seq) < 2:
                            continue
                        seq_dil = lstpm_train.create_dilated_rnn_input(list(seq), poi_distance_matrix)
                        max_len = len(seq)
                        padded_seq, mask_batch_ix, mask_batch_non_local = lstpm_train.pad_batch_of_lists_masks([seq], max_len)
                        padded_seq_t = torch.LongTensor(np.array(padded_seq)).to(device)
                        mask_ix_t = torch.FloatTensor(np.array(mask_batch_ix)).to(device)
                        mask_non_local_t = torch.FloatTensor(np.array(mask_batch_non_local)).to(device)
                        user_t = torch.LongTensor(np.array([u_idx])).to(device)

                        logp_seq = model(user_t, padded_seq_t, mask_non_local_t, [sid], [tim], False, poi_distance_matrix, [seq_dil])
                        predictions_logp = logp_seq[:, :-1] * mask_ix_t[:, :-1, None]
                        targets = padded_seq_t[:, 1:]
                        if predictions_logp.numel() == 0 or targets.numel() == 0:
                            continue
                        logp_next = torch.gather(predictions_logp, dim=2, index=targets[:, :, None])
                        nll_sum = float((-logp_next).sum().item())
                        tok_count = float(mask_ix_t[:, :-1].sum().item())
                        ppl = nll_sum / max(tok_count, 1.0)
                        line = f"{u_label},{sid},{ppl:.3f}"
                        if out_f:
                            out_f.write(line + "\n")
                        else:
                            print(line)

        if output_dir is not None and out_f:
            out_f.close()

    if single_out_f:
        single_out_f.close()


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    main()

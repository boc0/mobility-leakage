import os
import math
import argparse
import pickle
import json

import torch
import torch.nn as nn

from tqdm import tqdm

from train import RnnParameterData
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong

# from IPython import embed

# auto‐select MPS/CPU/CUDA
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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()

def process_trajectory(model, loc_seq, tim_seq, target_seq, model_mode, uid, mode='topk', k_values=[1,5,10]):
    """
    loc_seq, tim_seq: list of int prefix
    target_seq: list of int next‐locations
    """
    locs = torch.LongTensor(loc_seq).unsqueeze(1).to(device)
    tims = torch.LongTensor(tim_seq).unsqueeze(1).to(device)
    tgt  = torch.LongTensor(target_seq).to(device)

    with torch.no_grad():
        if model_mode in ['simple', 'simple_long']:
            scores = model(locs, tims)
        elif model_mode == 'attn_avg_long_user':
            # treat prefix as history too
            history_loc   = locs
            history_tim   = tims
            history_count = [1] * history_loc.size(0)
            uid_tensor    = torch.LongTensor([uid]).to(device)
            target_len    = tgt.size(0)
            scores = model(locs, tims,
                           history_loc, history_tim,
                           history_count, uid_tensor, target_len)
        else:  # attn_local_long
            target_len = tgt.size(0)
            scores = model(locs, tims, target_len)

        # align lengths
        if scores.size(0) > tgt.size(0):
            scores = scores[-tgt.size(0):]

        if mode == 'topk':
            # get top-k accuracy for each k in k_values
            topk = {}
            for k in k_values:
                topk_k = 0
                _, top_indices = scores.topk(k, dim=1)
                for i in range(tgt.size(0)):
                    if tgt[i] in top_indices[i]:
                        topk_k += 1
                topk[k] = topk_k / tgt.size(0)
            return topk
        
        elif mode == 'rank':
            # compute mean rank of true next locations
            ranks = []
            _, indices = scores.sort(dim=1, descending=True)
            for i in range(tgt.size(0)):
                rank = (indices[i] == tgt[i]).nonzero(as_tuple=True)[0].item() + 1
                ranks.append(rank)
            mean_rank = sum(ranks) / len(ranks)
            return mean_rank

        else:
            raise ValueError(f"Unknown mode: {mode}")

def pad_sequences(sequences, pad_value=0):
    """Pad sequences to the same length for batching"""
    if not sequences:
        return torch.tensor([])
    
    max_len = max(len(seq) for seq in sequences)
    padded = []
    masks = []
    
    for seq in sequences:
        pad_len = max_len - len(seq)
        padded_seq = seq + [pad_value] * pad_len
        mask = [1] * len(seq) + [0] * pad_len
        padded.append(padded_seq)
        masks.append(mask)
    
    return torch.LongTensor(padded), torch.BoolTensor(masks)

def batch_process_trajectories(model, batch_data, model_mode, device, mode='topk', k_values=[1,5,10]):
    """
    Process a batch of trajectories
    batch_data: list of (loc_seq, tim_seq, target_seq, uid_idx) tuples
    """
    if not batch_data:
        return []
    
    # Separate the batch components
    loc_seqs, tim_seqs, target_seqs, uid_indices = zip(*batch_data)
    batch_size = len(batch_data)
    max_k = max(k_values) if k_values else 1

    def empty_result():
        if mode == 'topk':
            return {k: 0.0 for k in k_values}
        return float('inf')

    with torch.no_grad():
        if model_mode in ['simple', 'simple_long']:
            padded_locs, loc_masks = pad_sequences(loc_seqs)
            padded_tims, _ = pad_sequences(tim_seqs)
            padded_targets, target_masks = pad_sequences(target_seqs)

            padded_locs = padded_locs.unsqueeze(2).to(device)
            padded_tims = padded_tims.unsqueeze(2).to(device)
            padded_targets = padded_targets.to(device)
            target_masks = target_masks.to(device)

            loc_batch = padded_locs.transpose(0, 1).squeeze(-1).contiguous()
            tim_batch = padded_tims.transpose(0, 1).squeeze(-1).contiguous()
            scores = model(loc_batch, tim_batch)
            if scores.dim() == 2:
                scores = scores.unsqueeze(1)
            scores = scores.transpose(0, 1).contiguous()

            results = []
            for i in range(batch_size):
                target_len = target_masks[i].sum().item()
                if target_len == 0:
                    results.append(empty_result())
                    continue

                seq_scores = scores[i][:target_len]
                seq_targets = padded_targets[i][:target_len]

                if mode == 'topk':
                    metrics = {}
                    top_indices = seq_scores.topk(max_k, dim=1).indices
                    target_exp = seq_targets.unsqueeze(1)
                    for k in k_values:
                        hits = (top_indices[:, :k] == target_exp).any(dim=1).float()
                        metrics[k] = hits.mean().item()
                    results.append(metrics)
                else:
                    sorted_idx = seq_scores.argsort(dim=1, descending=True)
                    matches = (sorted_idx == seq_targets.unsqueeze(1))
                    ranks = matches.float().argmax(dim=1).float() + 1.0
                    results.append(ranks.mean().item())

            return results

        results = [empty_result() for _ in range(batch_size)]

        if model_mode == 'attn_avg_long_user':
            groups = {}
            for idx, targets in enumerate(target_seqs):
                tgt_len = len(targets)
                if tgt_len == 0:
                    continue
                groups.setdefault(tgt_len, []).append(idx)

            for tgt_len, indices in groups.items():
                seq_max = max(len(loc_seqs[i]) for i in indices)
                group_size = len(indices)

                loc_batch = torch.zeros(seq_max, group_size, dtype=torch.long, device=device)
                tim_batch = torch.zeros(seq_max, group_size, dtype=torch.long, device=device)
                history_counts = []

                for col, data_idx in enumerate(indices):
                    loc_seq = torch.tensor(loc_seqs[data_idx], dtype=torch.long, device=device)
                    tim_seq = torch.tensor(tim_seqs[data_idx], dtype=torch.long, device=device)
                    length = loc_seq.size(0)
                    loc_batch[:length, col] = loc_seq
                    tim_batch[:length, col] = tim_seq
                    history_counts.append([1] * length)

                uid_tensor = torch.tensor([uid_indices[i] for i in indices], dtype=torch.long, device=device)
                target_lengths = [len(target_seqs[i]) for i in indices]

                scores = model(
                    loc_batch,
                    tim_batch,
                    loc_batch,
                    tim_batch,
                    history_counts,
                    uid_tensor,
                    target_lengths,
                )

                if scores.dim() == 2:
                    scores = scores.unsqueeze(1)
                scores = scores.transpose(0, 1).contiguous()

                for col, data_idx in enumerate(indices):
                    targets = torch.tensor(target_seqs[data_idx], dtype=torch.long, device=device)
                    seq_scores = scores[col][-targets.size(0):]

                    if mode == 'topk':
                        metrics = {}
                        top_indices = seq_scores.topk(max_k, dim=1).indices
                        target_exp = targets.unsqueeze(1)
                        for k in k_values:
                            hits = (top_indices[:, :k] == target_exp).any(dim=1).float()
                            metrics[k] = hits.mean().item()
                        results[data_idx] = metrics
                    else:
                        sorted_idx = seq_scores.argsort(dim=1, descending=True)
                        matches = (sorted_idx == targets.unsqueeze(1))
                        ranks = matches.float().argmax(dim=1).float() + 1.0
                        results[data_idx] = ranks.mean().item()

            return results

        # attn_local_long
        groups = {}
        for idx, targets in enumerate(target_seqs):
            tgt_len = len(targets)
            if tgt_len == 0:
                continue
            groups.setdefault(tgt_len, []).append(idx)

        for tgt_len, indices in groups.items():
            seq_max = max(len(loc_seqs[i]) for i in indices)
            group_size = len(indices)

            loc_batch = torch.zeros(seq_max, group_size, dtype=torch.long, device=device)
            tim_batch = torch.zeros(seq_max, group_size, dtype=torch.long, device=device)

            for col, data_idx in enumerate(indices):
                loc_seq = torch.tensor(loc_seqs[data_idx], dtype=torch.long, device=device)
                tim_seq = torch.tensor(tim_seqs[data_idx], dtype=torch.long, device=device)
                length = loc_seq.size(0)
                loc_batch[:length, col] = loc_seq
                tim_batch[:length, col] = tim_seq

            scores = model(loc_batch, tim_batch, tgt_len)

            if scores.dim() == 2:
                scores = scores.unsqueeze(1)
            scores = scores.transpose(0, 1).contiguous()

            for col, data_idx in enumerate(indices):
                targets = torch.tensor(target_seqs[data_idx], dtype=torch.long, device=device)
                seq_scores = scores[col][-targets.size(0):]

                if mode == 'topk':
                    metrics = {}
                    top_indices = seq_scores.topk(max_k, dim=1).indices
                    target_exp = targets.unsqueeze(1)
                    for k in k_values:
                        hits = (top_indices[:, :k] == target_exp).any(dim=1).float()
                        metrics[k] = hits.mean().item()
                    results[data_idx] = metrics
                else:
                    sorted_idx = seq_scores.argsort(dim=1, descending=True)
                    matches = (sorted_idx == targets.unsqueeze(1))
                    ranks = matches.float().argmax(dim=1).float() + 1.0
                    results[data_idx] = ranks.mean().item()

        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Test a pretrained DeepMove model."
    )
    parser.add_argument('--data_pk',     type=str, default=None,
                        help="path to a single processed .pk file")
    parser.add_argument('--data_dir',    type=str, default=None,
                        help="path to a directory of processed .pk files")
    parser.add_argument('--model_path',  type=str, required=True,
                        help="checkpoint .m file to load")
    parser.add_argument('--model_mode',  type=str, default='attn_avg_long_user',
                        choices=['simple','simple_long','attn_avg_long_user','attn_local_long'])
    parser.add_argument('--output',      type=str, default=None,
                        help="optional output. For a single file, a CSV file path. For a directory, an output directory path.")
    parser.add_argument('--metadata_json', type=str, default=None,
                        help="path to metadata json file (required for correct model size)")
    parser.add_argument('--merge_sessions', action='store_true',
                        help="merge all sessions per user into one long sequence before scoring", default=True)
    parser.add_argument('--no_merge', dest='merge_sessions', action='store_false',
                        help="do not merge sessions; score each session separately")
    parser.add_argument('--mode', choices=['topk', 'rank'], default='topk', help='Whether to get top-k accuracy (topk or rank)')
    # add additional argument for mode topk to define the k values
    parser.add_argument('--k_values', '--ks', type=int, nargs='+', default=[1,5,10], help='List of k values for top-k accuracy')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Also print each result line to stdout even when writing to an output file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for parallel processing of trajectories')
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')
    if not args.metadata_json:
        parser.error('--metadata_json is required')

    # get file list
    pk_files = []
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")
        pk_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pk')]
        if not pk_files:
            print(f"Warning: No .pk files found in {args.data_dir}")
    else:
        if not os.path.exists(args.data_pk):
            raise FileNotFoundError(f"File not found: {args.data_pk}")
        pk_files.append(args.data_pk)

    # build parameter object
    params = RnnParameterData(metadata=args.metadata_json)
    params.model_mode  = args.model_mode
    params.use_cuda    = (device.type != 'cpu')
    params.rnn_type    = getattr(params, 'rnn_type', 'LSTM')
    params.attn_type   = getattr(params, 'attn_type', 'dot')
    params.dropout_p   = getattr(params, 'dropout_p', 0.3)
    params.tim_size    = getattr(params, 'tim_size', 48) # Use default, since it's not in metadata

    # load metadata for sizes
    meta = json.load(open(args.metadata_json, 'r'))
    params.loc_size = len(meta.get('pid_mapping', {})) + 1
    params.uid_size = len(meta.get('users', []))
    if params.loc_size == 0 or params.uid_size == 0:
        raise ValueError("metadata.json is missing 'pid_mapping' or 'users' information.")
    # build user label → index mapping
    user_to_idx = {u: i for i, u in enumerate(meta.get('users', []))}

    # load model once
    model = load_model(params, args.model_mode, args.model_path)

    # --- Output Handling ---
    output_dir = None
    if args.data_dir:
        output_dir = args.output if args.output else 'likelihoods'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output for directory processing will be in: {output_dir}")

    # --- Main Processing Loop ---
    single_out_f = None
    if args.data_pk and args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        single_out_f = open(args.output, 'w')
        if args.mode == 'topk':
            header = "tid," + ",".join([f"top-{k}" for k in args.k_values]) + "\n"
            single_out_f.write(header)
        else:
            single_out_f.write("tid,mean_rank\n")

    for pk_file in pk_files:
        print(f"Processing {pk_file}...")

        # Determine current output file handle
        out_f = None
        if args.data_dir:
            basename = os.path.basename(pk_file)
            out_name = os.path.splitext(basename)[0] + '.csv'
            out_path = os.path.join(output_dir, out_name)
            out_f = open(out_path, 'w')
            if args.mode == 'topk':
                header = "tid," + ",".join([f"top-{k}" for k in args.k_values]) + "\n"
                out_f.write(header)
            else:
                out_f.write("tid,mean_rank\n")
        else: # single file mode
            out_f = single_out_f

        data = pickle.load(open(pk_file, 'rb'))
        sessions_all = data['data_neural']
        uid_list = data['uid_list']  # mapping: original_label -> [embedded_idx]
        # Build mapping: embedded_idx -> original_label
        idx_to_user = {v[0]: k for k, v in uid_list.items()}

        # Build metadata user->idx using string keys to avoid type mismatches
        # (metadata users may be strings; uid_list keys may be ints)
        # user_to_idx is already built above; rebuild with str keys for safety:
        meta = json.load(open(args.metadata_json, 'r'))
        user_to_idx = {str(u): i for i, u in enumerate(meta.get('users', []))}

        # Collect all trajectories for batch processing
        trajectories_batch = []
        labels_batch = []
        
        for u_idx, udata in sessions_all.items():
            # map embedded uid index -> original user label -> metadata uid index
            label = idx_to_user.get(u_idx, None)
            uid_idx = user_to_idx.get(str(label), None)
            if uid_idx is None:
                # User from pk not present in metadata.json; skip
                continue

            # Merge all sessions into one long sequence (chronological by session id)
            sess_ids = sorted(udata['sessions'].keys())
            merged = []
            for sid in sess_ids:
                merged.extend(udata['sessions'][sid])
            if len(merged) < 2:
                continue
            locs = [p[0] for p in merged]
            tims = [p[1] for p in merged]
            if args.model_mode == 'attn_local_long':
                loc_seq = locs
                tim_seq = tims
            else:
                loc_seq = locs[:-1]
                tim_seq = tims[:-1]
            target_loc = locs[1:]
            trajectories_batch.append((loc_seq, tim_seq, target_loc, uid_idx))
            labels_batch.append(label)
        
        # Process trajectories in batches
        for i in range(0, len(trajectories_batch), args.batch_size):
            batch = trajectories_batch[i:i+args.batch_size]
            batch_labels = labels_batch[i:i+args.batch_size]
            
            results = batch_process_trajectories(model, batch, args.model_mode, device, 
                                               mode=args.mode, k_values=args.k_values)
            
            for label, result in zip(batch_labels, results):
                if args.mode == 'topk':
                    values = [str(result[k]) for k in args.k_values]
                    line = f"{label}," + ",".join(values)
                else:
                    line = f"{label},{result}"
                
                if out_f:
                    out_f.write(line + "\n")
                if (not out_f) or args.verbose:
                    print(line)


        # Close file if in directory mode
        if args.data_dir and out_f:
            out_f.close()

    # Close file if in single file mode
    if single_out_f:
        single_out_f.close()

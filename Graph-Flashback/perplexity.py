import argparse
import os
import json
import pickle
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.sparse import coo_matrix, csr_matrix

from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from network import create_h0_strategy, RnnFactory


def load_graph(path, to="coo"):
    with open(path, 'rb') as f:
        g = pickle.load(f)
    return coo_matrix(g) if to == "coo" else csr_matrix(g)


def evaluate_perplexity(args, data_file, device, users_original):
    # Collect all user integer IDs present in the raw TXT before any filtering
    raw_user_ints = set()
    with open(data_file, 'r') as rf_raw:
        for line in rf_raw:
            parts = line.strip().split('\t')
            if len(parts) != 5:
                continue
            try:
                raw_user_ints.add(int(parts[0]))
            except ValueError:
                continue

    # Low min_checkins to include all users
    poi_loader = PoiDataloader(max_users=0, min_checkins=1)
    poi_loader.read(data_file)
    # Cap batch size by number of available users in this file
    eval_batch_size = min(args.batch_size, poi_loader.user_count()) if poi_loader.user_count() > 0 else 1
    dataset_test = poi_loader.create_dataset(
        sequence_length=args.sequence_length,
        batch_size=eval_batch_size,
        split=Split.TEST
    )
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    # Load graphs
    transition_graph = load_graph(args.trans_loc_file, to="coo")
    spatial_graph = load_graph(args.trans_loc_spatial_file, to="coo") if (args.use_spatial_graph and args.trans_loc_spatial_file) else None
    friend_graph = load_graph(args.trans_user_file, to="coo") if (args.use_graph_user and args.trans_user_file) else None
    interact_graph = load_graph(args.trans_interact_file, to="csr")

    # Local->global user id mapping
    local_to_global = [None] * len(poi_loader.user2id)
    for orig_uid, local_idx in poi_loader.user2id.items():
        if local_idx < len(local_to_global):
            local_to_global[local_idx] = orig_uid

    # Load checkpoint
    ckpt = torch.load(args.model_path, map_location=device)
    hidden_dim = ckpt.get('hidden_dim', args.hidden_dim)
    loc_count = ckpt.get('loc_count', transition_graph.shape[0])
    user_count = ckpt.get('user_count', interact_graph.shape[0])

    if poi_loader.locations() > loc_count:
        raise ValueError(
            f"Test file {data_file} has {poi_loader.locations()} POIs but model expects {loc_count}. "
            "Re-run preprocessing including this file before training to extend the global POI vocabulary.")

    rnn_factory = RnnFactory('rnn')
    is_lstm = rnn_factory.is_lstm()
    trainer = FlashbackTrainer(
        lambda_t=ckpt.get('lambda_t', 0.1),
        lambda_s=ckpt.get('lambda_s', 1000),
        lambda_loc=ckpt.get('lambda_loc', 1.0),
        lambda_user=ckpt.get('lambda_user', 1.0),
        use_weight=False,
        transition_graph=transition_graph,
        spatial_graph=spatial_graph,
        friend_graph=friend_graph,
        use_graph_user=args.use_graph_user,
        use_spatial_graph=args.use_spatial_graph,
        interact_graph=interact_graph,
    )
    trainer.prepare(loc_count, user_count, hidden_dim, rnn_factory, device)
    trainer.model.load_state_dict(ckpt['model_state_dict'])

    h0_strategy = create_h0_strategy(hidden_dim, is_lstm)
    dataset_test.reset()
    h = h0_strategy.on_init(eval_batch_size, device)

    # Accumulators per local user index
    nll_sum = np.zeros(poi_loader.user_count(), dtype=np.float64)
    token_counts = np.zeros(poi_loader.user_count(), dtype=np.float64)
    reset_count = torch.zeros(poi_loader.user_count())

    with torch.no_grad():
        for batch in dataloader_test:
            x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users = batch
            active_users = active_users.squeeze()
            # Reset hidden state for new users
            for j, reset in enumerate(reset_h):
                if reset:
                    if is_lstm:
                        hc = h0_strategy.on_reset(active_users[j])
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        h[0, j] = h0_strategy.on_reset(active_users[j])
                    reset_count[active_users[j]] += 1

            # Move tensors
            x = x.squeeze().to(device)
            t = t.squeeze().to(device)
            t_slot = t_slot.squeeze().to(device)
            s = s.squeeze().to(device)
            y = y.squeeze()  # target labels (seq_len, batch)
            y_t = y_t.squeeze().to(device)
            y_t_slot = y_t_slot.squeeze().to(device)
            y_s = y_s.squeeze().to(device)
            active_users = active_users.to(device)

            out, h = trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)
            # out shape: (batch, seq_len, loc_count)
            batch_size = eval_batch_size
            for j in range(batch_size):
                u_local = int(active_users[j].item())
                if reset_count[u_local] > 1:  # already processed full sequence once
                    continue
                seq_logits = out[j]  # (seq_len, loc_count)
                seq_true = y[:, j]
                # Convert to log probabilities
                log_probs = F.log_softmax(seq_logits, dim=-1)
                for k in range(seq_true.size(0)):
                    true_label = int(seq_true[k].item())
                    nll = -log_probs[k, true_label].item()
                    nll_sum[u_local] += nll
                    token_counts[u_local] += 1

    # Build global mapping results
    results_global = {}
    for local_idx in range(poi_loader.user_count()):
        global_uid = local_to_global[local_idx]
        denom = token_counts[local_idx]
        if denom > 0:
            ppl = math.exp(nll_sum[local_idx] / denom)
        else:
            ppl = float('nan')
        results_global[global_uid] = ppl

    # Add users missing due to filtering with NaN perplexity
    missing_globals = raw_user_ints - set(results_global.keys())
    for g in missing_globals:
        results_global[g] = float('nan')

    return results_global


def write_results(output_dir, base_name, results, users_original):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_perplexity.csv")
    with open(out_path, 'w') as f:
        f.write("tid,perplexity\n")
        for global_uid, ppl in sorted(results.items()):
            orig_id = users_original[global_uid] if global_uid is not None and global_uid < len(users_original) else str(global_uid)
            if isinstance(ppl, float) and (math.isnan(ppl) or math.isinf(ppl)):
                f.write(f"{orig_id},\n")
            else:
                f.write(f"{orig_id},{ppl:.6f}\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compute per-user sequence perplexity for Graph-Flashback model.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--data_file', type=str, help='path to a single TXT dataset file')
    src.add_argument('--data_dir', type=str, help='path to a directory with TXT dataset files')
    parser.add_argument('--model_path', type=str, required=True, help='checkpoint .pt file to load')
    parser.add_argument('--trans_loc_file', type=str, default='', help='transition POI graph (pickle). If omitted, infer per-file graph by name.')
    parser.add_argument('--trans_interact_file', type=str, default='', help='user-POI interact graph (pickle). If omitted, infer per-file graph by name.')
    parser.add_argument('--trans_loc_spatial_file', type=str, default='', help='spatial POI graph (optional)')
    parser.add_argument('--trans_user_file', type=str, default='', help='user-user friend graph (optional)')
    parser.add_argument('--use_spatial_graph', action='store_true', default=False)
    parser.add_argument('--use_graph_user', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, required=True, help='directory to save CSV results')
    parser.add_argument('--sequence_length', type=int, default=20, help='sequence length used to build dataset')
    parser.add_argument('--batch_size', type=int, default=200, help='internal user batch size')
    parser.add_argument('--hidden_dim', type=int, default=10, help='hidden dimension (fallback if not in checkpoint)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use; -1 for CPU')
    args = parser.parse_args()

    device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    if args.data_file:
        if not os.path.isfile(args.data_file):
            raise FileNotFoundError(args.data_file)
        base_dir = os.path.dirname(args.data_file)
        files = [args.data_file]
    else:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(args.data_dir)
        base_dir = args.data_dir
        files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.txt') and 'union' not in f]
        files.sort()

    metadata_path = os.path.join(base_dir, 'metadata.json')
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.json not found in {base_dir}; required for original id mapping")
    with open(metadata_path, 'r') as f_md:
        metadata = json.load(f_md)
    users_original = metadata.get('users', [])
    if not users_original:
        raise ValueError("metadata.json missing 'users' list for original id mapping")

    for fpath in files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        # Resolve per-file graph paths when not specified
        trans_loc = args.trans_loc_file if args.trans_loc_file else os.path.join(base_dir, f'trans_loc_{base}.pkl')
        if not os.path.isfile(trans_loc):
            raise FileNotFoundError(f"Per-file graph not found: {trans_loc}. Provide --trans_loc_file or generate per-file graphs.")
        trans_interact = args.trans_interact_file if args.trans_interact_file else os.path.join(base_dir, f'trans_interact_{base}.pkl')
        if not os.path.isfile(trans_interact):
            raise FileNotFoundError(f"Per-file graph not found: {trans_interact}. Provide --trans_interact_file or generate per-file graphs.")

        # Clone args with resolved graph paths
        class A: pass
        a = A()
        for k, v in vars(args).items():
            setattr(a, k, v)
        setattr(a, 'trans_loc_file', trans_loc)
        setattr(a, 'trans_interact_file', trans_interact)

        results = evaluate_perplexity(a, fpath, device, users_original)
        out_csv = write_results(args.output_dir, base, results, users_original)
        print(f"Wrote results: {out_csv}")


if __name__ == '__main__':
    main()
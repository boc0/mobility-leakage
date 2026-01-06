import argparse
import os
import json
import pickle
import numpy as np
import torch
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


def evaluate_file(args, data_file, device, users_original):
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

    # Load dataset (test split) (PoiDataloader may filter users by min_checkins/sequence length)
    # Use a low min_checkins to ensure all users present in the TXT are evaluated
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
    spatial_graph = None
    if args.use_spatial_graph and args.trans_loc_spatial_file:
        spatial_graph = load_graph(args.trans_loc_spatial_file, to="coo")
    friend_graph = None
    if args.use_graph_user and args.trans_user_file:
        friend_graph = load_graph(args.trans_user_file, to="coo")
    interact_graph = load_graph(args.trans_interact_file, to="csr")

    # Build mapping from local (re-enumerated) user indices back to original global user ids used in preprocessing
    # poi_loader.user2id maps original_user_int -> local_index
    local_to_global = [None] * len(poi_loader.user2id)
    for orig_uid, local_idx in poi_loader.user2id.items():
        if local_idx < len(local_to_global):
            local_to_global[local_idx] = orig_uid

    # Build trainer/model from checkpoint metadata
    ckpt = torch.load(args.model_path, map_location=device)
    hidden_dim = ckpt.get('hidden_dim', args.hidden_dim)
    loc_count = ckpt.get('loc_count', transition_graph.shape[0])
    user_count = ckpt.get('user_count', interact_graph.shape[0])

    # After loading model vocab sizes, ensure test file doesn't exceed them
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

    # Inference loop mirrors Evaluation but collects per-user results
    h0_strategy = create_h0_strategy(hidden_dim, is_lstm)
    dataset_test.reset()
    h = h0_strategy.on_init(eval_batch_size, device)

    # Accumulators
    reset_count = torch.zeros(poi_loader.user_count())
    user_topk_hits = {k: np.zeros(poi_loader.user_count(), dtype=np.float64) for k in args.k_values}
    user_denoms = np.zeros(poi_loader.user_count(), dtype=np.float64)
    user_ranks_sum = np.zeros(poi_loader.user_count(), dtype=np.float64)

    with torch.no_grad():
        for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users) in enumerate(dataloader_test):
            active_users = active_users.squeeze()
            # reset hidden states for newly added users
            for j, reset in enumerate(reset_h):
                if reset:
                    if is_lstm:
                        hc = h0_strategy.on_reset(active_users[j])
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
                    else:
                        # non-LSTM h shape: (1, batch_size, hidden_size)
                        h[0, j] = h0_strategy.on_reset(active_users[j])
                    reset_count[active_users[j]] += 1

            # move to device
            x = x.squeeze().to(device)
            t = t.squeeze().to(device)
            t_slot = t_slot.squeeze().to(device)
            s = s.squeeze().to(device)
            y = y.squeeze()
            y_t = y_t.squeeze().to(device)
            y_t_slot = y_t_slot.squeeze().to(device)
            y_s = y_s.squeeze().to(device)
            active_users = active_users.to(device)

            out, h = trainer.evaluate(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users)

            # iterate users in the active batch
            for j in range(active_users.size(0)):
                u = int(active_users[j].item())
                if reset_count[u] > 1:
                    continue  # already evaluated fully once
                o = out[j]  # (seq_len, loc_count)
                o_n = o.cpu().detach().numpy()
                y_j = y[:, j]

                # compute rank list once per timestep
                if args.mode == 'topk':
                    topK = max(args.k_values)
                    ind = np.argpartition(o_n, -topK, axis=1)[:, -topK:]
                    for k in range(len(y_j)):
                        t_label = int(y_j[k].item())
                        ind_k = ind[k]
                        r = ind_k[np.argsort(-o_n[k, ind_k], axis=0)]
                        user_denoms[u] += 1
                        for kv in args.k_values:
                            user_topk_hits[kv][u] += (t_label in r[:kv])
                else:  # rank
                    # get descending ranks indices per timestep
                    order = np.argsort(-o_n, axis=1)
                    for k in range(len(y_j)):
                        t_label = int(y_j[k].item())
                        # rank is 1-based index in sorted list
                        rank = int(np.where(order[k] == t_label)[0][0]) + 1
                        user_denoms[u] += 1
                        user_ranks_sum[u] += rank

    # Build per-user metrics dict
    results = {}
    for u in range(poi_loader.user_count()):
        denom = user_denoms[u] if user_denoms[u] > 0 else 1.0
        if args.mode == 'topk':
            results[u] = {kv: float(user_topk_hits[kv][u] / denom) for kv in args.k_values}
        else:
            results[u] = float(user_ranks_sum[u] / denom)

    # Convert local-index keyed results to global user id keyed results
    results_global = {}
    for local_idx, metrics in results.items():
        global_uid = local_to_global[local_idx]
        results_global[global_uid] = metrics

    # Add missing users (present in TXT but filtered out) with zero metrics
    missing_globals = raw_user_ints - set(results_global.keys())
    if missing_globals:
        if args.mode == 'topk':
            for g in missing_globals:
                results_global[g] = {kv: 0.0 for kv in args.k_values}
        else:
            for g in missing_globals:
                results_global[g] = 0.0

    return results_global


def write_results(output_dir, base_name, mode, results, k_values, users_original):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{base_name}_{mode}.csv")
    with open(out_path, 'w') as f:
        if mode == 'topk':
            header = "tid," + ",".join([f"top-{k}" for k in k_values]) + "\n"
            f.write(header)
            for global_uid, metrics in sorted(results.items()):
                orig_id = users_original[global_uid] if global_uid is not None and global_uid < len(users_original) else str(global_uid)
                row = [str(orig_id)] + [f"{metrics[k]:.6f}" for k in k_values]
                f.write(",".join(row) + "\n")
        else:
            f.write("tid,mean_rank\n")
            for global_uid, mean_rank in sorted(results.items()):
                orig_id = users_original[global_uid] if global_uid is not None and global_uid < len(users_original) else str(global_uid)
                f.write(f"{orig_id},{mean_rank:.6f}\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate a pretrained Graph-Flashback model on TXT data.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument('--data_file', type=str, help='path to a single TXT dataset file')
    src.add_argument('--data_dir', type=str, help='path to a directory with TXT dataset files')
    parser.add_argument('--model_path', type=str, required=True, help='checkpoint .pt file to load')
    parser.add_argument('--trans_loc_file', type=str, default='', help='transition POI graph (pickle). If omitted, infer per-file graph by name in the data directory.')
    parser.add_argument('--trans_interact_file', type=str, default='', help='user-POI interact graph (pickle). If omitted, infer per-file graph by name in the data directory.')
    parser.add_argument('--trans_loc_spatial_file', type=str, default='', help='spatial POI graph (optional)')
    parser.add_argument('--trans_user_file', type=str, default='', help='user-user friend graph (optional)')
    parser.add_argument('--use_spatial_graph', action='store_true', default=False)
    parser.add_argument('--use_graph_user', action='store_true', default=False)
    parser.add_argument('--output_dir', type=str, required=True, help='directory to save CSV results')
    parser.add_argument('--mode', choices=['topk', 'rank'], default='topk', help='metric to compute')
    parser.add_argument('--k_values', '--ks', type=int, nargs='+', default=[1, 5, 10], help='K values for top-k')
    parser.add_argument('--batch_size', type=int, default=200, help='internal user batch size')
    parser.add_argument('--sequence_length', type=int, default=20, help='sequence length used to build dataset')
    parser.add_argument('--hidden_dim', type=int, default=10, help='hidden dimension (fallback if not in checkpoint)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to use; -1 for CPU')
    args = parser.parse_args()

    device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    files = []
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

    # locate metadata.json
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
        if args.trans_loc_file:
            trans_loc = args.trans_loc_file
        else:
            trans_loc = os.path.join(base_dir, f'trans_loc_{base}.pkl')
            if not os.path.isfile(trans_loc):
                raise FileNotFoundError(f"Per-file graph not found: {trans_loc}. Provide --trans_loc_file or generate per-file graphs.")

        if args.trans_interact_file:
            trans_interact = args.trans_interact_file
        else:
            trans_interact = os.path.join(base_dir, f'trans_interact_{base}.pkl')
            if not os.path.isfile(trans_interact):
                raise FileNotFoundError(f"Per-file graph not found: {trans_interact}. Provide --trans_interact_file or generate per-file graphs.")

        # Clone args with resolved paths for this file
        class A: pass
        a = A()
        for k, v in vars(args).items():
            setattr(a, k, v)
        setattr(a, 'trans_loc_file', trans_loc)
        setattr(a, 'trans_interact_file', trans_interact)

        results = evaluate_file(a, fpath, device, users_original)
        out_csv = write_results(args.output_dir, base, args.mode, results, args.k_values, users_original)
        print(f"Wrote results: {out_csv}")


if __name__ == '__main__':
    main()

import argparse
import os
import pickle
import numpy as np
import torch

# Import training components to reuse Model and evaluate utilities
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
    # Try to coerce keys to int and ensure 1..n_items-1 coverage when possible
    coerced = {}
    try:
        for k, v in vid_lookup.items():
            ki = int(k)
            coerced[ki] = v
    except Exception:
        return fallback
    # Minimal sanity: have entries for some of the range 1..n_items-1
    ok_keys = [k for k in coerced.keys() if 1 <= k <= n_items]
    if len(ok_keys) == 0:
        return fallback
    # Build matrix using available coords; fill missing pairs with large distance
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
            # Use train’s geodistance helper via lstpm_train
            d = lstpm_train.geodistance(lon_i, lat_i, lon_j, lat_j)
            if d < 1:
                d = 1
            mat[i, j] = d
            mat[j, i] = d
    return mat


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTPM model on a dataset (.pk) and model checkpoint (.m)")
    parser.add_argument('--data_pk', required=True, type=str, help='Path to dataset .pk file')
    parser.add_argument('--model_m', required=True, type=str, help='Path to model .m checkpoint')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for evaluation')
    args = parser.parse_args()

    device = auto_device()
    print(f"Using device: {device}")

    # Load dataset
    vid_list, uid_list, data_neural, poi_coordinate = load_data(args.data_pk)
    n_items = len(vid_list)
    n_users = len(uid_list)

    # Prepare similarity/distance matrices
    time_sim_matrix = lstpm_train.calculate_time_sim(data_neural)
    # ensure float32
    time_sim_matrix = np.asarray(time_sim_matrix, dtype=np.float32)

    # Build POI distance matrix from vid_lookup when possible, else fallback
    poi_distance_matrix = build_distance_matrix(poi_coordinate, n_items)
    poi_distance_matrix = np.asarray(poi_distance_matrix, dtype=np.float32)

    # Wire globals expected by evaluate()
    lstpm_train.data_neural = data_neural
    lstpm_train.poi_distance_matrix = poi_distance_matrix

    # Build model and load weights
    model = lstpm_train.Model(n_users=n_users, n_items=n_items, data_neural=data_neural, tim_sim_matrix=time_sim_matrix).to(device)
    # Build model with sizes from checkpoint to avoid shape mismatch
    state = torch.load(args.model_m, map_location=device)
    n_items_ckpt = state['item_emb.weight'].shape[0]
    # Optional: infer users if user_emb exists; else fall back to dataset count
    n_users_ckpt = state.get('user_emb.weight', torch.empty(n_users, 1)).shape[0] if 'user_emb.weight' in state else n_users

    # Safety check: all item ids in data_neural must be < n_items_ckpt
    max_item_id = max(
        (s[0] for u in data_neural for sid in data_neural[u]['sessions'] for s in data_neural[u]['sessions'][sid]),
        default=-1
    )
    if max_item_id >= n_items_ckpt:
        raise ValueError(f"Data contains item id {max_item_id} >= checkpoint n_items {n_items_ckpt}. "
                         f"Use a dataset built with the training pid mapping.")

    model = lstpm_train.Model(
        n_users=n_users_ckpt,
        n_items=n_items_ckpt,
        data_neural=data_neural,
        tim_sim_matrix=time_sim_matrix
    ).to(device)

    model.load_state_dict(state)

    # Evaluate
    results = lstpm_train.evaluate(model, batch_size=args.batch_size)
    # results: [@1, @5, @10, ndcg@1, ndcg@5, ndcg@10]
    acc1, acc5, acc10 = results[0], results[1], results[2]
    print({
        'Acc@1': float(acc1),
        'Acc@5': float(acc5),
        'Acc@10': float(acc10),
    })


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    main()

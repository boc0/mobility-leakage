import argparse
import os
import pickle
from model import MarkovModel
import pandas as pd



def _parse_csv_sequences(csv_path):
    df = pd.read_csv(csv_path)
    required_cols = {"tid", "lat", "lon", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Build local pid mapping by unique lat,lon pairs
    loc_df = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    loc_df["pid"] = loc_df.index.astype(str)
    df = df.merge(loc_df, on=["lat", "lon"], how="left")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["tid", "timestamp"]).reset_index(drop=True)

    sequences = []
    for tid, g in df.groupby("tid", sort=True):
        pids = g["pid"].astype(str).tolist()
        if len(pids) > 1:
            sequences.append((tid, " ".join(pids)))
    return sequences


def test_file(model, file_path, output_csv_path, mode='topk', k_values=[1,5,10]):
    """
    Loads a .pk or .csv file, computes likelihoods for all its sequences using the specified
    function, and saves the results to a CSV file.
    """
    print(f"Processing {file_path}...")

    results = []
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pk":
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        except FileNotFoundError:
            print(f"  Error: Could not find file {file_path}")
            return
        data_neural = data.get('data_neural', {})
        user_map = {v[0]: k for k, v in data.get('uid_list', {}).items()}
        for user_idx, udata in data_neural.items():
            tid = user_map.get(user_idx, 'unknown_user')
            sequence_str = ' '.join([str(p[0]) for sess_id, sess in udata.get('sessions', {}).items() for p in sess])
            if len(sequence_str.split()) <= model.state_size:
                continue
            if mode == 'topk':
                topk = model.topk_accuracy(sequence_str, k_values=k_values)
                result = {"tid": tid, **{f'top-{k}': v for k, v in topk.items()}}
            elif mode == 'rank':
                rank = model.mean_true_rank(sequence_str)
                result = {'tid': tid, 'mean rank': rank}
            results.append(result)
    elif ext == ".csv":
        try:
            pairs = _parse_csv_sequences(file_path)
        except Exception as e:
            print(f"  Error parsing CSV {file_path}: {e}")
            return
        for tid, sequence_str in pairs:
            if len(sequence_str.split()) <= model.state_size:
                continue
            if mode == 'topk':
                topk = model.topk_accuracy(sequence_str, k_values=k_values)
                result = {"tid": tid, **{f'top-{k}': v for k, v in topk.items()}}
            elif mode == 'rank':
                rank = model.mean_true_rank(sequence_str)
                result = {'tid': tid, 'mean rank': rank}
            results.append(result)
    else:
        print(f"  Skipping unsupported file type: {file_path}")
        return

    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_csv_path, index=False)
        print(f"  Saved {len(results)} likelihoods to {output_csv_path}")
    else:
        print(f"  No valid sequences found in {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute trajectory likelihoods using a pre-trained Markov model."
    )
    parser.add_argument('--model', type=str, required=True,
                        help="Path to the trained markov_model.json file.")
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to the directory containing .pk or .csv data files.")
    parser.add_argument('--output_dir', type=str, default='markov_likelihoods',
                        help="Directory to save the output .csv files.")
    parser.add_argument('--mode', choices=['topk', 'rank'], default='topk', help='Whether to get top-k accuracy (topk or rank)')
    parser.add_argument('--k_values', '--ks', type=int, nargs='+', default=[1,5,10], help='Values of k for top-k accuracy (only for mode=topk)')
    
    args = parser.parse_args()

    # --- Load Model ---
    print(f"Loading Markov model from {args.model}...")
    model = MarkovModel.load_json(args.model)

    # --- Prepare Directories ---
    os.makedirs(args.output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(('.pk', '.csv'))]
    if not files:
        print(f"No .pk or .csv files found in directory: {args.data_dir}")
        return

    # --- Main Processing Loop ---
    for fname in files:
        input_path = os.path.join(args.data_dir, fname)
        output_filename = os.path.splitext(fname)[0] + f'_{args.mode}.csv'
        output_path = os.path.join(args.output_dir, output_filename)

        # ✅ Skip file if output already exists and is non-empty
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"⏭️ Skipping {fname} — results already exist at {output_path}")
            continue
        try:
            test_file(model, input_path, output_path, args.mode, args.k_values)
            print(f"✅ Done: {output_filename}")
        except Exception as e:
            print(f"❌ Error on {fname}: {e}")
            continue

if __name__ == '__main__':
    main()
import argparse
import os
import pickle
from model import MarkovModel
import pandas as pd



def _parse_csv_sequences(csv_path):
    """Parse a CSV file into (tid, sequence_str) pairs.
    Uses a local lat,lon -> pid mapping consistent with the training CSV prep.
    """
    df = pd.read_csv(csv_path)
    required_cols = {"tid", "lat", "lon", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Build local pid mapping by unique lat,lon pairs
    loc_df = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    loc_df["pid"] = loc_df.index.astype(str)
    df = df.merge(loc_df, on=["lat", "lon"], how="left")

    # sort and group
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["tid", "timestamp"]).reset_index(drop=True)

    sequences = []
    for tid, g in df.groupby("tid", sort=True):
        pids = g["pid"].astype(str).tolist()
        if len(pids) > 1:
            sequences.append((tid, " ".join(pids)))
    return sequences

def compute_likelihoods_for_file(model, file_path, output_csv_path, likelihood_func_name):
    """
    Loads a .pk or .csv file, computes likelihoods for all sequences using the specified
    function, and saves the results to a CSV file.
    """
    print(f"Processing {file_path}...")

    # Get the requested likelihood function from the model instance
    try:
        likelihood_function = getattr(model, likelihood_func_name)
    except AttributeError:
        print(f"  Error: Likelihood function '{likelihood_func_name}' not found in MarkovModel class.")
        return

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
            for sess_id, sess in udata.get('sessions', {}).items():
                if len(sess) <= model.state_size:
                    continue
                sequence_str = ' '.join([str(p[0]) for p in sess])
                metric = likelihood_function(sequence_str)
                row = {
                    'tid': tid,
                    'perplexity': metric,
                }
                results.append(row)
                print(row)
    elif ext == ".csv":
        try:
            pairs = _parse_csv_sequences(file_path)
        except Exception as e:
            print(f"  Error parsing CSV {file_path}: {e}")
            return
        for tid, sequence_str in pairs:
            if len(sequence_str.split()) <= model.state_size:
                continue
            metric = likelihood_function(sequence_str)
            row = {
                'tid': tid,
                'perplexity': metric,
            }
            results.append(row)
            print(row)
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
                        help="Path to the directory containing processed .pk data files.")
    parser.add_argument('--output_dir', type=str, default='markov_likelihoods',
                        help="Directory to save the output .csv files.")
    
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
        output_filename = os.path.splitext(fname)[0] + '_perplexity.csv'
        output_path = os.path.join(args.output_dir, output_filename)

        compute_likelihoods_for_file(model, input_path, output_path, 'perplexity')


if __name__ == '__main__':
    main()
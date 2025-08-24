import pickle
import argparse
import os
import pandas as pd
from model import MarkovModel



def prepare_corpus_from_deepmove(data_pickle_path):
    """
    Loads the processed data from sparse_traces.py and converts it
    into a corpus for the MarkovModel.
    """
    print(f"Loading data from {data_pickle_path}...")
    with open(data_pickle_path, 'rb') as f:
        data = pickle.load(f)

    data_neural = data['data_neural']
    corpus = []

    # Iterate through each user and their sessions
    for user_id in data_neural:
        sessions = data_neural[user_id]['sessions']
        for session_id in sessions:
            session = sessions[session_id]
            
            # Extract only the location IDs (vid) from the session
            # A session looks like: [[vid1, tid1], [vid2, tid2], ...]
            location_ids = [str(point[0]) for point in session]
            
            # A Markov "sentence" must have at least state_size + 1 words
            # If your state_size is 1, you need at least 2 locations.
            if len(location_ids) > 1:
                sentence = ' '.join(location_ids)
                corpus.append(sentence)
                
    print(f"Created a corpus with {len(corpus)} sequences.")
    return corpus


def prepare_corpus_from_csv(csv_path):
    """
    Loads a raw CSV (columns: tid, lat, lon, timestamp) and converts it
    into a corpus for the MarkovModel by mapping each unique (lat, lon)
    to a location ID and building per-user ordered sequences.

    Returns a list of space-separated location-id sentences, one per user.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Basic column checks
    required_cols = {"tid", "lat", "lon", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    # Build a stable location mapping for this CSV
    loc_df = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    loc_df["pid"] = loc_df.index.astype(str)

    # Join back to assign pids
    df = df.merge(loc_df, on=["lat", "lon"], how="left")

    # Ensure time ordering per user
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])  # drop unparsable timestamps
    df = df.sort_values(["tid", "timestamp"]).reset_index(drop=True)

    # Build corpus: one sentence per user over the whole time span
    corpus = []
    for tid, g in df.groupby("tid", sort=True):
        pids = g["pid"].astype(str).tolist()
        if len(pids) > 1:
            corpus.append(" ".join(pids))

    print(f"Created a corpus with {len(corpus)} user sequences from CSV.")
    return corpus

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MarkovModel on trajectory data and save to JSON')
    # Mutually exclusive inputs: either a DeepMove .pk or a raw CSV
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_pk', '--data', dest='data_pk', type=str,
                        help='Path to processed DeepMove .pk file (from sparse_traces.py)')
    group.add_argument('--data_csv', dest='data_csv', type=str,
                        help='Path to raw CSV (tid, lat, lon, timestamp)')
    parser.add_argument('--save_model', type=str, default='markov_model.json',
                        help='Path to save trained Markov model JSON (default: markov_model.json)')
    parser.add_argument('--state_size', type=int, default=1,
                        help='Markov state size (order). 1 = first-order (default)')
    args = parser.parse_args()

    # 1. Prepare the corpus from the chosen input
    if args.data_pk:
        trajectory_corpus = prepare_corpus_from_deepmove(args.data_pk)
    else:
        trajectory_corpus = prepare_corpus_from_csv(args.data_csv)

    # 3. Initialize and train the Markov Model
    # state_size=1 means it predicts the next location based on the last one.
    # state_size=2 would use the last two locations.
    markov = MarkovModel(state_size=args.state_size)
    print("Training Markov model...")
    markov.train(trajectory_corpus)
    print("Training complete.")

    # 4. Save the trained model
    model_save_path = args.save_model
    markov.save_json(model_save_path)

    # 5. Load the model from the file (sanity check)
    loaded_markov = MarkovModel.load_json(model_save_path)

    # Now the loaded model is ready to be used
    if loaded_markov.states:
        generated_sequence = loaded_markov.generate(length=max(4, args.state_size + 1))
        print(f"Generated sequence from loaded model: {generated_sequence}")
        likelihood = loaded_markov.likelihood_with_smoothing(generated_sequence)
        print(f"Smoothed likelihood of generated sequence: {likelihood:.6f}")
    else:
        print("No states were learned, cannot generate a sequence.")
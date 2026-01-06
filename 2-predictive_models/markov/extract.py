import os
import argparse
import pickle
from typing import List, Tuple

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable

from model import MarkovModel


def ensure_directory(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


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


def load_sequences_from_pk(pk_path: str) -> List[Tuple[str, List[str]]]:
    with open(pk_path, 'rb') as f:
        data = pickle.load(f)

    data_neural = data.get('data_neural', {})
    uid_list = data.get('uid_list', {})
    idx_to_user = {v[0]: k for k, v in uid_list.items()}

    sequences = []
    for user_idx in sorted(data_neural.keys()):
        udata = data_neural[user_idx]
        sessions = udata.get('sessions', {})
        merged = []
        for sid in sorted(sessions.keys()):
            merged.extend([str(p[0]) for p in sessions[sid]])
        if len(merged) <= 1:
            continue
        label = idx_to_user.get(user_idx, str(user_idx))
        sequences.append((label, merged))
    return sequences


def load_sequences_from_csv(csv_path: str) -> List[Tuple[str, List[str]]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required to process CSV inputs. Please install pandas or provide a .pk file instead.") from exc

    df = pd.read_csv(csv_path)
    required_cols = {"tid", "lat", "lon", "timestamp"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    loc_df = df[["lat", "lon"]].drop_duplicates().reset_index(drop=True)
    loc_df["pid"] = loc_df.index.astype(str)
    df = df.merge(loc_df, on=["lat", "lon"], how="left")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["tid", "timestamp"]).reset_index(drop=True)

    sequences = []
    for tid, group in df.groupby("tid", sort=True):
        pids = group["pid"].astype(str).tolist()
        if len(pids) > 1:
            sequences.append((str(tid), pids))
    return sequences


def compute_rank_proxies(sequences: List[Tuple[str, List[str]]], model: MarkovModel,
                          prefix_lengths: List[int], steps: int, alpha: float,
                          progress_label: str) -> List[Tuple[str, List[int]]]:
    records = []
    prefix_lengths = normalize_prefix_lengths(prefix_lengths)
    step_limit = None if steps is None else max(int(steps), 0)

    for label, tokens in tqdm(sequences, desc=progress_label):
        if len(tokens) <= model.state_size:
            continue
        sequence_str = " ".join(tokens)
        ranks_raw = model.predicted_ranks(sequence_str, alpha=alpha)
        if not isinstance(ranks_raw, list) or not ranks_raw:
            continue
        ranks = [max(1, int(round(r))) for r in ranks_raw if r > 0]
        if not ranks:
            continue
        if step_limit is not None:
            ranks = ranks[:step_limit]
        if not ranks:
            continue
        records.append((label, ranks))

    return records, prefix_lengths


def write_output(output_path: str, records: List[Tuple[str, List[int]]],
                 prefix_lengths: List[int], vocab_size: int) -> int:
    ensure_directory(output_path)
    header = ",".join(f"prefix-{p}" for p in prefix_lengths)

    if not records:
        with open(output_path, 'w') as out_f:
            out_f.write(f"tid,{header}\n")
        return 0

    max_len = max(len(r) for _, r in records)

    with open(output_path, 'w') as out_f:
        out_f.write(f"tid,{header}\n")
        for label, ranks in records:
            proxies = [
                str(trajectory_rank_proxy(ranks, vocab_size, max_len, prefix_len=p))
                for p in prefix_lengths
            ]
            out_f.write(f"{label},{','.join(proxies)}\n")

    return len(records)


def process_file(input_path: str, output_path: str, model: MarkovModel,
                 prefix_lengths: List[int], steps: int, alpha: float) -> int:
    ext = os.path.splitext(input_path)[1].lower()
    if ext == '.pk':
        sequences = load_sequences_from_pk(input_path)
    elif ext == '.csv':
        sequences = load_sequences_from_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type for extraction: {input_path}")

    records, normalized_prefix = compute_rank_proxies(
        sequences,
        model,
        prefix_lengths,
        steps,
        alpha,
        progress_label=os.path.basename(input_path),
    )

    vocab_size = max(1, len(model.vocab))
    return write_output(output_path, records, normalized_prefix, vocab_size)


def main():
    parser = argparse.ArgumentParser(description="Extract trajectory rank proxies using a Markov model")
    parser.add_argument('--model', type=str, required=True, help='Path to trained markov model JSON file')
    parser.add_argument('--data_pk', type=str, default=None, help='Path to a processed DeepMove .pk file')
    parser.add_argument('--data_csv', type=str, default=None, help='Path to a raw CSV (tid, lat, lon, timestamp)')
    parser.add_argument('--data_dir', type=str, default=None, help='Directory of .pk/.csv files to process')
    parser.add_argument('--output', type=str, default=None, help='Output CSV (for single file) or directory (for data_dir)')
    parser.add_argument('--steps', type=int, default=None, help='Maximum number of ranks to retain per trajectory')
    parser.add_argument('--prefix_lengths', '--prefix_lens', type=int, nargs='+', default=[0],
                        help='List of prefix lengths for rank proxy computation (non-negative integers)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Laplace smoothing alpha used for rank computation')
    args = parser.parse_args()

    inputs = [bool(args.data_pk), bool(args.data_csv), bool(args.data_dir)]
    if sum(inputs) != 1:
        parser.error('Provide exactly one of --data_pk, --data_csv, or --data_dir')

    if (args.data_pk or args.data_csv) and not args.output:
        parser.error('--output is required when processing a single file')

    print(f"Loading Markov model from {args.model}...")
    model = MarkovModel.load_json(args.model)

    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")
        files = [
            os.path.join(args.data_dir, fname)
            for fname in os.listdir(args.data_dir)
            if fname.lower().endswith(('.pk', '.csv'))
        ]
        if not files:
            raise FileNotFoundError(f"No .pk or .csv files found in {args.data_dir}")

        output_dir = args.output or 'markov_extraction'
        os.makedirs(output_dir, exist_ok=True)

        for path in sorted(files):
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(output_dir, f"{base}_extraction_proxy.csv")
            print(f"Processing {path} -> {out_path}")
            processed = process_file(
                path,
                out_path,
                model,
                args.prefix_lengths,
                args.steps,
                args.alpha,
            )
            if processed == 0:
                print(f"Warning: No valid trajectories found in {path}")
    else:
        input_path = args.data_pk or args.data_csv
        processed = process_file(
            input_path,
            args.output,
            model,
            args.prefix_lengths,
            args.steps,
            args.alpha,
        )
        if processed == 0:
            raise SystemExit("No valid trajectories found for extraction.")


if __name__ == '__main__':
    main()

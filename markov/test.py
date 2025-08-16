import argparse
import pickle
import json
import math
from collections import defaultdict

from model import MarkovModel


def parse_topk_list(spec: str):
    """
    Parse a comma-separated list like "1,5,10" or with percents like "1%,5%,10%".
    Returns a list of entries; each entry is a tuple (kind, value) where
    kind is 'abs' for absolute integer K, or 'pct' for percentage [0-100].
    """
    ks = []
    for part in spec.split(','):
        p = part.strip()
        if not p:
            continue
        if p.endswith('%'):
            try:
                pct = float(p[:-1])
                if pct <= 0:
                    continue
                ks.append(('pct', pct))
            except ValueError:
                raise ValueError(f"Invalid percentage in --topk: {p}")
        else:
            try:
                k = int(p)
                if k <= 0:
                    continue
                ks.append(('abs', k))
            except ValueError:
                raise ValueError(f"Invalid integer in --topk: {p}")
    if not ks:
        # fallback default
        ks = [('abs', 1), ('abs', 5), ('abs', 10)]
    return ks


essential_keys = ['data_neural']

def load_pk_dataset(pk_path: str):
    with open(pk_path, 'rb') as f:
        data = pickle.load(f)
    for k in essential_keys:
        if k not in data:
            raise ValueError(f"Missing key '{k}' in dataset {pk_path}")
    return data


def iter_eval_transitions(data_neural, mode='test'):
    """
    Yield (tokens_state, actual_next) pairs from sessions according to mode.
    tokens are string PIDs (location indices) from the DeepMove sessions, i.e., session = [[vid, tid], ...]
    """
    for u in data_neural:
        sessions = data_neural[u]['sessions']
        # choose ids per mode
        ids = []
        if mode == 'train':
            ids = data_neural[u].get('train', [])
        elif mode == 'test':
            ids = data_neural[u].get('test', [])
        elif mode == 'all':
            ids = list(sessions.keys())
        if not ids:
            # fallback: use all sessions if chosen split is empty
            ids = list(sessions.keys())
        for sid in ids:
            seq = sessions[sid]
            # tokens are the location ids (first element)
            tokens = [str(p[0]) for p in seq]
            yield tokens


def compute_topk_accuracy(model: MarkovModel, sequences, topk_spec, mode_desc: str):
    """
    Compute top-k accuracy for provided sequences.
    sequences: iterable of token lists (strings)
    topk_spec: list of ('abs'| 'pct', value)
    Returns: dict with accuracy per spec string, and counters.
    """
    # Prepare structures
    hit_counts = defaultdict(int)
    total_steps = 0
    unseen_states = 0

    # precompute friendly labels
    labels = []
    for kind, val in topk_spec:
        if kind == 'abs':
            labels.append(str(val))
        else:
            labels.append(f"{val:g}%")

    for tokens in sequences:
        if len(tokens) <= model.state_size:
            continue
        for i in range(model.state_size, len(tokens)):
            state = tuple(tokens[i - model.state_size:i])
            actual = tokens[i]
            next_counts = model.transitions.get(state)
            if not next_counts:
                unseen_states += 1
                total_steps += 1
                continue
            # sort candidates by count desc
            sorted_next = sorted(next_counts.items(), key=lambda kv: kv[1], reverse=True)
            cand_len = len(sorted_next)
            # evaluate all requested K
            for (kind, val), label in zip(topk_spec, labels):
                if kind == 'abs':
                    k = min(val, cand_len)
                else:
                    k = max(1, math.ceil((val / 100.0) * cand_len))
                    k = min(k, cand_len)
                top_pred = [s for (s, _) in sorted_next[:k]]
                if actual in top_pred:
                    hit_counts[label] += 1
            total_steps += 1

    accuracies = {label: (hit_counts[label] / total_steps) if total_steps > 0 else 0.0 for label in labels}
    return {
        'accuracies': accuracies,
        'total_steps': total_steps,
        'unseen_states': unseen_states,
        'seen_states': total_steps - unseen_states,
        'mode': mode_desc,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Markov model next-step top-k accuracy on a DeepMove .pk dataset')
    parser.add_argument('--model_json', type=str, required=True, help='Path to trained Markov model JSON (saved via save_json)')
    parser.add_argument('--data_pk', type=str, required=True, help='Path to DeepMove .pk file (from sparse_traces.py)')
    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'all'], help='Which split to evaluate; falls back to all if empty')
    parser.add_argument('--topk', type=str, default='1,5,10', help='Comma-separated list of K values, e.g., "1,5,10" or percents "1%,5%,10%"')
    args = parser.parse_args()

    # Load model
    model = MarkovModel.load_json(args.model_json)

    # Load dataset
    data = load_pk_dataset(args.data_pk)
    data_neural = data['data_neural']

    # Build sequences generator
    sequences = iter_eval_transitions(data_neural, mode=args.mode)

    # Parse Ks and evaluate
    topk_spec = parse_topk_list(args.topk)
    result = compute_topk_accuracy(model, sequences, topk_spec, mode_desc=args.mode)

    # Report
    print("\nMarkov next-step prediction evaluation")
    print(f"  model:   {args.model_json}")
    print(f"  data:    {args.data_pk}")
    print(f"  mode:    {result['mode']}")
    print(f"  state n: {model.state_size}")
    print(f"  steps:   {result['total_steps']} (seen: {result['seen_states']}, unseen: {result['unseen_states']})")
    for k_label, acc in result['accuracies'].items():
        print(f"  top-{k_label} accuracy: {acc:.4f}")


if __name__ == '__main__':
    main()

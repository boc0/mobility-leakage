import os
import argparse
import pickle
import json
from typing import List, Dict, Any, Tuple

import torch
from tqdm import tqdm

from train import RnnParameterData
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong


# auto-select MPS/CPU/CUDA
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
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def pad_sequences(sequences: List[List[int]], pad_value: int = 0):
    if not sequences:
        return torch.tensor([]), torch.tensor([])
    max_len = max(len(seq) for seq in sequences)
    padded = torch.full((max_len, len(sequences)), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for idx, seq in enumerate(sequences):
        length = len(seq)
        if length == 0:
            continue
        padded[:length, idx] = torch.tensor(seq, dtype=torch.long)
        mask[idx, :length] = True
    return padded, mask


def next_time_slot(tim_seq: List[int], tim_size: int) -> int:
    """Predict the next temporal slot given history.

    Uses the most recent step size when available, otherwise defaults to a
    single-slot increment, and wraps around the configured temporal vocabulary.
    """
    if tim_size <= 0:
        return tim_seq[-1] if tim_seq else 0

    if not tim_seq:
        return 0

    last = int(tim_seq[-1])
    if len(tim_seq) >= 2:
        prev = int(tim_seq[-2])
        delta = (last - prev) % tim_size
        if delta == 0:
            delta = 1
    else:
        delta = 1

    return (last + delta) % tim_size


def get_next_step_log_probs(model, loc_seq: List[int], tim_seq: List[int], uid_idx: int,
                            model_mode: str, device) -> torch.Tensor:
    """Return log-probabilities for the next location given current history."""

    loc_tensor = torch.tensor(loc_seq, dtype=torch.long, device=device).unsqueeze(1)
    tim_tensor = torch.tensor(tim_seq, dtype=torch.long, device=device).unsqueeze(1)

    if model_mode in ['simple', 'simple_long']:
        scores = model(loc_tensor, tim_tensor)
    elif model_mode == 'attn_avg_long_user':
        history_count = [1] * len(loc_seq)
        uid_tensor = torch.tensor([uid_idx], dtype=torch.long, device=device)
        scores = model(
            loc_tensor,
            tim_tensor,
            loc_tensor,
            tim_tensor,
            history_count,
            uid_tensor,
            target_len=1,
        )
    else:  # attn_local_long
        scores = model(loc_tensor, tim_tensor, target_len=1)

    if scores.dim() == 1:
        return scores
    if scores.dim() == 2:
        return scores[-1]
    # shape (tgt_len, batch, vocab)
    return scores[-1, 0, :]


def beam_search_predict(model, loc_seq: List[int], tim_seq: List[int], uid_idx: int,
                        model_mode: str, steps: int, tim_size: int, beam_width: int,
                        device) -> List[int]:
    if steps <= 0:
        return []

    beam_width = max(1, beam_width)
    initial = (list(loc_seq), list(tim_seq), [], 0.0)
    beams: List[Tuple[List[int], List[int], List[int], float]] = [initial]

    for _ in range(steps):
        candidates: List[Tuple[List[int], List[int], List[int], float]] = []
        for curr_loc, curr_tim, preds, logp in beams:
            if not curr_loc or not curr_tim:
                # model expects at least one element; skip invalid beam
                continue

            next_log_probs = get_next_step_log_probs(model, curr_loc, curr_tim, uid_idx, model_mode, device)
            vocab_size = next_log_probs.size(0)
            k = min(beam_width, vocab_size)
            topk = torch.topk(next_log_probs, k)
            next_time = next_time_slot(curr_tim, tim_size)

            for score, loc_id in zip(topk.values.tolist(), topk.indices.tolist()):
                loc_id = int(loc_id)
                new_loc = curr_loc + [loc_id]
                new_tim = curr_tim + [next_time]
                new_preds = preds + [loc_id]
                candidates.append((new_loc, new_tim, new_preds, logp + score))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[3], reverse=True)
        beams = candidates[:beam_width]

    if not beams:
        return []

    best = max(beams, key=lambda x: x[3])
    return best[2][:steps]


def predict_batch(model, batch_data: List[Dict[str, Any]], model_mode: str, steps: int,
                  tim_size: int, beam_width: int, device):
    if not batch_data:
        return []

    predictions = []
    for entry in batch_data:
        preds = beam_search_predict(
            model,
            entry['loc_seq'],
            entry['tim_seq'],
            entry['uid_idx'],
            model_mode,
            steps,
            tim_size,
            beam_width,
            device,
        )
        predictions.append(preds)

    return predictions


def ensure_parent_dir(path: str):
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Greedy trajectory prediction for pretrained DeepMove models."
    )
    parser.add_argument('--data_pk', type=str, default=None,
                        help="path to a single processed .pk file")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="path to a directory of processed .pk files")
    parser.add_argument('--model_path', type=str, required=True,
                        help="checkpoint .m file to load")
    parser.add_argument('--model_mode', type=str, default='attn_avg_long_user',
                        choices=['simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long'])
    parser.add_argument('--metadata_json', type=str, required=True,
                        help="path to metadata json file")
    parser.add_argument('--output', type=str, required=True,
                        help="output file (for single pk) or directory (for data_dir)")
    parser.add_argument('--steps', type=int, default=1,
                        help="number of future steps to predict")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="batch size for processing trajectories")
    parser.add_argument('--beam_width', type=int, default=5,
                        help="beam width for decoding (1 matches greedy)")
    parser.add_argument('--verbose', action='store_true', default=False,
                        help="print predictions to stdout")
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')
    if args.steps <= 0:
        parser.error('--steps must be positive')

    pk_files = []
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")
        pk_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pk')]
        if not pk_files:
            raise FileNotFoundError(f"No .pk files found in {args.data_dir}")
    else:
        if not os.path.exists(args.data_pk):
            raise FileNotFoundError(f"File not found: {args.data_pk}")
        pk_files.append(args.data_pk)

    params = RnnParameterData(metadata=args.metadata_json)
    params.model_mode = args.model_mode
    params.use_cuda = (device.type != 'cpu')
    params.rnn_type = getattr(params, 'rnn_type', 'LSTM')
    params.attn_type = getattr(params, 'attn_type', 'dot')
    params.dropout_p = getattr(params, 'dropout_p', 0.3)
    params.tim_size = getattr(params, 'tim_size', 48)

    meta = json.load(open(args.metadata_json, 'r'))
    params.loc_size = len(meta.get('pid_mapping', {})) + 1
    params.uid_size = len(meta.get('users', []))
    if params.loc_size == 0 or params.uid_size == 0:
        raise ValueError("metadata.json is missing required 'pid_mapping' or 'users' information")
    user_to_idx = {str(u): i for i, u in enumerate(meta.get('users', []))}

    model = load_model(params, args.model_mode, args.model_path)

    output_dir = None
    single_output_path = None
    if args.data_dir:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
    else:
        single_output_path = args.output
        ensure_parent_dir(single_output_path)

    single_out_f = None
    if single_output_path:
        single_out_f = open(single_output_path, 'w')
        header = 'tid,' + ','.join([f'step_{i+1}' for i in range(args.steps)]) + '\n'
        single_out_f.write(header)

    for pk_file in pk_files:
        print(f"Predicting for {pk_file}...")

        out_f = None
        if output_dir:
            basename = os.path.basename(pk_file)
            out_name = os.path.splitext(basename)[0] + 'predict.csv'
            out_path = os.path.join(output_dir, out_name)
            out_f = open(out_path, 'w')
            header = 'tid,' + ','.join([f'step_{i+1}' for i in range(args.steps)]) + '\n'
            out_f.write(header)
        else:
            out_f = single_out_f

        data = pickle.load(open(pk_file, 'rb'))
        sessions_all = data['data_neural']
        uid_list = data['uid_list']
        idx_to_user = {v[0]: k for k, v in uid_list.items()}

        trajectories = []
        labels = []

        for u_idx, udata in sessions_all.items():
            label = idx_to_user.get(u_idx, None)
            uid_idx = user_to_idx.get(str(label), None)
            if uid_idx is None:
                continue
            # merge sessions
            sess_ids = sorted(udata['sessions'].keys())
            merged = []
            for sid in sess_ids:
                merged.extend(udata['sessions'][sid])
            if len(merged) < 1:
                continue
            locs = [p[0] for p in merged]
            tims = [p[1] for p in merged]
            trajectories.append({'loc_seq': locs, 'tim_seq': tims, 'uid_idx': uid_idx})
            labels.append(label)

        for i in tqdm(range(0, len(trajectories), args.batch_size)):
            batch = trajectories[i:i + args.batch_size]
            batch_labels = labels[i:i + args.batch_size]
            preds = predict_batch(
                model,
                batch,
                args.model_mode,
                args.steps,
                params.tim_size,
                args.beam_width,
                device,
            )
            for label, pred_seq in zip(batch_labels, preds):
                padded_pred = pred_seq + [''] * max(0, args.steps - len(pred_seq))
                str_preds = [str(p) for p in padded_pred[:args.steps]]
                line = f"{label}," + ','.join(str_preds)
                if out_f:
                    out_f.write(line + "\n")
                if (not out_f) or args.verbose:
                    print(line)

        if output_dir and out_f:
            out_f.close()

    if single_out_f:
        single_out_f.close()

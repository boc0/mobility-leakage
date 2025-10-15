import os
import argparse
import pickle
import json
from typing import List, Dict, Any

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


def predict_batch(model, batch_data: List[Dict[str, Any]], model_mode: str, steps: int, tim_size: int, device):
    if steps <= 0 or not batch_data:
        return [[] for _ in batch_data]

    batch_size = len(batch_data)
    predictions = [[] for _ in range(batch_size)]

    loc_seqs = [list(entry['loc_seq']) for entry in batch_data]
    tim_seqs = [list(entry['tim_seq']) for entry in batch_data]

    if model_mode in ['simple', 'simple_long']:
        for _ in range(steps):
            lengths = [len(seq) for seq in loc_seqs]

            loc_tensor, _ = pad_sequences(loc_seqs, pad_value=0)
            tim_tensor, _ = pad_sequences(tim_seqs, pad_value=0)
            loc_tensor = loc_tensor.to(device)
            tim_tensor = tim_tensor.to(device)

            scores = model(loc_tensor, tim_tensor)
            if scores.dim() == 2:
                scores = scores.unsqueeze(1)
            scores = scores.transpose(0, 1).contiguous()

            length_tensor = torch.tensor(lengths, device=device)
            idx_tensor = length_tensor - 1
            batch_indices = torch.arange(batch_size, device=device)
            step_scores = scores[batch_indices, idx_tensor]
            pred_ids = step_scores.argmax(dim=-1)

            for i in range(batch_size):
                pred = int(pred_ids[i].item())
                predictions[i].append(pred)
                loc_seqs[i].append(pred)
                next_tim = next_time_slot(tim_seqs[i], tim_size)
                tim_seqs[i].append(next_tim)
        return predictions

    if model_mode == 'attn_avg_long_user':
        uid_tensor_template = torch.tensor([entry['uid_idx'] for entry in batch_data], dtype=torch.long, device=device)

        for _ in range(steps):
            lengths = [len(seq) for seq in loc_seqs]

            loc_tensor, _ = pad_sequences(loc_seqs, pad_value=0)
            tim_tensor, _ = pad_sequences(tim_seqs, pad_value=0)
            loc_tensor = loc_tensor.to(device)
            tim_tensor = tim_tensor.to(device)

            history_counts = [[1] * len(seq) for seq in loc_seqs]
            target_lengths = [1] * batch_size

            scores = model(
                loc_tensor,
                tim_tensor,
                loc_tensor,
                tim_tensor,
                history_counts,
                uid_tensor_template,
                target_lengths,
            )

            if scores.dim() == 1:
                scores = scores.view(1, 1, -1)
            elif scores.dim() == 2:
                scores = scores.unsqueeze(1)
            scores = scores.transpose(0, 1).contiguous()
            step_scores = scores[:, -1, :]
            pred_ids = step_scores.argmax(dim=-1)

            for i in range(batch_size):
                pred = int(pred_ids[i].item())
                predictions[i].append(pred)
                loc_seqs[i].append(pred)
                next_tim = next_time_slot(tim_seqs[i], tim_size)
                tim_seqs[i].append(next_tim)
        return predictions

    # attn_local_long (fallback to per-sample greedy)
    for i in range(batch_size):
        loc_seq = list(loc_seqs[i])
        tim_seq = list(tim_seqs[i])
        sample_predictions = []
        for _ in range(steps):
            loc_tensor = torch.tensor(loc_seq, dtype=torch.long, device=device).unsqueeze(1)
            tim_tensor = torch.tensor(tim_seq, dtype=torch.long, device=device).unsqueeze(1)
            scores = model(loc_tensor, tim_tensor, target_len=1)
            if scores.dim() == 1:
                step_scores = scores
            elif scores.dim() == 2:
                step_scores = scores[-1]
            else:
                step_scores = scores[-1, 0, :]
            pred = int(torch.argmax(step_scores).item())
            sample_predictions.append(pred)
            loc_seq.append(pred)
            next_tim = next_time_slot(tim_seq, tim_size)
            tim_seq.append(next_tim)
        predictions[i] = sample_predictions
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
    parser.add_argument('--merge_sessions', action='store_true', default=True,
                        help="merge sessions per user before predicting")
    parser.add_argument('--no_merge', dest='merge_sessions', action='store_false',
                        help="do not merge sessions; predict each session separately")
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

            if args.merge_sessions:
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
            else:
                for sess_id, sess in udata['sessions'].items():
                    if len(sess) < 1:
                        continue
                    locs = [p[0] for p in sess]
                    tims = [p[1] for p in sess]
                    trajectories.append({'loc_seq': locs, 'tim_seq': tims, 'uid_idx': uid_idx})
                    labels.append(f"{label}_{sess_id}")

        for i in tqdm(range(0, len(trajectories), args.batch_size)):
            batch = trajectories[i:i + args.batch_size]
            batch_labels = labels[i:i + args.batch_size]
            preds = predict_batch(model, batch, args.model_mode, args.steps, params.tim_size, device)
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

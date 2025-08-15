import os
import math
import argparse
import pickle
import json

import torch
import torch.nn as nn

from train import RnnParameterData
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong

# auto‐select MPS/CPU/CUDA
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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return model.to(device).eval()

def trajectory_perplexity(model, loc_seq, tim_seq, target_seq, model_mode, uid):
    """
    loc_seq, tim_seq: list of int prefix
    target_seq: list of int next‐locations
    """
    locs = torch.LongTensor(loc_seq).unsqueeze(1).to(device)
    tims = torch.LongTensor(tim_seq).unsqueeze(1).to(device)
    tgt  = torch.LongTensor(target_seq).to(device)

    with torch.no_grad():
        if model_mode in ['simple', 'simple_long']:
            scores = model(locs, tims)
        elif model_mode == 'attn_avg_long_user':
            # treat prefix as history too
            history_loc   = locs
            history_tim   = tims
            history_count = [1] * history_loc.size(0)
            uid_tensor    = torch.LongTensor([uid]).to(device)
            target_len    = tgt.size(0)
            scores = model(locs, tims,
                           history_loc, history_tim,
                           history_count, uid_tensor, target_len)
        else:  # attn_local_long
            target_len = tgt.size(0)
            scores = model(locs, tims, target_len)

        # align lengths
        if scores.size(0) > tgt.size(0):
            scores = scores[-tgt.size(0):]

        # sum negative log‐likelihood
        loss_fn = nn.NLLLoss(reduction='sum')
        nll = loss_fn(scores, tgt).item()

    # perplexity = exp( avg nll per token )
    return math.exp(nll / len(target_seq))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute trajectory perplexities for a pretrained DeepMove model."
    )
    parser.add_argument('--data_pk',     type=str, default=None,
                        help="path to a single processed .pk file")
    parser.add_argument('--data_dir',    type=str, default=None,
                        help="path to a directory of processed .pk files")
    parser.add_argument('--model_path',  type=str, required=True,
                        help="checkpoint .m file to load")
    parser.add_argument('--model_mode',  type=str, default='attn_avg_long_user',
                        choices=['simple','simple_long','attn_avg_long_user','attn_local_long'])
    parser.add_argument('--output',      type=str, default=None,
                        help="optional output. For a single file, a CSV file path. For a directory, an output directory path.")
    parser.add_argument('--metadata_json', type=str, default=None,
                        help="path to metadata json file (required for correct model size)")
    args = parser.parse_args()

    if not args.data_pk and not args.data_dir:
        parser.error('either --data_pk or --data_dir is required')
    if args.data_pk and args.data_dir:
        parser.error('cannot use both --data_pk and --data_dir')
    if not args.metadata_json:
        parser.error('--metadata_json is required')

    # get file list
    pk_files = []
    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")
        pk_files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.pk')]
        if not pk_files:
            print(f"Warning: No .pk files found in {args.data_dir}")
    else:
        if not os.path.exists(args.data_pk):
            raise FileNotFoundError(f"File not found: {args.data_pk}")
        pk_files.append(args.data_pk)

    # build parameter object
    params = RnnParameterData(metadata=args.metadata_json)
    params.model_mode  = args.model_mode
    params.use_cuda    = (device.type != 'cpu')
    params.rnn_type    = getattr(params, 'rnn_type', 'LSTM')
    params.attn_type   = getattr(params, 'attn_type', 'dot')
    params.dropout_p   = getattr(params, 'dropout_p', 0.3)
    params.tim_size    = getattr(params, 'tim_size', 48) # Use default, since it's not in metadata

    # load metadata for sizes
    meta = json.load(open(args.metadata_json, 'r'))
    params.loc_size = len(meta.get('pid_mapping', {}))
    params.uid_size = len(meta.get('users', []))
    if params.loc_size == 0 or params.uid_size == 0:
        raise ValueError("metadata.json is missing 'pid_mapping' or 'users' information.")
    # build user label → index mapping
    user_to_idx = {u: i for i, u in enumerate(meta.get('users', []))}

    # load model once
    model = load_model(params, args.model_mode, args.model_path)

    # --- Output Handling ---
    output_dir = None
    if args.data_dir:
        output_dir = args.output if args.output else 'likelihoods'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output for directory processing will be in: {output_dir}")

    # --- Main Processing Loop ---
    single_out_f = None
    if args.data_pk and args.output:
        single_out_f = open(args.output, 'w')
        single_out_f.write("tid,perplexity\n")

    for pk_file in pk_files:
        print(f"Processing {pk_file}...")

        # Determine current output file handle
        out_f = None
        if args.data_dir:
            basename = os.path.basename(pk_file)
            out_name = os.path.splitext(basename)[0] + '.csv'
            out_path = os.path.join(output_dir, out_name)
            out_f = open(out_path, 'w')
            out_f.write("tid,perplexity\n")
        else: # single file mode
            out_f = single_out_f

        data = pickle.load(open(pk_file, 'rb'))
        sessions_all = data['data_neural']
        tids = [int(k) for k in data['uid_list'].keys()]

        for (u, udata), tid in zip(sessions_all.items(), tids):
            # map user label to embedded index
            uid_idx = user_to_idx.get(u, None)
            if uid_idx is None:
                # This can happen if the user from the .pk file is not in the metadata.json
                # For example, when processing a test set with users not in the training set.
                # We can either skip or assign a default index. Skipping is safer.
                # print(f"Warning: User '{u}' not found in metadata.json, skipping.")
                continue

            # sessions stored as dict session_id → [(loc, tim), ...]
            for sess_id, sess in udata['sessions'].items():
                # prefix / target split
                if len(sess) < 2:
                    continue
                locs = [p[0] for p in sess]
                tims = [p[1] for p in sess]
                # predict from step 1…end
                loc_seq    = locs[:-1]
                tim_seq    = tims[:-1]
                target_loc = locs[1:]
                ppl = trajectory_perplexity(
                    model, loc_seq, tim_seq, target_loc,
                    args.model_mode, uid_idx
                )
                line = f"{tid},{ppl:.3f}" # sess_id is always 0 when we have 1 traj per user
                if out_f:
                    out_f.write(line + "\n")
                else:
                    print(line)

        # Close file if in directory mode
        if args.data_dir and out_f:
            out_f.close()

    # Close file if in single file mode
    if single_out_f:
        single_out_f.close()

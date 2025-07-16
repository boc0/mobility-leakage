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
    parser.add_argument('--data_pk',     type=str, default='data/foursquare_test.pk',
                        help="path to processed .pk file (data_neural, vid_list, uid_list)")
    parser.add_argument('--model_path',  type=str, default='results/res.m',
                        help="checkpoint .m file to load")
    parser.add_argument('--model_mode',  type=str, default='attn_avg_long_user',
                        choices=['simple','simple_long','attn_avg_long_user','attn_local_long'])
    parser.add_argument('--output',      type=str, default=None,
                        help="optional CSV output (user,session,perplexity)")
    parser.add_argument('--metadata_json', type=str, default='data/foursquare/metadata.json',
                        help="path to metadata json file")
    args = parser.parse_args()

    # load data
    data = pickle.load(open(args.data_pk, 'rb'))
    sessions_all = data['data_neural']
    vid_list     = data['vid_list']
    uid_list     = data['uid_list']

    # build parameter object
    params = RnnParameterData()
    params.data_neural = sessions_all
    params.tim_size    = data.get('tim_size', 48)
    params.loc_size    = len(vid_list)
    params.uid_size    = len(uid_list)
    params.model_mode  = args.model_mode
    params.use_cuda    = (device.type != 'cpu')
    params.rnn_type    = getattr(params, 'rnn_type', 'LSTM')
    params.attn_type   = getattr(params, 'attn_type', 'dot')
    params.dropout_p   = getattr(params, 'dropout_p', 0.3)
    # load metadata for sizes if provided
    if args.metadata_json:
        meta = json.load(open(args.metadata_json, 'r'))
        params.loc_size = len(meta.get('pid_mapping', {}))
        params.uid_size = len(meta.get('users', []))
        # build user label → index mapping
        user_to_idx = {u: i for i, u in enumerate(meta.get('users', []))}
    else:
        # fallback mapping assuming numeric u
        user_to_idx = {u: u for u in uid_list}

    # load model once
    model = load_model(params, args.model_mode, args.model_path)

    # prepare output
    if args.output:
        out_f = open(args.output, 'w')
        out_f.write("user,session,perplexity\n")
    else:
        out_f = None

    # iterate and compute
    for u, udata in sessions_all.items():
        # map user label to embedded index
        uid_idx = user_to_idx.get(u, None)
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
            line = f"{u},{sess_id},{ppl:.3f}"
            if out_f:
                out_f.write(line + "\n")
            else:
                print(line)

    if out_f:
        out_f.close()

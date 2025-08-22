import argparse
import os
import pickle
import json

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from train import RnnParameterData, run_simple, \
    generate_input_history, generate_input_long_history, generate_input_long_history2
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a pretrained DeepMove model on DeepMove .pk data (single file or directory)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data',      type=str,
                        help="path to a processed .pk file")
    group.add_argument('--data_dir',  type=str,
                        help="path to a directory containing .pk files")
    parser.add_argument('--model_path',   type=str, required=True,
                        help="path to trained .m model checkpoint")
    parser.add_argument('--model_mode',   type=str, required=True,
                        choices=['simple','simple_long',
                                 'attn_avg_long_user','attn_local_long'])
    parser.add_argument('--metadata_json',type=str, required=True,
                        help="path to metadata.json for loc/uid sizes")
    args = parser.parse_args()

    # resolve a representative .pk to initialize parameter sizes
    if args.data:
        rep_pk = args.data
    else:
        # pick the first .pk in the directory
        pk_files = [os.path.join(args.data_dir, f) for f in sorted(os.listdir(args.data_dir)) if f.endswith('.pk')]
        if not pk_files:
            raise FileNotFoundError(f"No .pk files found in directory: {args.data_dir}")
        rep_pk = pk_files[0]

    # build parameter object (loads .pk internally for sizes)
    parameters = RnnParameterData(data_path=rep_pk)
    parameters.model_mode = args.model_mode

    # override sizes from metadata.json
    meta = json.load(open(args.metadata_json, 'r'))
    parameters.loc_size = len(meta['pid_mapping'])
    parameters.uid_size = len(meta['users'])

    # set history_mode for non‐long models
    if 'max' in args.model_mode:
        parameters.history_mode = 'max'
    elif 'avg' in args.model_mode:
        parameters.history_mode = 'avg'
    else:
        parameters.history_mode = 'whole'

    # instantiate & load model (once)
    device = torch.device("mps") if torch.backends.mps.is_available() \
             else torch.device("cuda") if torch.cuda.is_available() \
             else torch.device("cpu")
    if args.model_mode in ['simple','simple_long']:
        model = TrajPreSimple(parameters).to(device)
    elif args.model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser(parameters).to(device)
    else:
        model = TrajPreLocalAttnLong(parameters).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # define loss & optimizer (optimizer is unused in test mode)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad],
                           lr=parameters.lr, weight_decay=parameters.L2)

    def eval_one(pk_path: str):
        ds = pickle.load(open(pk_path, 'rb'))
        data_neural = ds['data_neural']
        candidate = list(data_neural.keys())
        if 'long' in args.model_mode:
            if args.model_mode == 'simple_long':
                data_test, test_idx = generate_input_long_history2(
                    data_neural, 'test', candidate=candidate)
            else:
                data_test, test_idx = generate_input_long_history(
                    data_neural, 'test', candidate=candidate)
        else:
            data_test, test_idx = generate_input_history(
                data_neural, 'test', mode2=parameters.history_mode,
                candidate=candidate)
        avg_loss, avg_acc, _ = run_simple(
            data_test, test_idx, 'test',
            parameters.lr, parameters.clip,
            model, optimizer, criterion,
            parameters.model_mode
        )
        print(f"file: {pk_path}")
        print(f"  Test Loss: {avg_loss:.4f}    Test Accuracy: {avg_acc:.4f}")

    if args.data:
        eval_one(args.data)
    else:
        pk_files = [os.path.join(args.data_dir, f) for f in sorted(os.listdir(args.data_dir)) if f.endswith('.pk')]
        if not pk_files:
            print(f"No .pk files found in {args.data_dir}")
            return
        for pk in pk_files:
            eval_one(pk)


if __name__ == '__main__':
    main()
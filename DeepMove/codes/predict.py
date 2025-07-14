import torch
import pickle
import numpy as np
import argparse

from train import RnnParameterData
from model import TrajPreSimple, TrajPreAttnAvgLongUser, TrajPreLocalAttnLong

# auto-select MPS/CPU/CUDA
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def load_model(parameters, model_mode, path):
    if model_mode in ['simple', 'simple_long']:
        model = TrajPreSimple(parameters)
    elif model_mode == 'attn_avg_long_user':
        model = TrajPreAttnAvgLongUser(parameters)
    else:
        model = TrajPreLocalAttnLong(parameters)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()

def generate_next(model, loc_seq, tim_seq, steps, mode2, uid):
    """
    loc_seq, tim_seq: lists of ints, your seed trajectory
    steps: how many new points to generate
    returns full trajectory of length len(loc_seq)+steps
    """
    locs = loc_seq.copy()
    tims = tim_seq.copy()
    for _ in range(steps):
        x_loc = torch.LongTensor(locs).unsqueeze(1).to(device)
        x_tim = torch.LongTensor(tims).unsqueeze(1).to(device)
        with torch.no_grad():
            if mode2 in ['simple', 'simple_long']:
                scores = model(x_loc, x_tim)
            elif mode2 == 'attn_avg_long_user':
                # use seed as history
                history_loc = x_loc
                history_tim = x_tim
                history_count = [1] * history_loc.size(0)
                uid_tensor = torch.LongTensor([uid]).to(device)
                target_len = 1
                scores = model(x_loc, x_tim,
                               history_loc, history_tim,
                               history_count, uid_tensor, target_len)
            elif mode2 == 'attn_local_long':
                target_len = 1
                scores = model(x_loc, x_tim, target_len)
            else:
                raise ValueError(f"Unknown model_mode: {mode2}")
        # take the last prediction
        next_id = scores[-1].argmax().item()
        # append
        locs.append(next_id)
        tims.append((tims[-1]+1) % parameters.tim_size)  # or your own time logic
    return locs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_pk',     default='../data/foursquare.pk')
    parser.add_argument('--model_path',  default='../results/res.m')
    parser.add_argument('--model_mode',  default='attn_avg_long_user',
                        choices=['simple','simple_long','attn_avg_long_user','attn_local_long'])
    parser.add_argument('--user',        type=int, default=0,
                        help="which user id to seed")
    parser.add_argument('--session',     type=int, default=0,
                        help="which session index to seed")
    parser.add_argument('--seed_len',    type=int, default=5)
    parser.add_argument('--gen_steps',   type=int, default=10)
    args = parser.parse_args()

    # load raw pickle
    data = pickle.load(open(args.data_pk,'rb'))
    vid_list = data['vid_list']
    sessions = data['data_neural'][args.user]['sessions'][args.session]

    # seed
    seed = sessions[:args.seed_len]
    loc_seq = [p[0] for p in seed]
    tim_seq = [p[1] for p in seed]

    # build params & load model
    parameters = RnnParameterData(data_name='', data_path=args.data_pk.replace('.pk',''), save_path=args.model_path)
    parameters.data_neural = data['data_neural']
    parameters.tim_size = 48
    parameters.model_mode = args.model_mode
    parameters.loc_size = len(vid_list)
    parameters.uid_size = len(data['uid_list'])
    model = load_model(parameters, args.model_mode, args.model_path)

    # generate
    traj = generate_next(model, loc_seq, tim_seq,
                         args.gen_steps, args.model_mode, args.user)
    print("Generated trajectory (location IDs):", traj)
    # if you want original venue IDs, invert vid_list:
    id2pid = {v[0]:k for k,v in vid_list.items()}
    print("As original PIDs:", [id2pid[i] for i in traj])
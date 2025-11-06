"""Attack-style work location inference for trained DeepMove models."""

import argparse
import json
import os
import pickle
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from tqdm import tqdm

from model import TrajPreAttnAvgLongUser, TrajPreLocalAttnLong, TrajPreSimple
from train import RnnParameterData


if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def load_model(parameters: RnnParameterData, model_mode: str, ckpt_path: str) -> torch.nn.Module:
    if model_mode in ["simple", "simple_long"]:
        model = TrajPreSimple(parameters)
    elif model_mode == "attn_avg_long_user":
        model = TrajPreAttnAvgLongUser(parameters)
    else:
        model = TrajPreLocalAttnLong(parameters)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def next_time_slot(tim_seq: Sequence[int], tim_size: int) -> int:
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


def get_next_step_log_probs(
    model: torch.nn.Module,
    loc_seq: Sequence[int],
    tim_seq: Sequence[int],
    uid_idx: int,
    model_mode: str,
) -> torch.Tensor:
    loc_tensor = torch.tensor(loc_seq, dtype=torch.long, device=DEVICE).unsqueeze(1)
    tim_tensor = torch.tensor(tim_seq, dtype=torch.long, device=DEVICE).unsqueeze(1)

    if model_mode in ["simple", "simple_long"]:
        scores = model(loc_tensor, tim_tensor)
    elif model_mode == "attn_avg_long_user":
        history_count = [1] * len(loc_seq)
        uid_tensor = torch.tensor([uid_idx], dtype=torch.long, device=DEVICE)
        scores = model(
            loc_tensor,
            tim_tensor,
            loc_tensor,
            tim_tensor,
            history_count,
            uid_tensor,
            target_len=1,
        )
    else:
        scores = model(loc_tensor, tim_tensor, target_len=1)

    if scores.dim() == 1:
        return scores
    if scores.dim() == 2:
        return scores[-1]
    return scores[-1, 0, :]


def infer_location_from_slots(
    trajectory: Sequence[Tuple[int, int]],
    start_hour: int,
    end_hour: int,
    weekdays_only: bool,
) -> int:
    if not trajectory:
        raise ValueError("Trajectory is empty")

    counts: Counter[int] = Counter()
    has_weekday = False

    for loc, slot in trajectory:
        loc = int(loc)
        slot = int(slot)
        if loc <= 0:
            continue
        hour = slot % 24
        is_weekday = slot < 24
        if is_weekday:
            has_weekday = True
        if weekdays_only and not is_weekday:
            continue
        if start_hour <= hour < end_hour:
            counts[loc] += 1

    if weekdays_only and not has_weekday:
        raise ValueError("weekdays_only=True but trajectory contains no weekday data")
    if not counts:
        raise ValueError(f"No locations found in time window [{start_hour}, {end_hour})")

    return counts.most_common(1)[0][0]


def merge_sessions(sessions: Dict[int, List[List[int]]]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for sid in sorted(sessions.keys()):
        for loc, tim in sessions[sid]:
            merged.append((int(loc), int(tim)))
    return merged


def extract_prefix(trajectory: Sequence[Tuple[int, int]], cutoff_hour: int = 6) -> Tuple[List[int], List[int]]:
    loc_prefix: List[int] = []
    tim_prefix: List[int] = []
    for loc, slot in trajectory:
        hour = int(slot) % 24
        if hour < cutoff_hour:
            loc_prefix.append(int(loc))
            tim_prefix.append(int(slot))
        else:
            break
    return loc_prefix, tim_prefix


def deepmove_beam_search(
    model: torch.nn.Module,
    model_mode: str,
    uid_idx: int,
    base_loc: Sequence[int],
    base_tim: Sequence[int],
    beam_steps: int,
    beam_width: int,
    tim_size: int,
) -> List[Tuple[List[int], List[int], float]]:
    if beam_steps <= 0 or beam_width <= 0:
        return [(list(base_loc), list(base_tim), 0.0)]

    beam: List[Tuple[List[int], List[int], float]] = [(list(base_loc), list(base_tim), 0.0)]

    for _ in range(beam_steps):
        candidates: List[Tuple[List[int], List[int], float]] = []
        for loc_hist, tim_hist, logp in beam:
            if not loc_hist or not tim_hist:
                continue
            log_probs = get_next_step_log_probs(model, loc_hist, tim_hist, uid_idx, model_mode)
            vocab_size = log_probs.size(0)
            k = min(beam_width, vocab_size)
            values, indices = torch.topk(log_probs, k)
            next_tim = next_time_slot(tim_hist, tim_size)
            for score, loc_id in zip(values.tolist(), indices.tolist()):
                new_loc = loc_hist + [int(loc_id)]
                new_tim = tim_hist + [next_tim]
                candidates.append((new_loc, new_tim, logp + float(score)))

        if not candidates:
            break

        candidates.sort(key=lambda item: (-item[2], item[0]))
        beam = candidates[:beam_width]

    return beam


def rank_target_location(
    model: torch.nn.Module,
    model_mode: str,
    uid_idx: int,
    loc_hist: Sequence[int],
    tim_hist: Sequence[int],
    target_loc: int,
) -> int:
    log_probs = get_next_step_log_probs(model, loc_hist, tim_hist, uid_idx, model_mode)
    probs = log_probs.detach().cpu().tolist()
    order = list(range(len(probs)))
    order.sort(key=lambda idx: (-probs[idx], idx))
    try:
        return order.index(int(target_loc)) + 1
    except ValueError:
        return len(order)


def greedy_rollout_ranks(
    model: torch.nn.Module,
    model_mode: str,
    uid_idx: int,
    start_loc: Sequence[int],
    start_tim: Sequence[int],
    target_loc: int,
    greedy_steps: int,
    tim_size: int,
) -> List[int]:
    loc_hist = list(start_loc)
    tim_hist = list(start_tim)
    ranks: List[int] = []

    for _ in range(greedy_steps):
        rank = rank_target_location(model, model_mode, uid_idx, loc_hist, tim_hist, target_loc)
        ranks.append(rank)

        log_probs = get_next_step_log_probs(model, loc_hist, tim_hist, uid_idx, model_mode)
        probs = log_probs.detach().cpu().tolist()
        order = list(range(len(probs)))
        order.sort(key=lambda idx: (-probs[idx], idx))
        if not order:
            break

        best_next = order[0]
        next_tim = next_time_slot(tim_hist, tim_size)
        loc_hist.append(int(best_next))
        tim_hist.append(next_tim)

    return ranks


def attack_single_user(
    model: torch.nn.Module,
    model_mode: str,
    uid_idx: int,
    trajectory: Sequence[Tuple[int, int]],
    beam_width: int,
    tim_size: int,
    home_start: int = 0,
    home_end: int = 6,
    work_start: int = 10,
    work_end: int = 18,
) -> float:
    if len(trajectory) < 2:
        raise ValueError("Trajectory too short for attack")

    # Compute beam_steps (commute) and greedy_steps (work period)
    # Trajectory points are every 30 minutes = 2 per hour
    beam_steps = (work_start - home_end) * 2
    greedy_steps = (work_end - work_start) * 2

    work_loc = infer_location_from_slots(trajectory, start_hour=work_start, end_hour=work_end, weekdays_only=True)
    prefix_loc, prefix_tim = extract_prefix(trajectory, cutoff_hour=home_end)
    if not prefix_loc or not prefix_tim:
        raise ValueError(f"Prefix before {home_end}am is empty")

    beam_paths = deepmove_beam_search(
        model=model,
        model_mode=model_mode,
        uid_idx=uid_idx,
        base_loc=prefix_loc,
        base_tim=prefix_tim,
        beam_steps=beam_steps,
        beam_width=beam_width,
        tim_size=tim_size,
    )

    all_ranks: List[int] = []
    for loc_hist, tim_hist, _ in beam_paths:
        ranks = greedy_rollout_ranks(
            model=model,
            model_mode=model_mode,
            uid_idx=uid_idx,
            start_loc=loc_hist,
            start_tim=tim_hist,
            target_loc=work_loc,
            greedy_steps=greedy_steps,
            tim_size=tim_size,
        )
        all_ranks.extend(ranks)

    if not all_ranks:
        raise ValueError("No ranks computed for user")

    return sum(all_ranks) / len(all_ranks)


def evaluate_attack(
    data_pk: str,
    model_path: str,
    metadata_json: str,
    model_mode: str,
    beam_width: int,
    max_users: Optional[int],
    home_start: int = 0,
    home_end: int = 6,
    work_start: int = 10,
    work_end: int = 18,
) -> Tuple[pd.DataFrame, int]:
    if not os.path.exists(data_pk):
        raise FileNotFoundError(f".pk file not found: {data_pk}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(metadata_json):
        raise FileNotFoundError(f"metadata.json not found: {metadata_json}")

    params = RnnParameterData(metadata=metadata_json)
    params.model_mode = model_mode
    params.use_cuda = DEVICE.type != "cpu"
    params.rnn_type = getattr(params, "rnn_type", "LSTM")
    params.attn_type = getattr(params, "attn_type", "dot")
    params.dropout_p = getattr(params, "dropout_p", 0.3)
    params.tim_size = getattr(params, "tim_size", 48)

    with open(metadata_json, "r") as f:
        metadata = json.load(f)

    pid_mapping = metadata.get("pid_mapping", {})
    if not pid_mapping:
        raise ValueError("metadata.json missing pid_mapping")
    users = metadata.get("users", [])
    if not users:
        raise ValueError("metadata.json missing users list")

    vocab_size = len(pid_mapping)

    user_to_idx = {str(u): i for i, u in enumerate(users)}
    params.loc_size = len(pid_mapping) + 1  # reserve 0 for 'unk'
    params.uid_size = len(users)

    model = load_model(params, model_mode, model_path)
    model.eval()

    with open(data_pk, "rb") as f:
        data = pickle.load(f)

    sessions_all = data.get("data_neural", {})
    uid_list = data.get("uid_list", {})
    if not sessions_all or not uid_list:
        raise ValueError(".pk file missing required keys")

    idx_to_user = {value[0]: key for key, value in uid_list.items()}

    user_ids: List[str] = []
    avg_ranks: List[float] = []
    iterator: Iterable[Tuple[int, Dict[str, object]]] = sessions_all.items()
    if max_users is not None:
        iterator = list(iterator)[:max_users]

    for user_idx, udata in tqdm(list(iterator), desc="Users"):
        label = idx_to_user.get(user_idx)
        if label is None:
            continue
        uid_idx = user_to_idx.get(str(label))
        if uid_idx is None:
            continue

        sessions = udata.get("sessions", {})
        trajectory = merge_sessions(sessions)
        if not trajectory:
            continue

        filtered_traj = [(loc, tim) for loc, tim in trajectory if loc > 0]
        if not filtered_traj:
            continue

        try:
            infer_location_from_slots(trajectory, start_hour=0, end_hour=6, weekdays_only=False)
        except ValueError:
            continue

        try:
            avg_rank = attack_single_user(
                model=model,
                model_mode=model_mode,
                uid_idx=uid_idx,
                trajectory=filtered_traj,
                beam_width=beam_width,
                tim_size=params.tim_size,
                home_start=home_start,
                home_end=home_end,
                work_start=work_start,
                work_end=work_end,
            )
        except ValueError:
            continue

        avg_ranks.append(avg_rank)
        user_ids.append(str(label))

    if not avg_ranks:
        raise ValueError("No attack results computed")

    df = pd.DataFrame({"user_id": user_ids, "avg_work_rank": avg_ranks})
    df.sort_values("avg_work_rank", inplace=True)
    df = df[["user_id", "avg_work_rank"]]
    return df, vocab_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Attack difficulty evaluation for DeepMove models.")
    parser.add_argument("--data_pk", default=None, help="Path to processed .pk file")
    parser.add_argument("--data_dir", default=None, help="Path to directory with processed .pk files")
    parser.add_argument("--model_path", required=True, help="Path to trained DeepMove checkpoint (.m)")
    parser.add_argument("--metadata_json", required=True, help="Path to metadata.json containing pid mapping")
    parser.add_argument("--model_mode", default="attn_avg_long_user",
                        choices=["simple", "simple_long", "attn_avg_long_user", "attn_local_long"],
                        help="Model variant used during training")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width for the attack search")
    parser.add_argument("--max_users", type=int, default=None, help="Optional cap on number of users to evaluate")
    parser.add_argument("--output", default=None, help="Optional CSV output path for results")
    parser.add_argument("--home_start", type=int, default=0, help="Hour when home period starts (must be 0)")
    parser.add_argument("--home_end", type=int, default=6, help="Hour when home period ends (default: 6am)")
    parser.add_argument("--work_start", type=int, default=10, help="Hour when work period starts (default: 10am)")
    parser.add_argument("--work_end", type=int, default=18, help="Hour when work period ends (default: 6pm)")
    args = parser.parse_args()

    if bool(args.data_pk) == bool(args.data_dir):
        parser.error("Provide exactly one of --data_pk or --data_dir")

    if args.home_start != 0:
        parser.error("home_start must be 0 (non-zero home_start not yet implemented)")

    if not (args.home_start < args.home_end <= args.work_start < args.work_end <= 24):
        parser.error("Time periods must satisfy: home_start < home_end <= work_start < work_end <= 24")

    beam_width = max(1, args.beam_width)

    if args.data_dir:
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(f"Directory not found: {args.data_dir}")

        pk_files = sorted(
            os.path.join(args.data_dir, f)
            for f in os.listdir(args.data_dir)
            if f.endswith('.pk')
        )

        if not pk_files:
            raise FileNotFoundError(f"No .pk files found in {args.data_dir}")

        output_dir = args.output if args.output else 'attack_results'
        os.makedirs(output_dir, exist_ok=True)

        for pk_file in pk_files:
            print(f"Processing {pk_file}...")
            try:
                df, vocab_size = evaluate_attack(
                    data_pk=pk_file,
                    model_path=args.model_path,
                    metadata_json=args.metadata_json,
                    model_mode=args.model_mode,
                    beam_width=beam_width,
                    max_users=args.max_users,
                    home_start=args.home_start,
                    home_end=args.home_end,
                    work_start=args.work_start,
                    work_end=args.work_end,
                )
            except ValueError as err:
                print(f"  Skipping {pk_file}: {err}")
                continue

            df = df[["user_id", "avg_work_rank"]]
            df['max_rank'] = vocab_size
            mean_rank = df["avg_work_rank"].mean()
            basename = os.path.splitext(os.path.basename(pk_file))[0]
            out_path = os.path.join(output_dir, f"{basename}_attack.csv")
            df.to_csv(out_path, index=False)
            print(f"  ✓ Evaluated {len(df)} users. Mean rank: {mean_rank:.2f}")
            print(f"  Results written to {out_path}\n")
        return

    # Single file mode
    df, vocab_size = evaluate_attack(
        data_pk=args.data_pk,
        model_path=args.model_path,
        metadata_json=args.metadata_json,
        model_mode=args.model_mode,
        beam_width=beam_width,
        max_users=args.max_users,
        home_start=args.home_start,
        home_end=args.home_end,
        work_start=args.work_start,
        work_end=args.work_end,
    )
    df = df[["user_id", "avg_work_rank"]]
    df['max_rank'] = vocab_size
    mean_rank = df["avg_work_rank"].mean()
    print(f"\n✓ Evaluated {len(df)} users. Mean rank: {mean_rank:.2f}")

    if args.output:
        output_path = args.output
        if os.path.isdir(output_path):
            basename = os.path.splitext(os.path.basename(args.data_pk))[0]
            output_path = os.path.join(output_path, f"{basename}_attack.csv")
        else:
            parent = os.path.dirname(output_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
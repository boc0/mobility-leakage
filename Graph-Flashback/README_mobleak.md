# Usage (Graph-Flashback)

## Installation

We recommend using a virtual environment.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r Graph-Flashback/requirement.txt
```

## Data Format

Input trajectory CSV files should have at least the columns:

```
tid,timestamp,lat,lon
```

Each row is a point in a trajectory. A `tid` groups consecutive rows into one trajectory. All files you plan to use (training + evaluation subsets) must be preprocessed together so the global POI vocabulary includes every location that will appear at test time.

## Preprocessing

Preprocessing converts the CSV directory into Graph-Flashback raw TXT files and builds graphs plus `metadata.json`.

Script: `Graph-Flashback/preprocess.py`

Arguments:
- `--in_dir`: directory containing source CSV files
- `--out_dir`: output directory for TXT files, graphs, and `metadata.json`
- `--user_mode`: how to derive user id from `tid` (`tid_full` or `tid_prefix`)

Example:
```bash
python3 Graph-Flashback/preprocess.py \
  --in_dir path/to/csv_dir \
  --out_dir results/flashback/preprocessed \
  --user_mode tid_full
```

Outputs inside `out_dir`:
- One `<base>.txt` per input CSV (tab-separated: `user_id\ttimestamp\tlat\tlon\tpoi_id`)
- `metadata.json` with `users`, `pid_mapping`, counts, and graph file paths
- Global graphs: `trans_loc.pkl`, `trans_interact.pkl`
- Per-file graphs: `trans_loc_<base>.pkl`, `trans_interact_<base>.pkl`


## Training

Script: `Graph-Flashback/train.py` (uses `setting.py` CLI). You can train on a subset file (e.g., `training_set.txt`) while providing union-shaped graphs for full vocabulary support.

Key arguments (see `setting.py` for defaults):
- `--dataset` path to the TXT training file
- `--trans_loc_file` path to global (or subset) transition POI graph (pickled COO)
- `--trans_interact_file` path to global (or subset) user→POI interaction graph
- `--save_dir` directory for checkpoints and `res.txt`
- `--log_file` base path for log (timestamp appended)
- `--epochs` training epochs
- `--batch-size` user batch size (default 200 for Gowalla-style)
- `--gpu` set `-1` for CPU, otherwise GPU id

Example (union graphs + subset training file):
```bash
python3 Graph-Flashback/train.py \
  --dataset results/flashback/preprocessed/training_set.txt \
  --trans_loc_file results/flashback/preprocessed/trans_loc.pkl \
  --trans_interact_file results/flashback/preprocessed/trans_interact.pkl \
  --save_dir results/flashback/models \
  --log_file results/flashback/log_gowalla \
  --epochs 20 \
  --batch-size 200 \
```

Outputs in `save_dir`:
- Per-epoch checkpoints: `flashback_<timestamp>_epochN.pt`
- Rolling latest checkpoint: `flashback_<timestamp>_latest.pt`
- Learning curve: `res.txt` (JSON with train_loss, valid_loss, accuracy)

## Accuracy Evaluation (Top-k / Mean Rank)

Script: `Graph-Flashback/test.py`

Modes:
- `topk`: per-user top-k hit rates for each provided K
- `rank`: per-user average rank of the true next location

Automatic graph inference: If you omit `--trans_loc_file` / `--trans_interact_file`, the script looks for per-file graphs named `trans_loc_<base>.pkl` and `trans_interact_<base>.pkl` next to the TXT file.

Arguments:
- `--data_file` or `--data_dir` (mutually exclusive)
- `--model_path` checkpoint `.pt` model file from training
- `--output_dir` destination for CSVs
- `--mode` `topk` or `rank`
- `--k_values` list for top-k (default 1 5 10)
- Graph flags: `--use_spatial_graph`, `--use_graph_user` if corresponding graphs supplied

Example (directory, auto per-file graphs):
```bash
python3 Graph-Flashback/test.py \
  --data_dir results/flashback/preprocessed \
  --model_path results/flashback/models/flashback_<timestamp>_latest.pt \
  --output_dir results/flashback/eval \
  --mode topk --k_values 1 5 10 --gpu -1
```

Output per TXT: `<base>_topk.csv` with columns:
```
tid,top-1,top-5,top-10
```
IDs are original trajectory IDs from `metadata.json`. Use `verify_ids.py` to confirm mapping:
```bash
python3 Graph-Flashback/verify_ids.py \
  --txt results/flashback/preprocessed/cluster_0.txt \
  --csv results/flashback/eval/cluster_0_topk.csv \
  --meta results/flashback/preprocessed/metadata.json
```

## Perplexity

Script: `Graph-Flashback/perplexity.py`

Computes per-user sequence perplexity (exp of mean negative log-likelihood). Blank values appear where the user’s sequence produced no valid tokens.

Example:
```bash
python3 Graph-Flashback/perplexity.py \
  --data_dir results/flashback/preprocessed \
  --model_path results/flashback/models/flashback_<timestamp>_latest.pt \
  --output_dir results/flashback/eval --gpu -1
```

Output per TXT: `<base>_perplexity.csv`:
```
tid,perplexity
```

## Extraction Difficulty (Rank-Based Proxy)

Script: `Graph-Flashback/extract.py`

Computes proxy difficulty scores by ranking ground-truth next locations at different prefix truncations of the sequence (similar concept to DeepMove’s extraction). For each prefix length `P`, the script aggregates ranks over the remainder of the sequence; higher values ⇒ harder extraction.

Arguments:
- `--prefix_lengths` list of prefix truncation lengths (e.g. `0 3 5`)
- Same graph inference behavior as `test.py`
- `--data_file` or `--data_dir` (mutually exclusive)
- `--model_path`, `--output_dir`

Example:
```bash
python3 Graph-Flashback/extract.py \
  --data_dir results/flashback/preprocessed \
  --model_path results/flashback/models/flashback_<timestamp>_latest.pt \
  --output_dir results/flashback/eval \
  --prefix_lengths 0 3 5 --gpu -1
```

Output per TXT: `<base>_extraction.csv`:
```
tid,prefix-0,prefix-3,prefix-5
```
Empty cells indicate unavailable rank data for that user (e.g. insufficient sequence after truncation).

## Common Troubleshooting



## Example End-to-End (CPU)
```bash
# 1. Preprocess
python3 Graph-Flashback/preprocess.py --in_dir data/csv_all --out_dir results/flashback/preprocessed --user_mode tid_full

# 2. Train
python3 Graph-Flashback/train.py \
  --dataset results/flashback/preprocessed/training_set.txt \
  --trans_loc_file results/flashback/preprocessed/trans_loc.pkl \
  --trans_interact_file results/flashback/preprocessed/trans_interact.pkl \
  --save_dir results/flashback/models --epochs 20 --gpu -1

# 3. Accuracy
python3 Graph-Flashback/test.py --data_dir results/flashback/preprocessed \
  --model_path results/flashback/models/flashback_<timestamp>_latest.pt \
  --output_dir results/flashback/eval --mode topk --k_values 1 5 10 --gpu -1

# 4. Perplexity
python3 Graph-Flashback/perplexity.py --data_dir results/flashback/preprocessed \
  --model_path results/flashback/models/flashback_<timestamp>_latest.pt \
  --output_dir results/flashback/eval --gpu -1

# 5. Extraction Difficulty
python3 Graph-Flashback/extract.py --data_dir results/flashback/preprocessed \
  --model_path results/flashback/models/flashback_<timestamp>_latest.pt \
  --output_dir results/flashback/eval --prefix_lengths 0 3 5 --gpu -1

# 6. Verify IDs (example for one output)
python3 Graph-Flashback/verify_ids.py \
  --txt results/flashback/preprocessed/cluster_0.txt \
  --csv results/flashback/eval/cluster_0_topk.csv \
  --meta results/flashback/preprocessed/metadata.json
```


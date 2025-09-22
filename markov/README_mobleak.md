# Markov Baseline Usage

This guide explains how to train and evaluate the Markov baseline model on CSV trajectory data files.

## Install

We recommend using a virtual environment:

```bash
python3 -m venv markov/venv
source markov/venv/bin/activate
```

Install dependencies:

```bash
pip install -r markov/requirements.txt
```


## Data format (CSV)

Input files are sets of trajectories represented as a single series of points with columns: `tid` (trajectory/user id), `lat`, `lon`, `timestamp`. Each trajectory appears contiguously in the file. Example data:

```
tid,lat,lon,timestamp
126,40.799,-73.968,2025-07-05 00:00:00
126,40.794,-73.971,2025-07-05 01:00:00
126,40.833,-73.941,2025-07-05 02:00:00
126,40.757,-73.914,2025-07-05 14:00:00
126,40.749,-73.937,2025-07-05 17:00:00
127,40.834,-73.945,2025-06-30 13:00:00
127,40.567,-73.882,2025-06-30 19:00:00
127,40.689,-73.981,2025-06-30 23:00:00
127,40.708,-73.991,2025-06-30 23:00:00
127,40.833,-73.941,2025-07-01 14:00:00
127,40.708,-73.991,2025-07-01 17:00:00
```

## Train

Train a Markov model from a CSV:

```bash
python3 markov/train.py --data_csv path/to/train.csv --save_dir run0 --state_size 2
```

- `--state_size` controls the Markov order (number of previous locations used for prediction).
- The trained model is saved to `run0/model.json`.

## Evaluate

### Perplexity

Compute per-sequence perplexity for all files in a directory:

```bash
python3 markov/perplexity.py --model run_markov/model.json --data_dir path/to/eval_dir --output_dir run0/perplexity
```

- Scans `--data_dir` for `.csv` files.
- Writes one output CSV per input file to `--output_dir`, named `<input_basename>_perplexity.csv`.
- Each row in the output CSVs corresponds to one sequence indentified by `tid` and its computed perplexity.

### Accuracy

Compute top-k accuracy:

```bash
# Top-k accuracy (defaults internally to k = 1,5,10)
python3 markov/test.py --model run_markov/model.json --data_dir path/to/eval_dir --output_dir results/test --mode topk --k_values 1 5 10
```
- `--mode` can be `topk` or `mean_rank`.
- `--k_values` specifies the k values for top-k accuracy.
- Writes one CSV per input file to `--output_dir`, named `<input_basename>_topk.csv`.


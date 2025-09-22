# LSTPM Usage

This guide shows how to preprocess your trajectory CSVs into the pickle format used by LSTPM training, and how to evaluate a trained model. It mirrors the DeepMove pipeline but with an LSTPM-specific orchestrator and uses a shared distance matrix.

## Install

We recommend a virtual environment:

```bash
python3 -m venv LSTPM/venv
source LSTPM/venv/bin/activate
```

Install requirements:

```bash
pip install -r LSTPM/requirements.txt
```

## Data preprocessing

Data files are sets of trajectories represented as one single series of points containing latitude ('lat'), longitude ('lon'), timestamp ('ts'). The individual trajectories are then identified using a fourth trajectory id ('tid') column, with each trajectory appearing in the dataset contiguously. All training and testing data should be in this format, with the entire training dataset in one file. Example file:

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

Since the model predicts locations as classes, it needs to know the set of all possible locations, including all those contained in possible test data. Therefore, all the training and testing data should be preprocessed together. The following assumes a folder of data CSV files, with one of those intended as a training set with an appropriate name.

For each run, we recommend you create a folder that characterizes the run, e.g. by adding hyperparemeter values to the folder name. Intermediate files for that run can be stored in that folder. This guide assumes the folder name `run0`.

From the root directory of `mobleak_seq`, first preprocess your data directory into the format of LSTPM with

```bash
python3 LSTPM/train/preprocess.py --in_dir path/to/your_trajs --training_set_name train --out_dir run0
```

This will put the resulting files in a `preprocessed` subfolder of `run0`. It will also produce `metadata.json` and a `distance.pkl` file in `run0`. The resulting folder structure will be:

```
run0/
  metadata.json         # global pid/user mapping
  distance.pkl          # global (N+1)x(N+1) poi distance matrix
  txts/
    <each>.txt          # intermediate files
  preprocessed/
    <each>.pk           # datasets for training/eval
```


## Training

```bash
python3 LSTPM/train/train.py --data_pk run0/preprocessed/training_set.pk --metadata_json run0/metadata.json --distance run0/distance.pkl --save_dir run0/training --batch_size 512
```

This produces a model file in `run0/training/res.m` and checkpoints during training in `run0/training/checkpoint/`.

## Evaluation

You can test a single file or an entire directory of `.pk` files.

### Accuracy (top-k or mean rank)

- Single file:
```bash
python3 LSTPM/train/test.py --data_pk run0/preprocessed/cluster_0.pk --model_m run0/training/res.m --distance run0/distance.pkl --mode topk --k_values 1 5 10 --output run0/test/cluster_0.csv
```

- Directory:
```bash
python3 LSTPM/train/test.py --data_dir run0/preprocessed --model_m run0/training/res.m --distance run0/distance.pkl --mode topk --k_values 1 5 10 --output run0/test
```

### Perplexity

- Single file:
```bash
python3 LSTPM/train/perplexity.py --data_pk run0/preprocessed/cluster_0.pk --model_m run0/training/res.m --distance run0/distance.pkl --output run0/perplexities/cluster_0.csv
```

- Directory:
```bash
python3 LSTPM/train/perplexity.py --data_dir run0/preprocessed --model_m run0/training/res.m --distance run0/distance.pkl --output run0/perplexities
```

Both scripts can also be run on a single file too, by replacing the `--data_dir` argument with `--data_pk path/to/single/file.pk`.


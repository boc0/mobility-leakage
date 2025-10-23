# Usage

## Installation

We recommend using a virtual environment. Create and activate one with

```bash
python3 -m venv DeepMove/venv
source DeepMove/venv/bin/activate
```

Then install requirements with

```bash
pip install -r DeepMove/requirements.txt
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

From the root directory of `mobleak_seq`, first preprocess your data directory into the format of DeepMove with

```bash
python3 DeepMove/codes/preprocess.py --in_dir path/to/your_trajs --training_set_name <training_set_name> --out_dir run0
```

where `path/to/your_trajs` is the path to the folder containing your data files, and `<training_set_name>` is the name of the file you want to use as training data (without extension). This will create a metadata file at `run0/metadata.json` which is needed to use the model in the next steps and preprocessed data files in `run0/preprocessed/`.

## Training

Run the training script on your preprocessed training data with

```bash
python3 DeepMove/codes/main.py --metadata_json run0/metadata.json --model_mode <model_type> --data_path run0/preprocessed/<training_set_name.pk> --epoch_max 40 --save_dir run0/training
```

where `<train_pk>` is the path to the processed training data in pickle format, `<path_to_save_model>` is where you want to save the trained model, and `<model_type>` can be one of `simple`, `simple_long`, `attn_avg_long_user`, or `attn_local_long`.

The 'pretrain' flag is set to 0 to train a new model. You can also set the `--pretrain` flag to 1 to load a pretrained model -- but in this case, you need to specify the path to the pretrained model with `--pretrain_model_path <path_to_pretrained_model>`.

This produces a model file in `run0/training/res.m` and checkpoints during training in `run0/training/checkpoint/`.


## Evaluation

### Accuracy

```bash
python3 DeepMove/codes/test.py --metadata_json run0/metadata.json --model_mode <model_type> --model_path run0/training/res.m --data_dir run0/preprocessed/ --mode topk --k_values 1 5 10 20 --output run0/test                   
```

This produces a CSV file for each data file in `run0/test`, containing for each trajectory in that file the top-k accuracy across that trajectory for each of the selected k values (specified in `--k_values`).
The additional `--mode rank` produces instead one value per trajectory, the average rank of the true next location in the predicted ranking.


### Perplexity

```bash
python3 DeepMove/codes/perplexity.py --metadata_json run0/metadata.json --model_mode <model_type> --model_path run0/training/res.m --data_dir run0/preprocessed --output run0/perplexities               
```

where `<model_type>` is the same as before.

Both `test.py` and `perplexity.py` can be run a single file too, by replacing the `--data_dir` argument with `--data_pk path/to/single/file.pk` and `--output` with a target path for the single output file.

### Trajectory difficulty proxy

Use `DeepMove/codes/extract.py` to turn model predictions into rank-based extraction difficulty scores for each trajectory.

```bash
python3 DeepMove/codes/extract.py --metadata_json run0/metadata.json --model_mode simple --model_path run0/training/res.m --data_dir run0/preprocessed --prefix_lengths 0 3 5 --batch_size 64 --output run0/extraction
```

- Accepts either a single `.pk` via `--data_pk` (with `--output` as the destination file) or a directory with `--data_dir` (writes one CSV per input).
- Each output row contains the trajectory id and proxy columns `prefix-<N>` encoding ranks after truncating the history to that prefix length.
# Usage

## Data preprocessing

You need your dataset split in train and test sets in the default format of mobleak.

Then preprocess your data first with

```bash
python3 codes/convert.py --train_csv <path_to_train_csv> --test_csv <path_to_test_csv>
```

This saves two data files in the `data/foursquare/` directory: tweets_clean.txt (training data), tweets_test.txt (test data) and venues.json (mapping from venue id to location)

Then finally preprocess with

```bash
python3 codes/sparse_traces.py
```
For the training data and 

```bash
python3 codes/sparse_traces.py --data_path data/foursquare/tweets_test.txt --save_name foursquare_test
```
For the test data.

## Training

Run the training script with:

```bash
python3 codes/main.py --model_mode=<model_type> --pretrain=0 --epoch_max 100
```

where `<model_type>` can be one of `simple`, `simple_long`, `attn_avg_long_user`, or `attn_local_long`.

This produces a model file in results/res.m and checkpoints during training in results/checkpoint/.


## Perplexity evaluation

```bash
python3 codes/perplexity.py --data_pk data/foursquare_test.pk --model_path results/res.m --model_mode=<model_type>
```

where `<model_type>` is the same as before.
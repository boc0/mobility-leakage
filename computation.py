#!/usr/bin/env python
# coding: utf-8



from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import json

# Markov Wrapper functions
def train_markov_model(train_csv, save_dir, early_stopping=5, epoch_max=40, state_size=2):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # if model.json already exists, skip training
    model_json = save_dir / "model.json"
    if model_json.exists():
        print(f"⏭️ Model already trained at {model_json}, skipping training.")
        return model_json

    subprocess.run([
        "python3", "markov/train.py",
        "--data_csv", str(train_csv),
        "--save_dir", str(save_dir),
        "--state_size", str(state_size)
    ], check=True)
    print(f"✅ Model trained and saved to {model_json}")
    return model_json


def evaluate_perplexity(model_path, data_dir, run_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count CSVs in input and output dirs
    input_files = list(data_dir.glob("*.csv"))
    output_files = list(output_dir.glob("*.csv"))
    if len(input_files) == len(output_files):
        print(f"⏭️ Perplexity already computed for all files in {output_dir}, skipping.")
        return

    print(f"⚙️ Running perplexity on {len(input_files)} input files...")
    subprocess.run([
        "python3", "markov/perplexity.py",
        "--model", str(model_path),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
    ], check=True)

    print(f"✅ Perplexity results saved to {output_dir}")


def test_markov_model(model_path, data_dir, run_dir, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = list(data_dir.glob("*.csv"))
    output_files = list(output_dir.glob("*.csv"))

    if len(input_files) == len(output_files):
        print(f"⏭️ Test results already exist in {output_dir}, skipping.")
        return

    print(f"⚙️ Running top-k test on {len(input_files)} input files...")
    subprocess.run([
        "python3", "markov/test.py",
        "--model", str(model_path),
        "--data_dir", str(data_dir),
        "--output_dir", str(output_dir),
        "--mode", "topk",
        "--k_values", "1", "5", "10"
    ], check=True)

    print(f"✅ Test results saved to {output_dir}")

def extract_suffix_markov_difficulty(model_path, data_dir, run_dir, output_csv, prefix_lengths=(6, 12, 24), alpha=0.5):
    """
    Compute Markov model trajectory difficulty proxy for a single CSV.

    Args:
        model_path (Path): Path to trained Markov model.json
        data_path (Path): Path to a single evaluation CSV file
        output_csv (Path): Path where difficulty CSV should be written
    """
    data_csv = data_dir / "training_set.csv"
    output_csv = Path(output_csv) / "suffix_extraction.csv"
    print(output_csv)
    if output_csv.exists():
        print(f"⏭️ Markov difficulty already exists at {output_csv}, skipping.")
        return output_csv

    print(f"📊 Extracting Markov difficulty proxy for {data_csv.name}...")
    subprocess.run([
        "python3", "markov/extract.py",
        "--model", str(model_path),
        "--data_csv", str(data_csv),
        "--prefix_lengths", *map(str, prefix_lengths),
        "--alpha", str(alpha),
        "--output", str(output_csv)
    ], check=True)

    print(f"✅ Markov difficulty proxy saved to {output_csv}")
    return output_csv


# LSTPM Wrapper functions
def train_lstpm_model(train_csv, save_dir, early_stopping=5, epoch_max=40):
    """
    Preprocess + train LSTPM model, with skip checks.

    Args:
        train_csv (Path): path to training_set.csv
        save_dir (Path): directory to save model

    Returns:
        (Path to trained model .m, Path to preprocessed_dir)
    """
    save_dir = Path(save_dir)
    run_dir = save_dir.parent
    preprocessed_dir = run_dir / "preprocessed"
    metadata_path = run_dir / "metadata.json"
    distance_path = run_dir / "distance.pkl"

    model_path = save_dir / "res.m"

    # ----- 1. Check if model already trained -----
    if model_path.exists():
        print(f"⏭️ LSTPM model already trained at {model_path}, skipping training.")
        return model_path

    # ----- 2. Check if preprocessed already done -----
    input_csvs = list(train_csv.parent.glob("*.csv"))
    pk_files = list(preprocessed_dir.glob("*.pk"))

    if len(input_csvs) == len(pk_files) and len(pk_files) > 0:
        print(f"⏭️ Preprocessed data already present in {preprocessed_dir}, skipping preprocessing.")
    else:
        print("⚙️ Preprocessing LSTPM data...")
        result = subprocess.run([
            "python3", "LSTPM/train/preprocess.py",
            "--in_dir", str(train_csv.parent),
            "--training_set_name", train_csv.stem,
            "--out_dir", str(run_dir)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Preprocessing failed!")
            print("STDOUT:\n", result.stdout)
            print("STDERR:\n", result.stderr)
            raise RuntimeError("Preprocessing failed.")
        print(f"✅ Preprocessing completed, files saved to {preprocessed_dir}")

    # ----- 3. Train model -----
    print("🎯 Training LSTPM...")
    subprocess.run([
        "python3", "LSTPM/train/train.py",
        "--data_pk", str(preprocessed_dir / f"{train_csv.stem}.pk"),
        "--metadata_json", str(metadata_path),
        "--early_stopping", str(early_stopping),
        "--distance", str(distance_path),
        "--save_dir", str(save_dir),
        "--batch_size", "256",
        "--epochs", str(epoch_max)
    ], check=True)
    print(f"✅ LSTPM model saved at {model_path}")

    return model_path

def test_lstpm_model(model_path, data_dir, run_dir, output_dir):
    output_dir = Path(output_dir)
    if len(list(output_dir.glob("*.csv"))) == len(list(Path(data_dir).glob("*.csv"))):
        print(f"⏭️ LSTPM test results already exist in {output_dir}")
        return

    print(f"📊 Testing LSTPM...")
    subprocess.run([
        "python3", "LSTPM/train/test.py",
        "--data_dir", str(Path(run_dir) / "preprocessed"),
        "--model_m", str(model_path),
        "--distance", str(model_path.parent.parent / "distance.pkl"),
        "--mode", "topk",
        "--k_values", "1", "5", "10",
        "--output", str(output_dir)
    ], check=True)

def evaluate_lstpm_perplexity(model_path, data_dir, run_dir, output_dir):
    output_dir = Path(output_dir)
    preprocessed_dir = Path(run_dir) / "preprocessed"
    if len(list(output_dir.glob("*.csv"))) == len(list(Path(preprocessed_dir).glob("*.pk"))):
        print(f"⏭️ LSTPM perplexity already computed for {output_dir}")
        return

    print(f"📈 Evaluating LSTPM perplexity...")
    subprocess.run([
        "python3", "LSTPM/train/perplexity.py",
        "--data_dir", str(preprocessed_dir),
        "--model_m", str(model_path),
        "--distance", str(model_path.parent.parent / "distance.pkl"),
        "--output", str(output_dir)
    ], check=True)

def extract_suffix_lstpm_difficulty(model_path, data_dir, run_dir, output_csv,
                                    prefix_lengths=(6, 12, 24), batch_size=256):
    """
    Compute LSTPM model trajectory difficulty proxy for a single .pk file.

    Args:
        model_path (Path): Path to trained LSTPM .m model
        data_pk (Path): Path to preprocessed .pk file
        output_csv (Path): Path where difficulty CSV should be written
    """
    preprocessed_dir = Path(run_dir) / "preprocessed"
    training_pk = preprocessed_dir / "training_set.pk"
    output_csv = Path(output_csv) / "suffix_extraction.csv"
    if output_csv.exists():
        print(f"⏭️ LSTPM difficulty already exists at {output_csv}, skipping.")
        return output_csv

    print(f"📊 Extracting LSTPM difficulty proxy for {training_pk.name}...")
    subprocess.run([
        "python3", "LSTPM/train/extract.py",
        "--data_pk", str(training_pk),
        "--model_m", str(model_path),
        "--distance", str(model_path.parent.parent / "distance.pkl"),
        "--prefix_lengths", *map(str, prefix_lengths),
        #"--batch_size", str(batch_size),
        "--output", str(output_csv)
    ], check=True)

    print(f"✅ LSTPM difficulty proxy saved to {output_csv}")
    return output_csv


# DeepMove Wrapper functions
def train_deepmove_model(train_csv, save_dir, early_stopping=5, epoch_max=40, model_type="simple"):
    save_dir = Path(save_dir)
    run_dir = save_dir.parent
    preprocessed_dir = run_dir / "preprocessed"
    metadata_path = run_dir / "metadata.json"
    train_pk = preprocessed_dir / f"{train_csv.stem}.pk"

    # Skip if model already trained
    if (save_dir / "res.m").exists():
        print(f"⏭️ DeepMove ({model_type}) model already trained at {save_dir}, skipping.")
        return save_dir / "res.m"

    # Skip preprocessing if already done
    if not train_pk.exists():
        print("⚙️ Preprocessing DeepMove data...")
        subprocess.run([
            "python3", "DeepMove/codes/preprocess.py",
            "--in_dir", str(train_csv.parent),
            "--training_set_name", train_csv.stem,
            "--out_dir", str(run_dir)
        ], check=True)
    else:
        print(f"⏭️ Preprocessed file {train_pk} already exists, skipping preprocessing.")

    # Train
    print(f"🎯 Training DeepMove model ({model_type})...")
    subprocess.run([
        "python3", "DeepMove/codes/main.py",
        "--metadata_json", str(metadata_path),
        "--model_mode", model_type,
        "--data_path", str(train_pk),
        "--epoch_max", str(epoch_max),
        "--early_stopping", str(early_stopping),
        "--save_dir", str(save_dir),
        "--pretrain", "0"
    ], check=True)

    return save_dir / "res.m"

def test_deepmove_model(model_path, data_dir, run_dir, output_dir, model_type):
    output_dir = Path(output_dir)
    preprocessed_dir = Path(run_dir) / "preprocessed"
    if len(list(output_dir.glob("*.csv"))) == len(list(Path(preprocessed_dir).glob("*.pk"))):
        print(f"⏭️ DeepMove test results already exist in {output_dir}")
        return

    print(f"📊 Testing DeepMove ({model_type})...")
    subprocess.run([
        "python3", "DeepMove/codes/test.py",
        "--metadata_json", str(model_path.parent.parent / "metadata.json"),
        "--model_mode", model_type,
        "--model_path", str(model_path),
        "--data_dir", str(preprocessed_dir),
        "--mode", "topk",
        "--k_values", "1", "5", "10", "20",
        "--output", str(output_dir)
    ], check=True)

def perplexity_deepmove(model_path, data_dir, run_dir, output_dir, model_type):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dir = Path(model_path.parent.parent / "preprocessed")
    input_files = list(run_dir.glob("*.pk"))
    output_files = list(output_dir.glob("*.csv"))

    if len(input_files) == len(output_files):
        print(f"⏭️ DeepMove perplexity already computed for all files in {output_dir}")
        return

    print(f"📈 Evaluating DeepMove ({model_type}) perplexity on {len(input_files)} files...")

    for pk_file in input_files:
        out_file = output_dir / f"{pk_file.stem}.csv"
        if out_file.exists():
            print(f"⏭️ Skipping already computed file: {out_file.name}")
            continue

        print(f"Computing perplexity for {pk_file.name}...")
        subprocess.run([
            "python3", "DeepMove/codes/perplexity.py",
            "--metadata_json", str(model_path.parent.parent / "metadata.json"),
            "--model_mode", model_type,
            "--model_path", str(model_path),
            "--data_pk", str(pk_file),
            "--output", str(out_file)
        ], check=True)

    print(f"✅ DeepMove perplexity evaluation completed for all new files.")

def extract_suffix_deepmove_difficulty(model_path, data_dir, run_dir, output_csv,
                                       prefix_lengths=(6, 12, 24), batch_size=64, model_type="simple"):
    """
    Compute DeepMove model trajectory difficulty proxy for a single .pk file.

    Args:
        model_path (Path): Path to trained DeepMove .m model
        metadata_json (Path): Path to metadata.json
        model_type (str): Model type ('simple' or 'attn')
        data_pk (Path): Path to preprocessed .pk file
        output_csv (Path): Path where difficulty CSV should be written
    """
    training_pk = Path(run_dir / "preprocessed" / "training_set.pk")
    output_csv = Path(output_csv) / "suffix_extraction.csv"
    if output_csv.exists():
        print(f"⏭️ DeepMove difficulty already exists at {output_csv}, skipping.")
        return output_csv

    print(f"📊 Extracting DeepMove ({model_type}) difficulty proxy for {training_pk.name}...")
    subprocess.run([
        "python3", "DeepMove/codes/extract.py",
        "--metadata_json", str(model_path.parent.parent / "metadata.json"),
        "--model_mode", model_type,
        "--model_path", str(model_path),
        "--data_pk", str(training_pk),
        "--prefix_lengths", *map(str, prefix_lengths),
        "--batch_size", str(batch_size),
        "--output", str(output_csv)
    ], check=True)

    print(f"✅ DeepMove difficulty proxy saved to {output_csv}")
    return output_csv

def extract_work_deepmove_difficulty(model_path, data_dir, run_dir, output_dir, model_type="simple", beam_width=10):
    """
    Compute DeepMove model difficulty for inferring work location from home location.

    Args:
        model_path (Path): Path to trained DeepMove .m model
        run_dir (Path): Directory containing the 'preprocessed' data
        output_dir (Path): Directory where output CSV should be saved
        model_type (str): Model variant (e.g., 'simple', 'simple_long', 'attn_avg_long_user', 'attn_local_long')
        beam_width (int): Width of the beam search used during inference
    """
    output_dir = Path(output_dir)
    output_csv = output_dir / "infer_work.csv"
    preprocessed_dir = Path(run_dir) / "preprocessed"
    data_pk = preprocessed_dir / "training_set.pk"
    metadata_json = run_dir / "metadata.json"

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_csv.exists():
        print(f"⏭️ DeepMove infer_work results already exist at {output_csv}, skipping.")
        return output_csv

    print(f"📊 Running DeepMove infer_work for {model_type} on {data_pk.name}...")
    subprocess.run([
        "python3", "DeepMove/codes/infer_work.py",
        "--metadata_json", str(metadata_json),
        "--model_mode", model_type,
        "--model_path", str(model_path),
        "--data_pk", str(data_pk),
        "--output", str(output_csv),
        "--beam_width", str(beam_width)
    ], check=True)

    print(f"✅ DeepMove infer_work results saved to {output_csv}")
    return output_csv


# Graphflashback Wrapper functions
def train_graph_flashback_model(train_csv, save_dir,  epoch_max=40, user_mode="tid_full", batch_size=200, gpu="0", early_stopping=5):
        """
        Preprocess CSV trajectories and train a Graph-Flashback model.

        Args:
            train_dir (Path): Directory containing input CSVs.
            save_dir (Path): Directory to save model and logs.
            user_mode (str): Mode for deriving user_id from tid ('tid_full' or 'tid_prefix').
            epochs (int): Number of training epochs.
            batch_size (int): Batch size.
            gpu (str): GPU id to use; use "-1" for CPU.
        """
        train_dir = Path(train_csv).parent
        save_dir = Path(save_dir)
        preprocessed_dir = save_dir / "preprocessed"
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        training_file = preprocessed_dir / f"{train_csv.stem}.txt"
        trans_loc_file = preprocessed_dir / "trans_loc.pkl"
        trans_interact_file = preprocessed_dir / "trans_interact.pkl"
        log_file = save_dir / "log_flashback"

        # Skip if model already trained
        if (save_dir / "flashback_latest.pt").exists():
            print(f"⏭️ Graph flashback model already trained at {save_dir}, skipping.")
            return save_dir / "flashback_latest.pt"

        if not training_file.exists():
            print("⚙️ Preprocessing Graph-Flashback data...")
            subprocess.run([
                "python3", "Graph-Flashback/preprocess.py",
                "--in_dir", str(train_dir),
                "--out_dir", str(preprocessed_dir),
                "--user_mode", user_mode
            ], check=True)
        else:
            print(f"⏭️ Preprocessed file {training_file} already exists, skipping preprocessing.")


        print("🎯 Training Graph-Flashback model...")
        subprocess.run([
            "python3", "Graph-Flashback/train.py",
            "--dataset", str(training_file),
            "--trans_loc_file", str(trans_loc_file),
            "--trans_interact_file", str(trans_interact_file),
            "--save_dir", str(save_dir),
            "--log_file", str(log_file),
            "--epochs", str(epoch_max),
            "--batch-size", str(batch_size),
            "--gpu", str(gpu)
        ], check=True)

        print(f"✅ Graph-Flashback model trained and saved in {save_dir}")
        return save_dir / "flashback_latest.pt"


def test_graph_flashback_model(model_path, data_dir, run_dir, output_dir, mode="topk", k_values=(1, 5, 10), gpu="0"):
    preprocessed_dir = run_dir / "model"/"preprocessed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_flashback_base = Path("Graph-Flashback")
    print("📊 Evaluating Graph-Flashback accuracy...")
    subprocess.run([
        "python3", str(graph_flashback_base / "test.py"),
        "--data_dir", str(preprocessed_dir),
        "--model_path", str(model_path),
        "--output_dir", str(output_dir),
        "--mode", mode,
        "--k_values", *map(str, k_values),
        "--gpu", str(gpu)
    ], check=True)


def perplexity_graph_flashback(model_path, data_dir, run_dir, output_dir, gpu="0"):
    preprocessed_dir = run_dir / "model"/"preprocessed"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph_flashback_base = Path("Graph-Flashback")

    print("📈 Computing Graph-Flashback perplexity...")

    input_files = list(preprocessed_dir.glob("*.txt"))
    output_files = list(output_dir.glob("*.csv"))
    print(len(input_files), len(output_files))
    if len(input_files) -1 == len(output_files) or len(input_files) == len(output_files):
        print(f"⏭️ Graph-flashback perplexity already computed for all files in {output_dir}")
        return

    print(f"📈 Evaluating Graph-flashback perplexity on {len(input_files)} files...")

    subprocess.run([
        "python3", str(graph_flashback_base / "perplexity.py"),
        "--data_dir", str(preprocessed_dir),
        "--model_path", str(model_path),
        "--output_dir", str(output_dir),
        "--gpu", str(gpu)
    ], check=True)


def extract_difficulty_graph_flashback(model_path, data_dir,  run_dir, output_csv, prefix_lengths=(6, 12, 24), gpu="0"):

    graph_flashback_base = Path("Graph-Flashback")
    training_file = Path(run_dir / "model" / "preprocessed" / "training_set.txt")
    output_csv = Path(output_csv) / "suffix_extraction.csv"
    if output_csv.exists():
        print(f"⏭️ Graph-Flashback  difficulty already exists at {output_csv}, skipping.")
        return output_csv


    print("🔍 Computing Graph-Flashback extraction difficulty...")
    subprocess.run([
        "python3", str(graph_flashback_base / "extract.py"),
        "--data_file", str(training_file),
        "--model_path", str(model_path),
        "--output_dir", str(output_csv),
        "--prefix_lengths", *map(str, prefix_lengths),
        "--gpu", str(gpu)
    ], check=True)

    return output_csv

DATASETS = {
    "ShenzhenUrban": {
        "type1": "",
        "canaries": "",
        "type2_home": "",
        "type2_work": "",
        "type3": "",
    },
    "ShanghaiKaggle": {
        "type1": "",
        "canaries": "",
        "type2_home": "",
        "type2_work": "",
        "type3": "",
    },
    "YJMob100Kv3": {
        "type1": "",
        "canaries": "",
        "type2_home": "",
        "type2_work": "",
        "type3": "",
    },
}

MODELS = {
    "markov": {
        "train": train_markov_model,
        "test": test_markov_model,
        "perplexity": evaluate_perplexity,
        "extraction_suffix": extract_suffix_markov_difficulty,
    },
    "lstpm": {
        "train": train_lstpm_model,
        "test": test_lstpm_model,
        "perplexity": evaluate_lstpm_perplexity,
        "extraction_suffix": extract_suffix_lstpm_difficulty,
    },
    "deepmove_simple": {
        "train": lambda train_path, run_dir, epoch_max, early_stopping: train_deepmove_model(train_path, run_dir, early_stopping, epoch_max, "simple"),
        "test": lambda model_path, data_dir, run_dir, output_dir: test_deepmove_model(model_path, data_dir, run_dir, output_dir, "simple"),
        "perplexity": lambda model_path, data_dir, run_dir, output_dir: perplexity_deepmove(model_path, data_dir, run_dir, output_dir, "simple"),
        "extraction_suffix": lambda model_path, data_dir, run_dir, output_csv, prefix_lengths: extract_suffix_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, prefix_lengths, model_type="simple"),
        "extraction_work": lambda model_path, data_dir, run_dir, output_csv: extract_work_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, model_type="simple")
    },

    "deepmove_simple_long": {
        "train": lambda train_path, run_dir, epoch_max, early_stopping: train_deepmove_model(train_path, run_dir, early_stopping, epoch_max, "simple_long"),
        "test": lambda model_path, data_dir,run_dir, output_dir: test_deepmove_model(model_path, data_dir, run_dir, output_dir, "simple_long"),
        "perplexity": lambda model_path, data_dir, run_dir, output_dir: perplexity_deepmove(model_path, data_dir, run_dir, output_dir, "simple_long"),
        "extraction_suffix": lambda model_path, data_dir, run_dir, output_csv, prefix_lengths: extract_suffix_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, prefix_lengths, model_type="simple_long"),
        "extraction_work": lambda model_path, data_dir, run_dir, output_csv: extract_work_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, model_type="simple_long")
    },

    "deepmove_attn_avg_long_user": {
        "train": lambda train_path, run_dir, epoch_max, early_stopping: train_deepmove_model(train_path, run_dir, early_stopping, epoch_max, "attn_avg_long_user"),
        "test": lambda model_path, data_dir, run_dir, output_dir: test_deepmove_model(model_path, data_dir, run_dir, output_dir, "attn_avg_long_user"),
        "perplexity": lambda model_path, data_dir, run_dir, output_dir: perplexity_deepmove(model_path, data_dir, run_dir, output_dir, "attn_avg_long_user"),
        "extraction_suffix": lambda model_path, data_dir, run_dir, output_csv, prefix_lengths: extract_suffix_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, prefix_lengths, model_type="attn_avg_long_user"),
        "extraction_work": lambda model_path, data_dir, run_dir, output_csv: extract_work_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, model_type="attn_avg_long_user")

    },

    "deepmove_attn_local_long": {
        "train": lambda train_path, run_dir, epoch_max, early_stopping: train_deepmove_model(train_path, run_dir, early_stopping, epoch_max, "attn_local_long"),
        "test": lambda model_path, data_dir, run_dir, output_dir: test_deepmove_model(model_path, data_dir, run_dir, output_dir, "attn_local_long"),
        "perplexity": lambda model_path, data_dir, run_dir, output_dir: perplexity_deepmove(model_path, data_dir, run_dir, output_dir, "attn_local_long"),
        "extraction_suffix": lambda model_path, data_dir, run_dir, output_csv, prefix_lengths: extract_suffix_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, prefix_lengths, model_type="attn_local_long"),
        "extraction_work": lambda model_path, data_dir, run_dir, output_csv: extract_work_deepmove_difficulty(model_path,  data_dir, run_dir, output_csv, model_type="attn_local_long")
    },
    "graph_flashback": {
        "train": train_graph_flashback_model,
        "test": test_graph_flashback_model,
        "perplexity": perplexity_graph_flashback,
        "extraction_suffix": extract_difficulty_graph_flashback,
    },
}

# Unified results directory
OUTPUT_ROOT = Path("results/")



def compute_memorization_metrics(perplexity_dir, mapping_file):
    """
    Given a folder with <cluster_X_perplexity.csv> and training_set.csv,
    compute the 3 memorization metrics per training tid.
    """
    perplexity_dir = Path(perplexity_dir)

    # Support both filenames
    training_perp_path = perplexity_dir / "training_set_perplexity.csv"
    if not training_perp_path.exists():
        training_perp_path = perplexity_dir / "training_set.csv"

    training_df = pd.read_csv(training_perp_path)
    # print(training_df.head())

    # Support both column names: 'tid' or 'user'
    id_col = "tid" if "tid" in training_df.columns else "user"

    training_dict = training_df.set_index(id_col)["perplexity"].to_dict()
    mapping_df = pd.read_csv(mapping_file)

    if "cluster_file" in mapping_df.columns:
        type3 = False
        mapping_dict = mapping_df.set_index("cluster_file")["representant_tid"].to_dict()
    else:
        # type 3
        type3 = True
        mapping_df['reference_file'] = mapping_df['device_id'].apply(lambda x: f"{x}.csv")
        mapping_dict = mapping_df.set_index("reference_file")["training_tid"].to_dict()

    rows = []

    for ref_file in perplexity_dir.glob("*.csv"):
        cluster_id = ref_file.stem.replace("_perplexity", "") + ".csv"
        if cluster_id == "training_set.csv" or cluster_id == "training_set_perplexity.csv":
            continue
        print(ref_file)
        ref_df = pd.read_csv(ref_file)
        if ref_df.empty:
            continue

        # Adapt to tid/user here as well
        ref_id_col = "tid" if "tid" in ref_df.columns else "user"

        training_tid_val = mapping_dict.get(cluster_id)
        if training_tid_val not in training_dict:
            continue

        train_perp = training_dict[training_tid_val]

        result_row = {
            "tid": training_tid_val,
            "cluster_id": cluster_id,
        }

        if type3:
            for perturbation in ['substitute', 'stationary', 'shuffle']:
                mapping_df_perturbed = mapping_df[mapping_df['perturbation'] == perturbation]
                ref_df_perturbed = ref_df[ref_df[ref_id_col].isin(mapping_df_perturbed['reference_tid'])]

                ref_perps = ref_df_perturbed["perplexity"].values
                ref_mean = np.mean(ref_perps)
                rank = np.sum(ref_perps <= train_perp) + 1
                exposure = np.log2(len(ref_perps)) - np.log2(rank)
                percentile = (rank - 1) / len(ref_perps)
                gap = train_perp - ref_mean

                result_row.update({
                    f"train_perplexity_{perturbation}": train_perp,
                    f"mean_ref_perplexity_{perturbation}": ref_mean,
                    f"exposure_{perturbation}": exposure,
                    f"percentile_{perturbation}": percentile,
                    f"gap_{perturbation}": gap,
                })

        # General (non-perturbed) cluster
        ref_perps = ref_df["perplexity"].values
        ref_mean = np.mean(ref_perps)
        rank = np.sum(ref_perps <= train_perp) + 1
        exposure = np.log2(len(ref_perps)) - np.log2(rank)
        percentile = (rank - 1) / len(ref_perps)
        gap = train_perp - ref_mean

        result_row.update({
            "train_perplexity": train_perp,
            "mean_ref_perplexity": ref_mean,
            "exposure": exposure,
            "percentile": percentile,
            "gap": gap,
        })

        rows.append(result_row)

    return pd.DataFrame(rows)


def compute_memorization_per_window(perplexity_dir, mapping_file):
    perplexity_dir = Path(perplexity_dir)
    training_perp_path = perplexity_dir / "training_set_perplexity.csv"
    if not training_perp_path.exists():
        training_perp_path = perplexity_dir / "training_set.csv"

    training_df = pd.read_csv(training_perp_path)
    id_col = "tid" if "tid" in training_df.columns else "user"
    training_dict = training_df.set_index(id_col)["perplexity"].to_dict()

    mapping_df = pd.read_csv(mapping_file)
    mapping_df['reference_file'] = mapping_df['device_id'].apply(lambda x: f"{x}.csv")
    mapping_dict = mapping_df.set_index("reference_file")["training_tid"].to_dict()

    # Time window info for each reference_tid
    window_info = mapping_df.set_index("reference_tid")[["window_index", "window_start_hour"]].to_dict(orient="index")

    rows = []

    for ref_file in perplexity_dir.glob("*.csv"):
        cluster_id = ref_file.stem.replace("_perplexity", "") + ".csv"
        if cluster_id in ["training_set.csv", "training_set_perplexity.csv"]:
            continue

        ref_df = pd.read_csv(ref_file)
        if ref_df.empty:
            continue

        ref_id_col = "tid" if "tid" in ref_df.columns else "user"
        training_tid = mapping_dict.get(cluster_id)
        if training_tid not in training_dict:
            continue

        train_perp = training_dict[training_tid]

        # Add time window info
        ref_df["window_index"] = ref_df[ref_id_col].map(lambda tid: window_info.get(tid, {}).get("window_index", -1))
        ref_df["window_start_hour"] = ref_df[ref_id_col].map(lambda tid: window_info.get(tid, {}).get("window_start_hour", -1))
        ref_df["hour_of_day"] = ref_df["window_start_hour"] % 24  # <-- Group by hour of day (0–23)

        for hour, sub_df in ref_df.groupby("hour_of_day"):
            if hour == -1:
                continue

            ref_perps = sub_df["perplexity"].values
            if len(ref_perps) == 0:
                continue

            ref_mean = np.mean(ref_perps)
            rank = np.sum(ref_perps <= train_perp) + 1
            exposure = np.log2(len(ref_perps)) - np.log2(rank)
            percentile = (rank - 1) / len(ref_perps)
            gap = train_perp - ref_mean

            rows.append({
                "tid": training_tid,
                "cluster_id": cluster_id,
                "hour_of_day": hour,
                "train_perplexity": train_perp,
                "mean_ref_perplexity": ref_mean,
                "exposure": exposure,
                "percentile": percentile,
                "gap": gap
            })

    df = pd.DataFrame(rows)

    # Optional: aggregate across trajectories for smoother visualization
    #df_grouped = df.groupby("hour_of_day")[["exposure", "percentile", "gap"]].mean().reset_index()

    return df #, df_grouped



def compute_carlini_memorization_metric(perplexity_dir, canary_id="canary1"):
    """
    Compute Carlini-style exposure of a single canary against a reference set.

    Args:
        perplexity_dir (str or Path): Directory containing training_set_perplexity.csv and reference_set_perplexity.csv
        canary_id (str): The tid/user ID of the inserted canary
        output_file (str or Path): Where to save the exposure results (.json)
    """
    perplexity_dir = Path(perplexity_dir)

     # === Find training file ===
    train_file = perplexity_dir / "training_set_perplexity.csv"
    if not train_file.exists():
        train_file = perplexity_dir / "training_set.csv"
        if not train_file.exists():
            raise FileNotFoundError("Missing both training_set_perplexity.csv and training_set.csv")

    # === Find reference file ===
    ref_file = perplexity_dir / "reference_set_perplexity.csv"
    if not ref_file.exists():
        ref_file = perplexity_dir / "reference_set.csv"
        if not ref_file.exists():
            raise FileNotFoundError("Missing both reference_set_perplexity.csv and reference_set.csv")


    # Read training and reference perplexities
    df_train = pd.read_csv(train_file)
    df_ref = pd.read_csv(ref_file)

    id_col = "tid" if "tid" in df_train.columns else "user"

    # Canary perplexity
    try:
        canary_perp = df_train.set_index(id_col).loc[canary_id]["perplexity"]
    except KeyError:
        raise ValueError(f"Canary ID '{canary_id}' not found in training set.")

    # Reference perplexities
    ref_perps = df_ref["perplexity"].dropna().values
    if len(ref_perps) == 0:
        raise ValueError("Reference set is empty or invalid.")

    # === Compute metrics ===
    ref_mean = np.mean(ref_perps)
    rank = np.sum(ref_perps <= canary_perp) + 1
    exposure = np.log2(len(ref_perps)) - np.log2(rank)
    percentile = (rank - 1) / len(ref_perps)
    gap = canary_perp - ref_mean

    result = {
        "canary_id": canary_id,
        "canary_perplexity": float(canary_perp),
        "mean_ref_perplexity": float(ref_mean),
        "exposure": float(exposure),
        "percentile": float(percentile),
        "gap": float(gap),
        "num_reference": int(len(ref_perps))
    }

    return result



def run_memorization_test(dataset_name, type_name, dataset_path, model_name, test=False):
    print(f"\n🚀 Running: {model_name.upper()} | {dataset_name} | {type_name}")

    model = MODELS[model_name]
    dataset_path = Path(dataset_path)
    training_file = dataset_path / "training_set.csv" 
    mapping_file = dataset_path / "representant_mapping.txt"
    epoch_max = 100 if dataset_name == "YJMob100Kv3" else 40
    early_stopping = 20 if dataset_name == "YJMob100Kv3" else 5

    if not training_file.exists():
        print(f"⚠️ No training set found in {dataset_path}")
        return

    run_dir = OUTPUT_ROOT /   dataset_name / model_name/ type_name
    model_dir = run_dir / "model"
    perplexity_dir = run_dir / "perplexity"
    test_dir = run_dir / "test"

    # Train
    model_path = model["train"](training_file, model_dir, epoch_max=epoch_max, early_stopping=early_stopping)

    # Perplexity
    model["perplexity"](model_path, dataset_path, run_dir, perplexity_dir)

    #Metrics
    if type_name == "canaries":
        metrics_path = run_dir / "carlini_exposure.json"
        if metrics_path.exists():
            print(f"⏭️ Metrics already exist at {metrics_path}")
        else:
            result = compute_carlini_memorization_metric(perplexity_dir, canary_id="canary1")
            with open(metrics_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"✅ Carlini-style exposure saved to: {metrics_path}")
    else:
        metrics_path = run_dir / "memorization_metrics.csv"
        if metrics_path.exists():
            print(f"⏭️ Metrics already exist at {metrics_path}")
        else:
            metrics_df = compute_memorization_metrics(perplexity_dir, mapping_file)
            metrics_df.to_csv(metrics_path, index=False)
        print(f"✅ Metrics saved to: {metrics_path}")

        if type_name == "type3":
            per_window_path = run_dir / "memorization_metrics_per_window.csv"
            if per_window_path.exists():
                print(f"⏭️ Per-window metrics already exist at {per_window_path}")
            else:
                per_window_df = compute_memorization_per_window(perplexity_dir, mapping_file)
                per_window_df.to_csv(per_window_path, index=False)
            print(f"✅ Per-window metrics saved to: {per_window_path}")

    #Test
    if test:
        model["test"](model_path, dataset_path, run_dir, test_dir)

    #Extraction of difficulty proxies
    extraction_dir = run_dir / "difficulty_proxies"
    extraction_dir.mkdir(parents=True, exist_ok=True)
    if type_name == "type1":    
        model["extraction_suffix"](model_path, dataset_path, run_dir, extraction_dir, prefix_lengths=(12, 24, 48))
    if type_name == "type2_home":
        if model_name == "deepmove_simple":
            model["extraction_work"](model_path, dataset_path, run_dir, extraction_dir)


ALL_MODELS = [ "graph_flashback", "deepmove_simple", "markov", "deepmove_simple_long", "deepmove_attn_avg_long_user", "deepmove_attn_local_long", "lstpm"]
for dataset_name, type_paths in DATASETS.items():
    for model_name in ALL_MODELS:
        for type_name, path in type_paths.items():
            if type_name == "type1":
                run_memorization_test(dataset_name, type_name, path, model_name, test=True)
            else:
                run_memorization_test(dataset_name, type_name, path, model_name, test=False)



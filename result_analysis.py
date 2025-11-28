#!/usr/bin/env python
# coding: utf-8

# # Memorization Metrics Analysis
# 
# This notebook analyzes memorization behaviors across different models, datasets, and normalization types. It leverages previously computed results (e.g., memorization_metrics.csv, features.csv) to explore:
# - Correlation between mobility features and memorization (e.g., how radius of gyration or diversity relate to exposure).
# - Comparative strength of different memorization types (e.g., substitute vs. shuffle).
# - Cross-model comparison to assess which models are more prone to memorization.
# 
# The goal is to identify patterns, biases, or vulnerabilities in user memorization by location prediction models.

import pandas as pd
import seaborn as sns
from pathlib import Path
import pathlib
import json
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from collections import defaultdict
import numpy as np
from scipy.stats import gaussian_kde
from scipy.stats import spearmanr
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import joypy


DATASETS = {
    "ShenzhenUrban": {
        "type1": "",
        "type2_home": "",
        "type2_work": "",
        "type3": "",
    },
    "ShanghaiKaggle": {
        "type1": "",
        "type2_home": "",
        "type2_work": "",
        "type3": "",
    },
    "YJMob100Kv3": {
        "type1": "",
        "type2_home": "",
        "type2_work": "",
        "type3": "",
    }
}

ALL_MODELS = [ "deepmove_simple", "markov", "deepmove_simple_long", "deepmove_attn_avg_long_user", "deepmove_attn_local_long", "lstpm", "graph_flashback"]


# #### Question 0: Model performance and statistics

results = defaultdict(dict)        
result_errors = defaultdict(dict)  

for model_name in ALL_MODELS:
    print(f"Processing results for model: {model_name}")
    for dataset_name in DATASETS.keys():
        print(f"  Dataset: {dataset_name}")
        root_path = Path("results") / dataset_name / model_name / "type1" / "test"

        try:
            if not root_path.exists():
                raise FileNotFoundError(f"{root_path} does not exist.")

            all_dfs = []
            for file in root_path.glob("*.csv"):
                df = pd.read_csv(file)
                all_dfs.append(df)

            if not all_dfs:
                raise ValueError(f"No CSV files found in {root_path}")

            full_df = pd.concat(all_dfs, ignore_index=True)

            stats = full_df[["top-1", "top-5", "top-10"]].agg(["mean", "std"]).transpose()

            # ✅ Store only DataFrames here
            results[model_name][dataset_name] = stats

        except Exception as e:
            # ✅ Errors are stored separately, so `results` stays clean
            result_errors[model_name][dataset_name] = str(e)
            print(f"    [WARN] {e}")



def export_accuracy_macros(results, output_path="accuracy_macros.tex"):
    """
    Generate LaTeX accuracy macro commands from the `results` dictionary
    and write them to a file.

    Expects:
        results[model][dataset] = DataFrame with:
            index  = ["top-1", "top-5", "top-10"]
            columns = ["mean", "std"]
    """

    model_map = {
        "Markov (2nd)":         ("markov",                        "Markov"),
        "LSTM-simple":          ("deepmove_simple",               "LSTMsimple"),
        "LSTM-long":            ("deepmove_simple_long",          "LSTMlong"),
        "DeepMove-locallong":   ("deepmove_attn_local_long",      "DeepMovelocallong"),
        "DeepMove-avglonguser": ("deepmove_attn_avg_long_user",   "DeepMoveavglonguser"),
        "LSTPM":                ("lstpm",                         "LSTPM"),
        "Graph-Flashback":      ("graph_flashback",               "GraphFlashback"),
    }

    dataset_map = {
        "ShenzhenUrban": "SZ",
        "ShanghaiKaggle": "SH",
        "YJMob100Kv3":   "YJ",
    }

    topk_map = {
        "top-1": "One",
        "top-5": "Five",
    }

    output_path = pathlib.Path(output_path)

    with output_path.open("w") as f:
        for _, (results_key, macro_model) in model_map.items():
            model_results = results.get(results_key, {})

            for dataset_key, ds_short in dataset_map.items():
                stats_df = model_results.get(dataset_key)

                # Skip if missing or not a DataFrame (e.g., no data for this combination)
                if not isinstance(stats_df, pd.DataFrame):
                    continue

                for topk_key, top_suffix in topk_map.items():
                    if topk_key not in stats_df.index:
                        continue
                    if "mean" not in stats_df.columns:
                        continue

                    mean_val = float(stats_df.loc[topk_key, "mean"]) * 100.0
                    macro_name = f"\\acc{macro_model}{ds_short}T{top_suffix}"
                    f.write(f"\\newcommand{{{macro_name}}}{{{mean_val:.1f}}}\n")

    print(f"[✓] Wrote LaTeX macros to {output_path.resolve()}")


export_accuracy_macros(results, "analysis_outputs/accuracy_macros.tex")


# #### Question 1: Is there any memorization?


model_name = "deepmove_simple"
type_name = "type1"
png_output = True

# Pretty names for legends
DATASET_LABELS = {
    "ShenzhenUrban": "ShenzhenUrb",
    "ShanghaiKaggle": "ShanghaiTel",
    "YJMob100Kv3": "YJMob100K",
}

# Paths
ROOT_RESULTS = Path("results")
OUTPUT_DIR = Path("analysis_outputs") / "question1_memorization" / model_name / type_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Font sizes (adjust here globally)
label_fontsize = 15
tick_fontsize = 15
legend_fontsize = 15
linewidth_size = 3


# -----------------------------
# Load memorization data for ALL datasets and build one combined df
# -----------------------------
all_dfs = []
summaries = {}

for dataset_name in DATASETS.keys():
    input_file = ROOT_RESULTS / dataset_name / model_name / type_name / "memorization_metrics.csv"
    if not input_file.exists():
        print(f"[WARN] memorization_metrics.csv not found for {dataset_name} at {input_file}")
        continue

    df = pd.read_csv(input_file)

    # Create new metrics
    df["likAdvantage"] = -df["gap"]          # negative of gap
    df["likStanding"] = df["percentile"]     # rename percentile

    # Attach dataset label for plotting
    df["dataset"] = DATASET_LABELS.get(dataset_name, dataset_name)
    all_dfs.append(df)

    # Per-dataset summary (using new names)
    summary = {
        "num_trajectories": len(df),
        "likAdvantage_mean": float(df["likAdvantage"].mean()),
        "likAdvantage_median": float(df["likAdvantage"].median()),
        "percent_likAdvantage_above_zero": float((df["likAdvantage"] > 0).mean() * 100),
        "exposure_mean": float(df["exposure"].mean()),
        "exposure_median": float(df["exposure"].median()),
        "likStanding_mean": float(df["likStanding"].mean()),
        "percent_likStanding_below_0.1": float((df["likStanding"] < 0.1).mean() * 100),
    }
    summaries[dataset_name] = summary

    # Save per-dataset summary
    with open(OUTPUT_DIR / f"memorization_summary_{dataset_name}.json", "w") as f:
        json.dump(summary, f, indent=4)

# Combine for plotting
if not all_dfs:
    raise RuntimeError("No memorization_metrics.csv files found for any dataset.")

df_all = pd.concat(all_dfs, ignore_index=True)

# --- Plotting ---
sns.set_style('whitegrid')
palette = sns.color_palette("deep")
dataset_order = ["ShenzhenUrb", "ShanghaiTel", "YJMob100K"]
df_all["dataset"] = df_all["dataset"].replace({
    "ShenzhenUrban": "ShenzhenUrb",
    "ShanghaiKaggle": "ShanghaiTel",
    "YJMob100Kv3": "YJMob100K"
})
# ---------- 1) Exposure CDF (with legend) ----------
plt.figure(figsize=(3, 3))

for label, color in zip(dataset_order, palette):
    sub = df_all[df_all["dataset"] == label]
    if sub.empty:
        continue
    sns.ecdfplot(x=sub["exposure"], label=label, linewidth=linewidth_size, color=color)

plt.xlabel("Exposure", fontsize=label_fontsize)
plt.ylabel("CDF", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# Legend here (only here)
plt.legend(title="", fontsize=legend_fontsize)

plt.tight_layout()
if png_output:
    plt.savefig(OUTPUT_DIR / "exposure_cdf.png", dpi=300)
else:
    plt.savefig(OUTPUT_DIR / "exposure_cdf.pdf", dpi=300)
plt.close()

# ---------- 2) likAdvantage CDF (no legend) ----------
plt.figure(figsize=(3, 3))

for label, color in zip(dataset_order, palette):
    sub = df_all[df_all["dataset"] == label]
    if sub.empty:
        continue
    sns.ecdfplot(x=sub["likAdvantage"], linewidth=linewidth_size, color=color)

plt.axvline(0, color='red', linestyle='--')
plt.xlabel("likAdvantage", fontsize=label_fontsize)
plt.ylabel("CDF", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# No legend here
plt.tight_layout()
if png_output:
    plt.savefig(OUTPUT_DIR / "likAdvantage_cdf.png", dpi=300)
else:
    plt.savefig(OUTPUT_DIR / "likAdvantage_cdf.pdf", dpi=300)
plt.close()

# ---------- 3) likStanding CDF (no legend) ----------
plt.figure(figsize=(3, 3))

for label, color in zip(dataset_order, palette):
    sub = df_all[df_all["dataset"] == label]
    if sub.empty:
        continue
    sns.ecdfplot(x=sub["likStanding"], linewidth=linewidth_size, color=color)

plt.axvline(0.1, color='red', linestyle='--')
plt.xlabel("likStanding", fontsize=label_fontsize)
plt.ylabel("CDF", fontsize=label_fontsize)
plt.xticks(fontsize=tick_fontsize)
plt.yticks(fontsize=tick_fontsize)

# No legend here
plt.tight_layout()
if png_output:
    plt.savefig(OUTPUT_DIR / "likStanding_cdf.png", dpi=300)
else:
    plt.savefig(OUTPUT_DIR / "likStanding_cdf.pdf", dpi=300)
plt.close()

print("Summaries:", summaries)

# -----------------------------
# Learning curve plots (one per dataset)
# -----------------------------
for dataset_name in DATASETS.keys():
    res_file = ROOT_RESULTS / dataset_name / model_name / type_name / "model" / "res.txt"
    if not res_file.exists():
        print(f"[WARN] res.txt not found for {dataset_name} at {res_file}")
        continue

    with open(res_file, "r") as f:
        res_data = json.load(f)

    metrics = res_data.get("metrics", {})
    train_loss = metrics.get("train_loss", [])
    valid_loss = metrics.get("valid_loss", [])
    accuracy = metrics.get("accuracy", [])

    if not train_loss or not valid_loss or not accuracy:
        print(f"[WARN] Missing metrics in res.txt for {dataset_name}")
        continue

    epochs = list(range(1, len(train_loss) + 1))

    # Best epochs
    best_acc_epoch = int(np.argmax(accuracy)) + 1
    best_acc = accuracy[best_acc_epoch - 1]

    best_loss_epoch = int(np.argmin(valid_loss)) + 1
    best_loss = valid_loss[best_loss_epoch - 1]

    fig, ax1 = plt.subplots(figsize=(5, 3))

    # Loss
    ax1.plot(epochs, train_loss, label="Train Loss", marker="o", alpha=0.6)
    ax1.plot(epochs, valid_loss, label="Val. Loss", marker="o", alpha=0.6)
    ax1.axvline(x=best_acc_epoch, color="red", linestyle="--", alpha=0.6, label="Best Acc.")
    ax1.set_xlabel("Epoch", fontsize=label_fontsize)
    ax1.set_ylabel("Loss", fontsize=label_fontsize)
    ax1.tick_params(axis='both', labelsize=tick_fontsize)

    # Accuracy on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, accuracy, color="gray", linestyle="--", marker="x", alpha=0.5, label="Val. Acc.")
    ax2.set_ylabel("Accuracy", color="gray", fontsize=label_fontsize)
    ax2.tick_params(axis='y', labelcolor="gray", labelsize=tick_fontsize)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center", fontsize=legend_fontsize)

    plt.tight_layout()

    ds_label = DATASET_LABELS.get(dataset_name, dataset_name)
    fname = f"learning_curve_{ds_label}.png" if png_output else f"learning_curve_{ds_label}.pdf"
    plt.savefig(OUTPUT_DIR / fname, dpi=300)
    plt.close()


# #### Question 2: How do individual mobility behaviors correlate with memorization?
# Assess how mobility features (e.g., rg, diversity) correlate with memorization (exposure, gap, etc.)

#dataset_name = "YJMob100Kv3"
model_name = "deepmove_simple"
type_name = "type1"
label_fontsize = 14
tick_fontsize = 14
legend_fontsize = 14
linewidth_size=3
png_output= True
# Plot save format
ext = "png" if png_output else "pdf"

for dataset_name in DATASETS.keys():
    print(f"Processing dataset: {dataset_name}")
    # Paths
    MEMO_PATH = Path("results") / dataset_name / model_name / type_name / "memorization_metrics.csv"
    MOBILITY_PATH = Path(DATASETS[dataset_name][type_name]) / "mobility_characteristics.csv"
    OUTPUT_DIR = Path("analysis_outputs") /  "question2_mobility_correlation" / model_name / dataset_name / type_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_memo = pd.read_csv(MEMO_PATH)
    df_mob = pd.read_csv(MOBILITY_PATH)
    df = pd.merge(df_memo, df_mob, on="tid")

    # === 1. CDFs of mobility features ===
    mobility_metrics = ['diversity', 'rg_unique', 'stationarity', 'repetitiveness']
    for i, metric in enumerate(mobility_metrics):
        size = (3, 3)
        plt.figure(figsize=size)
        sns.ecdfplot(df[metric], linewidth=linewidth_size, color=sns.color_palette("deep")[i])
        xlabel = "Radius of Gyration (km)" if metric == "rg_unique" else metric.replace('_', ' ').title()
        plt.xlabel(xlabel, fontsize=label_fontsize+1)
        plt.ylabel("CDF", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize+1)
        plt.yticks(fontsize=tick_fontsize+1)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{metric}_cdf.{ext}", dpi=300)
        plt.close()

        if metric in ['diversity', 'rg_unique']:
            quantiles = df[metric].quantile([0.25, 0.5, 0.75, 1.0])
            print(f"{metric.title()} Quantiles:\nQ1={quantiles[0.25]:.3f}, Q2={quantiles[0.5]:.3f}, Q3={quantiles[0.75]:.3f}, Q4={quantiles[1.0]:.3f}")

    # Histogram of profiles
    plt.figure(figsize=(3, 2.5))
    profile_order = ["routiner", "regular", "scouter"]
    print(df.profile.value_counts(normalize=True))
    sns.countplot(data=df, x="profile", palette="Set2", order=profile_order)
    plt.xlabel("Mobility Profile", fontsize=label_fontsize)
    plt.ylabel("", fontsize=label_fontsize)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=tick_fontsize)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"profile_histogram.{ext}", dpi=300)
    plt.close()

    # === 2. Heatmap: Percentile by rg and diversity quartiles ===
    try:
        df["rg_q"] = pd.qcut(df["rg_unique"], 5, labels=[f"Q{i+1}" for i in range(5)])
        df["diversity_q"] = pd.qcut(df["diversity"], 5, labels=[f"Q{i+1}" for i in range(5)])

        heatmap_data = df.groupby(["diversity_q", "rg_q"])["percentile"].mean().unstack()

        plt.figure(figsize=(5, 3))
        ax = sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            annot_kws={"size": 12},
            cbar_kws={"label": "Avg Percentile"}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)  # Colorbar tick size
        cbar.ax.set_ylabel("Avg Percentile", fontsize=13)  # Colorbar label size

        plt.xlabel("Radius of Gyration (Q1 = Low)", fontsize=label_fontsize)
        plt.ylabel("Diversity (Q1 = Low)", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"heatmap_percentile_by_rg_diversity.{ext}", dpi=300)
        plt.close()
    except Exception as e:
        print("Heatmap plot failed:", e)

    # === 3. Contour plot: Repetitiveness vs Stationarity ===
    try:
        x = df["stationarity"]
        y = df["repetitiveness"]
        z = df["percentile"]

        xi = np.linspace(x.min(), x.max(), 100)
        yi = np.linspace(y.min(), y.max(), 100)
        xi, yi = np.meshgrid(xi, yi)
        zi = griddata((x, y), z, (xi, yi), method='linear')

        plt.figure(figsize=(4, 3))
        cp = plt.contourf(xi, yi, zi, levels=20, cmap="RdYlBu")
        plt.colorbar(cp, label="Avg likStanding")
        plt.xlabel("Stationarity", fontsize=label_fontsize)
        plt.ylabel("Repetitiveness", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"contour_percentile_by_stationarity_repetitiveness.{ext}", dpi=300)
        plt.close()
    except Exception as e:
        print("Contour plot failed:", e)

    # === 4. Ridge Plot: Percentile distribution by mobility profile ===
    try:
        import joypy
        plt.figure(figsize=(3, 3))
        fig, axes = joypy.joyplot(
            df,
            by="profile",
            column="percentile",
            kind="kde",
            fill=True,
            colormap=plt.cm.Blues,
            ylabelsize=label_fontsize,  # controls y-axis (profile) font size
            linewidth=1,
            fade=True,
            alpha=0.9,
            figsize=(3, 3)
        )
        plt.xlabel("likStanding", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"ridge_percentile_by_profile.{ext}", dpi=300)
        plt.close()
    except Exception as e:
        print("Ridge plot failed:", e)

    # CDF by profile
    plt.figure(figsize=(3, 3))
    for profile in df["profile"].unique():
        sns.ecdfplot(df[df["profile"] == profile]["percentile"], label=profile, linewidth=linewidth_size)
    plt.xlabel("likStanding", fontsize=label_fontsize)
    plt.ylabel("CDF", fontsize=label_fontsize)
    plt.legend(title="", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"likStanding_by_profile.{ext}", dpi=300)
    plt.close()


# #### Question 3: Are models more prone to certain memorization types?


# === Config ===
dataset_name = "ShenzhenUrban"
# model_name = "deepmove_simple"  # not used here directly
type_names = ["type1", "type2_home", "type2_work", "type3"]
metrics = ["exposure", "gap", "percentile"]

# Mapping from raw metric name to:
# - plotting column
# - axis label
# - filename suffix
metric_column_map = {
    "exposure": "exposure",
    "gap": "likAdvantage",   # plotted as -gap
    "percentile": "likStanding",
}

metric_axis_label = {
    "exposure": "Exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

metric_fname_map = {
    "exposure": "exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

# === Plot settings ===
label_fontsize = 14
tick_fontsize = 11
legend_fontsize = 12
png_output = True
ext = "png" if png_output else "pdf"

# === Memorization type display mapping ===
type_display = {
    "type1": "Location",
    "type2_home": "Anchor-home",
    "type2_work": "Anchor-work",
    "type3": "Segment-level"
}

#for model_name in ALL_MODELS:
for model_name in ['graph_flashback']:
    print(f"Processing model: {model_name}")
    INPUT_ROOT = Path("results") / dataset_name / model_name
    OUTPUT_DIR = Path("analysis_outputs") / "question3_memorization_types" / model_name / dataset_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load and tag data for all memorization types ---
    df_all_list = []
    for type_name in type_names:
        path = INPUT_ROOT / type_name / "memorization_metrics.csv"
        if path.exists():
            df = pd.read_csv(path)
            df["type"] = type_display.get(type_name, type_name)
            df_all_list.append(df)

    if not df_all_list:
        print(f"[WARN] No memorization_metrics.csv found for any type for model {model_name}")
        continue

    df_all = pd.concat(df_all_list, ignore_index=True)

    # Derived metrics for plotting
    if "gap" in df_all.columns:
        df_all["likAdvantage"] = -df_all["gap"]
    else:
        df_all["likAdvantage"] = np.nan

    if "percentile" in df_all.columns:
        df_all["likStanding"] = df_all["percentile"]
    else:
        df_all["likStanding"] = np.nan

    # === 1. CDF plots ===
    for metric in metrics:
        plot_col = metric_column_map[metric]
        axis_label = metric_axis_label[metric]
        fname_metric = metric_fname_map[metric]

        plt.figure(figsize=(4, 3))
        for mem_type in df_all["type"].unique():
            data = df_all[df_all["type"] == mem_type][plot_col].dropna()
            if len(data) < 2:
                continue
            kde = gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 500)
            cdf = np.cumsum(kde(x_vals))
            cdf /= cdf[-1]
            plt.plot(x_vals, cdf, label=mem_type, linewidth=2)
        plt.xlabel(axis_label, fontsize=label_fontsize)
        plt.ylabel("CDF", fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"cdf_{fname_metric}_by_type.{ext}", dpi=300)
        plt.close()

    # === 2. Stacked bar chart of mobility profiles per memorization type ===
    profile_counts_by_type = defaultdict(lambda: defaultdict(int))
    for type_name in type_names:
        mobility_file = Path(DATASETS[dataset_name][type_name]) / "mobility_characteristics.csv"
        df_mob = pd.read_csv(mobility_file)[["tid", "profile"]]
        counts = df_mob["profile"].value_counts()
        for profile, count in counts.items():
            profile_counts_by_type[type_display[type_name]][profile] += count

    df_counts = pd.DataFrame(profile_counts_by_type).fillna(0).T
    df_props = df_counts.div(df_counts.sum(axis=1), axis=0)

    plt.figure(figsize=(3, 3))
    ax = df_props[["routiner", "regular", "scouter"]].plot(
        kind="bar", stacked=True, colormap="Set3", width=0.7
    )
    plt.ylabel("", fontsize=12)
    plt.xlabel("", fontsize=12)
    plt.xticks(rotation=15, fontsize=8)
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=8,
        frameon=False
    )
    # Annotate percentages
    for i, row in enumerate(df_props[["routiner", "regular", "scouter"]].values):
        cum_height = 0
        for j, val in enumerate(row):
            if val > 0.01:
                ax.text(
                    i, cum_height + val / 2,
                    f"{val*100:.1f}%",
                    ha='center', va='center', fontsize=7, color="black"
                )
            cum_height += val

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"stacked_profiles_by_type.{ext}", dpi=300)
    plt.close()

    # === 3. Boxplots for gap, exposure, percentile by memorization type ===
    modes = {
        "": "all",
        "_substitute": "substitute",
        "_stationary": "stationary",
        "_shuffle": "shuffle"
    }
    df_type3 = df_all[df_all["type"] == "Segment-level"]

    for metric in metrics:
        axis_label = metric_axis_label[metric]
        fname_metric = metric_fname_map[metric]

        data = []
        for suffix, label in modes.items():
            column = f"{metric}{suffix}"
            if column in df_type3.columns:
                for value in df_type3[column].dropna():
                    # Transform gap → likAdvantage (negative), percentile unchanged
                    if metric == "gap":
                        val = -value
                    elif metric == "percentile":
                        val = value  # becomes likStanding in labeling
                    else:
                        val = value  # exposure unchanged
                    data.append({"mode": label, "value": val})
                    print(data)
        if not data:
            continue

        df_metric = pd.DataFrame(data)
        plt.figure(figsize=(4, 3))
        sns.boxplot(data=df_metric, x="mode", y="value", palette="deep")
        plt.xlabel("", fontsize=label_fontsize)
        plt.ylabel(axis_label, fontsize=label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"boxplot_type3_{fname_metric}_by_mode.{ext}", dpi=300)
        plt.close()

    # === 4. Radar Plot: median value of metrics per memorization type ===
    radar_cols = ["exposure", "likAdvantage", "likStanding"]
    radar_labels = ["Exposure", "likAdvantage", "likStanding"]

    avg_metrics = df_all.groupby("type")[radar_cols].median()

    labels = radar_labels
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # repeat first angle

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    for type_name_key, row in avg_metrics.iterrows():
        values = row[radar_cols].tolist()
        values += values[:1]
        ax.plot(angles, values, label=type_name_key, linewidth=2)
        ax.fill(angles, values, alpha=0.2)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"radar_median_metrics_by_type.{ext}", dpi=300)
    plt.close()

    # === 5. Type 3 memorization per hour of the day ===
    INPUT_FILE = Path("results") / dataset_name / model_name / "type3" / "memorization_metrics_per_window.csv"
    if INPUT_FILE.exists():
        df_win = pd.read_csv(INPUT_FILE)
        df_win["hour_bin"] = df_win["hour_of_day"] % 24
        df_win["hour_bin"] = pd.cut(
            df_win["hour_bin"],
            bins=[0, 4, 8, 12, 16, 20, 24],
            right=False,
            labels=["[0–3]", "[4–7]", "[8–11]", "[12–15]", "[16–19]", "[20–23]"]
        )

        # Derived metrics for window-level analysis
        if "gap" in df_win.columns:
            df_win["likAdvantage"] = -df_win["gap"]
        if "percentile" in df_win.columns:
            df_win["likStanding"] = df_win["percentile"]

        for metric in metrics:
            plot_col = metric_column_map[metric]
            axis_label = metric_axis_label[metric]
            fname_metric = metric_fname_map[metric]

            if plot_col not in df_win.columns:
                continue

            plt.figure(figsize=(4, 3))
            sns.violinplot(
                data=df_win,
                x="hour_bin",
                y=plot_col,
                inner="quartile",
                scale="width",
                cut=0,
                linewidth=0.8,
                palette="viridis"
            )
            plt.xlabel("Hour of day", fontsize=label_fontsize)
            plt.ylabel(axis_label, fontsize=label_fontsize)
            plt.xticks(rotation=0, fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"violin_{fname_metric}_by_hour_period.{ext}", dpi=300)
            plt.close()
    else:
        print(f"[WARN] memorization_metrics_per_window.csv not found for model {model_name}")


# === Config ===
dataset_name = "ShenzhenUrban"
type_pairs = [("type1", "type2_home"), ("type1", "type2_work"), ("type2_home", "type2_work")]

# Display settings
label_fontsize = 14
tick_fontsize = 14
legend_fontsize = 14
linewidth_size = 3
png_output = True
ext = "png" if png_output else "pdf"

# Type name mapping for paper
type_labels = {
    "type1": "Location",
    "type2_home": "Anchor-home",
    "type2_work": "Anchor-work",
    "type3": "Segment-level"
}

# Metric remapping:
# - gap          -> likAdvantage ( = -gap )
# - percentile   -> likStanding ( = percentile )
metric_axis_label = {
    "exposure": "Exposure",
    "percentile": "likStanding",
    "gap": "likAdvantage",
}

metric_fname_map = {
    "exposure": "exposure",
    "percentile": "likStanding",
    "gap": "likAdvantage",
}
#for model_name in ALL_MODELS:
for model_name in ['graph_flashback']:
    print(f"Processing model: {model_name}")
    INPUT_ROOT = Path("results") / dataset_name / model_name
    OUTPUT_DIR = Path("analysis_outputs") / "question3_memorization_types" / model_name / dataset_name / "contour_comparisons"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for metric in ["exposure", "percentile", "gap"]:
        axis_label = metric_axis_label[metric]
        fname_metric = metric_fname_map[metric]

        for t1, t2 in type_pairs:
            print(t1, t2)
            try:
                df1 = pd.read_csv(INPUT_ROOT / t1 / "memorization_metrics.csv")
                df2 = pd.read_csv(INPUT_ROOT / t2 / "memorization_metrics.csv")
                print(df1.shape, df2.shape)

                # Inner join on tid with raw metric columns
                df = df1[["tid", metric]].rename(columns={metric: f"{t1}_{metric}"}).merge(
                    df2[["tid", metric]].rename(columns={metric: f"{t2}_{metric}"}),
                    on="tid", how="inner"
                )
                print(df.shape)

                # Raw values
                x_raw = df[f"{t1}_{metric}"].to_numpy()
                y_raw = df[f"{t2}_{metric}"].to_numpy()

                # Transform according to metric:
                # - exposure     : unchanged
                # - percentile   : likStanding = percentile
                # - gap          : likAdvantage = -gap
                if metric == "gap":
                    x = -x_raw
                    y = -y_raw
                else:
                    x = x_raw
                    y = y_raw

                # z is the average of transformed values
                z = np.vstack([x, y]).mean(axis=0)

                # Define grid in transformed space
                xi = np.linspace(x.min(), x.max(), 100)
                yi = np.linspace(y.min(), y.max(), 100)
                xi, yi = np.meshgrid(xi, yi)

                # Interpolate z values on the grid
                zi = griddata((x, y), z, (xi, yi), method='linear')

                # Plot contour
                plt.figure(figsize=(4, 3))
                cp = plt.contourf(xi, yi, zi, levels=20, cmap="RdYlBu")
                cbar = plt.colorbar(cp)
                cbar.set_label(f"Avg {axis_label}", fontsize=label_fontsize)
                cbar.ax.tick_params(labelsize=tick_fontsize)

                plt.xlabel(f"{type_labels[t1]} {axis_label}", fontsize=label_fontsize)
                plt.ylabel(f"{type_labels[t2]} {axis_label}", fontsize=label_fontsize)
                plt.xticks(fontsize=tick_fontsize)
                plt.yticks(fontsize=tick_fontsize)

                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"contour_{t1}_vs_{t2}_{fname_metric}.{ext}", dpi=300)
                plt.close()

            except Exception as e:
                print(f"Contour plot failed for {t1} vs {t2} ({metric}):", e)


# #### Question 4: Are some models more prone to memorization than others?

question = "question4_model_comparison"
type_name = "type1"
metrics = ["exposure", "gap", "percentile"]

model_name_map = {
    # 'deepmove_attn_avg_long_user': 'DM-avglonguser',
    'graph_flashback': 'Graph-flashback',
    'deepmove_simple_long': 'LSTM-long',
    'deepmove_attn_local_long': 'DM-locallong',
    'lstpm': 'LSTPM',
    'markov': 'Markov',
    'deepmove_simple': 'LSTM-simple'
}

# Fixed display order for models (controls both column order and colors)
model_display_order = [
    "Markov",
    "LSTM-simple",
    "LSTM-long",
    "DM-locallong",
    # "DM-avglonguser",  # uncomment if you add it back in model_name_map
    "LSTPM",
    "Graph-flashback",
]

xtick_fontsize = 11
xlabel_fontsize = 14
ylabel_fontsize = 14
title_fontsize = 13
carlini_fontsize = 10
column_spacing = 0.05

png_output = False
ext = "png" if png_output else "pdf"

# Directory where all results are saved
#for dataset_name in DATASETS.keys():
for dataset_name in ['YJMob100Kv3']:
    print(f"Processing dataset: {dataset_name}")
    RESULTS_DIR = Path("results") / dataset_name
    OUTPUT_DIR = Path("analysis_outputs") / question / dataset_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Load and merge all models for the selected type ===
    dfs = []
    model_names = []
    carlini_values = {}

    for model_dir in RESULTS_DIR.iterdir():
        model_name = model_dir.name
        model_names.append(model_name)

        mem_file = model_dir / type_name / "memorization_metrics.csv"
        carlini_file = model_dir / "canaries" / "carlini_exposure.json"

        if mem_file.exists():
            df = pd.read_csv(mem_file)
            df["model"] = model_name
            df["type"] = type_name
            dfs.append(df)

            # Load Carlini exposure value
            if carlini_file.exists():
                with open(carlini_file, "r") as f:
                    data = json.load(f)
                    carlini_values[model_name] = {
                        "exposure": data["exposure"],
                        "gap": data["gap"],
                        "percentile": data["percentile"]
                    }

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["model"] = df_all["model"].map(model_name_map)

    # Drop any rows where model mapping failed (e.g., models not in model_name_map)
    df_all = df_all.dropna(subset=["model"])

    # Determine model order based on fixed display order and available models
    available_models = df_all["model"].unique().tolist()
    model_order = [m for m in model_display_order if m in available_models]

    # === Ridge plots for each metric by model with Carlini line ===
    for metric in metrics:
        g = sns.FacetGrid(
            df_all,
            col="model",
            col_wrap=3,
            hue="model",
            height=2.2,
            aspect=1.2,
            col_order=model_order,
            hue_order=model_order,  # ensure consistent colors across figures
            sharex=True,
            sharey=True,  # Share y-axis to compare scales
            palette="deep"
        )

        g.map(sns.kdeplot, metric, fill=True, alpha=0.6, linewidth=1.5)

        # Add Carlini line to each subplot (for exposure only)
        if metric == "exposure":
            for ax, model in zip(g.axes.flat, model_order):
                orig_model = [k for k, v in model_name_map.items() if v == model][0]
                if orig_model in carlini_values:
                    x_val = carlini_values[orig_model][metric]
                    ax.axvline(x=x_val, color="black", linestyle="--", linewidth=1)
                    ax.text(
                        x_val, ax.get_ylim()[1] * 0.85,
                        "Carlini", rotation=90, color="black", fontsize=10,
                        ha="right", va="center", fontweight="bold"
                    )

        # Set clean titles (without "model =")
        g.set_titles(col_template="{col_name}", size=title_fontsize)
        g.set_xlabels(metric.title(), fontsize=xlabel_fontsize)

        # Apply x-axis tick font size
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=xtick_fontsize)

        g.set_ylabels("Density", fontsize=ylabel_fontsize)

        # Only show y-axis ticks and labels on the first column
        for i, ax in enumerate(g.axes.flat):
            if i % 3 != 0:  # Not in first column
                ax.set_ylabel("")
                ax.tick_params(left=False, labelleft=False)  # Hide ticks + labels
            else:
                ax.tick_params(left=True, labelleft=True)    # Ensure ticks + labels are shown

        g.despine(left=True)
        plt.subplots_adjust(top=0.92, hspace=0.3, wspace=column_spacing)
        g.savefig(OUTPUT_DIR / f"ridgeplot_{metric}_by_model_with_carlini_grid_clean.{ext}", dpi=300)
        plt.close()


# #### Question 5: Assessing the Impact of Memorization on Cluster-Level Generalization

type_name = "type1"

png_output = False
ext = "png" if png_output else "pdf"

label_fontsize = 14
tick_fontsize = 14
legend_fontsize = 14
linewidth_size=3

# Mapping from raw metric name to plotting column, label, and filename suffix
metric_to_plot_col = {
    "exposure": "exposure",
    "gap": "likAdvantage",        # plot -gap
    "percentile": "likStanding",  # plot percentile as likStanding
}

metric_label = {
    "exposure": "Exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

metric_fname = {
    "exposure": "exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

#for model_name in ALL_MODELS:
for model_name in ['graph_flashback']:
    print(f"Processing model: {model_name}")
    for dataset_name in DATASETS.keys():
        print(f"  Dataset: {dataset_name}")
        BASE_PATH = Path("results") / dataset_name / model_name / type_name
        MEM_FILE = BASE_PATH / "memorization_metrics.csv"
        TEST_PATH = BASE_PATH / "test"

        OUTPUT_DIR = Path("analysis_outputs") / "question5_accuracy_vs_memorization" / model_name / dataset_name 
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load memorization file
        df_mem = pd.read_csv(MEM_FILE)
        df_mem["cluster_id"] = df_mem["cluster_id"].astype(str)

        # Initialize
        accuracy_metrics = ["top-1", "top-5", "top-10"]
        cluster_accuracy = {}

        # Extract accuracy per cluster from test files
        for cid in df_mem["cluster_id"].unique():
            cid_clean = cid.replace(".csv", "")
            test_file = TEST_PATH / f"{cid_clean}_topk.csv"
            if not test_file.exists():
                print(f"Test file for cluster {cid} not found, skipping.")
                continue
            df_test = pd.read_csv(test_file)
            cluster_accuracy[cid] = df_test[accuracy_metrics].mean().to_dict()

        # Convert accuracy dict to DataFrame
        df_acc = pd.DataFrame(cluster_accuracy).T
        df_acc.index.name = "cluster_id"
        df_acc.reset_index(inplace=True)

        # Merge with memorization metrics
        df_merged = df_mem.merge(df_acc, on="cluster_id", how="inner")

        # Add derived metrics
        df_merged["likAdvantage"] = -df_merged["gap"]          # negative gap
        df_merged["likStanding"] = df_merged["percentile"]      # percentile as likStanding

        # # Plot 1: Scatterplot with trendline per memorization metric vs top-1 accuracy
        # mem_metrics = ["exposure", "gap", "percentile"]
        # for metric in mem_metrics:
        #     plt.figure(figsize=(3, 3))
        #     sns.regplot(data=df_merged, x=metric, y="top-1", scatter_kws={"s": 10, "alpha": 0.5}, line_kws={"color": "red"})
        #     plt.xlabel(metric.title())
        #     plt.ylabel("Top-1 Accuracy")
        #     plt.tight_layout()
        #     plt.savefig(OUTPUT_DIR / f"scatter_{metric}_vs_top1.{ext}", dpi=300)
        #     plt.close()

        # Plot 1: Hexbin plot with trendline per memorization metric vs top-1 accuracy
        mem_metrics = ["exposure", "gap", "percentile"]
        for metric in mem_metrics:
            plot_col = metric_to_plot_col[metric]
            axis_label = metric_label[metric]
            fname_metric = metric_fname[metric]

            # ---- Filter out top 5% outliers on BOTH axes ----
            x05 = df_merged[plot_col].quantile(0.05)
            x95 = df_merged[plot_col].quantile(0.95)

            df_filt = df_merged[(df_merged[plot_col] >= x05) & (df_merged[plot_col] <= x95)].copy()
            # -------------------------------------------------

            plt.figure(figsize=(4, 3))
            hb = plt.hexbin(df_filt[plot_col], df_filt["top-1"], gridsize=40,
                            cmap="coolwarm", mincnt=1)
            cbar = plt.colorbar(hb)
            cbar.set_label("", fontsize=label_fontsize)
            cbar.ax.tick_params(labelsize=tick_fontsize)

            # Regression line
            sns.regplot(data=df_filt, x=plot_col, y="top-1", scatter=False,
                        line_kws={"color": "red"})

            plt.xlabel(axis_label, fontsize=label_fontsize)
            plt.ylabel("Top-1 Acc.", fontsize=label_fontsize)
            plt.xticks(fontsize=tick_fontsize)
            plt.yticks(fontsize=tick_fontsize)
            plt.tight_layout()

            plt.savefig(OUTPUT_DIR / f"hexbin_{fname_metric}_vs_top1.{ext}", dpi=300)
            print(f"Saved hexbin_{fname_metric}_vs_top1.{ext}")
            plt.close()


        # Plot 2: Boxplots of memorization metrics grouped by accuracy levels (binned top-k)
        quartiles = 5
        for metric in mem_metrics:
            plot_col = metric_to_plot_col[metric]
            axis_label = metric_label[metric]
            fname_metric = metric_fname[metric]

            df_temp = df_merged.copy()
            # Bin the memorization metric into quartiles (Q1–Q4)
            try:
                quart_col = f"{fname_metric}_quartile"
                df_temp[quart_col] = pd.qcut(df_temp[plot_col], quartiles, labels=[f"Q{i+1}" for i in range(quartiles)])
                # Melt accuracy columns to long format
                df_long = df_temp.melt(
                    id_vars=["tid", "cluster_id", quart_col],
                    value_vars=accuracy_metrics,
                    var_name="topk",
                    value_name="accuracy"
                )
                # Plot
                plt.figure(figsize=(3, 3))
                sns.boxplot(
                    data=df_long,
                    x=quart_col,
                    y="accuracy",
                    hue="topk",
                    palette="pastel"
                )
                #plt.xlabel(f"{metric.title()} Quartile", fontsize=12)
                plt.xlabel(f"{axis_label} Quartile (Q1 = Low)", fontsize=10)
                plt.ylabel("Top-k Accuracy", fontsize=12)
                plt.legend(
                    loc="upper center", 
                    bbox_to_anchor=(0.5, 1.1), 
                    ncol=len(accuracy_metrics),
                    fontsize=7,
                    title_fontsize=10,
                    frameon=False
                )
                plt.tight_layout()
                plt.savefig(OUTPUT_DIR / f"boxplot_accuracy_vs_{fname_metric}_quartile.{ext}", dpi=300)
                plt.close()
            except Exception as e:
                print(f"Boxplot failed for {metric}:", e)


# #### Question 6: Correlation with extractability


def safe_log10_series(series, k=5):
    """
    Compute log10(series) safely:
    - Try normal float log10
    - If inf/nan (because number is too big), use digit-based approximation
    """

    s = series.astype(str)           # keep original as string
    as_float = pd.to_numeric(series, errors='coerce')  # will be NaN for huge ints

    # Step 1: normal log10 where possible
    log_normal = np.log10(as_float + 1)

    # Mask of invalid results
    bad = log_normal.isna() | np.isinf(log_normal)

    if bad.any():
        # Number of digits
        n_digits = s.str.len()

        # Leading k digits (fits in float safely)
        leading = s.str[:k].astype(float)

        # Digit-based approximation for log10(x)
        log_approx = np.log10(leading) + (n_digits - k)

        # Fix only bad entries
        log_normal[bad] = log_approx[bad]

    return log_normal


# In[ ]:


# === Configurable Parameters ===
model_name = "deepmove_simple"
type_name = "type1"
png_output = False
ext = "png" if png_output else "pdf"

fontsize_label = 10
fontsize_tick = 9
fontsize_legend = 8

# Mapping from raw metric to derived plotting column / label / filename
metric_to_plot_col = {
    "exposure": "exposure",
    "gap": "likAdvantage",        # -gap
    "percentile": "likStanding",  # percentile
}

metric_label = {
    "exposure": "Exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

metric_fname = {
    "exposure": "exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

for dataset_name in DATASETS.keys():
    print(f"Processing dataset: {dataset_name}")
    # === Paths ===
    BASE_PATH = Path("results") / dataset_name / model_name / type_name
    MEM_FILE = BASE_PATH / "memorization_metrics.csv"
    DIFF_FILE = BASE_PATH / "difficulty_proxies" / "suffix_extraction.csv"
    OUTPUT_DIR = Path("analysis_outputs") /  "question6_memorization_vs_extractability" / model_name / dataset_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # === Load Data ===
    df_mem = pd.read_csv(MEM_FILE)
    df_diff = pd.read_csv(DIFF_FILE)
    df_mem["tid"] = df_mem["tid"].astype(str)
    df_diff["tid"] = df_diff["tid"].astype(str)
    df = df_mem.merge(df_diff, on="tid", how="inner")

    # Add derived memorization metrics
    df["likAdvantage"] = -df["gap"]          # negative gap
    df["likStanding"] = df["percentile"]     # percentile as likStanding

    # === Log-transform difficulty values ===
    difficulty_cols = [c for c in df_diff.columns if c.startswith("prefix-")]
    # for col in difficulty_cols:
    #     df[f"log_{col}"] = np.log10(df[col].astype(float) + 1)
    for col in difficulty_cols:
        df[f"log_{col}"] = safe_log10_series(df[col])


    # === Melt for plotting ===
    log_cols = [f"log_{c}" for c in difficulty_cols]
    df_long = df.melt(
        id_vars=["tid", "exposure", "likAdvantage", "likStanding"],
        value_vars=log_cols,
        var_name="prefix",
        value_name="log_difficulty"
    )
    df_long["prefix"] = df_long["prefix"].str.replace("log_prefix-", "").astype(int)

    # === Plot ===
    mem_metrics = ["exposure", "gap", "percentile"]
    quartiles = 4
    for metric in mem_metrics:
        try:
            plot_col = metric_to_plot_col[metric]
            axis_label = metric_label[metric]
            fname_metric = metric_fname[metric]

            df_long[f"{fname_metric}_quartile"] = pd.qcut(
                df_long[plot_col],
                quartiles,
                labels=[f"Q{i+1}" for i in range(quartiles)]
            )

            plt.figure(figsize=(3, 2.5))
            ax = sns.violinplot(
                data=df_long,
                x=f"{fname_metric}_quartile",
                y="log_difficulty",
                hue="prefix",
                inner="quartile",
                scale="width",
                width=0.9,
                cut=0,
                linewidth=0.7,
                palette="Set2"
            )

            handles, xlabels = ax.get_legend_handles_labels()
            ax.legend(
                handles, [f"p-{int(int(elt)/2)}" for elt in xlabels],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.2),
                ncol=len(xlabels),
                frameon=False,
                title=None,
                fontsize=fontsize_legend
            )

            ax.set_xlabel(f"{axis_label} Quartile (Q1 = Low)", fontsize=fontsize_label)
            ax.set_ylabel("Prefix-based Extractability", fontsize=fontsize_label)
            ax.tick_params(axis='x', labelsize=fontsize_tick)
            ax.set_title("")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"violin_combined_{metric}_by_prefix.{ext}", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Violin plot failed for {metric}:", e)
            continue


# In[ ]:


# Configuration variables
model_name = "deepmove_simple"
type_name = "type2_home"
png_output = False
ext = "png" if png_output else "pdf"
figsize = (3, 2.5)
xlabel_fontsize = 11
ylabel_fontsize = 11
legend_fontsize = 9
title_fontsize = 11
tick_fontsize = 9

# Mapping from raw metric to derived plotting column / label / filename
metric_to_plot_col = {
    "exposure": "exposure",
    "gap": "likAdvantage",        # -gap
    "percentile": "likStanding",  # percentile
}

metric_label = {
    "exposure": "Exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

metric_fname = {
    "exposure": "exposure",
    "gap": "likAdvantage",
    "percentile": "likStanding",
}

for dataset_name in DATASETS.keys():
    print(f"Processing dataset: {dataset_name}")
    # Paths
    BASE_PATH = Path("results") / dataset_name / model_name / type_name
    MEM_FILE = BASE_PATH / "memorization_metrics.csv"
    DIFF_FILE = BASE_PATH / "difficulty_proxies" / "infer_work.csv"
    OUTPUT_DIR = Path("analysis_outputs") / "question6_memorization_vs_extractability" / model_name / dataset_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_mem = pd.read_csv(MEM_FILE)
    df_diff = pd.read_csv(DIFF_FILE)
    df_mem["tid"] = df_mem["tid"].astype(str)
    df_diff["tid"] = df_diff["user_id"].astype(str)
    df = df_mem.merge(df_diff, on="tid", how="inner")

    # Derived memorization metrics
    df["likAdvantage"] = -df["gap"]          # negative gap
    df["likStanding"] = df["percentile"]     # percentile as likStanding

    # Melt long format
    rank_cols = ["avg_work_rank"]
    df_long = df.melt(
        id_vars=["tid", "exposure", "likAdvantage", "likStanding"],
        value_vars=rank_cols,
        var_name="rank_type",
        value_name="rank_value"
    )

    # Plot
    mem_metrics = ["exposure", "gap", "percentile"]
    quartiles = 4
    for metric in mem_metrics:
        try: 
            plot_col = metric_to_plot_col[metric]
            axis_label = metric_label[metric]
            fname_metric = metric_fname[metric]

            df_long[f"{fname_metric}_quartile"] = pd.qcut(
                df_long[plot_col],
                quartiles,
                labels=[f"Q{i+1}" for i in range(quartiles)]
            )

            plt.figure(figsize=figsize)
            ax = sns.violinplot(
                data=df_long,
                x=f"{fname_metric}_quartile",
                y="rank_value",
                hue="rank_type",
                inner="quartile",
                scale="width",
                width=0.9,
                cut=0,
                linewidth=0.7,
                palette="Set2"
            )

            ax.legend_.remove()
            ax.set_xlabel(f"{axis_label} Quartile (Q1 = Low)", fontsize=xlabel_fontsize)
            ax.set_ylabel("Anchor-based Extractability", fontsize=ylabel_fontsize)
            ax.set_title("", fontsize=title_fontsize)
            ax.tick_params(axis='x', labelsize=tick_fontsize)
            ax.tick_params(axis='y', labelsize=tick_fontsize)

            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"violin_home_{fname_metric}_by_ranktype.{ext}", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Violin plot failed for {metric}:", e)
            continue


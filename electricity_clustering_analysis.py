"""
Electricity Consumption Pattern Analysis
=========================================

Clusters households from the "Smart Meters in London" dataset by their electricity
consumption behaviour, validates the clusters with internal evaluation metrics,
visualizes them, analyzes peak/off-peak usage per cluster, and generates
cluster-specific energy-saving recommendations.

This script is a script version of `Electricity_Clustering_Analysis.ipynb`, organized
into functions so it can be run end-to-end from the command line and reused as a module.

Usage
-----
    python electricity_clustering_analysis.py \
        --data-dir data \
        --figures-dir figures \
        --output-dir outputs \
        --best-k 4

Optional true hour-of-day peak analysis (needs the much larger `hhblock_dataset` files
from the same Kaggle dataset, not included in this repo — see data/README.md):

    python electricity_clustering_analysis.py --hhblock-dir "path/to/hhblock_dataset"

Data
----
Required files (see data/README.md for where to get them and their columns):
    daily_dataset.csv, informations_households.csv
Optional (used only for cross-checking / discussion, not for clustering):
    acorn_details.csv, uk_bank_holidays.csv,
    weather_daily_darksky.csv, weather_hourly_darksky.csv
Optional (enables true hour-of-day peak/off-peak detection):
    hhblock_dataset/block_*.csv
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures to disk without needing a display
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

RANDOM_STATE = 42
FEATURE_COLS = [
    "avg_daily_mean",
    "avg_daily_std",
    "avg_daily_max",
    "avg_daily_min",
    "total_consumption",
    "variability_ratio",
    "weekend_to_weekday_ratio",
    "winter_to_summer_ratio",
]

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110


# --------------------------------------------------------------------------------------
# Step 1-2: Load data
# --------------------------------------------------------------------------------------
def load_data(data_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the daily consumption records and household metadata."""
    daily_df = pd.read_csv(os.path.join(data_dir, "daily_dataset.csv"), parse_dates=["day"])
    households_df = pd.read_csv(os.path.join(data_dir, "informations_households.csv"))

    print("daily_dataset shape:", daily_df.shape)
    print("date range:", daily_df["day"].min(), "to", daily_df["day"].max())
    print("unique households:", daily_df["LCLid"].nunique())
    print("\nMissing values per column:")
    print(daily_df.isna().sum())
    return daily_df, households_df


# --------------------------------------------------------------------------------------
# Step 3: Clean
# --------------------------------------------------------------------------------------
def clean_data(daily_df: pd.DataFrame, min_days: int = 200) -> pd.DataFrame:
    """
    1. Drop rows with a missing energy_mean/max/min (unrecoverable day-level gaps).
    2. Keep only households with >= min_days recorded days, so a household's average
       isn't based on a handful of unrepresentative days.
    3. Drop exact duplicate (LCLid, day) rows.
    """
    daily_df = daily_df.dropna(subset=["energy_mean", "energy_max", "energy_min"])

    day_counts = daily_df.groupby("LCLid")["day"].count()
    valid_ids = day_counts[day_counts >= min_days].index
    daily_df = daily_df[daily_df["LCLid"].isin(valid_ids)].copy()

    daily_df = daily_df.drop_duplicates(subset=["LCLid", "day"])

    print("After cleaning:", daily_df.shape)
    print("Households remaining:", daily_df["LCLid"].nunique())
    return daily_df


# --------------------------------------------------------------------------------------
# Step 4: Merge metadata
# --------------------------------------------------------------------------------------
def merge_metadata(daily_df: pd.DataFrame, households_df: pd.DataFrame) -> pd.DataFrame:
    """Attach the Acorn demographic group (kept aside, not used for clustering)."""
    return daily_df.merge(households_df[["LCLid", "Acorn_grouped"]], on="LCLid", how="left")


# --------------------------------------------------------------------------------------
# Step 5: Feature engineering
# --------------------------------------------------------------------------------------
def engineer_features(daily_df: pd.DataFrame, households_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-day rows into one behavioural feature row per household."""
    household_features = daily_df.groupby("LCLid").agg(
        avg_daily_mean=("energy_mean", "mean"),
        avg_daily_std=("energy_mean", "std"),
        avg_daily_max=("energy_max", "mean"),
        avg_daily_min=("energy_min", "mean"),
        total_consumption=("energy_sum", "sum"),
        n_days=("day", "count"),
    ).reset_index()

    household_features["variability_ratio"] = (
        household_features["avg_daily_std"] / household_features["avg_daily_mean"]
    )

    daily_df = daily_df.copy()
    daily_df["day_of_week"] = daily_df["day"].dt.dayofweek
    daily_df["is_weekend"] = daily_df["day_of_week"].isin([5, 6])
    daily_df["month"] = daily_df["day"].dt.month
    daily_df["season"] = daily_df["month"].map(
        lambda m: "winter" if m in [12, 1, 2] else ("summer" if m in [6, 7, 8] else "shoulder")
    )

    weekday_weekend = daily_df.groupby(["LCLid", "is_weekend"])["energy_mean"].mean().unstack()
    weekday_weekend.columns = ["weekday_avg", "weekend_avg"]
    weekday_weekend["weekend_to_weekday_ratio"] = (
        weekday_weekend["weekend_avg"] / weekday_weekend["weekday_avg"]
    )

    seasonal = daily_df.groupby(["LCLid", "season"])["energy_mean"].mean().unstack()
    seasonal["winter_to_summer_ratio"] = seasonal["winter"] / seasonal["summer"]

    household_features = household_features.merge(
        weekday_weekend[["weekend_to_weekday_ratio"]], on="LCLid", how="left"
    )
    household_features = household_features.merge(
        seasonal[["winter_to_summer_ratio"]], on="LCLid", how="left"
    )
    household_features = household_features.merge(
        households_df[["LCLid", "Acorn_grouped"]], on="LCLid", how="left"
    )

    # Zero summer/weekday usage creates an infinite ratio; treat as missing, fill later.
    household_features = household_features.replace([np.inf, -np.inf], np.nan)

    print(household_features.shape)
    return household_features, daily_df


# --------------------------------------------------------------------------------------
# Step 6: Scale
# --------------------------------------------------------------------------------------
def scale_features(household_features: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = household_features[feature_cols].fillna(household_features[feature_cols].median())
    scaled = StandardScaler().fit_transform(X)
    print("Scaled feature matrix shape:", scaled.shape)
    return scaled


# --------------------------------------------------------------------------------------
# Step 7: PCA (for visualization only)
# --------------------------------------------------------------------------------------
def add_pca(household_features: pd.DataFrame, scaled: np.ndarray) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    pcs = pca.fit_transform(scaled)
    household_features["pca_1"] = pcs[:, 0]
    household_features["pca_2"] = pcs[:, 1]
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print(f"Together: {pca.explained_variance_ratio_.sum():.1%} of variance.")
    return household_features


# --------------------------------------------------------------------------------------
# Step 8: Choose k
# --------------------------------------------------------------------------------------
def select_k(scaled: np.ndarray, figures_dir: str, k_range=range(2, 7)) -> pd.DataFrame:
    """Scan k=2..6 for K-Means and plot elbow / silhouette / Davies-Bouldin curves."""
    inertias, silhouettes, db_scores = [], [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(scaled, km.labels_))
        db_scores.append(davies_bouldin_score(scaled, km.labels_))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_title("Elbow Method")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="green")
    axes[1].set_title("Silhouette Score (higher = better)")
    axes[1].set_xlabel("k")

    axes[2].plot(list(k_range), db_scores, marker="o", color="orange")
    axes[2].set_title("Davies-Bouldin Index (lower = better)")
    axes[2].set_xlabel("k")

    plt.tight_layout()
    out_path = os.path.join(figures_dir, "01_kmeans_k_selection.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")

    results = pd.DataFrame(
        {"k": list(k_range), "inertia": inertias, "silhouette": silhouettes, "davies_bouldin": db_scores}
    )
    for _, row in results.iterrows():
        print(f"k={int(row.k)}: silhouette={row.silhouette:.3f}, davies_bouldin={row.davies_bouldin:.3f}")
    return results


# --------------------------------------------------------------------------------------
# Step 9: Cluster with K-Means, Agglomerative, DBSCAN
# --------------------------------------------------------------------------------------
def run_clustering(
    household_features: pd.DataFrame,
    scaled: np.ndarray,
    best_k: int,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 15,
) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10).fit(scaled)
    household_features["cluster_kmeans"] = kmeans.labels_

    hier = AgglomerativeClustering(n_clusters=best_k).fit(scaled)
    household_features["cluster_hier"] = hier.labels_

    # DBSCAN needs no fixed k; points labelled -1 are noise/outlier households.
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(scaled)
    household_features["cluster_dbscan"] = dbscan.labels_

    print("K-Means cluster sizes:")
    print(household_features["cluster_kmeans"].value_counts().sort_index())
    print("\nDBSCAN cluster sizes (-1 = noise/outlier households):")
    print(household_features["cluster_dbscan"].value_counts().sort_index())
    return household_features


def plot_dendrogram(scaled: np.ndarray, figures_dir: str, sample_size: int = 200) -> None:
    sample_idx = np.random.RandomState(RANDOM_STATE).choice(
        scaled.shape[0], size=min(sample_size, scaled.shape[0]), replace=False
    )
    linked = linkage(scaled[sample_idx], method="ward")

    plt.figure(figsize=(12, 5))
    dendrogram(linked, truncate_mode="lastp", p=20)
    plt.title(f"Hierarchical Clustering Dendrogram (sample of {sample_size} households, last 20 merges)")
    plt.xlabel("Cluster size")
    plt.ylabel("Distance")
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "02_dendrogram.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------------------
# Model evaluation: compare K-Means / Agglomerative / DBSCAN on the chosen k
# --------------------------------------------------------------------------------------
def evaluate_models(scaled: np.ndarray, household_features: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Internal cluster-validity comparison across all three algorithms, using the same
    two metrics used for k-selection: silhouette score (higher is better) and the
    Davies-Bouldin index (lower is better). DBSCAN's noise points (label -1) are
    excluded before scoring, since silhouette/DB are undefined for a "noise" label.
    """
    rows = []
    for name, col in [
        ("KMeans", "cluster_kmeans"),
        ("Agglomerative", "cluster_hier"),
        ("DBSCAN", "cluster_dbscan"),
    ]:
        labels = household_features[col].to_numpy()
        mask = labels != -1
        n_clusters = len(set(labels[mask]))
        if n_clusters < 2:
            rows.append({"model": name, "n_clusters": n_clusters, "silhouette": np.nan, "davies_bouldin": np.nan,
                         "n_noise": int((~mask).sum())})
            continue
        sil = silhouette_score(scaled[mask], labels[mask])
        db = davies_bouldin_score(scaled[mask], labels[mask])
        rows.append(
            {
                "model": name,
                "n_clusters": n_clusters,
                "silhouette": round(sil, 4),
                "davies_bouldin": round(db, 4),
                "n_noise": int((~mask).sum()),
            }
        )

    results = pd.DataFrame(rows)
    print("\nModel evaluation (silhouette: higher is better, Davies-Bouldin: lower is better):")
    print(results.to_string(index=False))

    out_path = os.path.join(output_dir, "model_evaluation.csv")
    results.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    return results


# --------------------------------------------------------------------------------------
# Step 10: Visualize and profile clusters
# --------------------------------------------------------------------------------------
def plot_pca_scatter(household_features: pd.DataFrame, best_k: int, figures_dir: str) -> None:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=household_features,
        x="pca_1",
        y="pca_2",
        hue="cluster_kmeans",
        palette="tab10",
        s=35,
        alpha=0.7,
    )
    plt.title(f"Households in PCA Space, Colored by K-Means Cluster (k={best_k})")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "03_pca_cluster_scatter.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")


def profile_clusters(household_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cluster_profile = household_features.groupby("cluster_kmeans")[feature_cols].mean().round(3)
    print("\nAverage feature values per cluster:")
    print(cluster_profile)
    return cluster_profile


def crosstab_acorn(household_features: pd.DataFrame) -> pd.DataFrame:
    acorn_crosstab = pd.crosstab(household_features["cluster_kmeans"], household_features["Acorn_grouped"])
    print("\nCluster vs. Acorn demographic group (validation only, not used to build clusters):")
    print(acorn_crosstab)
    return acorn_crosstab


def plot_cluster_levels(cluster_profile: pd.DataFrame, figures_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    cluster_profile[["avg_daily_mean", "avg_daily_max", "avg_daily_min"]].plot(kind="bar", ax=ax)
    ax.set_title("Average Daily Consumption Level per Cluster")
    ax.set_ylabel("kWh")
    ax.set_xlabel("Cluster")
    plt.xticks(rotation=0)
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "04_cluster_consumption_levels.png")
    plt.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# --------------------------------------------------------------------------------------
# Step 11: Peak / off-peak usage
# --------------------------------------------------------------------------------------
def peak_offpeak_daytype(daily_df: pd.DataFrame, household_features: pd.DataFrame) -> pd.DataFrame:
    """Approximate peak/off-peak at the season x weekday/weekend level (always available)."""
    daily_with_cluster = daily_df.merge(
        household_features[["LCLid", "cluster_kmeans"]], on="LCLid", how="inner"
    )
    peak_summary = daily_with_cluster.groupby(["cluster_kmeans", "season", "is_weekend"])["energy_mean"].mean()
    peak_summary = peak_summary.reset_index().sort_values(
        ["cluster_kmeans", "energy_mean"], ascending=[True, False]
    )

    print("\nHighest and lowest average day-type per cluster:")
    for c in sorted(daily_with_cluster["cluster_kmeans"].unique()):
        sub = peak_summary[peak_summary["cluster_kmeans"] == c]
        top, low = sub.iloc[0], sub.iloc[-1]
        print(
            f"Cluster {c}: PEAK -> {top['season']} / {'weekend' if top['is_weekend'] else 'weekday'} "
            f"({top['energy_mean']:.3f} kWh avg)   "
            f"OFF-PEAK -> {low['season']} / {'weekend' if low['is_weekend'] else 'weekday'} "
            f"({low['energy_mean']:.3f} kWh avg)"
        )
    return peak_summary


def _hh_to_time(col: str) -> str:
    n = int(col.split("_")[1])
    h, m = divmod(n * 30, 60)
    return f"{h:02d}:{m:02d}"


def peak_offpeak_hourly(household_features: pd.DataFrame, hhblock_dir: str, figures_dir: str) -> pd.DataFrame | None:
    """
    True hour-of-day peak/off-peak detection using the half-hourly `hhblock_dataset`
    (48 half-hourly columns per household per day). Reads block files one at a time and
    accumulates a running sum/count per cluster per half-hour, so memory stays small.
    """
    hh_cols = [f"hh_{i}" for i in range(48)]
    hh_files = sorted(glob.glob(os.path.join(hhblock_dir, "block_*.csv")))
    if not hh_files:
        print(f"No block_*.csv files found in {hhblock_dir}; skipping hourly peak analysis.")
        return None
    print(f"\nFound {len(hh_files)} block files")

    cluster_ids = sorted(household_features["cluster_kmeans"].unique())
    cluster_map = household_features.set_index("LCLid")["cluster_kmeans"]

    cluster_sum = pd.DataFrame(0.0, index=cluster_ids, columns=hh_cols)
    cluster_count = pd.DataFrame(0.0, index=cluster_ids, columns=hh_cols)

    for i, f in enumerate(hh_files, start=1):
        block = pd.read_csv(f)
        block = block[block["LCLid"].isin(cluster_map.index)].copy()
        if block.empty:
            continue
        block["cluster_kmeans"] = block["LCLid"].map(cluster_map)

        # "Null" strings -> proper NaN so they aren't counted as real zero-usage readings.
        block[hh_cols] = block[hh_cols].apply(pd.to_numeric, errors="coerce")

        grouped_sum = block.groupby("cluster_kmeans")[hh_cols].sum(min_count=1)
        grouped_count = block.groupby("cluster_kmeans")[hh_cols].count()

        cluster_sum = cluster_sum.add(grouped_sum, fill_value=0)
        cluster_count = cluster_count.add(grouped_count, fill_value=0)

        if i % 20 == 0 or i == len(hh_files):
            print(f"  processed {i}/{len(hh_files)} files")

    cluster_load_curve = cluster_sum / cluster_count.replace(0, np.nan)
    valid_clusters = cluster_load_curve.dropna(how="all").index

    plt.figure(figsize=(11, 5))
    for c in valid_clusters:
        plt.plot(range(48), cluster_load_curve.loc[c], label=f"Cluster {c}")
    plt.xlabel("Half-hour of day")
    plt.ylabel("Average consumption (kWh)")
    plt.title("Average 24-Hour Load Curve per Cluster")
    plt.xticks(range(0, 48, 4), [f"{h:02d}:00" for h in range(0, 24, 2)], rotation=45)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(figures_dir, "05_cluster_load_curve.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved {out_path}")

    print("\nTrue hour-of-day peak / off-peak per cluster:")
    skipped = [c for c in cluster_load_curve.index if c not in valid_clusters]
    if skipped:
        print(f"(skipping cluster(s) {skipped} — no matching households found in the hh-block files)")
    for c in valid_clusters:
        peak_hh = cluster_load_curve.loc[c].idxmax()
        offpeak_hh = cluster_load_curve.loc[c].idxmin()
        print(
            f"Cluster {c}: PEAK at {_hh_to_time(peak_hh)} ({cluster_load_curve.loc[c, peak_hh]:.3f} kWh avg)   "
            f"OFF-PEAK at {_hh_to_time(offpeak_hh)} ({cluster_load_curve.loc[c, offpeak_hh]:.3f} kWh avg)"
        )
    return cluster_load_curve


# --------------------------------------------------------------------------------------
# Step 12: Recommendation engine
# --------------------------------------------------------------------------------------
def recommend_for_cluster(row: pd.Series, household_features: pd.DataFrame) -> str:
    recs = []
    if row["winter_to_summer_ratio"] > 1.8:
        recs.append(
            "Strong winter increase — likely electric heating. Recommend a smart "
            "thermostat / heating schedule to avoid unnecessary overnight heating."
        )
    if row["weekend_to_weekday_ratio"] > 1.3:
        recs.append(
            "Weekend-heavy usage — recommend running high-load appliances (washing "
            "machine, dishwasher) during weekday off-peak hours instead."
        )
    elif row["weekend_to_weekday_ratio"] < 0.8:
        recs.append(
            "Weekday-heavy usage, likely home unoccupied on weekends — recommend "
            "checking for standby/phantom load during empty weekday daytime hours."
        )
    if row["variability_ratio"] > 1.0:
        recs.append(
            "High day-to-day variability — recommend a consumption-tracking app to "
            "help the household notice and flag unusually high days."
        )
    if row["avg_daily_mean"] < household_features["avg_daily_mean"].quantile(0.25):
        recs.append(
            "Already a low, steady consumer — no major intervention needed; "
            "candidate for a small loyalty/efficiency reward under a demand-response program."
        )
    if not recs:
        recs.append(
            "Consumption pattern is close to average — general efficiency tips apply "
            "(LED lighting, standby power audit)."
        )
    return " ".join(recs)


def generate_recommendations(household_features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    cluster_summary = household_features.groupby("cluster_kmeans")[feature_cols].mean()
    cluster_summary["recommendation"] = cluster_summary.apply(
        lambda row: recommend_for_cluster(row, household_features), axis=1
    )
    print("\nEnergy-saving recommendations:")
    for c, row in cluster_summary.iterrows():
        n = int((household_features["cluster_kmeans"] == c).sum())
        print(f"--- Cluster {c} (n={n}) ---")
        print(row["recommendation"])
        print()
    return cluster_summary


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Electricity Consumption Pattern Analysis")
    parser.add_argument("--data-dir", default="data", help="Folder with daily_dataset.csv etc.")
    parser.add_argument("--figures-dir", default="figures", help="Folder to save .png figures to")
    parser.add_argument("--output-dir", default="outputs", help="Folder to save .csv outputs to")
    parser.add_argument(
        "--hhblock-dir",
        default=None,
        help="Optional folder with block_*.csv files for true hour-of-day peak analysis",
    )
    parser.add_argument("--best-k", type=int, default=4, help="Number of clusters for K-Means/Agglomerative")
    parser.add_argument("--min-days", type=int, default=200, help="Minimum recorded days to keep a household")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    Path(args.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    daily_df, households_df = load_data(args.data_dir)
    daily_df = clean_data(daily_df, min_days=args.min_days)
    daily_df = merge_metadata(daily_df, households_df)
    household_features, daily_df = engineer_features(daily_df, households_df)

    scaled = scale_features(household_features, FEATURE_COLS)
    household_features = add_pca(household_features, scaled)

    select_k(scaled, args.figures_dir)
    household_features = run_clustering(household_features, scaled, args.best_k)
    plot_dendrogram(scaled, args.figures_dir)
    evaluate_models(scaled, household_features, args.output_dir)

    plot_pca_scatter(household_features, args.best_k, args.figures_dir)
    cluster_profile = profile_clusters(household_features, FEATURE_COLS)
    crosstab_acorn(household_features)
    plot_cluster_levels(cluster_profile, args.figures_dir)

    peak_offpeak_daytype(daily_df, household_features)
    if args.hhblock_dir:
        peak_offpeak_hourly(household_features, args.hhblock_dir, args.figures_dir)

    generate_recommendations(household_features, FEATURE_COLS)

    out_path = os.path.join(args.output_dir, "household_clusters_output.csv")
    household_features.to_csv(out_path, index=False)
    print(f"\nSaved {out_path} with {household_features.shape[0]} households and their cluster labels.")


if __name__ == "__main__":
    main()

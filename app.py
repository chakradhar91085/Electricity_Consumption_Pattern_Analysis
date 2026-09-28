"""
Streamlit dashboard for the Electricity Consumption Pattern Analysis project.

Reuses every data-loading, feature-engineering, clustering, evaluation, and
recommendation function from `electricity_clustering_analysis.py` as-is — this file
only adds interactive controls (algorithm choice, k) and Streamlit-native rendering
on top of those same functions.

Run with:
    streamlit run app.py
"""

import tempfile

import pandas as pd
import streamlit as st

from electricity_clustering_analysis import (
    FEATURE_COLS,
    add_pca,
    clean_data,
    crosstab_acorn,
    engineer_features,
    evaluate_models,
    generate_recommendations,
    load_data,
    merge_metadata,
    peak_offpeak_daytype,
    profile_clusters,
    run_clustering,
    scale_features,
    select_k,
)

st.set_page_config(page_title="Electricity Consumption Clustering", page_icon="⚡", layout="wide")

CLUSTER_COL_BY_ALGO = {
    "K-Means": "cluster_kmeans",
    "Agglomerative (Ward)": "cluster_hier",
    "DBSCAN": "cluster_dbscan",
}


@st.cache_data(show_spinner="Loading and engineering household features…")
def load_pipeline(min_days: int):
    daily_df, households_df = load_data("data")
    daily_df = clean_data(daily_df, min_days=min_days)
    daily_df = merge_metadata(daily_df, households_df)
    household_features, daily_df = engineer_features(daily_df, households_df)
    scaled = scale_features(household_features, FEATURE_COLS)
    household_features = add_pca(household_features, scaled)
    return household_features, daily_df, scaled


@st.cache_data(show_spinner="Scanning k = 2..6…")
def cached_select_k(_scaled):
    with tempfile.TemporaryDirectory() as tmp:
        return select_k(_scaled, tmp)


@st.cache_data(show_spinner="Clustering…")
def cached_clustering(household_features: pd.DataFrame, _scaled, k: int, eps: float, min_samples: int):
    hf = household_features.copy()
    return run_clustering(hf, _scaled, best_k=k, dbscan_eps=eps, dbscan_min_samples=min_samples)


@st.cache_data(show_spinner=False)
def cached_evaluation(_scaled, household_features: pd.DataFrame):
    with tempfile.TemporaryDirectory() as tmp:
        return evaluate_models(_scaled, household_features, tmp)


def static_figure(path: str, caption: str) -> None:
    from pathlib import Path

    if Path(path).exists():
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not available: `{path}`")


# --------------------------------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------------------------------
st.sidebar.title("⚡ Controls")
algo = st.sidebar.selectbox("Clustering algorithm", list(CLUSTER_COL_BY_ALGO.keys()))

if algo == "DBSCAN":
    k = 4  # unused by DBSCAN but kept for run_clustering's K-Means/Agglomerative columns
    eps = st.sidebar.slider("DBSCAN eps", 0.5, 3.0, 1.2, 0.1)
    min_samples = st.sidebar.slider("DBSCAN min_samples", 5, 30, 15)
else:
    k = st.sidebar.slider("Number of clusters (k)", 2, 8, 4)
    eps, min_samples = 1.2, 15

st.sidebar.caption(
    "Data: bundled 400-household sample of the Smart Meters in London dataset. "
    "See README for the full 5,530-household results."
)

household_features, daily_df, scaled = load_pipeline(min_days=200)
hf = cached_clustering(household_features, scaled, k, eps, min_samples)
active = hf.copy()
active["cluster_kmeans"] = active[CLUSTER_COL_BY_ALGO[algo]]

# --------------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------------
st.title("Electricity Consumption Pattern Analysis")
st.caption(
    "Clustering London households by electricity consumption behaviour — "
    "Smart Meters in London dataset (UK Power Networks / Kaggle)."
)

tabs = st.tabs(
    ["Overview", "Data & Features", "Choosing k", "Clusters", "Profiles & Validation",
     "Peak / Off-Peak", "Recommendations"]
)

# --------------------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------------------
with tabs[0]:
    st.markdown(
        """
Groups ~5,500 London households into behavioural segments (not just "big house vs.
small house"), validates the clusters statistically, visualizes them, profiles their
peak/off-peak usage, and generates a cluster-specific energy-saving recommendation for
each segment.

**Method**
1. Load & clean half-hourly-derived daily readings; keep households with ≥200 recorded days.
2. Engineer 8 behavioural features per household (overall level, variability, weekend/weekday
   ratio, winter/summer ratio).
3. Scale with `StandardScaler`; reduce to 2D with PCA for visualization only.
4. Scan k = 2..6 with K-Means (elbow, silhouette, Davies-Bouldin).
5. Cluster three ways — K-Means, Agglomerative (Ward), DBSCAN — and compare.
6. Profile clusters, cross-check against Acorn demographic group (validation only).
7. Analyze peak/off-peak usage and generate a plain-language recommendation per cluster.

Use the sidebar to switch clustering algorithm and (for K-Means/Agglomerative) the
number of clusters — every tab below recomputes live.
        """
    )

# --------------------------------------------------------------------------------------
# Data & Features
# --------------------------------------------------------------------------------------
with tabs[1]:
    st.subheader("Household feature table")
    st.caption(f"{len(household_features)} households after cleaning (≥200 recorded days).")
    st.dataframe(household_features[["LCLid", *FEATURE_COLS, "Acorn_grouped"]].head(50), use_container_width=True)
    st.subheader("Feature summary statistics")
    st.dataframe(household_features[FEATURE_COLS].describe().round(3), use_container_width=True)

# --------------------------------------------------------------------------------------
# Choosing k
# --------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("K-Means k-selection scan (k = 2..6)")
    k_results = cached_select_k(scaled)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.line_chart(k_results.set_index("k")[["inertia"]])
        st.caption("Elbow (lower isn't always better — look for the bend)")
    with c2:
        st.line_chart(k_results.set_index("k")[["silhouette"]])
        st.caption("Silhouette — higher is better")
    with c3:
        st.line_chart(k_results.set_index("k")[["davies_bouldin"]])
        st.caption("Davies-Bouldin — lower is better")
    st.dataframe(k_results.round(4), use_container_width=True)
    static_figure("figures/02_dendrogram.png", "Hierarchical clustering dendrogram (sampled, full-dataset run)")

# --------------------------------------------------------------------------------------
# Clusters
# --------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader(f"Households in PCA space — colored by {algo}")
    plot_df = active[["pca_1", "pca_2", "cluster_kmeans"]].copy()
    plot_df["cluster"] = plot_df["cluster_kmeans"].astype(str)
    st.scatter_chart(plot_df, x="pca_1", y="pca_2", color="cluster")

    st.subheader("Cluster sizes")
    st.bar_chart(active["cluster_kmeans"].value_counts().sort_index())

    st.subheader("Model evaluation — K-Means vs. Agglomerative vs. DBSCAN")
    st.caption("Same k / DBSCAN params as the sidebar. Silhouette: higher is better. Davies-Bouldin: lower is better.")
    st.dataframe(cached_evaluation(scaled, hf), use_container_width=True)

# --------------------------------------------------------------------------------------
# Profiles & Validation
# --------------------------------------------------------------------------------------
with tabs[4]:
    st.subheader(f"Average feature values per cluster ({algo})")
    cluster_profile = profile_clusters(active, FEATURE_COLS)
    st.dataframe(cluster_profile, use_container_width=True)
    st.bar_chart(cluster_profile[["avg_daily_mean", "avg_daily_max", "avg_daily_min"]])

    st.subheader("Cluster vs. Acorn demographic group (validation only — not used to build clusters)")
    st.dataframe(crosstab_acorn(active), use_container_width=True)

# --------------------------------------------------------------------------------------
# Peak / Off-Peak
# --------------------------------------------------------------------------------------
with tabs[5]:
    st.subheader(f"Day-type usage per cluster ({algo})")
    peak_summary = peak_offpeak_daytype(daily_df, active)
    peak_summary = peak_summary.copy()
    peak_summary["day_type"] = peak_summary["season"] + " / " + peak_summary["is_weekend"].map(
        {True: "weekend", False: "weekday"}
    )
    pivot = peak_summary.pivot(index="day_type", columns="cluster_kmeans", values="energy_mean")
    st.bar_chart(pivot)
    st.dataframe(peak_summary.drop(columns="day_type"), use_container_width=True)

    st.divider()
    st.subheader("True hour-of-day load curve (reference)")
    st.caption(
        "Requires the optional `hhblock_dataset` (not bundled). Figure below, if present, is from a "
        "prior full-dataset run and doesn't change with the sidebar controls."
    )
    static_figure("figures/05_cluster_load_curve.png", "24-hour load curve per cluster")

# --------------------------------------------------------------------------------------
# Recommendations
# --------------------------------------------------------------------------------------
with tabs[6]:
    st.subheader(f"Energy-saving recommendations ({algo})")
    recs = generate_recommendations(active, FEATURE_COLS)
    for cluster_id, row in recs.iterrows():
        n = int((active["cluster_kmeans"] == cluster_id).sum())
        label = "Noise / outliers" if cluster_id == -1 else f"Cluster {cluster_id}"
        with st.container(border=True):
            st.markdown(f"**{label}** (n={n})")
            st.write(row["recommendation"])

    st.divider()
    st.subheader("How this supports smart energy management")
    st.markdown(
        """
- **Targeted demand response** — the highest-consuming, earliest-peaking cluster is the
  best target for a demand-response alert aimed at flattening the grid-wide evening peak.
- **Tariff design** — clusters with a high winter/summer ratio are candidates for a
  heating-season-aware tariff rather than a flat rate.
- **Infrastructure planning** — relative cluster sizes give a rough read on what share of
  a neighbourhood falls into each usage pattern, useful for transformer/feeder sizing.
- **Personalized guidance** — a distinct recommendation per cluster instead of one
  generic message to every household.
- **Outlier handling** — DBSCAN's noise-labelled households are worth a manual look
  (possibly faulty meters, or genuinely unusual consumption) rather than being forced
  into the nearest K-Means cluster.
        """
    )

    with st.expander("Limitations"):
        st.markdown(
            """
- Dataset covers Nov 2011–Feb 2014 only — no post-pandemic, EV, or heat-pump-scale patterns.
- This dashboard runs on the bundled 400-household sample; headline README numbers are
  from the full 5,530-household Kaggle dataset.
- True hour-of-day peak analysis needs the optional `hhblock_dataset`, not bundled here.
- K-Means assumes roughly spherical, similarly-sized clusters — DBSCAN handles outliers
  more honestly by labelling them noise instead of forcing them into a cluster.
- Acorn demographic data is used only for post-hoc validation, never as a clustering input.
            """
        )

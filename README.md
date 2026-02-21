# test
import argparse
import os
import sys
import pandas as pd


# -----------------------------
# Helpers
# -----------------------------
def _has(df, col: str) -> bool:
    return col in df.columns


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _safe_value_counts(df: pd.DataFrame, col: str, topn: int = 20):
    if not _has(df, col):
        print(f"[skip] missing column: {col}")
        return
    vc = df[col].value_counts(dropna=False).head(topn)
    print(vc)


def _coerce_boolish(s: pd.Series) -> pd.Series:
    # handles True/False, "True"/"False", 1/0, NaN
    if s.dtype == bool:
        return s.fillna(False)
    return (
        s.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def _coerce_month(s: pd.Series) -> pd.Series:
    # expects MONAT like 202506 or "202506"
    out = pd.to_numeric(s, errors="coerce")
    return out.astype("Int64")


def _cluster_cols(k: int):
    return (
        [f"cluster_id_{i}" for i in range(1, k + 1)],
        [f"cluster_score_{i}" for i in range(1, k + 1)],
    )


def _topk_long(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return long-format rows: one row per original record per rank."""
    id_cols, score_cols = _cluster_cols(k)
    present_ids = [c for c in id_cols if _has(df, c)]
    present_scores = [c for c in score_cols if _has(df, c)]

    if not present_ids:
        return pd.DataFrame()

    # build long rows
    rows = []
    base_cols = [c for c in ["MONAT", "TEXT_CLEAN", "TEXT", "TEXT_PRE", "TEXT_KEY_CLEAN", "CLIENT", "PRODUCT", "SUBCATEGORY", "RATING", "cluster_type", "noise_reassigned_any", "contains_pii"] if _has(df, c)]
    for i, idc in enumerate(id_cols, start=1):
        if not _has(df, idc):
            continue
        sc = f"cluster_score_{i}"
        tmp = df[base_cols + [idc]].copy()
        tmp.rename(columns={idc: "cluster_id"}, inplace=True)
        tmp["rank"] = i
        if _has(df, sc):
            tmp["cluster_score"] = pd.to_numeric(df[sc], errors="coerce")
        else:
            tmp["cluster_score"] = pd.NA
        rows.append(tmp)

    out = pd.concat(rows, axis=0, ignore_index=True)
    out = out.dropna(subset=["cluster_id"])
    # normalize cluster_id to string (your ids like 202506_78)
    out["cluster_id"] = out["cluster_id"].astype(str)
    return out


def _write_csv(df: pd.DataFrame, path: str, sep: str = ";"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep=sep, index=False)
    print(f"[saved] {path}")


# -----------------------------
# Analyses
# -----------------------------
def basic_overview(df: pd.DataFrame):
    _print_section("BASIC OVERVIEW")
    print(f"Rows: {len(df):,}")
    print(f"Cols: {len(df.columns):,}")
    print("Columns:", list(df.columns))


def missingness(df: pd.DataFrame):
    _print_section("MISSINGNESS (top 25)")
    miss = (df.isna().mean() * 100).sort_values(ascending=False).head(25)
    print(miss.to_string())


def pii_overview(df: pd.DataFrame):
    _print_section("PII OVERVIEW")
    if not _has(df, "contains_pii"):
        print("[skip] missing column: contains_pii")
        return
    s = _coerce_boolish(df["contains_pii"])
    print(s.value_counts(dropna=False))


def noise_overview(df: pd.DataFrame):
    _print_section("NOISE / REASSIGN OVERVIEW")
    if _has(df, "noise_reassigned_any"):
        s = _coerce_boolish(df["noise_reassigned_any"])
        print("noise_reassigned_any:")
        print(s.value_counts(dropna=False))
    else:
        print("[skip] missing column: noise_reassigned_any")

    if _has(df, "cluster_type"):
        print("\ncluster_type:")
        _safe_value_counts(df, "cluster_type", topn=50)
    else:
        print("[skip] missing column: cluster_type")


def month_overview(df: pd.DataFrame):
    _print_section("MONTH OVERVIEW")
    if not _has(df, "MONAT"):
        print("[skip] missing column: MONAT")
        return
    m = _coerce_month(df["MONAT"])
    tmp = df.copy()
    tmp["MONAT"] = m
    print("Rows per MONAT:")
    print(tmp["MONAT"].value_counts(dropna=False).sort_index().to_string())


def cluster_combo_overview(df: pd.DataFrame):
    _print_section("TOP-K COMBINATION OVERVIEW (cluster_ids)")
    if not _has(df, "cluster_ids"):
        print("[skip] missing column: cluster_ids")
        return
    print("Unique cluster_ids combinations:", df["cluster_ids"].nunique(dropna=True))
    print("\nMost common combinations:")
    print(df["cluster_ids"].value_counts().head(20).to_string())


def cluster_size_overview(df: pd.DataFrame, k: int):
    _print_section("CLUSTER SIZE OVERVIEW (using cluster_id_1)")
    cid1 = "cluster_id_1"
    if not _has(df, cid1):
        print(f"[skip] missing column: {cid1}")
        return

    # normalize string
    c = df[cid1].astype(str)
    print("Unique cluster_id_1:", c.nunique(dropna=True))
    print("\nTop 25 cluster_id_1 sizes:")
    print(c.value_counts().head(25).to_string())

    if _has(df, "MONAT"):
        tmp = df.copy()
        tmp["MONAT"] = _coerce_month(tmp["MONAT"])
        tmp[cid1] = tmp[cid1].astype(str)
        print("\nTop 10 cluster_id_1 per month (size):")
        per_m = (
            tmp.groupby("MONAT")[cid1]
            .value_counts()
            .groupby(level=0)
            .head(10)
        )
        print(per_m.to_string())


def topk_usage_overview(df: pd.DataFrame, k: int):
    _print_section("TOP-K USAGE OVERVIEW (how many ranks filled)")
    id_cols, _ = _cluster_cols(k)
    present = [c for c in id_cols if _has(df, c)]
    if not present:
        print("[skip] no cluster_id_1..k columns found")
        return

    # count filled ranks per row
    filled = pd.DataFrame({c: df[c].notna() & (df[c].astype(str).str.len() > 0) for c in present})
    n_filled = filled.sum(axis=1)
    print(n_filled.value_counts().sort_index().to_string())

    # optional: compare with cluster_ranks string
    if _has(df, "cluster_ranks"):
        print("\n(cluster_ranks) sample & dtype:")
        print(df["cluster_ranks"].head(5))
        print(type(df["cluster_ranks"].iloc[0]))


def score_overview(df: pd.DataFrame, k: int):
    _print_section("SCORE OVERVIEW (cluster_score_1..k)")
    _, score_cols = _cluster_cols(k)
    present = [c for c in score_cols if _has(df, c)]
    if not present:
        print("[skip] no cluster_score_1..k columns found")
        return

    tmp = df[present].apply(pd.to_numeric, errors="coerce")
    print(tmp.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_string())


def examples_per_rank(df: pd.DataFrame, k: int, per_cluster: int = 3, top_clusters: int = 10):
    """
    Fachlich: zeige Beispieltexte für häufige Cluster, getrennt nach Rank 1/2/3.
    """
    _print_section("EXAMPLE TEXTS PER RANK (for top clusters)")
    text_col = "TEXT_CLEAN" if _has(df, "TEXT_CLEAN") else ("TEXT" if _has(df, "TEXT") else None)
    if text_col is None:
        print("[skip] no text column found (TEXT_CLEAN/TEXT)")
        return

    long = _topk_long(df, k)
    if long.empty:
        print("[skip] cannot build long format (missing cluster_id_*)")
        return

    # take top clusters by rank=1 size
    top = (
        long[long["rank"] == 1]["cluster_id"]
        .value_counts()
        .head(top_clusters)
        .index
        .tolist()
    )
    for cid in top:
        print("\n" + "-" * 80)
        print(f"Cluster: {cid}")
        for r in range(1, k + 1):
            sub = long[(long["cluster_id"] == cid) & (long["rank"] == r)]
            if sub.empty:
                continue
            print(f"\n  Rank {r} examples (n={len(sub)}; showing {per_cluster}):")
            # prefer highest score samples if available
            if "cluster_score" in sub.columns:
                sub = sub.sort_values("cluster_score", ascending=False)
            for t in sub[text_col].dropna().astype(str).head(per_cluster).tolist():
                print("   -", t[:300].replace("\n", " "))


def build_cluster_samples_for_llm(df: pd.DataFrame, k: int, out_dir: str, sep: str):
    """
    Export: pro cluster_id_1 (Top-1) eine Datei mit repräsentativen Texten.
    Heuristik: top N nach cluster_score_1 (oder falls fehlt: zufällige / erste).
    """
    _print_section("EXPORT: CLUSTER SAMPLES FOR LLM TITLES")
    text_col = "TEXT_CLEAN" if _has(df, "TEXT_CLEAN") else ("TEXT" if _has(df, "TEXT") else None)
    if text_col is None:
        print("[skip] no text column found (TEXT_CLEAN/TEXT)")
        return
    if not _has(df, "cluster_id_1"):
        print("[skip] missing column: cluster_id_1")
        return

    tmp = df.copy()
    tmp["cluster_id_1"] = tmp["cluster_id_1"].astype(str)

    # optional: filter out PII rows if you want
    if _has(tmp, "contains_pii"):
        pii = _coerce_boolish(tmp["contains_pii"])
        # keep all, but you could exclude:
        # tmp = tmp[~pii]
        tmp["contains_pii"] = pii

    # score_1 optional
    if _has(tmp, "cluster_score_1"):
        tmp["cluster_score_1"] = pd.to_numeric(tmp["cluster_score_1"], errors="coerce")
    else:
        tmp["cluster_score_1"] = pd.NA

    # remove obvious junk clusters if present
    bad = set(["NOISE", "SCHROTT", "-1", "-2", "nan", "None"])
    tmp = tmp[~tmp["cluster_id_1"].isin(bad)]

    # create one compact table: cluster_id_1, MONAT, count, sample_texts (joined)
    sample_n = 20
    def pick_texts(grp: pd.DataFrame):
        g = grp.dropna(subset=[text_col]).copy()
        if g.empty:
            return ""
        if g["cluster_score_1"].notna().any():
            g = g.sort_values("cluster_score_1", ascending=False)
        texts = g[text_col].astype(str).head(sample_n).tolist()
        # keep them separated by " || " for downstream prompts
        return " || ".join([t.replace("\n", " ").strip()[:500] for t in texts])

    agg = tmp.groupby("cluster_id_1", dropna=False).apply(
        lambda g: pd.Series({
            "count": len(g),
            "months": ",".join(sorted(g["MONAT"].dropna().astype(str).unique().tolist())) if _has(g, "MONAT") else "",
            "sample_texts": pick_texts(g),
            "share_noise_reassigned": float(_coerce_boolish(g["noise_reassigned_any"]).mean()) if _has(g, "noise_reassigned_any") else pd.NA,
        })
    ).reset_index()

    agg = agg.sort_values("count", ascending=False)

    out_path = os.path.join(out_dir, "cluster_samples_for_llm.csv")
    _write_csv(agg, out_path, sep=sep)

    print("\nTop 10 clusters exported (preview):")
    print(agg.head(10).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/containeruser/data/initial_pipeline/results_hdbscan.csv")
    parser.add_argument("--sep", type=str, default=";")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default="/home/containeruser/data/initial_pipeline/analysis_out")
    args = parser.parse_args()

    # pretty display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", 200)

    if not os.path.exists(args.path):
        print(f"[error] file not found: {args.path}")
        sys.exit(1)

    df = pd.read_csv(args.path, sep=args.sep)
    # normalize key columns if present
    if _has(df, "MONAT"):
        df["MONAT"] = _coerce_month(df["MONAT"])
    if _has(df, "noise_reassigned_any"):
        df["noise_reassigned_any"] = _coerce_boolish(df["noise_reassigned_any"])
    if _has(df, "contains_pii"):
        df["contains_pii"] = _coerce_boolish(df["contains_pii"])

    # run analyses
    basic_overview(df)
    missingness(df)
    pii_overview(df)
    noise_overview(df)
    month_overview(df)

    cluster_combo_overview(df)
    cluster_size_overview(df, args.k)
    topk_usage_overview(df, args.k)
    score_overview(df, args.k)

    examples_per_rank(df, args.k, per_cluster=3, top_clusters=10)
    build_cluster_samples_for_llm(df, args.k, args.out_dir, args.sep)


if __name__ == "__main__":
    main()


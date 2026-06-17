#!/usr/bin/env python3
"""
fix_myc_tf_clean_regulon_wrapper.py
================================================================================
Wrapper ejecutable: corre el Módulo 19 con el regulón LIMPIO sin modificar
myc_tf_activity_decoupler.py.

Estrategia: importa el módulo original, monkey-patches las dos funciones
problemáticas en su namespace, luego ejecuta main().

OUTPUTS (misma ruta que el original, sufijo _clean):
  results/myc_tf_activity/myc_tf_activity_results_clean.csv
  results/myc_tf_activity/myc_tf_activity_by_phenotype_clean.csv
  results/myc_tf_activity/fig_myc_tf_violin_clean.png
  results/myc_tf_activity/fig_myc_tf_binary_desert_clean.png
  results/myc_tf_activity/fig_myc_tf_vs_mrna_clean.png
  data/processed/adata_with_myc_tf_clean.h5ad

COMPARACIÓN:
  Después de correr, comparar resultados _clean vs originales (sin sufijo).
  La diferencia entre los dos d(Desert vs Inflamed) cuantifica el artefacto.
================================================================================
"""

import sys
import logging
import time
import warnings
from pathlib import Path

try:
    import torch as _torch_preload  # noqa: F401
except ImportError:
    pass  # torch no instalado → sin problema TLS, continuar
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Añadir directorio de códigos al path ─────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ── Importar módulo original ──────────────────────────────────────────────────
try:
    import myc_tf_activity_decoupler as _orig
except ImportError as e:
    print(f"[ERROR] No se pudo importar myc_tf_activity_decoupler: {e}")
    print("  Asegúrate de ejecutar desde el directorio codigos/")
    sys.exit(1)

# ── Importar el fix ───────────────────────────────────────────────────────────
try:
    from fix_myc_tf_clean_regulon import (
        MYC_COLLECTRI_CLEAN_POSITIVE,
        READOUT_GENES,
        audit_regulon_contamination,
        _build_backup_regulon_CLEAN,
        get_myc_regulon_CLEAN,
        analysis_B_correlations_CLEAN,
    )
except ImportError:
    # Si fix_myc_tf_clean_regulon.py está en results/
    sys.path.insert(0, str(_SCRIPT_DIR.parent / "results"))
    from fix_myc_tf_clean_regulon import (
        MYC_COLLECTRI_CLEAN_POSITIVE,
        READOUT_GENES,
        audit_regulon_contamination,
        _build_backup_regulon_CLEAN,
        get_myc_regulon_CLEAN,
        analysis_B_correlations_CLEAN,
    )

# ── Paths — idénticos al original salvo sufijo _clean ─────────────────────────
_BASE            = _SCRIPT_DIR.parent
DATA_PROCESSED   = _BASE / "data" / "processed"
RESULTS_DIR      = _BASE / "results" / "myc_tf_activity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADATA_INPUT      = DATA_PROCESSED / "adata_with_mechanism.h5ad"
ADATA_OUTPUT     = DATA_PROCESSED / "adata_with_myc_tf_clean.h5ad"

SEED = 42
np.random.seed(SEED)

# ── Labels (igual que el original) ───────────────────────────────────────────
PHENOTYPE_COL  = "Phenotype"
DESERT_LABEL   = "Immune_Desert"
EXCLUDED_LABEL = "Immune_Excluded"
INFLAMED_LABEL = "Inflamed"
STROMA_LABEL   = "Normal_Stroma"


# ── Helpers copiados del original (para evitar dependencia implícita) ─────────

def safe_toarray(X):
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


def cohens_d_pooled(g1, g2):
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var_pool = (
        ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
    )
    if var_pool == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / np.sqrt(var_pool)


def fdr_correct(pvals):
    pvals_arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals_arr)
    qvals = np.full_like(pvals_arr, np.nan)
    if valid.sum() > 0:
        _, q, _, _ = multipletests(pvals_arr[valid], alpha=0.05, method="fdr_bh")
        qvals[valid] = q
    return qvals


def get_expression_matrix_raw(adata):
    if adata.raw is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names)
        return X, var_names
    return adata.X, adata.var_names


def compute_gene_score(adata, genes, score_name, X=None, var_names=None):
    if var_names is None or X is None:
        X, var_names = get_expression_matrix_raw(adata)
    available = [g for g in genes if g in var_names]
    absent    = [g for g in genes if g not in var_names]
    if absent:
        logging.warning(f"  [{score_name}] ausentes: {absent}")
    if not available:
        return np.full(adata.n_obs, np.nan)
    logging.info(f"  [{score_name}] {len(available)}/{len(genes)} genes usados: {available}")
    vecs = []
    for g in available:
        idx = var_names.get_loc(g)
        col = X[:, idx]
        vecs.append(safe_toarray(col).ravel())
    return np.mean(np.stack(vecs, axis=1), axis=1)


def get_gene_vector(adata, gene, X=None, var_names=None):
    if var_names is None or X is None:
        X, var_names = get_expression_matrix_raw(adata)
    if gene in var_names:
        idx = var_names.get_loc(gene)
        return safe_toarray(X[:, idx]).ravel()
    logging.warning(f"  Gen '{gene}' no encontrado → zeros")
    return np.zeros(adata.n_obs, dtype=float)


# ── ULM manual (copiado del original) ────────────────────────────────────────

def _ulm_manual(expr_df, net):
    targets = net["target"].values
    weights = net["weight"].values.astype(float)
    available_mask = np.isin(targets, expr_df.columns)
    targets = targets[available_mask]
    weights = weights[available_mask]
    if len(targets) < 5:
        logging.error("  ULM manual: <5 targets → NaN")
        return pd.Series(np.nan, index=expr_df.index)
    X = expr_df[targets].values.astype(float)
    w = weights / (np.sqrt(np.sum(weights ** 2)) + 1e-9)
    scores = X @ w
    scores = (scores - scores.mean()) / (scores.std() + 1e-9)
    return pd.Series(scores, index=expr_df.index)


def compute_myc_tf_activity_clean(adata, net):
    """Calcula MYC TF activity con el regulón limpio."""
    X_raw, var_names = get_expression_matrix_raw(adata)
    available_targets = [t for t in net["target"].values if t in var_names]
    logging.info(f"  Targets MYC disponibles para ULM (limpio): {len(available_targets)}")

    if len(available_targets) < 10:
        raise ValueError(
            f"Solo {len(available_targets)} targets disponibles. Mínimo: 10."
        )

    # Intentar decoupler primero
    try:
        import decoupler as dc
        logging.info("  Intentando dc.run_ulm() con regulón limpio...")
        target_genes = [t for t in available_targets]
        gene_idx = [list(var_names).index(g) for g in target_genes]
        X_sub = X_raw[:, gene_idx]
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
        X_sub = X_sub.astype(np.float32)
        adata_sub = sc.AnnData(X=X_sub, obs=adata.obs[[PHENOTYPE_COL]].copy())
        adata_sub.var_names = pd.Index(target_genes)
        dc.run_ulm(
            mat=adata_sub, net=net,
            source="source", target="target", weight="weight",
            verbose=False, use_raw=False, min_n=5,
        )
        if "ulm_estimate" in adata_sub.obsm:
            acts_df = adata_sub.obsm["ulm_estimate"]
            if "MYC" in acts_df.columns:
                scores = acts_df["MYC"].values.astype(float)
                logging.info(
                    f"  dc.run_ulm() OK (limpio): mean={scores.mean():.4f}, "
                    f"std={scores.std():.4f}"
                )
                return scores
    except Exception as e:
        logging.warning(f"  dc.run_ulm() falló ({e}) → ULM manual")

    # Fallback ULM manual
    logging.info("  Ejecutando ULM manual (regulón limpio)...")
    target_genes = [t for t in net["target"].values if t in var_names]
    gene_idx = [list(var_names).index(g) for g in target_genes]
    X_sub = safe_toarray(X_raw[:, gene_idx])
    expr_df = pd.DataFrame(X_sub, columns=target_genes, index=adata.obs_names)
    net_filtered = net[net["target"].isin(target_genes)].copy()
    scores_series = _ulm_manual(expr_df, net_filtered)
    scores = scores_series.values.astype(float)
    logging.info(
        f"  ULM manual OK (limpio): mean={scores.mean():.4f}, std={scores.std():.4f}"
    )
    return scores


# ── Análisis A ────────────────────────────────────────────────────────────────

def analysis_A(adata, results, logger):
    scores = adata.obs["MYC_TF_activity_clean"].values
    phenos = adata.obs[PHENOTYPE_COL].values
    groups = {
        DESERT_LABEL:   scores[phenos == DESERT_LABEL],
        EXCLUDED_LABEL: scores[phenos == EXCLUDED_LABEL],
        INFLAMED_LABEL: scores[phenos == INFLAMED_LABEL],
    }
    for gname, gvals in groups.items():
        logger.info(
            f"  {gname}: n={len(gvals):,}, mean={gvals.mean():.4f}, "
            f"median={np.median(gvals):.4f}, std={gvals.std():.4f}"
        )

    comparisons = [
        (DESERT_LABEL,   INFLAMED_LABEL,  "MYC_TF_Desert_vs_Inflamed"),
        (EXCLUDED_LABEL, INFLAMED_LABEL,  "MYC_TF_Excluded_vs_Inflamed"),
        (DESERT_LABEL,   EXCLUDED_LABEL,  "MYC_TF_Desert_vs_Excluded"),
    ]
    pvals = []
    for g1_name, g2_name, test_id in comparisons:
        g1, g2 = groups[g1_name], groups[g2_name]
        stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
        d = cohens_d_pooled(g1, g2)
        pvals.append(pval)
        results.append({
            "analysis": "A", "test_id": test_id,
            "cohens_d": float(d), "p_value": float(pval),
            "n1": len(g1), "n2": len(g2),
            "group1": g1_name, "group2": g2_name,
            "regulon": "CLEAN (solo positivos)",
        })
        logger.info(f"  {test_id}: d={d:.4f}, p={pval:.3e}")

    qvals = fdr_correct(pvals)
    a_results = [r for r in results if r["analysis"] == "A"]
    for i, r in enumerate(a_results):
        r["q_value"]         = float(qvals[i])
        r["fdr_significant"] = bool(qvals[i] < 0.05)
        logger.info(f"  {r['test_id']}: q={qvals[i]:.3e}, sig={r['fdr_significant']}")

    return {p: float(v.mean()) for p, v in groups.items()}


# ── Análisis B (limpio) ───────────────────────────────────────────────────────

def analysis_B(adata, results, logger, net_df, X_raw, var_names):
    """Correlaciones con auditoría explícita de circularidad."""
    logger.info("=" * 70)
    logger.info("ANÁLISIS B (CLEAN): Correlaciones MYC_TF vs readouts")

    neg_targets = set(net_df.loc[net_df["weight"] < 0, "target"].values)
    logger.info(
        f"  Regulón limpio: {len(net_df)} genes | negativos: {len(neg_targets)}"
    )

    desert_mask = adata.obs[PHENOTYPE_COL] == DESERT_LABEL
    tf_vals = adata.obs.loc[desert_mask, "MYC_TF_activity_clean"].values
    logger.info(f"  Spots Desert: {desert_mask.sum():,}")

    score_defs = [
        ("ISG_score",
         ["IFIT1","IFIT2","IFIT3","ISG15","MX1","MX2","OAS1","OAS2","RSAD2","IFI44L"],
         "ISG score"),
        ("MHC_I_score",
         ["HLA-A","HLA-B","HLA-C","B2M","TAP1","TAP2","TAPBP"],
         "MHC-I score"),
        ("Chemokine_score",
         ["CCL5","CXCL9","CXCL10"],
         "Chemokine score"),
        ("CD47_expr",
         ["CD47"],
         "CD47 expression"),
    ]

    pvals = []
    for score_col, genes, description in score_defs:
        overlap = set(genes) & neg_targets
        pct_overlap = 100 * len(overlap) / max(len(genes), 1)
        is_circular = pct_overlap > 10.0

        if is_circular:
            logger.warning(
                f"  [{score_col}] CIRCULAR: {len(overlap)}/{len(genes)} genes "
                f"en peso negativo → {sorted(overlap)}"
            )
            results.append({
                "analysis": "B", "test_id": f"MYC_TF_vs_{score_col}_Desert",
                "statistic": np.nan, "p_value": np.nan, "q_value": np.nan,
                "fdr_significant": False, "circular": True,
                "circular_genes": sorted(overlap),
                "pct_overlap": round(pct_overlap, 1),
                "regulon": "CLEAN",
            })
            pvals.append(np.nan)
            continue

        if score_col not in adata.obs.columns:
            vals = compute_gene_score(adata, genes, score_col, X=X_raw, var_names=var_names)
            adata.obs[score_col] = vals

        score_desert = adata.obs.loc[desert_mask, score_col].values
        valid = ~(np.isnan(tf_vals) | np.isnan(score_desert))
        if valid.sum() < 30:
            pvals.append(np.nan)
            continue

        rho, pval = spearmanr(tf_vals[valid], score_desert[valid])
        pvals.append(pval)
        results.append({
            "analysis": "B",
            "test_id": f"MYC_TF_vs_{score_col}_Desert",
            "statistic": float(rho), "p_value": float(pval),
            "n1": int(valid.sum()), "circular": False,
            "pct_overlap": round(pct_overlap, 1),
            "regulon": "CLEAN",
        })
        logger.info(
            f"  MYC_TF vs {score_col}: ρ={rho:.4f}, p={pval:.3e} "
            f"[NO circular, overlap={pct_overlap:.0f}%]"
        )

    qvals = fdr_correct(pvals)
    b_results = [r for r in results if r.get("analysis") == "B"]
    for i, r in enumerate(b_results):
        if not r.get("circular", False) and not np.isnan(qvals[i]):
            r["q_value"] = float(qvals[i])
            r["fdr_significant"] = bool(qvals[i] < 0.05)


# ── Análisis C (split intra-Desert) ──────────────────────────────────────────

def analysis_C(adata, results, logger):
    """Binary intra-Desert usando el score limpio."""
    logger.info("=" * 70)
    logger.info("ANÁLISIS C (CLEAN): Binary intra-Desert (MYC_TF high vs low)")

    desert_mask = adata.obs[PHENOTYPE_COL] == DESERT_LABEL
    tf_desert   = adata.obs.loc[desert_mask, "MYC_TF_activity_clean"].values
    threshold   = np.median(tf_desert)
    high_mask   = tf_desert >= threshold
    low_mask    = ~high_mask
    logger.info(f"  Umbral (mediana): {threshold:.4f}")
    logger.info(f"  high: {high_mask.sum():,} | low: {low_mask.sum():,}")

    outcomes = [
        ("ISG_score",    "ISG score"),
        ("MHC_I_score",  "MHC-I score"),
        ("Chemokine_score", "Chemokine score"),
        ("CD47_expr",    "CD47 expression"),
    ]

    # Añadir CD8 abundance si disponible
    c2l_df = adata.obsm.get("means_cell_abundance_w_sf", None)
    if c2l_df is not None and hasattr(c2l_df, "columns"):
        cd8_candidates = [c for c in c2l_df.columns if "CD8" in c]
        if cd8_candidates:
            adata.obs["CD8_T_abundance"] = c2l_df[cd8_candidates[0]].values
            outcomes.append(("CD8_T_abundance", "CD8+ T cell abundance"))

    pvals = []
    for score_col, description in outcomes:
        if score_col not in adata.obs.columns:
            continue
        vals = adata.obs.loc[desert_mask, score_col].values
        g_high = vals[high_mask][~np.isnan(vals[high_mask])]
        g_low  = vals[low_mask][~np.isnan(vals[low_mask])]
        if len(g_high) < 10 or len(g_low) < 10:
            continue
        _, pval = mannwhitneyu(g_high, g_low, alternative="two-sided")
        d = cohens_d_pooled(g_high, g_low)
        pvals.append(pval)
        results.append({
            "analysis": "C", "test_id": f"MYC_TF_BinaryDesert_{score_col}",
            "cohens_d": float(d), "p_value": float(pval),
            "n1": len(g_high), "n2": len(g_low),
            "regulon": "CLEAN",
        })
        logger.info(f"  {score_col}: d={d:.4f}, p={pval:.3e}")

    qvals = fdr_correct(pvals)
    c_results = [r for r in results if r.get("analysis") == "C"]
    for i, r in enumerate(c_results[: len(qvals)]):
        r["q_value"] = float(qvals[i])
        r["fdr_significant"] = bool(qvals[i] < 0.05)


# ── Análisis D (sanity) ───────────────────────────────────────────────────────

def analysis_D(adata, results, logger, X_raw, var_names):
    logger.info("=" * 70)
    logger.info("ANÁLISIS D (CLEAN): Sanity check TF activity vs MYC mRNA")
    tf_vals  = adata.obs["MYC_TF_activity_clean"].values
    myc_mrna = get_gene_vector(adata, "MYC", X=X_raw, var_names=var_names)
    valid    = ~(np.isnan(tf_vals) | np.isnan(myc_mrna))
    rho, pval = spearmanr(tf_vals[valid], myc_mrna[valid])
    level = ("OK" if 0.05 <= rho <= 0.85
              else "WARNING_LOW" if rho < 0.05 else "WARNING_HIGH")
    logger.info(f"  ρ(TF_activity_clean, MYC_mRNA) = {rho:.4f}, p={pval:.3e} [{level}]")
    if rho < 0.05:
        logger.warning(
            "  ρ MUY BAJO. Con regulón solo-positivos (proliferación pura)\n"
            "  puede ser bajo si MYC mRNA no correlaciona con la actividad\n"
            "  de sus genes target de proliferación en este dataset."
        )
    results.append({
        "analysis": "D", "test_id": "MYC_TF_clean_vs_MYC_mRNA",
        "statistic": float(rho), "p_value": float(pval),
        "q_value": float(pval), "fdr_significant": pval < 0.05,
        "n1": int(valid.sum()), "interpretation": level,
        "regulon": "CLEAN",
    })


# ── Figura violín ─────────────────────────────────────────────────────────────

def plot_comparison(adata, means_orig, means_clean, out_dir):
    """
    Genera figura comparativa: regulón original (contaminado) vs limpio.
    Visualiza directamente el artefacto de circularidad.
    """
    import matplotlib.pyplot as plt

    phenotypes = [INFLAMED_LABEL, EXCLUDED_LABEL, DESERT_LABEL]
    labels     = ["Inflamed", "Excluded", "Desert"]
    colors_orig  = ["#E74C3C", "#E67E22", "#3498DB"]
    colors_clean = ["#C0392B", "#D35400", "#1A5276"]

    x = np.arange(len(phenotypes))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "MYC TF Activity: Regulón original (contaminado) vs Limpio (solo positivos)\n"
        "Comparación directa del artefacto de circularidad",
        fontsize=12, fontweight="bold"
    )

    # Panel A: medias
    ax = axes[0]
    orig_vals  = [means_orig.get(p, np.nan) for p in phenotypes]
    clean_vals = [means_clean.get(p, np.nan) for p in phenotypes]
    bars1 = ax.bar(x - width/2, orig_vals,  width, label="Original (backup+neg)",
                   color=["#E74C3C", "#E67E22", "#3498DB"], alpha=0.7,
                   edgecolor="black", linewidth=0.8)
    bars2 = ax.bar(x + width/2, clean_vals, width, label="Limpio (solo positivos)",
                   color=["#C0392B", "#D35400", "#1A5276"], alpha=0.9,
                   edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Mean MYC TF Activity (z-score)", fontsize=10)
    ax.set_title("Media por fenotipo", fontsize=11)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=9)
    # Anotar Cohen's d sobre las barras
    for xi, (o, c) in enumerate(zip(orig_vals, clean_vals)):
        if not np.isnan(o) and not np.isnan(c):
            ax.text(xi, max(o, c) + 0.02, f"↓{abs(o-c):.2f}", ha="center",
                    fontsize=8, color="gray")

    # Panel B: violines
    ax2 = axes[1]
    phenotypes_subset = [INFLAMED_LABEL, DESERT_LABEL]
    for i, (pheno, color) in enumerate(zip(phenotypes_subset, ["#E74C3C", "#3498DB"])):
        mask = adata.obs[PHENOTYPE_COL] == pheno

        # Submuestra eficiente
        idx_pheno = np.where(mask)[0]
        np.random.seed(SEED)
        idx_sample = np.random.choice(
            idx_pheno, min(3000, len(idx_pheno)), replace=False
        )

        orig_vals_v  = adata.obs.iloc[idx_sample]["MYC_TF_activity"].values
        clean_vals_v = adata.obs.iloc[idx_sample]["MYC_TF_activity_clean"].values

        ax2.violinplot(
            [orig_vals_v, clean_vals_v],
            positions=[i * 3, i * 3 + 1],
            showmedians=True, widths=0.8
        )

    ax2.set_xticks([0, 1, 3, 4])
    ax2.set_xticklabels(["Infl\nOrig", "Infl\nClean", "Des\nOrig", "Des\nClean"],
                         fontsize=8)
    ax2.set_ylabel("MYC TF Activity (z-score)", fontsize=10)
    ax2.set_title("Distribución (Inflamed vs Desert)", fontsize=11)
    ax2.axhline(0, color="gray", linewidth=0.6, linestyle="--")

    plt.tight_layout()
    fpath = out_dir / "fig_myc_tf_orig_vs_clean_comparison.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Figura: {fpath}")


# ── Resumen de diferencia entre versiones ─────────────────────────────────────

def print_comparison_summary(results_orig_path, results_clean_df, logger):
    """
    Compara resultados orig vs clean y cuantifica el artefacto.
    """
    logger.info("\n" + "=" * 70)
    logger.info("COMPARACIÓN: Original (contaminado) vs Limpio")
    logger.info("=" * 70)

    # Cargar resultados originales si existen
    orig_csv = results_orig_path
    if orig_csv.exists():
        df_orig = pd.read_csv(orig_csv)
    else:
        logger.warning(f"  Resultados originales no encontrados: {orig_csv}")
        return

    key_tests = [
        "MYC_TF_Desert_vs_Inflamed",
        "MYC_TF_vs_ISG_score_Desert",
        "MYC_TF_vs_MHC_I_score_Desert",
        "MYC_TF_vs_Chemokine_score_Desert",
    ]

    logger.info(f"\n  {'Test':<40} {'Original':>12} {'Limpio':>12} {'Diferencia':>12}")
    logger.info("  " + "-" * 78)

    for test in key_tests:
        row_orig  = df_orig[df_orig["test_id"] == test]
        row_clean_list = [
            r for r in results_clean_df
            if r.get("test_id") == test
        ]

        orig_val = (
            row_orig["cohens_d"].iloc[0]
            if not row_orig.empty and "cohens_d" in row_orig.columns
            else row_orig["statistic"].iloc[0] if not row_orig.empty else np.nan
        )
        clean_val = (
            row_clean_list[0].get("cohens_d",
            row_clean_list[0].get("statistic", np.nan))
            if row_clean_list else np.nan
        )

        diff = clean_val - orig_val if not (np.isnan(clean_val) or np.isnan(orig_val)) else np.nan
        diff_str = f"{diff:+.4f}" if not np.isnan(diff) else "N/A (circular)"

        orig_str  = f"{orig_val:.4f}"  if not np.isnan(orig_val)  else "N/A"
        clean_str = f"{clean_val:.4f}" if not np.isnan(clean_val) else "circular"

        logger.info(f"  {test:<40} {orig_str:>12} {clean_str:>12} {diff_str:>12}")

    logger.info("\n  INTERPRETACIÓN:")
    logger.info(
        "  Si Limpio ≈ 0 donde Original >> 0 para Desert_vs_Inflamed:\n"
        "  → El d=+0.622 original era ENTERAMENTE artefacto de circularidad.\n"
        "  → La hipótesis MYC→Desert no tiene soporte en actividad TF limpia.\n"
        "\n"
        "  Si Limpio es significativo (d > 0.3) con regulón solo-positivos:\n"
        "  → Evidencia genuina. Reportar con nota sobre el fix aplicado."
    )


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(RESULTS_DIR / "myc_tf_activity_clean.log", mode="w"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger("myc_tf_clean")

    logger.info("=" * 70)
    logger.info("MYC TF ACTIVITY SCORING — REGULÓN LIMPIO (solo positivos)")
    logger.info("Fix: eliminados 31 genes neg (ISG/MHC-I/STING) del regulón backup")
    logger.info(f"Output: {RESULTS_DIR}")
    logger.info(f"Input : {ADATA_INPUT}")
    logger.info("=" * 70)

    # ── Cargar datos ─────────────────────────────────────────────────────────
    if not ADATA_INPUT.exists():
        fallback = DATA_PROCESSED / "adata_with_phenotypes.h5ad"
        if fallback.exists():
            logger.warning(f"  Usando fallback: {fallback.name}")
            adata = sc.read_h5ad(fallback)
        else:
            logger.error("FATAL: no se encontró adata procesado")
            sys.exit(1)
    else:
        logger.info(f"  Cargando {ADATA_INPUT.name}...")
        adata = sc.read_h5ad(ADATA_INPUT)

    logger.info(f"  Spots: {adata.n_obs:,} | Genes .X: {adata.n_vars:,}")
    if adata.raw is not None:
        logger.info(f"  Genes en .raw: {adata.raw.n_vars:,}")

    dist = adata.obs[PHENOTYPE_COL].value_counts()
    logger.info(f"  Fenotipos:\n{dist.to_string()}")

    # ── Expresión desde .raw ─────────────────────────────────────────────────
    X_raw, var_names = get_expression_matrix_raw(adata)

    # ── Construir regulón LIMPIO ──────────────────────────────────────────────
    logger.info("\n  Construyendo regulón MYC LIMPIO...")

    # Intentar CollecTRI real primero
    try:
        import decoupler as dc
        _DECOUPLER_AVAILABLE = True
        logger.info(f"  decoupler disponible: v{dc.__version__}")
    except ImportError:
        dc = None
        _DECOUPLER_AVAILABLE = False
        logger.warning("  decoupler no disponible → regulón limpio backup")

    net = get_myc_regulon_CLEAN(var_names, _DECOUPLER_AVAILABLE, dc)
    logger.info(f"  Regulón final: {len(net)} genes")

    # Auditoría final del regulón usado
    audit = audit_regulon_contamination(net)
    logger.info(
        f"\n  AUDITORÍA REGULÓN LIMPIO:\n"
        f"    Genes totales: {audit['n_total']}\n"
        f"    Positivos: {audit['n_positive']} | Negativos: {audit['n_negative']}\n"
        f"    Overlap neg-readout: {audit['n_overlap_neg']} genes\n"
        f"    Veredicto: {audit['verdict']}"
    )
    if audit["is_circular"]:
        logger.error(
            "FATAL: regulón limpio tiene contaminación residual. "
            "Revisar fix_myc_tf_clean_regulon.py"
        )
        sys.exit(1)

    # ── Calcular MYC TF activity LIMPIO ──────────────────────────────────────
    logger.info("\n  Calculando MYC TF activity (regulón limpio)...")
    t_ulm = time.time()
    myc_tf_clean = compute_myc_tf_activity_clean(adata, net)
    logger.info(f"  ULM completado en {time.time()-t_ulm:.1f}s")
    logger.info(
        f"  MYC_TF_activity_clean: mean={myc_tf_clean.mean():.4f}, "
        f"std={myc_tf_clean.std():.4f}, "
        f"range=[{myc_tf_clean.min():.3f}, {myc_tf_clean.max():.3f}]"
    )

    # Guardar ambas versiones para comparación directa
    adata.obs["MYC_TF_activity_clean"] = myc_tf_clean

    # Si la versión original (contaminada) está en adata.obs, conservarla
    if "MYC_TF_activity" not in adata.obs.columns:
        logger.warning(
            "  MYC_TF_activity (original) no está en adata.obs — "
            "ejecutar myc_tf_activity_decoupler.py primero para comparación"
        )
    else:
        # Diferencia entre versiones en Desert vs Inflamed
        phenos = adata.obs[PHENOTYPE_COL].values
        d_mask = phenos == DESERT_LABEL
        i_mask = phenos == INFLAMED_LABEL
        d_orig_mean  = adata.obs.loc[d_mask, "MYC_TF_activity"].mean()
        i_orig_mean  = adata.obs.loc[i_mask, "MYC_TF_activity"].mean()
        d_clean_mean = myc_tf_clean[d_mask].mean()
        i_clean_mean = myc_tf_clean[i_mask].mean()
        logger.info(
            f"\n  Comparación Desert vs Inflamed:\n"
            f"    Original  (contaminado): Desert={d_orig_mean:.4f}, "
            f"Inflamed={i_orig_mean:.4f}\n"
            f"    Limpio    (solo posit.): Desert={d_clean_mean:.4f}, "
            f"Inflamed={i_clean_mean:.4f}"
        )

    # ── Pre-calcular scores funcionales ──────────────────────────────────────
    logger.info("\n  Pre-calculando scores funcionales...")
    for score_col, genes in [
        ("ISG_score",      ["IFIT1","IFIT2","IFIT3","ISG15","MX1","MX2",
                             "OAS1","OAS2","RSAD2","IFI44L"]),
        ("MHC_I_score",    ["HLA-A","HLA-B","HLA-C","B2M","TAP1","TAP2","TAPBP"]),
        ("Chemokine_score", ["CCL5","CXCL9","CXCL10"]),
    ]:
        if score_col not in adata.obs.columns:
            adata.obs[score_col] = compute_gene_score(
                adata, genes, score_col, X=X_raw, var_names=var_names
            )
    adata.obs["CD47_expr"] = get_gene_vector(adata, "CD47", X=X_raw, var_names=var_names)

    # ── Análisis ─────────────────────────────────────────────────────────────
    all_results = []

    logger.info("\n" + "=" * 70)
    logger.info("ANÁLISIS A: distribución por fenotipo")
    means_clean = analysis_A(adata, all_results, logger)

    analysis_B(adata, all_results, logger, net, X_raw, var_names)
    analysis_C(adata, all_results, logger)
    analysis_D(adata, all_results, logger, X_raw, var_names)

    # ── Guardar resultados ────────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    out_csv = RESULTS_DIR / "myc_tf_activity_results_clean.csv"
    df_results.to_csv(out_csv, index=False)
    logger.info(f"\n  Resultados: {out_csv}")
    logger.info(f"  Total tests: {len(df_results)}")

    n_sig      = df_results["fdr_significant"].sum() if "fdr_significant" in df_results.columns else 0
    n_circular = df_results["circular"].sum() if "circular" in df_results.columns else 0
    logger.info(f"  FDR-sig (no circulares): {n_sig - n_circular}/{len(df_results)}")
    logger.info(f"  Tests marcados CIRCULAR: {n_circular}")

    # Tabla de medias
    df_means = pd.DataFrame([
        {"phenotype": p, "mean_MYC_TF_activity_clean": m}
        for p, m in means_clean.items()
    ])
    df_means.to_csv(RESULTS_DIR / "myc_tf_activity_by_phenotype_clean.csv", index=False)

    # ── Figura comparativa ────────────────────────────────────────────────────
    try:
        means_orig = {}
        orig_csv = RESULTS_DIR / "myc_tf_activity_by_phenotype.csv"
        if orig_csv.exists():
            df_orig_means = pd.read_csv(orig_csv)
            means_orig = dict(zip(
                df_orig_means["phenotype"],
                df_orig_means["mean_MYC_TF_activity"]
            ))
        if means_orig and "MYC_TF_activity" in adata.obs.columns:
            plot_comparison(adata, means_orig, means_clean, RESULTS_DIR)
    except Exception as e:
        logger.warning(f"  Figura comparativa: {e}")

    # ── Comparación resumen ───────────────────────────────────────────────────
    print_comparison_summary(
        RESULTS_DIR / "myc_tf_activity_results.csv",
        all_results,
        logger
    )

    # ── Guardar adata ─────────────────────────────────────────────────────────
    logger.info(f"\n  Guardando adata → {ADATA_OUTPUT}")
    try:
        adata.write_h5ad(ADATA_OUTPUT)
        logger.info("  adata guardado correctamente")
    except Exception as e:
        logger.warning(f"  No se pudo guardar: {e}")

    # ── Resumen ejecutivo ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN EJECUTIVO — MYC TF ACTIVITY (REGULÓN LIMPIO)")
    logger.info("=" * 70)

    a_tests = [r for r in all_results if r.get("analysis") == "A"]
    for r in a_tests:
        logger.info(
            f"  {r['test_id']}: d={r.get('cohens_d', r.get('statistic', 'N/A')):.4f}, "
            f"q={r.get('q_value', float('nan')):.3e}, sig={r['fdr_significant']}"
        )

    b_tests = [r for r in all_results if r.get("analysis") == "B"]
    for r in b_tests:
        if r.get("circular"):
            logger.info(f"  {r['test_id']}: CIRCULAR (omitido)")
        else:
            logger.info(
                f"  {r['test_id']}: ρ={r.get('statistic', float('nan')):.4f}, "
                f"q={r.get('q_value', float('nan')):.3e}, sig={r.get('fdr_significant', False)}"
            )

    logger.info(f"\n  Tiempo total: {time.time()-t0:.1f}s")
    logger.info("  STATUS: COMPLETADO")
    logger.info(
        "\n  CÓMO LEER ESTOS RESULTADOS:\n"
        "  Análisis A: mide actividad proliferación-MYC pura (sin genes inmunes).\n"
        "    d ≈ 0 → la hipótesis MYC no tiene soporte en actividad TF limpia.\n"
        "    d > 0.3, sig → evidencia genuina. Reportar con nota de metodología.\n"
        "\n"
        "  Análisis B: correlaciones ISG/MHC-I marcadas CIRCULAR si eran\n"
        "    circulares. Solo CD47 y Chemokine son potencialmente válidos aquí.\n"
        "\n"
        "  Comparar log_19_myc_tf.txt (original) vs este log para cuantificar\n"
        "  exactamente cuánto del d=+0.622 era artefacto."
    )


if __name__ == "__main__":
    main()

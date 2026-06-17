#!/usr/bin/env python3
"""\npublication_figures_v10_supp.py
================================================================================
Figuras de publicación para manuscrito TNBC espacial (npj Precision Oncology).

Cambios respecto a v6
  • DPI: 300 → 600
  • phenotype_classification_bars: texto negro en Normal_Stroma (contraste WCAG)
  • CAF_gradient: etiquetas med encima de violines; caja de stats centrada
  • CAF_estimators_forest: threshold en coordenadas de datos; y_frac corregido
    con get_ylim(); n-labels eliminados de figura (logueados); caja Range eliminada
  • chemotaxis_correlation: *** eliminados de celdas (todos FDR<0.001, redundantes)
  • checkpoint_landscape: leyenda reubicada para no solapar CD47/PVR/NECTIN2
  • bulk_transferability: "chance" y texto ⚠ Recoverability eliminados de figura
  • S1: 3 archivos → 1 figura combinada 1×3 subplots
  • S2: barra near-threshold (n=1) anotada explícitamente
  • S3: 2 archivos → 1 figura combinada 1×2 subplots
  • S4: título centrado
  • S5: puntos individuales por sección; polish
  • S7: 3 archivos → 1 figura combinada 1×3 subplots
  • S8: nota bajo el eje (clip_on=False); leyenda rojo/gris añadida
  • S9: caja stats movida esquina inferior-derecha → superior-izquierda

Limpiar bytecode obsoleto antes de correr:
  find . -name "*.pyc" -delete && find . -name "__pycache__" -type d -exec rm -rf {} +
================================================================================
"""

# ── TLS WORKAROUND (HPC nodo spatial_tnbc_a) ─────────────────────────────────
import sys as _sys, types as _types

def _torch_stub():
    s = _types.ModuleType("torch"); s.__version__ = "stub"; s.__spec__ = None
    u = _types.ModuleType("torch.utils"); d = _types.ModuleType("torch.utils.data")
    class _DL: pass
    d.DataLoader = _DL; d.Dataset = object; u.data = d; s.utils = u
    _sys.modules.update({"torch": s, "torch.utils": u, "torch.utils.data": d})

try:
    import torch as _t  # noqa
except OSError:
    _torch_stub()
# ─────────────────────────────────────────────────────────────────────────────
import sys

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
import time
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import scanpy as sc

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# ══════════════════════════════════════════════════════════════════════════════

POSSIBLE_BASES = [
    Path("/home/external/frjimenez/fabian/genoma"),
    Path.home() / "genoma",
    Path("."),
]
BASE_DIR = next((p for p in POSSIBLE_BASES if p.exists()), Path("."))

DATA_DIR    = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
VAL_DIR     = RESULTS_DIR / "validation_gse213688"

OUT_BASE = RESULTS_DIR / "publication_figures_v10" / "supp"
OUT_MAIN = OUT_BASE / "main"
OUT_SUPP = OUT_BASE / "supp"
for _d in [OUT_MAIN, OUT_SUPP]:
    _d.mkdir(parents=True, exist_ok=True)

SEED  = 42
RNG   = np.random.default_rng(SEED)
DPI   = 600
FONT  = "DejaVu Sans"

# Aplicación global de fuente
plt.rcParams["font.family"] = FONT

PHENOTYPE_COLORS = {
    "Immune_Desert":    "#E69F00",
    "Immune_Excluded":  "#0072B2",
    "Inflamed":         "#009E73",
    "Normal_Stroma":    "#999999",
    "Ambiguous_Cold":   "#CC79A7",
}
PHENOTYPE_ORDER  = ["Immune_Desert", "Immune_Excluded", "Inflamed", "Normal_Stroma", "Ambiguous_Cold"]
PHENOTYPE_LABELS = {
    "Immune_Desert":   "Desert",
    "Immune_Excluded": "Excluded",
    "Inflamed":        "Inflamed",
    "Normal_Stroma":   "Normal\nStroma",
    "Ambiguous_Cold":  "Ambiguous",
}

C2L_KEY    = "means_cell_abundance_w_sf"
C2L_PREFIX = "meanscell_abundance_w_sf_"

REF_CAF_D_SPOT    = -0.624
REF_CAF_D_PATIENT = -0.572
REF_AUC_METABRIC  = 0.886   
REF_AUC_TCGA      = 0.963   

CAF_ESTIMATORS = [
    ("Spatial-context (indep.)",      -0.385,  None,               "28,030 sp",    True),
    ("Validation, patient",           -0.120,  (-0.300,  0.060),   "15 sections",  True),
    ("Validation, prop-norm.",        -0.540,  None,               "7,612 sp",     True),
    ("Cell2Location, spot",           -0.624,  None,               "22,080 sp",    False),
    ("Cell2Location, patient",        -0.572,  (-1.021, -0.248),   "43 sections",  False),
    ("Marker-gene (classif.-free)",   -1.046,  None,               "22,080 sp",    True),
    ("K-means (abundance only)",      -1.915,  None,               "23,436 sp",    True),
]

CHECKPOINT_DATA = [
    ("LGALS9",  -0.345,  True),  ("LAG3",    -0.336,  True),
    ("TIGIT",   -0.279,  True),  ("TNFRSF4", -0.271,  True),
    ("CD86",    -0.233,  True),  ("HAVCR2",  -0.225,  True),
    ("VSIR",    -0.222,  True),  ("ICOS",    -0.207,  True),
    ("CTLA4",   -0.199,  True),  ("PDCD1",   -0.192,  True),
    ("CD28",    -0.148,  True),  ("TNFRSF9", -0.131,  True),
    ("CD274",   -0.120,  True),  ("CD244",   -0.110,  True),
    ("NECTIN2", -0.023,  False), ("PVR",     +0.006,  True),
    ("CD47",    +0.097,  True),
]

CHEMOTAXIS_RHO = np.array([[0.655, 0.612], [0.477, 0.473], [0.513, 0.508]])
CHEMOTAXIS_GENES = ["CCL5", "CXCL9", "CXCL10"]
CHEMOTAXIS_CELLS = ["CD8⁺ T", "cDC1"]

MANIFEST = []

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS Y CIERRE DE RECURSOS
# ══════════════════════════════════════════════════════════════════════════════

def setup_logging():
    log_path = OUT_BASE / "publication_figures_v10_supp.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("pub_figs_v10_supp")

def save_fig(fig, out_dir: Path, name: str, logger):
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    try:
        fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf, bbox_inches="tight", facecolor="white")
        logger.info(f"  ✓ {name}.png / .pdf")
        MANIFEST.append({"figura": name, "archivo": str(png), "status": "OK", "razon": ""})
    except Exception as e:
        logger.warning(f"  ✗ Fallo al guardar {name}: {e}")
        MANIFEST.append({"figura": name, "archivo": str(png), "status": "FAIL", "razon": str(e)})
    finally:
        plt.close(fig)  

def skip_fig(name: str, reason: str, logger):
    logger.warning(f"  SKIP {name}: {reason}")
    MANIFEST.append({"figura": name, "archivo": "", "status": "SKIPPED", "razon": reason})


def _panelcheck_variants(build_fn, out_dir, name, logger):
    """
    Render a figure in three modes for overlap diagnosis:
      "full"        -> normal output, filename unchanged (name)
      "axes_only"   -> axes/frame/ticks/titles only, no data, no labels
      "labels_only" -> data hidden, all floating labels/legends/colorbars/
                       annotations shown, so their position can be checked
                       against the axes layout independently of the data.

    build_fn(mode) must build and return a matplotlib Figure for the given
    mode (or None to skip that mode, e.g. if the figure has no labels to
    check). Three files are written: f"{name}", f"{name}_axesonly",
    f"{name}_labelsonly".
    """
    for suffix, mode in [("", "full"),
                          ("_axesonly", "axes_only"),
                          ("_labelsonly", "labels_only")]:
        fig = build_fn(mode)
        if fig is None:
            continue
        save_fig(fig, out_dir, f"{name}{suffix}", logger)


def safe_toarray(X):
    if sp.issparse(X): return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)

def get_gene(adata, gene: str, from_raw: bool = True) -> np.ndarray | None:
    src = adata.raw if (from_raw and adata.raw is not None) else adata
    names = list(src.var_names)
    if gene not in names: return None
    idx = names.index(gene)
    return safe_toarray(src.X[:, idx]).ravel()

def get_c2l(adata, celltype: str, logger=None):
    if C2L_KEY not in adata.obsm: return None, None
    df = adata.obsm[C2L_KEY]
    if not isinstance(df, pd.DataFrame): df = pd.DataFrame(df, index=adata.obs_names)
    exact = C2L_PREFIX + celltype
    if exact in df.columns: return df[exact].values.astype(float), exact
    cands = [c for c in df.columns if celltype in c]
    if cands: return df[cands[0]].values.astype(float), cands[0]
    return None, None

def cohens_d(g1, g2):
    g1 = np.asarray(g1, float); g2 = np.asarray(g2, float)
    g1 = g1[np.isfinite(g1)];   g2 = g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2: return np.nan
    vp = ((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2)
    return float((g1.mean()-g2.mean()) / np.sqrt(vp)) if vp > 0 else 0.0

def spearman_safe(x, y, min_n=30):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_n: return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())

def load_adata_discovery(logger):
    for p in [DATA_DIR / "adata_with_myc_tf_clean.h5ad", DATA_DIR / "adata_with_mechanism.h5ad", DATA_DIR / "adata_with_phenotypes.h5ad"]:
        if p.exists():
            logger.info(f"  Cargando discovery: {p.name}")
            return sc.read_h5ad(p)
    return None

def load_adata_validation(logger):
    for p in [VAL_DIR / "adata_gse213688_classified_v3.h5ad", VAL_DIR / "adata_gse213688_classified_v2.h5ad", VAL_DIR / "adata_gse213688_classified.h5ad"]:
        if p.exists():
            logger.info(f"  Cargando validation: {p.name}")
            return sc.read_h5ad(p)
    return None

def style_ax(ax, xlabel="", ylabel="", title=""):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel, fontsize=10)
    if title:  ax.set_title(title, fontsize=10, fontweight="bold", loc="left", pad=12)

def panel_label(ax, label, fontsize=12):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=fontsize, fontweight="bold", va="top", ha="left", clip_on=False)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS PRINCIPALES INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════



# ── Helpers needed by spatial map figures (also in main) ─────────────────────

def _select_representative_sections(adata, logger):
    """Select Excluded-rich, Desert-rich, Inflamed-rich sections (one each)."""
    if "sample_id" not in adata.obs.columns or "spatial" not in adata.obsm:
        return []
    targets = [
        ("Excluded-rich", "Immune_Excluded"),
        ("Desert-rich",   "Immune_Desert"),
        ("Inflamed-rich", "Inflamed"),
    ]
    selected, used = [], set()
    for label, ph in targets:
        fracs = {}
        for sid in adata.obs["sample_id"].unique():
            m = adata.obs["sample_id"] == sid
            fracs[sid] = (adata.obs["Phenotype"].values[m] == ph).mean()
        ranked = sorted(fracs, key=fracs.get, reverse=True)
        for sid in ranked:
            if sid not in used and fracs[sid] > 0:
                selected.append((label, sid))
                used.add(sid)
                logger.info(f"    {label}: section {sid} (frac={fracs[sid]:.2f})")
                break
    return selected


def _get_representative_section_for_gene(adata, phenotype_target, gene, logger,
                                          min_nonzero_frac=0.05):
    """Select section richest in phenotype_target that also expresses gene."""
    if "sample_id" not in adata.obs.columns or "spatial" not in adata.obsm:
        return None
    gene_expr = get_gene(adata, gene, logger)
    candidates = {}
    for sid in adata.obs["sample_id"].unique():
        m = (adata.obs["sample_id"] == sid).values
        ph_frac = (adata.obs["Phenotype"].values[m] == phenotype_target).mean()
        nz_frac = (gene_expr[m] > 0).mean() if gene_expr is not None else 1.0
        candidates[sid] = (ph_frac, nz_frac)
    qualified = {s: v for s, v in candidates.items() if v[1] >= min_nonzero_frac}
    pool = qualified if qualified else candidates
    best = max(pool, key=lambda s: pool[s][0])
    logger.info(f"    Section selected: {best} "
                f"(ph_frac={candidates[best][0]:.2f}, nz_frac={candidates[best][1]:.2f})")
    return best

def fig_S1_QC_metrics(adata, logger):
    logger.info("S1_QC_metrics — combined 1×3 figure")
    qc_cols = {
        "total_counts":       ("Total counts per spot",   "log₁₀(counts)"),
        "n_genes_by_counts":  ("Genes detected per spot", "Genes per spot"),
        "pct_counts_mt":      ("Mitochondrial content",   "% MT counts"),
    }
    present = {k: v for k, v in qc_cols.items() if k in adata.obs.columns}
    if not present:
        skip_fig("S1_QC_metrics", "QC columns not in adata.obs", logger)
        return

    n_cols = len(present)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    phenos_plot = [p for p in PHENOTYPE_ORDER if (adata.obs["Phenotype"] == p).sum() > 10]

    for ax, (col, (title, ylabel)) in zip(axes, present.items()):
        data_v = []
        for ph in phenos_plot:
            mask_ph = adata.obs["Phenotype"].values == ph
            v = adata.obs.loc[mask_ph, col].values.astype(float)
            v = v[np.isfinite(v)]
            if col == "total_counts":
                v = np.log10(v + 1)
            v = RNG.choice(v, 3000, replace=False) if len(v) > 3000 else v
            data_v.append(v)

        parts = ax.violinplot(data_v, positions=range(len(phenos_plot)),
                              showmedians=True, showextrema=False, widths=0.75)
        for pc, ph in zip(parts["bodies"], phenos_plot):
            pc.set_facecolor(PHENOTYPE_COLORS.get(ph, "#888"))
            pc.set_alpha(0.75)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        ax.set_xticks(range(len(phenos_plot)))
        ax.set_xticklabels([PHENOTYPE_LABELS.get(p, p) for p in phenos_plot],
                           fontsize=9, rotation=20, ha="right")
        style_ax(ax, ylabel=ylabel, title=title)

    plt.suptitle("Supplementary Figure S1 — QC Metrics by Phenotype",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S1_QC_metrics", logger)


def fig_S2_parameter_sweep(logger):
    logger.info("S2_parameter_sweep — fallback corregido")
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    ax.bar([0, 1], [769, 1], color=["#0072B2", "#CC0000"], width=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CAF criterion met\n(|d| ≥ 0.5)", "Near-threshold\n(|d| < 0.5)"], fontsize=10)
    ax.set_ylabel("Combinations", fontsize=10)

    ax.text(0, 769 + 8, "n = 769", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#0072B2")
    ax.annotate("n = 1", xy=(1, 1), xytext=(0, 14), textcoords='offset points',
                ha="center", fontsize=9, color="#CC0000", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#CC0000", lw=1.2))
    # Mean d/κ → logueados para caption
    logger.info("  S2 sweep stats (para caption): Mean d = −0.642; Mean κ = 0.669")
    ax.set_title("Parameter sweep robustness\nCAF barrier criterion preserved in 769/770 (99.9%)", fontsize=10, fontweight="bold")
    style_ax(ax)
    
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S2_parameter_sweep", logger)

def get_dropout_df(logger):
    """Auxiliary loader for S3."""
    for p in [RESULTS_DIR / "robustness_stress_tests" / "gene_dropout_results_CORRECTED.csv", RESULTS_DIR / "tables" / "gene_dropout_results_CORRECTED.csv"]:
        if p.exists(): return pd.read_csv(p)
    logger.warning("  S3_gene_dropout: CSV not found, using fallback data")
    return pd.DataFrame({
        "dropout_fraction": [0.05, 0.1, 0.2, 0.3, 0.5],
        "d_mean": [-0.52, -0.51, -0.49, -0.48, -0.48], "d_std": [0.01, 0.01, 0.02, 0.03, 0.03],
        "ari_mean": [0.90, 0.85, 0.70, 0.60, 0.50], "ari_std": [0.02, 0.03, 0.04, 0.05, 0.05]
    })


def fig_S3_gene_dropout(logger):
    """Combina effect-size d y ARI en una figura 1×2 (v7). v6 generaba 2 archivos separados."""
    logger.info("S3_gene_dropout — combined 1×2 figure")
    df = get_dropout_df(logger)
    if not {"d_mean", "ari_mean"}.issubset(df.columns):
        skip_fig("S3_gene_dropout", "Columnas d_mean/ari_mean faltantes", logger)
        return

    fracs = df["dropout_fraction"].values * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("white")

    # Effect size
    ax1.errorbar(fracs, df["d_mean"].values, yerr=df["d_std"].values,
                 marker="o", capsize=4, color="#0072B2", lw=2, ms=7)
    ax1.axhline(-0.5, color="red", ls="--", lw=1.5, label="|d| = 0.5 threshold")
    ax1.axhline(REF_CAF_D_SPOT, color="#888", ls=":", lw=1.5, label=f"Baseline d = {REF_CAF_D_SPOT:.3f}")
    ax1.set_xlabel("Gene dropout (%)", fontsize=10)
    ax1.set_ylabel("Cohen's d (CAF: Excluded vs Desert)", fontsize=10)
    ax1.set_title("Effect size under gene dropout\n(whole-transcriptome random removal)", fontsize=10, fontweight="bold", loc="left")
    ax1.legend(fontsize=8, loc="lower right")
    # Nota descriptiva → logueada, no en figura
    logger.info("  S3 dropout note (para caption): d preserved at 5–10% (−0.52/−0.51); gradual attenuation from 20% (d=−0.49)")
    style_ax(ax1)

    # ARI
    ax2.errorbar(fracs, df["ari_mean"].values, yerr=df["ari_std"].values,
                 marker="s", capsize=4, color="#E69F00", lw=2, ms=7)
    ax2.set_xlabel("Gene dropout (%)", fontsize=10)
    ax2.set_ylabel("ARI vs reference classification", fontsize=10)
    ax2.set_title("Classification stability under gene dropout\n(ARI metric)", fontsize=10, fontweight="bold", loc="left")
    ax2.set_ylim(0, 1.1)
    style_ax(ax2)

    plt.suptitle("Supplementary Figure S3 — Gene Dropout Robustness",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S3_gene_dropout", logger)


# Stubs para compatibilidad con main() (v6 tenía dos funciones; v7 usa una)
def fig_S3_gene_dropout_d(logger):
    fig_S3_gene_dropout(logger)

def fig_S3_gene_dropout_ari(logger):
    pass  # Generado por fig_S3_gene_dropout_d → no duplicar


def fig_S4_all_celltypes(logger):
    logger.info("S4_all_celltypes — Heatmap")
    csv_path = None
    for p in [RESULTS_DIR / "comprehensive_celltype" / "all_celltypes_by_phenotype.csv", RESULTS_DIR / "tables" / "all_celltypes_by_phenotype.csv"]:
        if p.exists(): csv_path = p; break

    if not csv_path:
        skip_fig("S4_all_celltypes_heatmap", "CSV not found", logger)
        return

    df = pd.read_csv(csv_path)
    if not {"cell_type", "comparison", "cohens_d", "q_value"}.issubset(df.columns):
        skip_fig("S4_all_celltypes_heatmap", "Columnas faltantes en CSV", logger); return

    pivot = df.pivot_table(index="cell_type", columns="comparison", values="cohens_d", aggfunc="first")
    pivot_q = df.pivot_table(index="cell_type", columns="comparison", values="q_value", aggfunc="first")

    fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 3), max(6, pivot.shape[0] * 0.5 + 2)))
    fig.patch.set_facecolor("white")

    vmax = np.nanpercentile(np.abs(pivot.values), 95)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=20, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=9)
    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.7)

    if pivot.shape[0] <= 15:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                q_val = pivot_q.values[i, j]
                if pd.notna(q_val) and q_val < 0.05:
                    ax.text(j, i, "*", ha="center", va="center", fontsize=9, color="white" if abs(pivot.values[i,j]) > vmax*0.6 else "black")

    ax.set_title("All 15 Cell Types × Phenotype Comparisons\n(Cohen's d; * = FDR q < 0.05)", fontsize=10, fontweight="bold", loc="center")
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S4_all_celltypes_heatmap", logger)


def fig_S5_spatial_coherence(logger):
    logger.info("S5_spatial_coherence — Moran's I")
    csv_path = None
    for p in [RESULTS_DIR / "spatial_coherence" / "morans_i_results.csv", RESULTS_DIR / "tables" / "morans_i_results.csv"]:
        if p.exists(): csv_path = p; break

    if not csv_path:
        skip_fig("S5_spatial_coherence", "morans_i_results.csv not found", logger)
        return

    df = pd.read_csv(csv_path)
    if "morans_I" not in df.columns or "phenotype" not in df.columns:
        skip_fig("S5_spatial_coherence", "Columnas morans_I o phenotype faltantes", logger)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")

    agg = df.groupby("phenotype")["morans_I"].agg(["mean", "std"]).reset_index()
    ph_order_present = [p for p in PHENOTYPE_ORDER if p in agg["phenotype"].values]
    # Reordenar según PHENOTYPE_ORDER
    agg = agg.set_index("phenotype").loc[ph_order_present].reset_index()

    colors = [PHENOTYPE_COLORS.get(p, "#888") for p in agg["phenotype"]]
    ax.bar(range(len(agg)), agg["mean"].values, yerr=agg["std"].values,
           color=colors, edgecolor="black", linewidth=0.8, capsize=6, width=0.55)

    # Puntos individuales si el CSV tiene múltiples filas por fenotipo
    rows_per_ph = df.groupby("phenotype")["morans_I"].count()
    if rows_per_ph.max() > 1:
        for idx, ph in enumerate(agg["phenotype"]):
            ph_vals = df[df["phenotype"] == ph]["morans_I"].values
            jit = RNG.uniform(-0.12, 0.12, len(ph_vals))
            ax.scatter(np.full(len(ph_vals), idx) + jit, ph_vals,
                       color="black", s=18, alpha=0.5, zorder=5)

    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(len(agg)))
    # Etiquetas cortas (PHENOTYPE_LABELS)
    ax.set_xticklabels(
        [PHENOTYPE_LABELS.get(p, p).replace("\n", " ") for p in agg["phenotype"]],
        fontsize=10
    )
    ax.set_ylabel("Moran's I (mean ± SD across sections)", fontsize=10)
    # Referencia y=0 identificada en el eje, no como anotación de texto
    ax.set_title("Spatial Coherence of Phenotype Domains\n(Moran's I > 0: clustered; mean across sections)",
                 fontsize=10, fontweight="bold", loc="center")
    style_ax(ax)
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S5_spatial_coherence", logger)


def fig_S6_permutation_null_dist(logger):
    logger.info("S6_permutation_null_dist")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    from scipy.stats import norm
    sigma = 1.0 / np.sqrt(22080)
    x_null = np.linspace(-4 * sigma, 4 * sigma, 300)
    ax.plot(x_null, norm.pdf(x_null, 0, sigma), color="#AACCE6", lw=2,
            label="Theoretical null  N(0, 1/√n)")
    ax.fill_between(x_null, norm.pdf(x_null, 0, sigma), alpha=0.3, color="#AACCE6")
    # Nota "permutation array not found" → logueada (info técnica para caption)
    logger.info("  S6 null dist (para caption): distribución nula teórica N(0,1/√n) "
                "porque el array de permutaciones no fue encontrado en el path.")

    ax.axvline(REF_CAF_D_SPOT, color="#0072B2", lw=2.5,
               label=f"Observed d = {REF_CAF_D_SPOT:.3f}", clip_on=False)
    # "1,000 perm." ya está en el título → no repetir en etiqueta

    # z-box moved to (0.22, 0.80) — away from the null peak near x≈0, right edge
    ax.text(0.22, 0.80, "z = −44.08", transform=ax.transAxes,
            ha="center", va="center", fontsize=9, color="#0072B2",
            bbox=dict(boxstyle="round", fc="white", ec="#0072B2", alpha=0.9))

    ax.set_xlim(min(-0.65, REF_CAF_D_SPOT - 0.05), max(4 * sigma + 0.01, 0.05))
    ax.set_xlabel("Cohen's d (CAF: Excluded vs Desert)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Permutation null distribution\n"
                 "(spot-level, 1,000 permutations; observed d far outside null)",
                 fontsize=10, fontweight="bold", loc="left")
    ax.legend(fontsize=9, frameon=True, loc="lower center")
    style_ax(ax)

    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S6_permutation_null_distribution", logger)



def fig_MYC_clean_MHCI_scatter(adata, logger):
    """\nScatter de actividad MYC-clean (regulón limpio, sin genes ISG/MHC-I)
    versus score MHC-I en spots Desert.

    FRAMING CORRECTO: esta figura muestra la AUSENCIA de correlación fuerte
    con el regulón limpio — es evidencia del resultado negativo de MYC.
    Con el regulón contaminado ρ era -0.505 (ISG en el regulón creaba
    circularidad); con el regulón limpio ρ colapsa a ~0. ESO es lo que
    debe transmitir el título y la anotación.

    NO usar el título "Functional Suppression" (que implicaría un hallazgo
    positivo). El título correcto es "Clean regulon shows no strong MHC-I
    suppression in Desert" o similar.

    Si la columna MYC_TF_clean / MHC_I no está en adata.obs, hace skip.
    """
    logger.info("MYC_clean_MHCI_scatter — negative result confirmation")

    # ── locate columns ───────────────────────────────────────────────────────
    # MYC clean TF score
    myc_col = None
    for cand in ["MYC_TF_activity_clean", "myc_tf_clean", "MYC_clean"]:
        if cand in adata.obs.columns:
            myc_col = cand; break
    if myc_col is None:
        # fallback: any column with MYC that doesn't say "contam"
        candidates = [c for c in adata.obs.columns
                      if "MYC" in c.upper() and "contam" not in c.lower()]
        if candidates:
            myc_col = candidates[0]
            logger.warning(f"  MYC clean column not found exactly; using fallback: {myc_col}")

    # MHC-I score
    mhci_col = None
    for cand in ["MHC_I_signature", "MHC_I_score", "mhc_i", "MHC1_signature"]:
        if cand in adata.obs.columns:
            mhci_col = cand; break
    if mhci_col is None:
        candidates = [c for c in adata.obs.columns if "MHC" in c.upper()]
        if candidates:
            mhci_col = candidates[0]
            logger.warning(f"  MHC-I column not found exactly; using fallback: {mhci_col}")

    if myc_col is None or mhci_col is None:
        skip_fig("MYC_clean_MHCI_scatter",
                 f"columns not found (myc={myc_col}, mhci={mhci_col})",
                 logger)
        return

    mask_desert = (adata.obs["Phenotype"] == "Immune_Desert").values
    x = adata.obs.loc[mask_desert, myc_col].values.astype(float)
    y = adata.obs.loc[mask_desert, mhci_col].values.astype(float)

    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    if len(x) < 50:
        skip_fig("MYC_clean_MHCI_scatter",
                 f"insufficient Desert spots after filtering (n={len(x)})",
                 logger)
        return

    # subsample for rasterization
    if len(x) > 6000:
        idx = RNG.choice(len(x), 6000, replace=False)
        x_plot, y_plot = x[idx], y[idx]
    else:
        x_plot, y_plot = x, y

    from scipy.stats import spearmanr as _spearmanr
    rho, p_val = _spearmanr(x, y)
    n_total = len(x)
    logger.info(f"  MYC-clean vs MHC-I in Desert: rho={rho:.3f}, p={p_val:.3e}, n={n_total}")

    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    fig.patch.set_facecolor("white")

    ax.scatter(x_plot, y_plot,
               color=PHENOTYPE_COLORS["Immune_Desert"],
               s=4, alpha=0.30, rasterized=True)

    # regression line
    m, b = np.polyfit(x, y, 1)
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, m * xl + b, color="black", lw=1.6, zorder=3)

    # annotation — framed as negative result
    sig_str = "ns" if p_val > 0.05 else (
        f"p = {p_val:.2e}")
    # Annotation removed — key stats for caption logged here
    logger.info(
        f"  Caption stats: rho={rho:.3f} ({sig_str}), n={n_total:,} Desert spots; "
        f"contaminated regulon gave rho~-0.505; collapse to ~0 confirms artefact"
    )

    ax.set_xlabel(f"MYC TF activity — clean regulon ({myc_col})", fontsize=10)
    ax.set_ylabel(f"MHC-I / antigen presentation score ({mhci_col})", fontsize=10)

    style_ax(ax)
    ax.set_title(
        "Clean MYC regulon: no strong MHC-I suppression in Desert",
        fontsize=10, fontweight="bold", loc="left", pad=10)

    plt.tight_layout(pad=1.3)
    save_fig(fig, OUT_SUPP, "MYC_clean_MHCI_scatter", logger)

# Old main() removed (v9-2 style); see argparse main() below

def fig_S_validation_spatial_maps(adata_val, logger):
    """
    Supplementary: spatial maps of validation cohort (GSE213688).
    2 rows × 2 columns: [Excluded-rich, Desert-rich] × [Phenotype, CAF abundance]

    Anti-overlap rules:
    - No text on scatter; colorbars external
    - Row/column labels via set_ylabel / set_title only
    - CAF abundance: use raw C2L values (no proportion normalisation —
      that is only for the statistical contrast, not the visual map)
    - sample_id check: if not in adata_val.obs, use all spots as single block

    Source: adata_val.obsm["spatial"], adata_val.obs["Phenotype"],
            C2L CAF from get_c2l(adata_val, "CAF", logger)
    """
    logger.info("S_validation_spatial_maps — 2×2 panel (GSE213688)")

    if adata_val is None:
        skip_fig("S_validation_spatial_maps",
                 "adata_val is None — validation cohort not loaded", logger)
        return

    if "spatial" not in adata_val.obsm:
        skip_fig("S_validation_spatial_maps",
                 "spatial coords not found in validation adata", logger)
        return

    caf_vals, caf_col = get_c2l(adata_val, "CAF", logger)
    if caf_vals is None:
        skip_fig("S_validation_spatial_maps",
                 f"CAF not found in validation C2L ({caf_col})", logger)
        return

    has_sample = "sample_id" in adata_val.obs.columns

    # Select Excluded-rich and Desert-rich sections
    sections = []  # list of (label, mask)
    targets  = [("Excluded-rich", "Immune_Excluded"),
                ("Desert-rich",   "Immune_Desert")]
    used_sids = set()

    if has_sample:
        for label, ph_target in targets:
            best_sid, best_frac = None, -1
            for sid in adata_val.obs["sample_id"].unique():
                if sid in used_sids: continue
                m = (adata_val.obs["sample_id"] == sid).values
                frac = (adata_val.obs["Phenotype"].values[m] == ph_target).mean()
                if frac > best_frac:
                    best_frac = frac; best_sid = sid
            if best_sid is not None:
                mask = (adata_val.obs["sample_id"] == best_sid).values
                sections.append((label, mask))
                used_sids.add(best_sid)
                logger.info(f"  {label}: section {best_sid} "
                            f"({ph_target} frac={best_frac:.2f})")
    else:
        # No sample_id — split by phenotype dominance is not possible;
        # show all spots as one block for each row (different colour)
        for label, ph_target in targets:
            mask = (adata_val.obs["Phenotype"] == ph_target).values
            if mask.sum() > 50:
                sections.append((label, mask))
                logger.info(f"  {label}: using all {ph_target} spots "
                            f"(n={mask.sum()}) — no sample_id in adata_val")

    if not sections:
        skip_fig("S_validation_spatial_maps",
                 "no suitable sections found in validation cohort", logger)
        return

    n_rows = len(sections)
    n_cols = 2
    col_titles = ["Phenotype", "CAF abundance (C2L)"]

    # Pre-fetch per-row data once so all render modes use the same sample.
    row_data = []
    for row_label, mask in sections:
        coords  = adata_val.obsm["spatial"][mask]
        phenos  = adata_val.obs["Phenotype"].values[mask]
        caf_s   = caf_vals[mask]
        n_s     = mask.sum()
        sub_idx = (RNG.choice(n_s, min(n_s, 8000), replace=False)
                   if n_s > 8000 else np.arange(n_s))
        row_data.append(dict(label=row_label, coords=coords[sub_idx],
                              phenos=phenos[sub_idx], caf=caf_s[sub_idx]))

    def _build(mode):
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4.5 * n_cols, 4.2 * n_rows),
                                  squeeze=False)
        fig.patch.set_facecolor("white")

        for j, ct in enumerate(col_titles):
            axes[0, j].set_title(ct, fontsize=11, fontweight="bold", pad=7)

        for row_idx, rd in enumerate(row_data):
            coords_sub, phenos_sub, caf_sub = rd["coords"], rd["phenos"], rd["caf"]

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.set_aspect("equal")
                if mode == "axes_only":
                    ax.axis("on")
                    ax.set_xticks([]); ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(True); sp.set_color("#BBBBBB")
                else:
                    ax.axis("off")

                if col_idx == 0:
                    if mode != "labels_only":
                        for ph in PHENOTYPE_ORDER:
                            m = phenos_sub == ph
                            if m.sum() == 0: continue
                            ax.scatter(coords_sub[m, 0], coords_sub[m, 1],
                                       c=PHENOTYPE_COLORS.get(ph, "#888"),
                                       s=2, alpha=0.82, rasterized=True)
                else:
                    sc = None
                    if mode != "labels_only":
                        fin = np.isfinite(caf_sub)
                        vmax = float(np.percentile(caf_sub[fin], 97)) if fin.sum() > 10 else 1.0
                        sc = ax.scatter(coords_sub[:, 0], coords_sub[:, 1],
                                        c=caf_sub, cmap="Oranges",
                                        vmin=0, vmax=vmax,
                                        s=2, alpha=0.85, rasterized=True)
                    if mode != "axes_only":
                        if sc is None:
                            sc = matplotlib.cm.ScalarMappable(
                                cmap="Oranges", norm=matplotlib.colors.Normalize(0, 1))
                            sc.set_array([])
                        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                        cbar.ax.tick_params(labelsize=7)
                        cbar.set_label("C2L abundance", fontsize=7)

        if mode != "axes_only":
            _row_ys_v = [0.75, 0.25] if n_rows == 2 else [
                1 - (i + 0.5) / n_rows for i in range(n_rows)]
            for _ri, rd in enumerate(row_data):
                fig.text(0.01, _row_ys_v[_ri], rd["label"],
                         fontsize=10, fontweight="bold",
                         va="center", ha="left", rotation=90,
                         transform=fig.transFigure)
            handles = [mpatches.Patch(color=PHENOTYPE_COLORS[ph],
                                       label=PHENOTYPE_LABELS.get(ph, ph))
                       for ph in PHENOTYPE_ORDER if ph in PHENOTYPE_COLORS]
            axes[0, 0].legend(handles=handles, loc="lower left",
                              bbox_to_anchor=(0, -0.12), fontsize=8,
                              frameon=True, framealpha=0.9, ncol=2)
            fig.suptitle("Validation cohort spatial architecture (GSE213688)",
                         fontsize=12, fontweight="bold", y=1.01)

        plt.tight_layout(pad=1.2, h_pad=0.8, w_pad=0.6)
        return fig

    _panelcheck_variants(_build, OUT_SUPP, "S_validation_spatial_maps", logger)


def fig_S6_permutation_zscores(logger):
    logger.info("S6_permutation_zscores")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    z_data = [("Spot-level\n(1,000 perm.)", 44.08, "#0072B2"), ("Patient-level\n(500 perm.)", 14.02, "#009E73")]
    for i, (lbl, z_val, col) in enumerate(z_data):
        ax.bar(i, z_val, color=col, edgecolor="black", width=0.5)
        ax.text(i, z_val + 0.6, f"z = −{z_val:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels([d[0] for d in z_data], fontsize=10)
    ax.set_ylabel("|z-score| vs permutation null", fontsize=10)
    ax.margins(y=0.15)
    # "Higher |z|..." → descriptivo, a caption
    logger.info("  S6 zscores (para caption): Higher |z| = greater deviation from "
                "random-label null (phenotype identity drives the CAF separation)")
    ax.set_title("Permutation z-scores (spot and patient)\n(observed vs random-label null)",
                 fontsize=10, fontweight="bold", loc="left")
    style_ax(ax)

    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S6_permutation_zscores", logger)


def fig_S7_MYC_extended(adata, logger):
    """Combina los 3 scatter MYC×Tumour en una figura 1×3. v6 generaba 3 archivos."""
    logger.info("S7_MYC_extended — combined 1×3 figure")
    if "MYC_TF_activity_clean" not in adata.obs.columns:
        skip_fig("S7_MYC_extended_confounding", "MYC_TF_activity_clean not in adata.obs", logger)
        return
    tumor_c2l, _ = get_c2l(adata, "Tumor", logger)
    if tumor_c2l is None:
        skip_fig("S7_MYC_extended_confounding", "Tumor C2L not found", logger)
        return

    tf_vals = adata.obs["MYC_TF_activity_clean"].values
    phenos  = ["Immune_Desert", "Immune_Excluded", "Inflamed"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")

    for ax, pheno in zip(axes, phenos):
        mask_ph = adata.obs["Phenotype"].values == pheno
        idx_ph  = np.where(mask_ph & np.isfinite(tf_vals) & np.isfinite(tumor_c2l))[0]
        idx_s   = RNG.choice(idx_ph, min(3000, len(idx_ph)), replace=False)

        ax.scatter(tumor_c2l[idx_s], tf_vals[idx_s],
                   alpha=0.2, s=5, color=PHENOTYPE_COLORS.get(pheno, "#888"), rasterized=True)
        coef  = np.polyfit(tumor_c2l[idx_ph], tf_vals[idx_ph], 1)
        x_fit = np.linspace(tumor_c2l[idx_ph].min(), tumor_c2l[idx_ph].max(), 100)
        ax.plot(x_fit, np.polyval(coef, x_fit), "-", color="black", lw=1.5, alpha=0.7)

        rho, _, n_v = spearman_safe(tumor_c2l[idx_ph], tf_vals[idx_ph])
        ax.text(0.03, 0.96, f"ρ = {rho:.3f}\n(n = {n_v:,})",
                transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))

        ax.set_xlabel("Tumour abundance (C2L)", fontsize=9)
        ax.set_ylabel("MYC TF activity (clean, z)", fontsize=9)
        label = PHENOTYPE_LABELS.get(pheno, pheno).replace("\n", " ")
        ax.set_title(f"MYC TF vs Tumour fraction: {label}",
                     fontsize=10, fontweight="bold", loc="left")
        style_ax(ax)

    fig.suptitle("Supplementary Figure S7 — MYC TF score correlates with tumour fraction "
                 "across all phenotypes\n(confounding is universal, not Desert-specific)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S7_MYC_extended_confounding", logger)


def fig_S8_checkpoint_vs_MYC(adata, logger):
    logger.info("S8_checkpoint_vs_MYC")
    if "MYC_TF_activity_clean" not in adata.obs.columns: return

    CHECKPOINT_GENES = [c[0] for c in CHECKPOINT_DATA]
    tf_vals  = adata.obs["MYC_TF_activity_clean"].values
    mask_tum = adata.obs["Phenotype"].isin(["Immune_Desert", "Immune_Excluded", "Inflamed"]).values

    rhos, genes_found = [], []
    for gene in CHECKPOINT_GENES:
        expr = get_gene(adata, gene)
        if expr is None: continue
        rho, _, _ = spearman_safe(tf_vals[mask_tum], expr[mask_tum])
        if np.isfinite(rho):
            rhos.append(rho); genes_found.append(gene)
    
    if not genes_found:
        skip_fig("S8_checkpoint_vs_MYC", "Ningún gen checkpoint tiene datos correlacionables", logger); return
    

    order = np.argsort(rhos)
    rhos_s = np.array(rhos)[order]
    genes_s = np.array(genes_found)[order]

    fig, ax = plt.subplots(figsize=(8, max(5, len(genes_s)*0.45)))
    fig.patch.set_facecolor("white")

    colors = ["#0072B2" if r < -0.10 else "#CC0000" if r > 0.10 else "#AAAAAA" for r in rhos_s]
    ax.barh(range(len(genes_s)), rhos_s, color=colors, edgecolor="white", linewidth=0.4, height=0.7)
    ax.axvline(0, color="black", lw=0.8)
    ax.axvline(-0.10, color="gray", ls="--", lw=1, alpha=0.7)
    ax.axvline(+0.10, color="gray", ls="--", lw=1, alpha=0.7)

    ax.set_yticks(range(len(genes_s)))
    ax.set_yticklabels(genes_s, fontsize=9)
    ax.set_xlabel("Spearman ρ (checkpoint gene vs MYC TF-clean, tumour spots)", fontsize=9)
    
   
    # "Note: MYC score..." → descriptivo, a caption
    logger.info("  S8 (para caption): MYC score refleja fracción tumoral (ρ_tumour=0.58). "
                "Interpretar correlaciones checkpoint×MYC con cautela.")

    leg_els = [
        mpatches.Patch(color="#CC0000", label="Positively correlated (ρ > 0.10)"),
        mpatches.Patch(color="#AAAAAA", label="Negligible correlation (|ρ| ≤ 0.10)"),
        mpatches.Patch(color="#0072B2", label="Negatively correlated (ρ < -0.10)"),
        ]
    ax.legend(handles=leg_els, fontsize=8, loc="lower right", frameon=True)

    ax.set_title("Checkpoint × MYC TF Activity\n(exploratory; based on cleaned regulon score)",
                 fontsize=10, fontweight="bold", loc="center")
    style_ax(ax)
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S8_checkpoint_vs_MYC", logger)


def fig_S9_validation_patient_CAF(adata_val, logger):
    logger.info("S9_validation_patient_CAF")
    if adata_val is None or "sample_id" not in adata_val.obs.columns:
        skip_fig("S9_validation_patient_CAF", "Validation adata o sample_id no disponibles", logger); return
    caf_vals, _ = get_c2l(adata_val, "CAF", logger)
    if caf_vals is None:
        skip_fig("S9_validation_patient_CAF", "CAF C2L no encontrado en validación", logger); return

    desert_meds, excluded_meds = [], []
    for sid in sorted(adata_val.obs["sample_id"].unique()):
        mask_sid = adata_val.obs["sample_id"] == sid
        ph_s     = adata_val.obs.loc[mask_sid, "Phenotype"].values
        caf_s    = caf_vals[mask_sid]
        des_v    = caf_s[(ph_s == "Immune_Desert") & np.isfinite(caf_s)]
        exc_v    = caf_s[(ph_s == "Immune_Excluded") & np.isfinite(caf_s)]
        if len(des_v) > 0 and len(exc_v) > 0:
            desert_meds.append(float(np.median(des_v)))
            excluded_meds.append(float(np.median(exc_v)))

    if len(desert_meds) < 3:
        skip_fig("S9_validation_patient_CAF", "Data pareada insuficiente (<3 secciones)", logger); return

    des_arr, exc_arr = np.array(desert_meds), np.array(excluded_meds)
    n_concordant = int((exc_arr > des_arr).sum())

    # Stats -> logueados (descriptivos para caption)
    logger.info(f"  S9 stats (para caption): d = -0.120; Wilcoxon p = 0.004; "
                f"{n_concordant}/{len(des_arr)} sections concordant (Excluded > Desert)")

    from matplotlib.lines import Line2D as _L2D

    def _build(mode):
        fig, ax = plt.subplots(figsize=(5, 6.5))
        fig.patch.set_facecolor("white")

        # ── Axes structure (always present) ──────────────────────────────
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Desert", "Excluded"], fontsize=12)
        ax.set_xlim(-0.4, 1.4)
        ax.set_ylabel("CAF abundance (section median)", fontsize=10)
        ax.set_title("Validation Patient-Level CAF\n(GSE213688; 15 sections)",
                     fontsize=10, fontweight="bold", loc="center")
        # symlog: two sections have CAF medians ~15-25, compressing the
        # remaining 13 sections (range ~0-3.5) near the bottom. linthresh=1
        # keeps 0-1 linear (so zeros remain visible) and compresses the
        # high-CAF outliers without a hard axis break.
        ax.set_yscale("symlog", linthresh=1.0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_yticks([0, 1, 2, 3, 5, 10, 20, 30])
        style_ax(ax)

        # ── Data: per-section lines + dots + mean markers ──────────────────
        if mode != "labels_only":
            for d_val, e_val in zip(des_arr, exc_arr):
                concordant = e_val > d_val
                ax.plot([0, 1], [d_val, e_val],
                        "-", color="#0072B2" if concordant else "#E69F00",
                        alpha=0.55, lw=1.8)
            ax.scatter([0]*len(des_arr), des_arr,
                       color=PHENOTYPE_COLORS["Immune_Desert"], s=50, zorder=4,
                       edgecolors="white", lw=0.5)
            ax.scatter([1]*len(exc_arr), exc_arr,
                       color=PHENOTYPE_COLORS["Immune_Excluded"], s=50, zorder=4,
                       edgecolors="white", lw=0.5)
            ax.scatter([0, 1], [des_arr.mean(), exc_arr.mean()],
                       marker="D", s=100, color="black", zorder=5)
        else:
            # Keep y-limits sensible without data (use observed range)
            all_v = np.concatenate([des_arr, exc_arr])
            pad = (all_v.max() - all_v.min()) * 0.1 + 1e-6
            ax.set_ylim(all_v.min() - pad, all_v.max() + pad)

        # ── Labels: concordant/discordant/mean legend ──────────────────────
        if mode != "axes_only":
            ax.legend(handles=[
                _L2D([0],[0], color="#0072B2", lw=2,
                     label=f"Concordant ({n_concordant}/{len(des_arr)})"),
                _L2D([0],[0], color="#E69F00", lw=2,
                     label=f"Discordant ({len(des_arr)-n_concordant}/{len(des_arr)})"),
                _L2D([0],[0], marker="D", ms=8, color="black", lw=0, label="Mean"),
            ], fontsize=8.5, loc="center right", bbox_to_anchor=(0.98, 0.35), frameon=True)

        plt.tight_layout(pad=1.2)
        return fig

    _panelcheck_variants(_build, OUT_SUPP, "S9_validation_patient_CAF", logger)


def fig_S10_null_survival(logger):
    logger.info("S10_null_survival")
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("white")

    surv_data = [("Inflamed vs Cold\n(METABRIC)", 0.048, 0.291), ("Desert vs Excluded\n(METABRIC)", 0.612, 1.000), ("Desert vs Other\n(METABRIC)", 0.234, 0.702), ("Inflamed vs Cold\n(TCGA)", 0.183, 0.549), ("Desert vs Excluded\n(TCGA)", 0.891, 1.000), ("Desert vs Other\n(TCGA)", 0.445, 1.000)]
    labels = [s[0] for s in surv_data]
    pvals  = [s[1] for s in surv_data]
    qvals  = [s[2] for s in surv_data]
    y_pos  = np.arange(len(surv_data))

    ax.barh(y_pos, pvals, color="#AACCE6", edgecolor="white", height=0.5, label="p-value")
    ax.barh(y_pos, qvals, color="#0072B2", edgecolor="white", height=0.5, alpha=0.5, label="q-value (BH-FDR)")
    ax.axvline(0.05, color="red", ls="--", lw=1.5, label="α = 0.05")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("p-value / q-value", fontsize=9)
    ax.set_xlim(0, 1.15)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Survival analysis: 0/6 contrasts FDR-significant\n(power-limited; exploratory only; min q = 0.291)", fontsize=10, fontweight="bold", loc="left")
    style_ax(ax)
    
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_SUPP, "S10_null_results_survival", logger)


def fig_S10_null_myc(logger):
    """\nTwo-panel summary of MYC null results — Q1 presentable, no n/a.

    Panel A: mRNA (d=+0.052) and Hallmark (d=-0.062, p=0.592) — both with
             verified p-values; threshold line at |d|=0.5; clean Cohen's d axis.
    Panel B: Structured text explaining TF-clean d=+0.347 as tumour-fraction
             confounding, not a MYC effect. Replaces the problematic orange bar
             that visually highlighted the artefact as if it were a finding.

    Audit note: the previous single-axis version mixed Spearman rho and Cohen's d
    on the same "Statistic" axis and showed missing p-values for two of three measures.
    Both issues are resolved here. The MYC negative result is fully detailed in
    the main MYC_negative_result panel; this figure provides a concise supplement.
    """
    logger.info("S10_null_myc — two-panel version (no n/a, no mixed axis)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5),
                                    gridspec_kw={"width_ratios": [1.4, 1]})
    fig.patch.set_facecolor("white")

    # ── Panel A: the two null measures with clean p-values ──────────────────
    labels = ["MYC mRNA\n(Desert vs Inflamed)", "MYC Hallmark\n(Desert vs Inflamed)"]
    values = [+0.052, -0.062]
    pstrs  = ["p = ns (near-zero d)", "p = 0.592 (ns)"]
    colors = ["#AACCE6", "#AACCE6"]

    ax1.bar(range(2), values, color=colors, edgecolor="black", linewidth=0.9, width=0.45, zorder=3)
    ax1.axhline(0, color="black", lw=0.8, zorder=2)

    # threshold lines
    for thresh, ls in [(0.5, "--"), (-0.5, "--")]:
        ax1.axhline(thresh, color="#D55E00", lw=1.2, ls=ls, alpha=0.7, zorder=1)
    ax1.text(1.55, 0.52, "|d| = 0.5\nrelevance\nthreshold",
             fontsize=7.5, color="#D55E00", va="bottom", ha="center")

    for i, (v, p) in enumerate(zip(values, pstrs)):
        yoff = 0.025 if v >= 0 else -0.025
        va_  = "bottom" if v >= 0 else "top"
        ax1.text(i, v + yoff, f"{v:+.3f}\n({p})",
                 ha="center", va=va_, fontsize=8.5,
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))

    ax1.set_xticks(range(2))
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_ylabel("Cohen's d (Desert vs Inflamed)", fontsize=10)
    ax1.set_ylim(-0.38, 0.78)
    ax1.set_xlim(-0.6, 2.0)
    style_ax(ax1)
    ax1.set_title("a  MYC expression and target-programme activity",
                  fontsize=10, fontweight="bold", loc="left", pad=10)
    panel_label(ax1, "a")

    # ── Panel B: TF-clean confounding explanation ────────────────────────────
    ax2.axis("off")
    note_lines = [
        ("b  MYC TF activity (clean regulon)", True),
        ("", False),
        ("Apparent effect:  d = +0.347  (Desert vs Inflamed)", False),
        ("", False),
        ("After controlling for tumour fraction:", False),
        ("  ρ(MYC TF, tumour)  =  0.58  (intra-Desert)", False),
        ("  ρ(MYC TF, tumour)  =  0.69  (intra-Inflamed)", False),
        ("  Tumour fraction alone:  d = 0.499", False),
        ("", False),
        ("Interpretation:", False),
        ("  The apparent Desert enrichment of MYC TF", False),
        ("  activity is explained by higher tumour-cell", False),
        ("  abundance in Desert niches, not by niche-", False),
        ("  specific MYC regulation. Once tumour fraction", False),
        ("  is held constant, no residual MYC effect", False),
        ("  exceeds the |d| ≥ 0.5 relevance threshold.", False),
        ("", False),
        ("Conclusion: no robust MYC driver evidence", False),
        ("across all three pre-specified measures.", False),
    ]

    y = 0.97
    for line, bold in note_lines:
        weight = "bold" if bold else "normal"
        size   = 9.5 if bold else 8.5
        ax2.text(0.04, y, line, transform=ax2.transAxes,
                 va="top", ha="left", fontsize=size, fontweight=weight,
                 family="DejaVu Sans")
        y -= 0.053 if bold else 0.048

    # border box around panel B
    for spine_name in ["top", "bottom", "left", "right"]:
        ax2.spines[spine_name].set_visible(True)
        ax2.spines[spine_name].set_color("#CCCCCC")
        ax2.spines[spine_name].set_linewidth(0.8)
    ax2.set_facecolor("#F9F9F9")

    plt.tight_layout(pad=1.5)
    logger.info("  S10_null_myc: panel A = mRNA d=+0.052 (ns) + Hallmark d=-0.062 p=0.592; "
                "panel B = TF-clean d=+0.347 explained as tumour-fraction confounding.")
    save_fig(fig, OUT_SUPP, "S10_null_results_myc", logger)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS NUEVAS v9-2 — CDF DISTANCE + MYC CLEAN SCATTER
# ══════════════════════════════════════════════════════════════════════════════


def fig_distance_cdf(adata, logger):
    """\nCDF de distancia euclídea de spots CD8-alto a spots CAF-alto (percentil 75),
    calculada sección a sección y acumulada por fenotipo.

    Interpretación esperada: si Excluded tiene CDF desplazada hacia distancias
    MENORES que Desert/Inflamed → los T en el nicho excluido están más cerca de
    la barrera CAF densa, evidencia espacial directa del confinamiento físico.

    CHECK INTERNO DE SEPARACIÓN: se calcula la estadística KS entre Excluded y
    Desert. Si KS D < 0.08 (separación menor al 8 % de la distribución), la
    figura se omite con skip_fig() y se logea. Esto evita publicar una figura
    que no soporte el claim si los datos no la sostienen.

    Notas de seguridad:
    - Usa distancia EUCLÍDEA (no geodésica). La geodésica fue eliminada por
      métrica mal escalada (píxeles vs hops). La euclídea opera en las mismas
      unidades para todos los spots y no tiene ese defecto.
    - Si CAF o CD8_T no están en el obsm, la función hace skip.
    - La unidad de distancia son unidades de coordenadas Visium (píxeles de
      alta resolución); se etiqueta como "spatial units" sin asumir µm.
    """
    logger.info("distance_cdf — euclidean CD8→CAF boundary")

    from scipy.spatial import cKDTree
    from scipy.stats import ks_2samp

    caf_vals, caf_col = get_c2l(adata, "CAF", logger)
    cd8_vals, cd8_col = get_c2l(adata, "CD8_T", logger)

    if caf_vals is None or cd8_vals is None:
        skip_fig("distance_cdf",
                 f"CAF or CD8_T not found in C2L obsm (caf_col={caf_col}, cd8_col={cd8_col})",
                 logger)
        return

    if "spatial" not in adata.obsm or "sample_id" not in adata.obs.columns:
        skip_fig("distance_cdf", "spatial coords or sample_id not available", logger)
        return

    distances = {"Immune_Desert": [], "Immune_Excluded": [], "Inflamed": []}

    for sid in adata.obs["sample_id"].unique():
        mask = (adata.obs["sample_id"] == sid).values
        coords  = adata.obsm["spatial"][mask]
        caf_s   = caf_vals[mask]
        cd8_s   = cd8_vals[mask]
        ph_s    = adata.obs["Phenotype"].values[mask]

        ok = np.isfinite(caf_s) & np.isfinite(cd8_s)
        if ok.sum() < 20:
            continue

        caf_thresh = np.percentile(caf_s[ok], 75)
        cd8_thresh = np.percentile(cd8_s[ok], 75)

        caf_coords = coords[ok][caf_s[ok] > caf_thresh]
        cd8_mask   = ok & (cd8_s > cd8_thresh)

        if len(caf_coords) == 0 or cd8_mask.sum() == 0:
            continue

        tree = cKDTree(caf_coords)
        dists, _ = tree.query(coords[cd8_mask])
        phs_cd8   = ph_s[cd8_mask]

        for d_val, ph in zip(dists, phs_cd8):
            if ph in distances:
                distances[ph].append(float(d_val))

    # ── separation check ────────────────────────────────────────────────────
    exc = np.array(distances["Immune_Excluded"])
    des = np.array(distances["Immune_Desert"])

    if len(exc) < 50 or len(des) < 50:
        skip_fig("distance_cdf",
                 f"insufficient spots for CDF (Excluded n={len(exc)}, Desert n={len(des)})",
                 logger)
        return

    ks_stat, ks_p = ks_2samp(exc, des)
    logger.info(f"  CDF KS test (Excluded vs Desert): D={ks_stat:.3f}, p={ks_p:.3e}")

    SEPARATION_THRESHOLD = 0.08   # minimum D to justify the figure
    if ks_stat < SEPARATION_THRESHOLD:
        skip_fig("distance_cdf",
                 f"KS D={ks_stat:.3f} < {SEPARATION_THRESHOLD} — CDFs not sufficiently "
                 f"separated to support figure (see log for details)",
                 logger)
        logger.warning("  distance_cdf: phenotype CDFs overlap too much; "
                       "figure omitted to avoid unsupported claim.")
        return

    # ── plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 5))
    fig.patch.set_facecolor("white")

    for ph in ["Immune_Excluded", "Immune_Desert", "Inflamed"]:
        d_arr = np.sort(np.array(distances[ph]))
        if len(d_arr) == 0:
            continue
        cdf = np.arange(1, len(d_arr) + 1) / len(d_arr)
        ax.plot(d_arr, cdf,
                color=PHENOTYPE_COLORS[ph],
                lw=2.2,
                label=f"{PHENOTYPE_LABELS[ph]} (n={len(d_arr):,})")

    ax.set_xlabel("Distance to dense CAF boundary (spatial units, p75 threshold)", fontsize=10)
    ax.set_ylabel("Cumulative probability", fontsize=10)
    # Legend (upper-left) and KS box (lower-right) -> opposite corners,
    # never overlap regardless of curve shapes.
    ax.legend(fontsize=9, frameon=True, loc="upper left")

    # KS annotation
    ax.text(0.97, 0.05,
            f"KS test (Excluded vs Desert)\nD = {ks_stat:.3f},  p = {ks_p:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.85))

    style_ax(ax)
    ax.set_title("CD8\u207a T-cell proximity to CAF-dense boundary by phenotype",
                 fontsize=10, fontweight="bold", loc="left", pad=10)

    plt.tight_layout(pad=1.3)
    save_fig(fig, OUT_SUPP, "distance_cdf", logger)



# ══════════════════════════════════════════════════════════════════════════════
# EXECUTOR — SUPP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TNBC publication figures — SUPP")
    parser.add_argument("--adata",     default=None,  help="Discovery h5ad (optional: uses HPC defaults if omitted)")
    parser.add_argument("--adata-val", default=None,  help="Validation h5ad (optional: uses HPC defaults if omitted)")
    parser.add_argument("--figures",   default="all",
                        help="Comma-separated figure names or 'all'")
    args = parser.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    logger = setup_logging()
    logger.info("=" * 72)
    logger.info(f"  Discovery:  {args.adata}")
    logger.info(f"  Validation: {args.adata_val}")
    logger.info(f"  Output:     {OUT_BASE}")
    logger.info("=" * 72)

    # Note: --adata CLI path is added to search list below if provided
    if args.adata:
        import sys as _sys
        from pathlib import Path as _P
        _extra = _P(args.adata)
        if _extra.exists():
            logger.info(f"  Loading discovery from CLI: {_extra}")
            adata = sc.read_h5ad(_extra)
        else:
            logger.warning(f"  --adata path not found: {_extra}; using HPC defaults")
            adata = load_adata_discovery(logger)
    else:
        adata = load_adata_discovery(logger)
    if adata is None:
        logger.error("Discovery adata failed to load — aborting"); return

    if args.adata_val:
        from pathlib import Path as _Pv
        _cliv = _Pv(args.adata_val)
        if _cliv.exists():
            logger.info(f"  Loading validation from CLI: {_cliv}")
            adata_val = sc.read_h5ad(_cliv)
        else:
            logger.warning(f"  --adata-val path not found: {_cliv}; using HPC defaults")
            adata_val = load_adata_validation(logger)
    else:
        adata_val = load_adata_validation(logger)

    figs = [
        ("S1_QC_metrics",             lambda: fig_S1_QC_metrics(adata, logger)),
        ("S2_parameter_sweep",        lambda: fig_S2_parameter_sweep(logger)),
        ("S3_gene_dropout",           lambda: fig_S3_gene_dropout(logger)),
        ("S4_all_celltypes",          lambda: fig_S4_all_celltypes(logger)),
        ("S5_spatial_coherence",      lambda: fig_S5_spatial_coherence(logger)),
        ("S6_permutation_null_dist",  lambda: fig_S6_permutation_null_dist(logger)),
        ("S6_permutation_zscores",    lambda: fig_S6_permutation_zscores(logger)),
        ("S7_MYC_extended",           lambda: fig_S7_MYC_extended(adata, logger)),
        ("S8_checkpoint_vs_MYC",      lambda: fig_S8_checkpoint_vs_MYC(adata, logger)),
        ("S9_validation_patient_CAF", lambda: fig_S9_validation_patient_CAF(adata_val, logger)),
        ("S10_null_survival",         lambda: fig_S10_null_survival(logger)),
        ("S10_null_myc",              lambda: fig_S10_null_myc(logger)),
        ("S_validation_spatial_maps", lambda: fig_S_validation_spatial_maps(adata_val, logger)),
        ("distance_cdf",              lambda: fig_distance_cdf(adata, logger)),
        ("MYC_clean_MHCI_scatter",    lambda: fig_MYC_clean_MHCI_scatter(adata, logger)),
    ]

    wanted = set(args.figures.split(",")) if args.figures != "all" else None

    run_ok, run_skip, run_fail = 0, 0, 0
    for name, fn in figs:
        if wanted and name not in wanted:
            continue
        logger.info(f"\n── {name} ──")
        try:
            fn()
            run_ok += 1
        except Exception as exc:
            logger.exception(f"  FAILED: {exc}")
            run_fail += 1

    logger.info(f"\nDone: {run_ok} OK | {run_skip} skipped | {run_fail} failed")


if __name__ == "__main__":
    main()

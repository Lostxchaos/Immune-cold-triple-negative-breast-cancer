#!/usr/bin/env python3
"""\npublication_figures_v10_main.py
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

OUT_BASE = RESULTS_DIR / "publication_figures_v10" / "main"
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
    log_path = OUT_BASE / "publication_figures_v10_main.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, mode="w"), logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("pub_figs_v10_main")

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


def fig_phenotype_classification_map(adata, logger):
    logger.info("phenotype_classification_map — spatial scatter")
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")

    if "spatial" not in adata.obsm or "sample_id" not in adata.obs.columns:
        ax.text(0.5, 0.5, "spatial coords / sample_id not available", ha="center", va="center", transform=ax.transAxes, fontsize=9, color="red")
    else:
        # Reuse the same Excluded-rich section shown in spatial_maps_panel /
        # gene_expression_overlay, so Fig 1a anchors the paper's headline
        # archetype (CAF-barrier Excluded) rather than the most disordered
        # section. Falls back to highest-entropy section if unavailable.
        best_sid = None
        try:
            _reps = _select_representative_sections(adata, logger)
            for _label, _sid in _reps:
                if _label == "Excluded-rich":
                    best_sid = _sid
                    break
        except Exception:
            best_sid = None

        title_suffix = "representative section, discovery"
        if best_sid is None:
            def _pheno_entropy(grp):
                counts = grp["Phenotype"].value_counts(normalize=True)
                return float(-np.sum(counts * np.log(counts + 1e-10)))
            try:
                best_sid = (adata.obs.groupby("sample_id", group_keys=False).apply(_pheno_entropy).idxmax())
            except TypeError:
                best_sid = (adata.obs.groupby("sample_id").apply(_pheno_entropy).idxmax())
            title_suffix = "representative section, discovery"

        mask_s = adata.obs["sample_id"] == best_sid
        adata_s = adata[mask_s]
        coords = adata_s.obsm["spatial"]
        phenos_s = adata_s.obs["Phenotype"].values

        for ph in PHENOTYPE_ORDER:
            m = phenos_s == ph
            if m.sum() == 0: continue
            ax.scatter(coords[m, 0], coords[m, 1], c=PHENOTYPE_COLORS.get(ph, "#888"), s=3, alpha=0.7, rasterized=True, label=PHENOTYPE_LABELS.get(ph, ph))

        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Section {best_sid} ({title_suffix})", fontsize=10, fontweight="bold", loc="left", pad=3)
        
        # Corrección: Leyenda anclada fuera del eje
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, markerscale=2.5, frameon=True, framealpha=0.85, title="Phenotype", title_fontsize=9)

    # Corrección: Ajuste del lienzo para acomodar la leyenda externa
    plt.tight_layout(rect=[0, 0, 0.82, 1], pad=1.2)
    save_fig(fig, OUT_MAIN, "phenotype_classification_map", logger)



def fig_phenotype_classification_bars(logger):
    logger.info("phenotype_classification_bars — stacked proportions")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    data_pct = {
        "Immune_Desert":    [19.0, 18.0],
        "Immune_Excluded":  [10.8, 11.4],
        "Inflamed":         [ 8.4,  9.0],
        "Normal_Stroma":    [60.0, 60.0],
        "Ambiguous_Cold":   [ 1.8,  1.6],
    }
    cohorts = ["Discovery\n(GSE210616)", "Validation\n(GSE213688)"]
    x = np.arange(len(cohorts))
    bottom = np.zeros(len(cohorts))

    for ph in PHENOTYPE_ORDER:
        vals = np.array(data_pct[ph])
        ax.bar(x, vals, bottom=bottom, color=PHENOTYPE_COLORS[ph], label=PHENOTYPE_LABELS[ph].replace("\n", " "), edgecolor="white", linewidth=0.5)
        for i, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 5:
                # Normal_Stroma (#999999) vs blanco: contraste 2.85:1 < WCAG 4.5:1
                # → usar negro para Normal_Stroma, blanco para el resto
                txt_color = "black" if ph == "Normal_Stroma" else "white"
                ax.text(x[i], b + v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=txt_color, fontweight="bold")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(cohorts, fontsize=10)
    ax.set_ylabel("% of spots", fontsize=10)
    ax.set_ylim(0, 105)
    style_ax(ax)
    ax.set_title("Phenotype proportions (discovery vs validation)", fontsize=10, fontweight="bold", loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8.5, frameon=False)
    
    plt.tight_layout(rect=[0, 0, 0.82, 1], pad=1.2)
    save_fig(fig, OUT_MAIN, "phenotype_classification_bars", logger)


def fig_CAF_gradient(adata, logger):
    logger.info("CAF_gradient — monotonic CAF abundance gradient")
    caf_vals, _ = get_c2l(adata, "CAF", logger)
    if caf_vals is None:
        skip_fig("CAF_gradient", "CAF C2L not found", logger)
        return

    phenos_grad = ["Immune_Desert", "Inflamed", "Immune_Excluded"]
    VERIFIED_MEDS = {"Immune_Desert": 0.848, "Inflamed": 1.391, "Immune_Excluded": 1.874}

    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("white")

    data_violin = []
    # Corrección: Se eliminó el cálculo de label_ys
    for ph in phenos_grad:
        mask_ph = adata.obs["Phenotype"].values == ph
        v_clean = caf_vals[mask_ph]
        v_clean = v_clean[np.isfinite(v_clean)]
        v_sub = RNG.choice(v_clean, 5000, replace=False) if len(v_clean) > 5000 else v_clean
        data_violin.append(v_sub)

    parts = ax.violinplot(data_violin, positions=range(3), showmedians=True, showextrema=False, widths=0.7)
    for pc, ph in zip(parts["bodies"], phenos_grad):
        pc.set_facecolor(PHENOTYPE_COLORS[ph])
        pc.set_alpha(0.75)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(2)
    # Clip Y to 99th pctile to reduce empty headroom
    _all_v = np.concatenate([d for d in data_violin if len(d) > 0])
    ax.set_ylim(0, float(np.percentile(_all_v, 99)) * 1.15)

    ax.set_xticks(range(3))
    ax.set_xticklabels([PHENOTYPE_LABELS[p] for p in phenos_grad], fontsize=11)
    ax.set_ylabel("CAF abundance (Cell2Location)", fontsize=11)

    # Corrección: Anotación anclada explícitamente a la mediana real
    for i, ph in enumerate(phenos_grad):
        med_val = VERIFIED_MEDS[ph]
        ax.annotate(
            f"med = {med_val:.3f}",
            xy=(i, med_val),
            xytext=(0, 8),  textcoords='offset points',
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.8),
            ha="center", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="lightgray", alpha=0.9)
        )

    ax.annotate("", xy=(0.87, -0.09), xycoords="axes fraction", xytext=(0.13, -0.09), textcoords="axes fraction", arrowprops=dict(arrowstyle="->, head_width=0.25, head_length=0.05", color="#555555", lw=1.5), annotation_clip=False)
    ax.text(0.50, -0.13, "increasing CAF barrier", ha="center", va="top", fontsize=8.5, color="#555555", transform=ax.transAxes, clip_on=False)

    logger.info("  CAF_gradient stats (para caption): "
                "d (Excluded vs Desert) = −0.624; 95% CI [−1.021, −0.248]; "
                "p = 2.27×10⁻¹³; 14/15 sections concordant (p = 0.004)")

    ax.set_title("CAF abundance gradient across immune phenotypes\n(Cell2Location, discovery; 43 sections, 74,131 spots)", fontsize=10, fontweight="bold", loc="center")
    style_ax(ax)
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_MAIN, "CAF_gradient", logger)


def fig_CAF_estimators_forest(logger):
    logger.info("CAF_estimators_forest — 7-estimator forest plot")
    from matplotlib.lines import Line2D

    n_est = len(CAF_ESTIMATORS)
    y_pos = np.arange(n_est)

    # n-labels logueados (etiquetas descriptivas → van a caption, no a figura)
    logger.info("  CAF forest n-labels (para caption/log, no en figura):")
    for name, d, ci, n_label, circ_free in CAF_ESTIMATORS:
        logger.info(f"    {name}: {n_label}")

    def _build(mode):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.patch.set_facecolor("white")

        # ── Axes structure (always present) ──────────────────────────────
        ax.axvline(0, color="black", lw=0.9, ls="-")
        ax.axvline(-0.5, color="#888", lw=0.8, ls="--", alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([e[0] for e in CAF_ESTIMATORS], fontsize=9)
        ax.set_xlabel("Cohen's d (Excluded vs Desert; negative = more CAF in Excluded)",
                       fontsize=10)
        ax.set_xlim(-2.6, 0.6)
        ax.set_title("Seven convergent CAF estimators — Excluded vs Desert\n"
                      "(all d < 0; anti-circularity controls marked ◆)",
                      fontsize=10, fontweight="bold", loc="left")
        style_ax(ax)

        # ── Data: markers + CI bars + per-point d= text ───────────────────
        if mode != "labels_only":
            for i, (name, d, ci, n_label, circ_free) in enumerate(CAF_ESTIMATORS):
                color = "#0072B2" if circ_free else "#555555"
                marker = "D" if circ_free else "o"
                ax.plot(d, i, marker, color=color, ms=9, zorder=3, clip_on=False)
                if ci:
                    ax.plot([ci[0], ci[1]], [i, i], "-", color=color, lw=2.5, zorder=2)
                x_label = (min(d, ci[0]) if ci else d) - 0.10
                ax.text(x_label, i, f"d = {d:.3f}", ha="right", va="center",
                        fontsize=8, color=color)

        # ── Labels: legend ─────────────────────────────────────────────────
        if mode != "axes_only":
            leg_els = [
                Line2D([0], [0], marker="D", color="#0072B2", ms=8, lw=0,
                       label="Classifier-independent (anti-circularity controls)  ◆"),
                Line2D([0], [0], marker="o", color="#555555", ms=8, lw=0,
                       label="Cell2Location-based"),
            ]
            ax.legend(handles=leg_els, fontsize=8.5, loc="upper right",
                      bbox_to_anchor=(1.0, 1.0), frameon=True)

        plt.tight_layout(rect=[0, 0, 0.93, 1], pad=1.2)
        return fig

    _panelcheck_variants(_build, OUT_MAIN, "CAF_estimators_forest", logger)


def fig_chemotaxis_correlation(logger):
    logger.info("chemotaxis_correlation — Spearman heatmap")
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white")

    im = ax.imshow(CHEMOTAXIS_RHO, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(CHEMOTAXIS_CELLS, fontsize=12)
    ax.set_yticks(range(len(CHEMOTAXIS_GENES)))
    ax.set_yticklabels(CHEMOTAXIS_GENES, fontsize=12)

    for i in range(len(CHEMOTAXIS_GENES)):
        for j in range(2):
            val = CHEMOTAXIS_RHO[i, j]
            txt_color = "white" if abs(val) > 0.45 else "black"
            # *** eliminados: todos tienen FDR < 0.001 (significancia uniforme → redundante).
            # La información de significancia está en el título y en la nota de validación.
            ax.text(j, i, f"ρ = {val:.3f}", ha="center", va="center",
                    fontsize=10.5, color=txt_color, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.75, pad=0.03)
    cbar.ax.tick_params(labelsize=8)

    ax.text(0.5, -0.12, "Validation: chemokine–CD8 ρ = 0.628 | chemokine–cDC1 ρ = 0.546", ha="center", va="top", fontsize=7.5, color="#555", transform=ax.transAxes, style="italic", clip_on=False)    
    ax.set_title("Chemokine co-localisation with cytotoxic immune cells\n(Spearman ρ; all 6 correlations FDR < 0.001, discovery cohort)", fontsize=10, fontweight="bold", loc="left")
    
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_MAIN, "chemotaxis_correlation", logger)


def fig_checkpoint_landscape(logger):
    logger.info("checkpoint_landscape — 17 checkpoint genes")
    checkpoints_sorted = sorted(CHECKPOINT_DATA, key=lambda x: x[1])
    names = [c[0] for c in checkpoints_sorted]
    ds    = [c[1] for c in checkpoints_sorted]
    sigs  = [c[2] for c in checkpoints_sorted]
    y_pos = np.arange(len(names))
    colors = ["#E69F00" if name == "CD47" else
              "#0072B2" if (d_val < 0 and sig) else
              "#AACCE6" if (d_val < 0 and not sig) else
              "#CC79A7"
              for d_val, sig, name in zip(ds, sigs, names)]

    logger.info("  checkpoint_landscape summary (para caption): "
                "14/17 genes lower in Desert FDR q<0.05; NECTIN2 q=0.755 ns; "
                "CD47 d=+0.097 q=3.10e-12; PVR d=+0.006 q=2.21e-9")

    def _build(mode):
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor("white")

        # ── Axes structure (always present) ──────────────────────────────
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.axvline(0, color="black", lw=0.9)
        ax.axvline(-0.5, color="#888", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("Cohen's d (Desert vs Inflamed)", fontsize=10)
        ax.margins(x=0.15)
        ax.set_title("Immune checkpoint landscape — Desert vs Inflamed\n"
                     "(n = 17 co-regulatory genes; Desert is broadly checkpoint-low)",
                     fontsize=10, fontweight="bold", loc="center")
        style_ax(ax)

        # ── Data: bars + significance asterisks ───────────────────────────
        if mode != "labels_only":
            ax.barh(y_pos, ds, color=colors, edgecolor="white", linewidth=0.4, height=0.72)
            for i, (d_val, sig) in enumerate(zip(ds, sigs)):
                if not sig: continue
                gap = 0.013 if d_val >= 0 else -0.013
                ha  = "left" if d_val >= 0 else "right"
                ax.text(d_val + gap, i, "*", ha=ha, va="center", fontsize=9, color="black")

        # ── Labels: threshold note + legend ────────────────────────────────
        if mode != "axes_only":
            leg_els = [
                mpatches.Patch(color="#0072B2", label="Lower in Desert (FDR q < 0.05)"),
                mpatches.Patch(color="#AACCE6", label="Lower in Desert (NS)"),
                mpatches.Patch(color="#E69F00", label="CD47 — elevated in Desert"),
                mpatches.Patch(color="#CC79A7", label="PVR — higher, |d| negligible"),
                Line2D([0], [0], color="none", label="* FDR q < 0.05"),
            ]
            ax.legend(handles=leg_els, fontsize=8, loc="lower left",
                      bbox_to_anchor=(0.0, 0.0), frameon=True)

        plt.tight_layout(pad=1.2)
        return fig

    _panelcheck_variants(_build, OUT_MAIN, "checkpoint_landscape", logger)


def fig_bulk_transferability(logger):
    logger.info("bulk_transferability — AUC verificado")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor("white")

    datasets = ["METABRIC\n(Basal-like\nn = 44)", "TCGA-BRCA\n(Basal-like\nn = 40)"]
    aucs = [REF_AUC_METABRIC, REF_AUC_TCGA]
    n_labels = ["31 Desert / 13 Excluded", "30 Desert / 10 Excluded"]
    delta_loo = [0.000, 0.007]
    colors = ["#2C6E7A", "#8A3B47"]  # neutral grays (avoid Excluded/Desert palette colours)

    ax.bar(range(2), aucs, color=colors, edgecolor="black", linewidth=0.8, width=0.5)
    ax.axhline(0.5, color="#888", lw=1, ls="--", alpha=0.7)
    # "chance" eliminado de figura: la dashed line a y=0.5 es autoexplicativa con el eje
    # Texto ⚠ Recoverability eliminado de figura → va en caption (logueado abajo)

    for i, (auc_val, delta, n_lbl) in enumerate(zip(aucs, delta_loo, n_labels)):
        ax.text(i, auc_val + 0.010, f"AUC = {auc_val:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(i, 0.44, n_lbl,
                ha="center", va="top", fontsize=7, color="white", style="italic")
    # ΔAUC LOO → logueado para caption
    logger.info("  bulk_transferability ΔAUC LOO (para caption): "
                "METABRIC ΔAUC = 0.000; TCGA ΔAUC = 0.007")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylabel("AUC (Barrier−Silencing continuous score)", fontsize=10)
    ax.set_ylim(0.38, 1.00)
    ax.set_title("Bulk signature recoverability — Desert vs Excluded\n(continuous score AUC; ΔAUC LOO-CV confirms no single-sample dominance)", fontsize=10, fontweight="bold", loc="center")
    style_ax(ax)

    logger.info("  bulk_transferability (para caption): Recoverability — NOT independent validation. "
                "The spatial-label scores and the bulk index share gene sets. "
                "AUC = 0.886 (METABRIC) / 0.963 (TCGA) confirms bulk computability, not phenotypic independence.")
    
    plt.tight_layout(pad=1.2)
    save_fig(fig, OUT_MAIN, "bulk_transferability", logger)


def fig_MYC_negative_result(adata, logger):
    logger.info("MYC_negative_result — resultado negativo y único multipanel permitido")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("white")

    # Panel a
    ax = axes[0]
    panel_label(ax, "a")
    metrics = [("MYC\nmRNA", +0.052, False, None), ("Hallmark\n(NS)", -0.062, False, None), ("TF-clean", +0.347, False, None), ("TF-contaminated\n(circular — excl.)", +0.622, True, "//")]
    for i, (label, d_val, circular, hatch) in enumerate(metrics):
        color, edgecolor, alpha = ("#CCCCCC", "#888888", 0.6) if circular else ("#009E73", "black", 0.85) if d_val < 0 else ("#E69F00", "black", 0.85)
        ax.bar(i, d_val, color=color, edgecolor=edgecolor, linewidth=1.2, alpha=alpha, hatch=hatch, width=0.6)
        yoff, va = (0.022, "bottom") if d_val >= 0 else (-0.04, "top")
        ax.text(i, d_val + yoff, f"d = {d_val:+.3f}", ha="center", va=va, fontsize=8.5, color="#333")
        # "circular artefact" y "No cleaned metric..." → descriptivos, a caption

    logger.info("  MYC panel a (para caption): "
                "No cleaned metric supports niche-specific MYC activity. "
                "TF-contaminado (d=+0.622) es artefacto circular.")

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([m[0] for m in metrics], fontsize=8.5)
    ax.set_ylabel("Cohen's d (Desert vs Inflamed)", fontsize=10)
    ax.set_ylim(-0.25, 0.92)
    ax.set_title("Three MYC metrics — none robust\n(d after removing read-out genes ↓↓)", fontsize=10, fontweight="bold", loc="left")

    # Legend explaining bar colour/hatch coding (bottom-right, empty quadrant)
    _myc_leg = [
        mpatches.Patch(facecolor="#E69F00", edgecolor="black", label="d ≥ 0"),
        mpatches.Patch(facecolor="#009E73", edgecolor="black", label="d < 0"),
        mpatches.Patch(facecolor="#CCCCCC", edgecolor="#888888", hatch="//",
                       alpha=0.6, label="Circular artefact (excluded)"),
    ]
    ax.legend(handles=_myc_leg, fontsize=7.5, loc="lower right",
              bbox_to_anchor=(1.0, 0.0), frameon=True, framealpha=0.9)
    style_ax(ax)

    # Panel b
    ax = axes[1]
    panel_label(ax, "b")
    has_tf = "MYC_TF_activity_clean" in adata.obs.columns
    tumor_c2l, _ = get_c2l(adata, "Tumor", logger)
    desert_mask = adata.obs["Phenotype"].values == "Immune_Desert"

    if has_tf and tumor_c2l is not None:
        tf_vals = adata.obs["MYC_TF_activity_clean"].values
        idx_des = np.where(desert_mask & np.isfinite(tf_vals) & np.isfinite(tumor_c2l))[0]
        idx_s = RNG.choice(idx_des, min(3000, len(idx_des)), replace=False)
        ax.scatter(tumor_c2l[idx_s], tf_vals[idx_s], alpha=0.25, s=5, color=PHENOTYPE_COLORS["Immune_Desert"], rasterized=True)
        coef = np.polyfit(tumor_c2l[idx_des], tf_vals[idx_des], 1)
        x_fit = np.linspace(tumor_c2l[idx_des].min(), tumor_c2l[idx_des].max(), 100)
        ax.plot(x_fit, np.polyval(coef, x_fit), "-", color="black", lw=1.5, alpha=0.8)
        rho, _, n_v = spearman_safe(tumor_c2l[idx_des], tf_vals[idx_des])
        # Stats box → logueada para caption (descriptiva)
        logger.info(f"  MYC panel b (para caption): "
                    f"Spearman ρ = {rho:.3f} (Desert, n = {n_v:,}); "
                    "Tumour fraction alone d = 0.499; MYC TF-clean d = 0.347")
        ax.set_xlabel("Tumour cell abundance (Cell2Location)", fontsize=10)
        ax.set_ylabel("MYC TF activity (clean, z-score)", fontsize=10)
        ax.set_title("MYC TF score ≈ tumour fraction\n(intra-Desert confounding — not niche-specific)", fontsize=10, fontweight="bold", loc="left")
        style_ax(ax)
    else:
        missing = []
        if not has_tf: missing.append("MYC_TF_activity_clean")
        if tumor_c2l is None: missing.append("Tumor C2L")
        ax.text(0.5, 0.5, f"Not available:\n{', '.join(missing)}", ha="center", va="center", transform=ax.transAxes, color="red", fontsize=9)
        ax.axis("off")


    # Panel c
    ax = axes[2]
    panel_label(ax, "c")
    corr_data = {"ISG\n(contam.)": -0.505, "MHC-I\n(contam.)": -0.717, "ISG\n(cleaned)": +0.045, "MHC-I\n(cleaned)": -0.173}
    bar_colors = ["#CC0000", "#CC0000", "#2166AC", "#2166AC"]
    
    for i, (label, val) in enumerate(corr_data.items()):
        alpha, hatch = (0.50, "//") if val < -0.2 else (1.00, None)
        ax.bar(i, val, color=bar_colors[i], alpha=alpha, hatch=hatch, edgecolor="black", linewidth=0.8, width=0.65)
        yoff, va = (0.022, "bottom") if val >= 0 else (-0.04, "top")
        ax.text(i, val + yoff, f"ρ = {val:+.3f}", ha="center", va=va, fontsize=8.5)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(list(corr_data.keys()), fontsize=8.5)
    ax.set_ylabel("Spearman ρ with MYC TF score\n(intra-Desert)", fontsize=9)
    ax.set_ylim(-0.88, 0.34)
    leg_els = [mpatches.Patch(facecolor="#CC0000", alpha=0.5, hatch="//", edgecolor="black", label="Contaminated regulon"), mpatches.Patch(facecolor="#2166AC", alpha=1.0, edgecolor="black", label="Cleaned regulon")]
    ax.legend(handles=leg_els, fontsize=7.5, loc="lower right", frameon=True)
    ax.set_title("Sign reversal after removing read-out genes\n(ISG: ρ = −0.51 → +0.04; artefact explained)", fontsize=10, fontweight="bold", loc="left")
    style_ax(ax)

    # Padding ajustado para evitar clipping del suptitle
    fig.suptitle("MYC transcription-factor activity: negative result\n(d = +0.622 was a circular artefact; cleaned score reflects tumour fraction only)", fontsize=11, fontweight="bold", y=1.08)
    
    plt.tight_layout(pad=1.2)
    fig.subplots_adjust(top=0.90) 
    save_fig(fig, OUT_MAIN, "MYC_negative_result", logger)

# ══════════════════════════════════════════════════════════════════════════════
# TANDA 3 — SUPLEMENTARIAS INDIVIDUALES
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# FIGURAS NUEVAS v9 — SPATIAL MAPS PANEL + NICHE COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════

def _select_representative_sections(adata, logger):
    """\nSelecciona 3 secciones representativas:
      - highest_excluded: sección con mayor fracción de spots Excluded
      - highest_desert:   sección con mayor fracción de spots Desert
      - highest_inflamed: sección con mayor fracción de spots Inflamed
    Devuelve lista de (label, sample_id) en ese orden.
    Si alguna no tiene coordenadas espaciales, se omite.
    """
    if "sample_id" not in adata.obs.columns or "spatial" not in adata.obsm:
        return []

    targets = [
        ("Excluded-rich", "Immune_Excluded"),
        ("Desert-rich",   "Immune_Desert"),
        ("Inflamed-rich", "Inflamed"),
    ]
    selected = []
    used = set()

    for label, ph in targets:
        fracs = {}
        for sid in adata.obs["sample_id"].unique():
            m = adata.obs["sample_id"] == sid
            ph_counts = adata.obs.loc[m, "Phenotype"].value_counts(normalize=True)
            fracs[sid] = ph_counts.get(ph, 0.0)
        # pick highest not already selected
        ranked = sorted(fracs, key=fracs.get, reverse=True)
        for sid in ranked:
            if sid not in used and fracs[sid] > 0:
                selected.append((label, sid))
                used.add(sid)
                logger.info(f"    {label}: section {sid}  ({ph} frac = {fracs[sid]:.2f})")
                break

    return selected  # list of (label, sample_id)



def fig_spatial_maps_panel(adata, logger):
    """\nPanel multipanel (única excepción junto a MYC): 3 secciones × 4 columnas.
    Columnas: Phenotype | Tumor abundance | CD8_T abundance | CAF abundance
    Secciones: representativas de Excluded-rich, Desert-rich, Inflamed-rich.

    Reglas de seguridad:
      - Paleta de fenotipos: canónica v9 (Desert naranja, Excluded azul).
      - Labels de columna en inglés.
      - Heatmaps de abundancia con colormap secuencial neutro (no divergente).
      - Si una sección carece de coordenadas espaciales, se omite y se logea.
      - Si get_c2l devuelve None para un tipo celular, la celda queda en gris
        con nota "data not available" — no silencia el fallo.
    """
    logger.info("spatial_maps_panel — 3 sections × 4 columns")

    sections = _select_representative_sections(adata, logger)
    if not sections:
        skip_fig("spatial_maps_panel", "spatial coords or sample_id not available", logger)
        return

    n_rows = len(sections)  # 2 or 3
    n_cols = 4
    col_titles = ["Phenotype", "Tumour", "CD8⁺ T cell", "CAF"]
    cell_types  = [None, "Tumor", "CD8_T", "CAF"]  # None = phenotype map

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 4.0 * n_rows),
                             squeeze=False)
    fig.patch.set_facecolor("white")

    # Column headers
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct, fontsize=11, fontweight="bold", pad=6)

    for row_idx, (sec_label, sid) in enumerate(sections):
        mask_s  = adata.obs["sample_id"] == sid
        adata_s = adata[mask_s]
        coords  = adata_s.obsm["spatial"]
        phenos  = adata_s.obs["Phenotype"].values
        n_spots = mask_s.sum()

        # subsample for rasterization speed (max 8 000 spots per cell)
        if n_spots > 8000:
            idx_sub = RNG.choice(n_spots, 8000, replace=False)
        else:
            idx_sub = np.arange(n_spots)

        coords_sub = coords[idx_sub]
        phenos_sub = phenos[idx_sub]

        for col_idx, ct in enumerate(cell_types):
            ax = axes[row_idx, col_idx]
            ax.set_aspect("equal")
            ax.axis("off")

            if col_idx == 0:
                # Phenotype map — categorical colours
                for ph in PHENOTYPE_ORDER:
                    m = phenos_sub == ph
                    if m.sum() == 0:
                        continue
                    ax.scatter(coords_sub[m, 0], coords_sub[m, 1],
                               c=PHENOTYPE_COLORS.get(ph, "#888"),
                               s=2, alpha=0.80, rasterized=True,
                               label=PHENOTYPE_LABELS.get(ph, ph))
                # Row label on leftmost column
                ax.set_ylabel(sec_label, fontsize=9, labelpad=4)
                ax.yaxis.set_visible(True)
                ax.yaxis.label.set_visible(True)
                ax.tick_params(left=False, labelleft=False)
            else:
                # Abundance heatmap
                vals, col_used = get_c2l(adata_s, ct, logger)
                if vals is None:
                    ax.text(0.5, 0.5, ct + "\nnot available",
                            ha="center", va="center", transform=ax.transAxes,
                            fontsize=8, color="#888")
                    ax.set_facecolor("#F0F0F0")
                    logger.warning(f"    {ct} abundance not found for section {sid}")
                    continue

                vals_sub = vals[idx_sub]
                finite   = vals_sub[np.isfinite(vals_sub)]
                vmax     = np.percentile(finite, 97) if len(finite) > 10 else 1.0
                vmax     = max(vmax, 0.01)

                sc_kw = dict(
                    c=vals_sub, cmap="YlOrRd" if ct == "Tumor" else
                               "Blues"       if ct == "CD8_T"  else
                               "Oranges",
                    vmin=0, vmax=vmax, s=2, alpha=0.85, rasterized=True
                )
                scat = ax.scatter(coords_sub[:, 0], coords_sub[:, 1], **sc_kw)
                # small colorbar per cell
                cbar = fig.colorbar(scat, ax=ax, fraction=0.046, pad=0.04,
                                    orientation="vertical")
                cbar.ax.tick_params(labelsize=7)
                cbar.set_label("C2L abundance", fontsize=7)

    # Shared phenotype legend below the last Phenotype column
    handles = [
        mpatches.Patch(color=PHENOTYPE_COLORS[ph], label=PHENOTYPE_LABELS[ph])
        for ph in PHENOTYPE_ORDER if ph != "Ambiguous_Cold"
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=len(handles), fontsize=8.5, frameon=True,
               framealpha=0.9, bbox_to_anchor=(0.23, -0.02))

    fig.suptitle("Spatial architecture of representative TNBC sections",
                 fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout(pad=1.2, h_pad=0.8, w_pad=0.5)
    save_fig(fig, OUT_MAIN, "spatial_maps_panel", logger)



def fig_niche_composition(adata, logger):
    """\nComposición celular media por fenotipo (Desert, Excluded, Inflamed).
    Stacked bar: cada barra es la fracción relativa media de 5 tipos celulares.

    Tipos: Tumor, CAF, CD8_T, Macrophage, B_Cell
    (si alguno no está disponible, se omite de la barra y se logea).

    Seguridad:
      - No mezcla ningún score de MYC.
      - Usa get_c2l de v9 (C2L_KEY / C2L_PREFIX canónicos).
      - Si un fenotipo tiene <10 spots, se omite.
    """
    logger.info("niche_composition — stacked bar per phenotype")

    CELL_TYPES_ORDERED = ["Tumor", "CAF", "CD8_T", "Macrophage", "B_Cell"]
    CELL_COLORS = {
        "Tumor":      "#C0392B",
        "CAF":        "#E67E22",
        "CD8_T":      "#2980B9",
        "Macrophage": "#27AE60",
        "B_Cell":     "#8E44AD",
    }

    phenotypes = ["Immune_Desert", "Immune_Excluded", "Inflamed"]
    ph_labels  = [PHENOTYPE_LABELS[p] for p in phenotypes]

    # Build mean abundance per phenotype per cell type
    found_cts = []
    mat = {}
    for ct in CELL_TYPES_ORDERED:
        vals, col_used = get_c2l(adata, ct, logger)
        if vals is None:
            logger.warning(f"    {ct} not found in C2L obsm — skipped from composition")
            continue
        found_cts.append(ct)
        mat[ct] = vals

    if not found_cts:
        skip_fig("niche_composition", "no C2L cell-type abundances found", logger)
        return

    # Mean per phenotype
    means = {}
    for ph in phenotypes:
        mask = (adata.obs["Phenotype"] == ph).values
        n_ph = mask.sum()
        if n_ph < 10:
            logger.warning(f"    {ph}: only {n_ph} spots, skipped")
            means[ph] = {ct: 0.0 for ct in found_cts}
            continue
        row = {}
        for ct in found_cts:
            v = mat[ct][mask]
            row[ct] = float(np.nanmean(v[np.isfinite(v)])) if np.isfinite(v).sum() > 0 else 0.0
        total = sum(row.values()) or 1.0
        means[ph] = {ct: v / total * 100 for ct, v in row.items()}

    # Build stacked bars
    fig, ax = plt.subplots(figsize=(6, 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(phenotypes))
    bottoms = np.zeros(len(phenotypes))

    for ct in found_cts:
        heights = np.array([means[ph][ct] for ph in phenotypes])
        bars = ax.bar(x, heights, bottom=bottoms,
                      color=CELL_COLORS[ct], edgecolor="white",
                      linewidth=0.6, width=0.55, label=ct.replace("_", " "))
        # Label inside bar if segment > 8%
        for xi, (h, b) in enumerate(zip(heights, bottoms)):
            if h > 8:
                ax.text(xi, b + h / 2, f"{h:.0f}%",
                        ha="center", va="center", fontsize=7.5,
                        color="white", fontweight="bold")
        bottoms += heights

    ax.set_xticks(x)
    ax.set_xticklabels(ph_labels, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Mean estimated cell fraction (% of total C2L abundance)", fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend(bbox_to_anchor=(1.02, 0.5), loc="center left",
              fontsize=8.5, frameon=True, framealpha=0.9,
              title="Cell type", title_fontsize=9)
    style_ax(ax)
    ax.set_title("In situ cellular composition by immune phenotype",
                 fontsize=10, fontweight="bold", loc="left", pad=10)

    plt.tight_layout(pad=1.3)
    save_fig(fig, OUT_MAIN, "niche_composition", logger)



# ══════════════════════════════════════════════════════════════════════════════
# NUEVAS FIGURAS PRINCIPALES v10
# ══════════════════════════════════════════════════════════════════════════════

def _get_representative_section_for_gene(adata, phenotype_target, gene, logger,
                                          min_nonzero_frac=0.05):
    """
    Select the section richest in phenotype_target that also has
    min_nonzero_frac of spots expressing `gene` (raw > 0).
    Fallback: richest in phenotype_target regardless of gene expression.
    Returns sample_id or None.
    """
    if "sample_id" not in adata.obs.columns or "spatial" not in adata.obsm:
        return None
    gene_expr = get_gene(adata, gene, logger)
    candidates = {}
    for sid in adata.obs["sample_id"].unique():
        m = (adata.obs["sample_id"] == sid).values
        ph_frac = (adata.obs["Phenotype"].values[m] == phenotype_target).mean()
        if gene_expr is not None:
            nz_frac = (gene_expr[m] > 0).mean()
        else:
            nz_frac = 1.0
        candidates[sid] = (ph_frac, nz_frac)
    # prefer sections where both phenotype fraction is high and gene is expressed
    qualified = {sid: v for sid, v in candidates.items()
                 if v[1] >= min_nonzero_frac}
    pool = qualified if qualified else candidates
    best = max(pool, key=lambda s: pool[s][0])
    logger.info(f"    Section selected: {best} "
                f"(ph_frac={candidates[best][0]:.2f}, "
                f"nz_frac={candidates[best][1]:.2f})")
    return best


def fig_gene_expression_overlay(adata, logger):
    """
    Fig 3 — 2 rows × 3 columns:
      Row 0: Excluded-rich section  | Phenotype map | COL1A1 | CD8A
      Row 1: Desert-rich section    | Phenotype map | COL1A1 | CD8A

    Anti-overlap rules:
    - No text overlaid on scatter maps
    - Column headers only above row 0
    - Row labels on left margin (ylabel of col 0)
    - Colorbars external (right of each abundance panel)
    - vmax = percentile(expr[expr>0], 97)  (zeros excluded to avoid compression)
    - All annotations via ax.set_title / ax.set_ylabel, zero ax.text on data area

    Sources: adata.raw (genes), adata.obsm["spatial"] (coords),
             adata.obs["Phenotype"] (labels), adata.obs["sample_id"]
    """
    logger.info("gene_expression_overlay — 2×3 panel (phenotype+COL1A1+CD8A)")

    if "spatial" not in adata.obsm:
        skip_fig("gene_expression_overlay", "spatial coords not found", logger)
        return

    GENES = ["COL1A1", "CD8A"]
    CMAPS = ["Oranges", "Blues"]
    COL_LABELS = ["Phenotype", "COL1A1 expression\n(log1p)", "CD8A expression\n(log1p)"]
    ROW_TARGETS = ["Immune_Excluded", "Immune_Desert"]
    ROW_LABELS  = ["Excluded-rich section", "Desert-rich section"]

    # Select one section per row
    sections = []
    used = set()
    for ph_target in ROW_TARGETS:
        # prefer section rich in target phenotype AND expressing COL1A1
        sid = _get_representative_section_for_gene(
            adata, ph_target, "COL1A1", logger)
        if sid is None or sid in used:
            # fallback: any unused section
            for s in adata.obs["sample_id"].unique():
                if s not in used:
                    sid = s; break
        sections.append(sid)
        used.add(sid)
        logger.info(f"  Row ({ph_target}): section {sid}")

    if not sections:
        skip_fig("gene_expression_overlay", "no sections available", logger)
        return

    # Pre-fetch gene expression (from adata.raw)
    gene_arrays = {}
    for gene in GENES:
        arr = get_gene(adata, gene, logger)
        if arr is None:
            logger.warning(f"  {gene} not found in adata.raw — column will be blank")
        gene_arrays[gene] = arr

    n_rows = len(sections)
    n_cols  = 3

    # Pre-fetch per-row data once (mask, coords, subsample) so all three
    # render modes use the identical sample.
    row_data = []
    for sid in sections:
        mask = (adata.obs["sample_id"] == sid).values
        adata_s = adata[mask]
        coords  = adata_s.obsm["spatial"]
        phenos  = adata_s.obs["Phenotype"].values
        n_s     = mask.sum()
        sub_idx = (RNG.choice(n_s, min(n_s, 8000), replace=False)
                   if n_s > 8000 else np.arange(n_s))
        row_data.append(dict(mask=mask, coords=coords[sub_idx],
                              phenos=phenos[sub_idx], sub_idx=sub_idx))

    def _build(mode):
        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(4.5 * n_cols, 4.2 * n_rows),
                                  squeeze=False)
        fig.patch.set_facecolor("white")

        # Column headers (top row only) — structural, always shown
        for j, cl in enumerate(COL_LABELS):
            axes[0, j].set_title(cl, fontsize=11, fontweight="bold", pad=7)

        for row_idx, rd in enumerate(row_data):
            mask, coords_s, phenos_s, sub_idx = (
                rd["mask"], rd["coords"], rd["phenos"], rd["sub_idx"])

            for col_idx in range(n_cols):
                ax = axes[row_idx, col_idx]
                ax.set_aspect("equal")
                # In axes_only mode, turn the frame ON so panel boundaries
                # are visible (the real figure uses axis("off") for spatial
                # panels, which would otherwise show nothing at all here).
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
                            m = phenos_s == ph
                            if m.sum() == 0: continue
                            ax.scatter(coords_s[m, 0], coords_s[m, 1],
                                       c=PHENOTYPE_COLORS.get(ph, "#888"),
                                       s=2, alpha=0.80, rasterized=True)
                else:
                    gene = GENES[col_idx - 1]
                    cmap = CMAPS[col_idx - 1]
                    arr  = gene_arrays.get(gene)

                    if mode != "labels_only" and arr is not None:
                        vals_s = arr[mask][sub_idx].astype(float)
                        nonzero = vals_s[vals_s > 0]
                        vmax = float(np.percentile(nonzero, 97)) if len(nonzero) > 10 else 1.0
                        vmax = max(vmax, 0.01)
                        sc = ax.scatter(coords_s[:, 0], coords_s[:, 1],
                                        c=vals_s, cmap=cmap,
                                        vmin=0, vmax=vmax,
                                        s=2, alpha=0.85, rasterized=True)
                    elif mode != "labels_only" and arr is None:
                        ax.text(0.5, 0.5, f"{gene}\nnot available",
                                ha="center", va="center",
                                transform=ax.transAxes, fontsize=9, color="#888")
                        ax.set_facecolor("#F4F4F4")
                        sc = None
                    else:
                        sc = None

                    # Colorbar is a "label" element; build a dummy mappable
                    # in labels_only mode (vmax unknown -> placeholder 0..1)
                    if mode != "axes_only":
                        if sc is None:
                            sc = matplotlib.cm.ScalarMappable(
                                cmap=cmap, norm=matplotlib.colors.Normalize(0, 1))
                            sc.set_array([])
                        cbar = fig.colorbar(sc, ax=ax, fraction=0.046,
                                            pad=0.04, orientation="vertical")
                        cbar.ax.tick_params(labelsize=7)
                        cbar.set_label("log1p counts", fontsize=7)

        # Row labels: fig.text between rows (a "label" element)
        if mode != "axes_only":
            _row_ys = [0.75, 0.25] if n_rows == 2 else [
                1 - (i + 0.5) / n_rows for i in range(n_rows)]
            for _ri, _rl in enumerate(ROW_LABELS[:n_rows]):
                fig.text(0.01, _row_ys[_ri], _rl,
                         fontsize=10, fontweight="bold",
                         va="center", ha="left",
                         rotation=90,
                         transform=fig.transFigure)
            # Phenotype legend below row 0, col 0
            handles = [mpatches.Patch(color=PHENOTYPE_COLORS[ph],
                                       label=PHENOTYPE_LABELS.get(ph, ph))
                       for ph in PHENOTYPE_ORDER if ph in PHENOTYPE_COLORS]
            axes[0, 0].legend(handles=handles, loc="lower left",
                              bbox_to_anchor=(0, -0.12), fontsize=8,
                              frameon=True, framealpha=0.9, ncol=2)
            fig.suptitle("In situ expression of barrier and immune markers",
                         fontsize=12, fontweight="bold", y=1.01)

        plt.tight_layout(pad=1.2, h_pad=0.8, w_pad=0.6)
        return fig

    _panelcheck_variants(_build, OUT_MAIN, "gene_expression_overlay", logger)


def fig_c2l_correlation_matrix(adata, logger):
    """
    Fig 7b-d — Spearman correlation matrices of Cell2Location abundances,
    stratified by phenotype (Desert / Excluded / Inflamed), 3 sub-panels.

    Anti-overlap rules:
    - No text inside matrix cells (color only)
    - Axis labels rotated 45° on X, horizontal on Y; fontsize scaled to n_types
    - Diagonal set to NaN (excluded from colormap)
    - vmin=-1 / vmax=1 fixed for comparability across all three panels
    - Subsample max 10,000 spots (seed=42) for speed; result is consistent
    - No p-value asterisks (N>1000 → all significant; pattern is the argument)

    Sources: adata.obsm["means_cell_abundance_w_sf"],
             adata.obs["Phenotype"]
    """
    logger.info("c2l_correlation_matrix — 3 subpanels by phenotype")

    from scipy.stats import spearmanr as _spr

    # Build C2L abundance dataframe
    obsm_key = C2L_KEY  # "means_cell_abundance_w_sf"
    if obsm_key not in adata.obsm:
        skip_fig("c2l_correlation_matrix",
                 f"obsm key '{obsm_key}' not found", logger)
        return

    c2l_df = pd.DataFrame(
        adata.obsm[obsm_key],
        index=adata.obs_names,
        columns=[c.replace(C2L_PREFIX, "")
                 for c in adata.obsm[obsm_key].dtype.names]
        if hasattr(adata.obsm[obsm_key], 'dtype')
        else [f"ct_{i}" for i in range(adata.obsm[obsm_key].shape[1])]
    )

    # Attempt to get column names from the DataFrame if obsm is already a df
    if hasattr(adata.obsm[obsm_key], 'columns'):
        c2l_df = adata.obsm[obsm_key].copy()
        c2l_df.columns = [c.replace(C2L_PREFIX, "") for c in c2l_df.columns]
    elif hasattr(adata.obsm[obsm_key], 'dtype') and adata.obsm[obsm_key].dtype.names:
        cols = [n.replace(C2L_PREFIX, "") for n in adata.obsm[obsm_key].dtype.names]
        c2l_df = pd.DataFrame(adata.obsm[obsm_key].tolist(),
                               columns=cols, index=adata.obs_names)
    else:
        # numpy array — use obs column approach
        obs_cols = [c for c in adata.obs.columns if c.startswith(C2L_PREFIX)]
        if not obs_cols:
            skip_fig("c2l_correlation_matrix",
                     "C2L columns not found in obsm or obs", logger)
            return
        c2l_df = adata.obs[obs_cols].copy()
        c2l_df.columns = [c.replace(C2L_PREFIX, "") for c in obs_cols]

    # Filter to abundant cell types (mean abundance > 0.1 across all spots)
    abundant = [c for c in c2l_df.columns
                if c2l_df[c].mean() > 0.1]
    if len(abundant) < 3:
        abundant = list(c2l_df.columns[:10])
    c2l_df = c2l_df[abundant]
    logger.info(f"  Cell types included: {abundant}")

    phenotypes = ["Immune_Desert", "Immune_Excluded", "Inflamed"]
    titles     = ["a  Immune Desert", "b  Immune Excluded", "c  Inflamed"]

    fig, axes = plt.subplots(1, 3,
                              figsize=(5.5 * 3, 5.5),
                              squeeze=False)
    fig.patch.set_facecolor("white")

    for col_idx, (ph, title) in enumerate(zip(phenotypes, titles)):
        ax = axes[0, col_idx]
        mask = (adata.obs["Phenotype"] == ph).values
        n_ph = mask.sum()

        if n_ph < 30:
            ax.text(0.5, 0.5, f"Insufficient spots\n(n={n_ph})",
                    transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title, fontsize=10, fontweight="bold", loc="left")
            continue

        # Subsample
        if n_ph > 10000:
            rng_sub = np.random.default_rng(42)
            sub = rng_sub.choice(np.where(mask)[0], 10000, replace=False)
            df_ph = c2l_df.iloc[sub]
        else:
            df_ph = c2l_df.loc[mask]

        # Spearman correlation matrix
        n_ct = len(abundant)
        corr = np.zeros((n_ct, n_ct))
        for i in range(n_ct):
            for j in range(n_ct):
                if i == j:
                    corr[i, j] = np.nan  # diagonal excluded
                    continue
                rho, _ = _spr(df_ph.iloc[:, i], df_ph.iloc[:, j])
                corr[i, j] = rho

        # Plot heatmap
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1,
                       aspect="auto")

        # Axis labels — fontsize scaled to label count
        fs = max(6, 9 - max(0, n_ct - 8))
        ct_labels = [c.replace("_", " ") for c in abundant]
        ax.set_xticks(range(n_ct))
        ax.set_xticklabels(ct_labels, rotation=45, ha="right", fontsize=fs)
        ax.set_yticks(range(n_ct))
        ax.set_yticklabels(ct_labels, fontsize=fs)

        # Colorbar on rightmost panel only
        if col_idx == 2:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("Spearman ρ", fontsize=9)
            cbar.ax.tick_params(labelsize=8)

        # Panel info in upper-left corner (transAxes — no overlap with heatmap)
        ax.text(0.02, 0.98, f"n = {n_ph:,} spots",
                transform=ax.transAxes, va="top", ha="left",
                fontsize=8, color="#444",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="none", alpha=0.85))

        ax.set_title(title, fontsize=10, fontweight="bold",
                     loc="left", pad=8)
        panel_label(ax, "abcd"[col_idx])

    fig.suptitle(
        "Cell-type co-localization (Spearman ρ) by phenotype",
        fontsize=11, fontweight="bold", y=1.02
    )
    plt.tight_layout(pad=1.5, w_pad=1.0)
    save_fig(fig, OUT_MAIN, "c2l_correlation_matrix", logger)


def fig_marker_dotplot(adata, logger):
    """
    Dot plot of canonical marker genes across phenotypes.
    dot size  = fraction of spots expressing the gene (raw > 0)
    dot color = mean log1p expression (from adata.raw)

    Anti-overlap strategy (manual matplotlib, not sc.pl.dotplot):
    - Gene labels on Y axis, fontsize=8.5 (fixed for 16 genes)
    - Phenotype labels on X axis, rotated 0°
    - Colorbar external (right side), no text on dots
    - Group separator lines between gene groups (hairlines)
    - No asterisks inside the grid

    Source: adata.raw, adata.obs["Phenotype"]
    """
    logger.info("marker_dotplot — 16 genes × 3 phenotypes")

    GENE_GROUPS = {
        "CAF / barrier":  ["FAP", "COL1A1", "ACTA2", "POSTN", "TGFB1"],
        "Immune":         ["CD8A", "GZMB", "CCL5", "CXCL9"],
        "Checkpoint":     ["PDCD1", "HAVCR2", "TIGIT", "LAG3", "CD47"],
        "Tumour":         ["EPCAM", "MKI67"],
    }
    ALL_GENES = [g for grp in GENE_GROUPS.values() for g in grp]
    PHENOTYPES = ["Immune_Desert", "Immune_Excluded", "Inflamed"]
    PH_LABELS  = ["Desert", "Excluded", "Inflamed"]

    # Check which genes are available in adata.raw
    if adata.raw is None:
        skip_fig("marker_dotplot", "adata.raw is None", logger)
        return
    avail_genes = set(adata.raw.var_names)
    missing = [g for g in ALL_GENES if g not in avail_genes]
    if missing:
        logger.warning(f"  Genes not in adata.raw: {missing}; "
                       "replacing MKI67→MYC if absent")
        if "MKI67" in missing and "MYC" in avail_genes:
            GENE_GROUPS["Tumour"] = ["EPCAM", "MYC"]
            missing = [g for g in missing if g != "MKI67"]
    genes_to_plot = [g for g in ALL_GENES if g in avail_genes]
    if len(genes_to_plot) < 4:
        skip_fig("marker_dotplot",
                 f"too few genes found in adata.raw ({genes_to_plot})", logger)
        return

    # Compute mean expression and % expressing per gene per phenotype
    expr_mat  = np.zeros((len(genes_to_plot), len(PHENOTYPES)))
    frac_mat  = np.zeros((len(genes_to_plot), len(PHENOTYPES)))

    gene_idx_raw = {g: list(adata.raw.var_names).index(g)
                    for g in genes_to_plot}

    for ph_i, ph in enumerate(PHENOTYPES):
        mask = (adata.obs["Phenotype"] == ph).values
        if mask.sum() == 0:
            continue
        raw_ph = adata.raw.X[mask, :]
        # Convert to dense if sparse
        if hasattr(raw_ph, "toarray"):
            raw_ph = raw_ph.toarray()
        for g_i, gene in enumerate(genes_to_plot):
            col = gene_idx_raw[gene]
            vals = raw_ph[:, col].astype(float)
            expr_mat[g_i, ph_i] = float(np.mean(vals))
            frac_mat[g_i, ph_i] = float((vals > 0).mean())

    n_genes = len(genes_to_plot)
    n_ph    = len(PHENOTYPES)
    fig, ax = plt.subplots(figsize=(5.5, max(6.5, 0.42 * n_genes + 2.5)))
    fig.patch.set_facecolor("white")

    # Max dot size = 400 (full circle at 100%)
    max_dot = 400
    xx, yy = np.meshgrid(np.arange(n_ph), np.arange(n_genes))

    sc = ax.scatter(
        xx.ravel(), yy.ravel(),
        s     = frac_mat.ravel() * max_dot,
        c     = expr_mat.ravel(),
        cmap  = "YlOrRd",
        vmin  = 0,
        vmax  = float(np.percentile(expr_mat[expr_mat > 0], 95))
                if (expr_mat > 0).any() else 1.0,
        edgecolors = "#555", linewidths = 0.3,
        zorder = 3
    )

    # Group separator lines (thin, below gene group boundaries)
    group_sizes = [len(v) for v in GENE_GROUPS.values()]
    sep = 0
    for gs in group_sizes[:-1]:
        sep += gs
        ax.axhline(sep - 0.5, color="#CCCCCC", lw=0.8, zorder=1)

    # Group labels removed from figure body (were extending the right margin
    # and pushing the plotted data toward the left of the canvas).
    # Gene groups, in row order top-to-bottom, for the figure caption:
    logger.info("  marker_dotplot gene groups (for caption, top-to-bottom): "
                 + "; ".join(f"{g} ({', '.join(genes)})"
                              for g, genes in GENE_GROUPS.items()))

    ax.set_xticks(range(n_ph))
    ax.set_xticklabels(PH_LABELS, fontsize=10)
    ax.set_yticks(range(n_genes))
    ax.set_yticklabels(genes_to_plot, fontsize=8.5, fontstyle="italic")
    ax.set_xlim(-0.8, n_ph - 0.2)
    ax.set_ylim(-0.6, n_genes - 0.4)
    ax.invert_yaxis()  # top gene = index 0

    # Colorbars — external, no overlap
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.18,
                        orientation="vertical")
    cbar.set_label("Mean log1p expression", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Dot size legend — placed below plot
    for frac, label in [(0.25, "25%"), (0.5, "50%"), (1.0, "100%")]:
        ax.scatter([], [], s=frac * max_dot, c="#888",
                   edgecolors="#555", linewidths=0.3,
                   label=f"{label} spots expressing")
    ax.legend(title="% spots expressing", title_fontsize=8,
              fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.16),
              frameon=True, framealpha=0.95, edgecolor="#CCCCCC",
              ncol=3, handlelength=1.4, columnspacing=1.4,
              handletextpad=0.6, borderpad=0.8)

    style_ax(ax)
    ax.set_title("Canonical marker gene expression by phenotype",
                 fontsize=10, fontweight="bold", loc="left", pad=10)
    plt.tight_layout(pad=1.5)
    save_fig(fig, OUT_MAIN, "marker_dotplot", logger)

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTOR — MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="TNBC publication figures — MAIN")
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
        # ── Classification ──────────────────────────────────────────────────
        ("phenotype_classification_map",  lambda: fig_phenotype_classification_map(adata, logger)),
        ("phenotype_classification_bars", lambda: fig_phenotype_classification_bars(logger)),
        # ── Spatial architecture ────────────────────────────────────────────
        ("spatial_maps_panel",            lambda: fig_spatial_maps_panel(adata, logger)),
        ("gene_expression_overlay",       lambda: fig_gene_expression_overlay(adata, logger)),
        # ── CAF barrier ─────────────────────────────────────────────────────
        ("CAF_gradient",                  lambda: fig_CAF_gradient(adata, logger)),
        ("CAF_estimators_forest",         lambda: fig_CAF_estimators_forest(logger)),
        # ── Landscape ───────────────────────────────────────────────────────
        ("niche_composition",             lambda: fig_niche_composition(adata, logger)),
        ("checkpoint_landscape",          lambda: fig_checkpoint_landscape(logger)),
        ("chemotaxis_correlation",        lambda: fig_chemotaxis_correlation(logger)),
        # ── MYC negative ────────────────────────────────────────────────────
        ("MYC_negative_result",           lambda: fig_MYC_negative_result(adata, logger)),
        # ── Bulk recoverability + correlation ───────────────────────────────
        ("bulk_transferability",          lambda: fig_bulk_transferability(logger)),
        ("c2l_correlation_matrix",        lambda: fig_c2l_correlation_matrix(adata, logger)),
        # ── Marker dot plot (supporting) ─────────────────────────────────
        ("marker_dotplot",                lambda: fig_marker_dotplot(adata, logger)),
    ]

    # Filter if --figures was specified
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
    # Write manifest
    import csv as _csv
    mf = OUT_BASE / "manifest_figures_main.csv"
    with open(mf, "w", newline="") as _f:
        w = _csv.DictWriter(_f, fieldnames=["figura", "archivo", "status", "razon"])
        w.writeheader(); w.writerows(MANIFEST)
    logger.info(f"  Manifest: {mf}")


if __name__ == "__main__":
    main()

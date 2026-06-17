#!/usr/bin/env python3

import sys as _sys
import types as _types

def _make_torch_stub():
    """Crea módulo torch mínimo que satisface anndata sin cargar .so."""
    stub = _types.ModuleType("torch")
    stub.__version__ = "stub"
    stub.__spec__ = None
    utils = _types.ModuleType("torch.utils")
    utils_data = _types.ModuleType("torch.utils.data")
    class _FakeDataLoader:
        pass
    utils_data.DataLoader = _FakeDataLoader
    utils_data.Dataset = object
    utils.data = utils_data
    stub.utils = utils
    _sys.modules["torch"] = stub
    _sys.modules["torch.utils"] = utils
    _sys.modules["torch.utils.data"] = utils_data

try:
    import torch as _t  # noqa
except OSError:
    _make_torch_stub()
# ─────────────────────────────────────────────────────────────────────────────

"""
fix_myc_tf_proliferation_confound.py
================================================================================
TEST: ¿El d=+0.347 (MYC_TF_clean Desert vs Inflamed) refleja actividad MYC
      genuina o simplemente fracción tumoral/proliferación?

PREGUNTA CIENTÍFICA
-------------------
Con el regulón limpio (53 genes positivos, solo proliferación/metabolismo MYC),
el score midió Desert > Inflamed (d=+0.347). Pero el regulón está dominado por
genes de ciclo celular (CDK, E2F, MCM, TYMS, PCNA, MKI67). Desert, por
definición, contiene más tejido tumoral que Inflamed (que está infiltrado de
linfocitos). Entonces el +0.347 puede ser:

  Hipótesis A (confounding): MYC_TF_clean ≈ fracción tumoral / proliferación
    Predicción: ρ(MYC_TF_clean, tumor_marker) alto intra-Desert
                MYC_TF_clean vs Inflamed: similar en intensidad a
                tumor_marker vs Inflamed

  Hipótesis B (señal MYC genuina): MYC_TF_clean mide algo sobre MYC más allá
    de la fracción tumoral
    Predicción: ρ(MYC_TF_clean, tumor_marker) bajo intra-Desert
                ρ(MYC_TF_clean, tumor_marker) << d diferencia entre fenotipos

DISEÑO LIBRE DE CIRCULARIDAD
------------------------------
Variable independiente: MYC_TF_activity_clean
  → Calculada sobre 53 genes de proliferación/metabolismo MYC
  → Cargada desde adata_with_myc_tf_clean.h5ad (generado por el wrapper)

Marcadores tumorales/proliferación INDEPENDIENTES (0 genes en regulón limpio):
  Epiteliales: EPCAM, KRT18, KRT19        (marcadores epiteliales, no en regulón)
  Proliferación: TOP2A                     (G2/M, no en regulón)
  Score combinado: TUMOR_SCORE_4G = mean(EPCAM, KRT18, KRT19, TOP2A)
  
  EXCLUIDO: MKI67 — está en el regulón limpio (peso +1). Usarlo como
  control crearía una correlación artefactual.

  Tumor_C2L: meanscell_abundance_w_sf_Tumor (deconvolución, completamente
  independiente de expresión génica).
  → Si la columna no existe en obsm, se usa TUMOR_SCORE_4G como fallback.

Controles de especificidad (Tests 2 y 3):
  Test 2: misma correlación intra-Inflamed. Si ρ_Desert ≈ ρ_Inflamed → efecto
          es composición general, no Desert-específico.
  Test 3: ρ(MYC_Hallmark_Combined, tumor_marker) intra-Desert. Si
          ρ_Hallmark ≈ ρ_TF_clean → ambos miden proliferación, no MYC.
          Si ρ_TF_clean >> ρ_Hallmark → TF_clean tiene señal MYC adicional.

SOLAPAMIENTOS VERIFICADOS
--------------------------
MYC_TF_clean ∩ EPCAM/KRT18/KRT19/TOP2A = ∅  (cero solapamiento)
MYC_TF_clean ∩ SILENCING_REPRESSORS = {MYC}  (1 gen, autoregulación; no afecta
                                               tests de correlación vs Tumor)
MYC_TF_clean ∩ PHYSICAL_BARRIER     = ∅
MYC_TF_clean ∩ CD8_T_CELLS          = ∅
MYC_TF_clean ∩ ISG_score            = ∅
MYC_TF_clean ∩ MHC_I_score          = ∅

INPUT
------
  data/processed/adata_with_myc_tf_clean.h5ad
  (generado por fix_myc_tf_clean_regulon_wrapper.py)
  Fallback: adata_with_mechanism.h5ad + recalculo de MYC_TF_activity_clean

OUTPUT
------
  results/myc_tf_activity/
    myc_tf_proliferation_confound.csv     (todos los tests)
    myc_tf_proliferation_confound.json    (resumen interpretable)
    fig_myc_tf_vs_tumor_confound.png      (scatter + barras)

EJECUCIÓN
----------
  cd /home/external/frjimenez/fabian/genoma/codigos
  python fix_myc_tf_proliferation_confound.py \
    2>&1 | tee ../results/log_myc_tf_proliferation_confound.txt

================================================================================
"""

import logging
import sys
import time
import warnings
from pathlib import Path

import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

import scanpy as sc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR    = Path(__file__).resolve().parent
_BASE          = _SCRIPT_DIR.parent
DATA_PROCESSED = _BASE / "data" / "processed"
RESULTS_DIR    = _BASE / "results" / "myc_tf_activity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADATA_CLEAN   = DATA_PROCESSED / "adata_with_myc_tf_clean.h5ad"
ADATA_FALLBACK = DATA_PROCESSED / "adata_with_mechanism.h5ad"

SEED = 42
np.random.seed(SEED)

# ── Constantes del pipeline ───────────────────────────────────────────────────
PHENOTYPE_COL  = "Phenotype"
DESERT_LABEL   = "Immune_Desert"
EXCLUDED_LABEL = "Immune_Excluded"
INFLAMED_LABEL = "Inflamed"

# Genes de tumor/proliferación INDEPENDIENTES del regulón limpio.
# Verificado: ninguno de estos está en MYC_COLLECTRI_CLEAN_POSITIVE.
# MKI67 está EXCLUIDO porque sí está en el regulón limpio.
TUMOR_MARKERS_INDEPENDENT = ["EPCAM", "KRT18", "KRT19", "TOP2A"]

# Regulón limpio — para verificar solapamiento en runtime
MYC_TF_CLEAN_GENES = {
    "CCNA2","CCNB1","CCNB2","CCND1","CCNE1",
    "CDK1","CDK2","CDK4","CDK6",
    "E2F1","E2F2","E2F3",
    "MKI67","PCNA","MCM2","MCM4","MCM5","MCM6",
    "TYMS","DHFR","DHODH","TK1","CAD","UMPS",
    "NPM1","NCL","FBL",
    "EIF4E","EIF4A1","EIF2S1",
    "RPL5","RPL11","RPL13","RPS6","RPS14",
    "LDHA","GLS","SLC7A5","SLC1A5",
    "PRDX1","NME1","NME2",
    "ODC1","SRM",
    "MDM2","MAX","MYC","TERT","VEGFA","HIF1A",
    "HSPA4","HSP90AA1","CD47",
}

# Obsm key Cell2Location (confirmado en logs del pipeline)
C2L_OBSM_KEY = "means_cell_abundance_w_sf"
# Prefijo de columnas dentro del DataFrame de obsm (confirmado en logs)
C2L_COL_PREFIX = "meanscell_abundance_w_sf_"


# ── Setup logging ─────────────────────────────────────────────────────────────
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(
                RESULTS_DIR / "myc_tf_proliferation_confound.log", mode="w"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("prolif_confound")


# ── Helpers de expresión ──────────────────────────────────────────────────────
def safe_toarray(X):
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


def get_gene_score(adata, genes, score_name, logger):
    """
    Score = media de expresión de genes disponibles en .raw.
    No usa scanpy.tl.score_genes para evitar dependencia de referencia génica.
    Devuelve array de longitud n_obs con NaN donde todos los genes faltan.
    """
    if adata.raw is not None:
        X = adata.raw.X
        var_names = list(adata.raw.var_names)
    else:
        X = adata.X
        var_names = list(adata.var_names)

    available = [g for g in genes if g in var_names]
    missing   = [g for g in genes if g not in var_names]

    if missing:
        logger.warning(f"  [{score_name}] ausentes: {missing}")
    if not available:
        logger.error(f"  [{score_name}] 0/{len(genes)} genes — score=NaN")
        return np.full(adata.n_obs, np.nan)

    logger.info(f"  [{score_name}] {len(available)}/{len(genes)} genes: {available}")
    vecs = []
    for g in available:
        idx = var_names.index(g)
        col = X[:, idx]
        vecs.append(safe_toarray(col).ravel())

    return np.mean(np.stack(vecs, axis=1), axis=1)


def get_c2l_column(adata, cell_type, logger):
    """
    Busca columna de abundancia Cell2Location para un tipo celular.
    Patrón exacto: obsm[C2L_OBSM_KEY][C2L_COL_PREFIX + cell_type]
    Devuelve (array, nombre_columna) o (None, None) si no existe.
    """
    if C2L_OBSM_KEY not in adata.obsm:
        logger.warning(f"  obsm['{C2L_OBSM_KEY}'] no encontrado")
        return None, None

    c2l = adata.obsm[C2L_OBSM_KEY]
    if not isinstance(c2l, pd.DataFrame):
        c2l = pd.DataFrame(c2l, index=adata.obs_names)

    # Buscar columna exacta primero
    exact = C2L_COL_PREFIX + cell_type
    if exact in c2l.columns:
        logger.info(f"  C2L {cell_type}: columna '{exact}'")
        return c2l[exact].values.astype(float), exact

    # Búsqueda por substring (case-sensitive)
    candidates = [c for c in c2l.columns if cell_type in c]
    if candidates:
        col = candidates[0]
        logger.info(
            f"  C2L {cell_type}: columna '{col}' (búsqueda substring)"
        )
        return c2l[col].values.astype(float), col

    logger.warning(
        f"  C2L {cell_type}: no encontrada en obsm. "
        f"Columnas disponibles: {list(c2l.columns)}"
    )
    return None, None


def spearman_safe(x, y, min_n=50):
    """
    Correlación de Spearman con filtrado de NaN/Inf y mínimo de n.
    Retorna (rho, pval, n_valid) o (nan, nan, 0) si insuficiente.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    n = valid.sum()
    if n < min_n:
        return np.nan, np.nan, int(n)
    rho, pval = spearmanr(x[valid], y[valid])
    return float(rho), float(pval), int(n)


def cohens_d_pooled(g1, g2):
    """Cohen's d pooled, ddof=1. Canónico del pipeline."""
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var_pool = (
        ((n1 - 1) * np.var(g1, ddof=1) + (n2 - 1) * np.var(g2, ddof=1))
        / (n1 + n2 - 2)
    )
    if var_pool < 1e-15:
        return 0.0
    return float((g1.mean() - g2.mean()) / np.sqrt(var_pool))


# ── Auditoría de solapamiento ──────────────────────────────────────────────────
def audit_independence(regulon_genes, marker_genes, name, logger):
    """
    Verifica que los marcadores tumorales son independientes del regulón.
    Aborta si hay solapamiento — previene circularidad silenciosa.
    """
    overlap = regulon_genes & set(marker_genes)
    if overlap:
        logger.error(
            f"  CIRCULARIDAD DETECTADA: {name} comparte genes con regulón limpio:\n"
            f"  {sorted(overlap)}\n"
            f"  Estos genes deben eliminarse de TUMOR_MARKERS_INDEPENDENT."
        )
        sys.exit(1)
    logger.info(
        f"  Auditoría [{name}]: 0 solapamientos con regulón — LIMPIO"
    )


# ── Carga de datos ────────────────────────────────────────────────────────────
def load_adata(logger):
    """
    Carga preferentemente adata_with_myc_tf_clean.h5ad (tiene MYC_TF_activity_clean).
    Si no existe, carga adata_with_mechanism.h5ad y recalcula MYC_TF_activity_clean
    usando el regulón limpio del fix anterior.
    """
    if ADATA_CLEAN.exists():
        logger.info(f"  Cargando {ADATA_CLEAN.name}...")
        adata = sc.read_h5ad(ADATA_CLEAN)
        if "MYC_TF_activity_clean" not in adata.obs.columns:
            logger.error(
                "  'MYC_TF_activity_clean' no está en adata.obs.\n"
                "  Ejecutar fix_myc_tf_clean_regulon_wrapper.py primero."
            )
            sys.exit(1)
        logger.info("  MYC_TF_activity_clean: presente en adata.obs ✓")
        return adata

    # Fallback: recalcular
    logger.warning(
        f"  {ADATA_CLEAN.name} no encontrado — "
        f"cargando {ADATA_FALLBACK.name} y recalculando MYC_TF_activity_clean"
    )
    if not ADATA_FALLBACK.exists():
        logger.error(f"  FATAL: tampoco existe {ADATA_FALLBACK}")
        sys.exit(1)

    adata = sc.read_h5ad(ADATA_FALLBACK)

    # Importar el wrapper y recalcular
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        from fix_myc_tf_clean_regulon import (
            MYC_COLLECTRI_CLEAN_POSITIVE,
            audit_regulon_contamination,
        )
        from fix_myc_tf_clean_regulon_wrapper import (
            compute_myc_tf_activity_clean,
            get_expression_matrix_raw,
        )
        import pandas as _pd

        net = _pd.DataFrame([
            {"source": "MYC", "target": g, "weight": float(w), "mor": float(w)}
            for g, w in MYC_COLLECTRI_CLEAN_POSITIVE.items()
            if g in (list(adata.raw.var_names) if adata.raw else list(adata.var_names))
        ])
        audit = audit_regulon_contamination(net)
        if audit["is_circular"]:
            logger.error(f"  Regulón fallback circular: {audit['verdict']}")
            sys.exit(1)
        logger.info(f"  Regulón fallback: {len(net)} genes — {audit['verdict']}")

        scores = compute_myc_tf_activity_clean(adata, net)
        adata.obs["MYC_TF_activity_clean"] = scores
        logger.info("  MYC_TF_activity_clean recalculado OK")
    except ImportError as e:
        logger.error(
            f"  No se pudo importar el wrapper para recalcular: {e}\n"
            f"  Asegurar que fix_myc_tf_clean_regulon.py y "
            f"fix_myc_tf_clean_regulon_wrapper.py están en el mismo directorio."
        )
        sys.exit(1)

    return adata


# ── Tests principales ─────────────────────────────────────────────────────────

def test_proliferation_confound(adata, logger):
    """
    Ejecuta los 3 tests de confounding proliferativo.
    Retorna lista de dicts con resultados.
    """
    results = []
    phenos_arr = adata.obs[PHENOTYPE_COL].values

    # ── Pre-auditoría en runtime ──────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("PRE-AUDITORÍA: solapamiento regulón vs marcadores")
    audit_independence(
        MYC_TF_CLEAN_GENES,
        TUMOR_MARKERS_INDEPENDENT,
        "TUMOR_MARKERS_INDEPENDENT (EPCAM/KRT18/KRT19/TOP2A)",
        logger,
    )
    # MYC está en el regulón; anotarlo sin abortar (no afecta el test)
    logger.info(
        "  Nota: MYC ∈ regulón limpio (autoregulación, peso +1) y en "
        "SILENCING_REPRESSORS. No afecta los tests de correlación vs Tumor "
        "porque MYC no es un marcador tumoral morfológico."
    )

    # ── Construir marcador de tumor independiente ─────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("CONSTRUYENDO MARCADORES TUMORALES INDEPENDIENTES")

    # Intentar Tumor_C2L primero
    tumor_c2l, tumor_c2l_col = get_c2l_column(adata, "Tumor", logger)
    if tumor_c2l is None:
        # Intentar "Normal_Epithelial" (aparece en log_12 como proxy tumoral)
        tumor_c2l, tumor_c2l_col = get_c2l_column(adata, "Normal_Epithelial", logger)

    if tumor_c2l is not None:
        logger.info(
            f"  Tumor_C2L disponible ({tumor_c2l_col}): "
            f"mean={np.nanmean(tumor_c2l):.4f}, "
            f"n_nonzero={np.sum(tumor_c2l > 0):,}"
        )
    else:
        logger.warning(
            "  Tumor_C2L no disponible. "
            "Usando TUMOR_SCORE_4G (EPCAM+KRT18+KRT19+TOP2A) como proxy."
        )

    # Score de expresión génica tumoral (independiente del regulón)
    tumor_expr = get_gene_score(
        adata, TUMOR_MARKERS_INDEPENDENT, "TUMOR_SCORE_4G", logger
    )

    # Si C2L disponible, comparar para validación cruzada interna
    if tumor_c2l is not None:
        rho_c2l_expr, pval_c2l_expr, n_c2l = spearman_safe(tumor_c2l, tumor_expr)
        logger.info(
            f"  Correlación Tumor_C2L vs TUMOR_SCORE_4G (global): "
            f"ρ={rho_c2l_expr:.3f}, p={pval_c2l_expr:.2e}, n={n_c2l:,}"
        )
        if rho_c2l_expr < 0.2:
            logger.warning(
                "  ρ < 0.2 entre C2L y expresión génica. "
                "Posiblemente 'Tumor' en C2L no corresponde a células epiteliales. "
                "Usando TUMOR_SCORE_4G como medida primaria."
            )

    # ── Elegir medida primaria ────────────────────────────────────────────────
    # Preferir C2L si está disponible y correlaciona bien con expresión
    use_c2l = (
        tumor_c2l is not None
        and not np.isnan(rho_c2l_expr if tumor_c2l is not None else np.nan)
        and rho_c2l_expr > 0.2
    ) if tumor_c2l is not None else False

    tumor_primary = tumor_c2l if use_c2l else tumor_expr
    tumor_primary_name = tumor_c2l_col if use_c2l else "TUMOR_SCORE_4G"
    logger.info(f"  Medida primaria: {tumor_primary_name}")

    # ── MYC_TF_activity_clean ─────────────────────────────────────────────────
    tf_clean = adata.obs["MYC_TF_activity_clean"].values

    # ── MYC_Hallmark_Combined (control Test 3) ────────────────────────────────
    hallmark = None
    hallmark_col = None
    if "MYC_Hallmark_Combined" in adata.obs.columns:
        hallmark = adata.obs["MYC_Hallmark_Combined"].values
        hallmark_col = "MYC_Hallmark_Combined"
        logger.info("  MYC_Hallmark_Combined: presente en adata.obs ✓")
    else:
        logger.warning(
            "  MYC_Hallmark_Combined no en adata.obs — "
            "Test 3 se omitirá (requiere ejecutar mechanism_additions.py primero)"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: ρ(MYC_TF_clean, tumor_primary) intra-Desert
    # Hipótesis confounding: ρ alto → d=+0.347 es fracción tumoral
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("TEST 1: ρ(MYC_TF_clean, Tumor) intra-Desert")
    logger.info(f"  Variable tumor: {tumor_primary_name}")

    desert_mask = phenos_arr == DESERT_LABEL
    rho1, pval1, n1 = spearman_safe(
        tf_clean[desert_mask], tumor_primary[desert_mask]
    )
    logger.info(
        f"  ρ = {rho1:.4f}, p = {pval1:.3e}, n = {n1:,}"
    )

    # Interpretación cuantitativa
    if np.isfinite(rho1):
        if abs(rho1) >= 0.5:
            interp1 = "CONFOUNDING FUERTE: MYC_TF_clean correlaciona altamente con fracción tumoral"
        elif abs(rho1) >= 0.3:
            interp1 = "CONFOUNDING MODERADO: correlación apreciable con fracción tumoral"
        elif abs(rho1) >= 0.1:
            interp1 = "CONFOUNDING DÉBIL: correlación pequeña pero presente"
        else:
            interp1 = "SIN CONFOUNDING: MYC_TF_clean es independiente de fracción tumoral intra-Desert"
    else:
        interp1 = "DATOS INSUFICIENTES"

    logger.info(f"  → {interp1}")

    results.append({
        "test": "T1_rho_TFclean_vs_Tumor_intraDesert",
        "variable_x": "MYC_TF_activity_clean",
        "variable_y": tumor_primary_name,
        "phenotype": DESERT_LABEL,
        "spearman_rho": rho1,
        "p_value": pval1,
        "n": n1,
        "interpretation": interp1,
        "circular": False,
    })

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: ρ(MYC_TF_clean, tumor_primary) intra-Inflamed
    # Control: si ρ_Desert ≈ ρ_Inflamed → efecto es composición general
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: ρ(MYC_TF_clean, Tumor) intra-Inflamed [control]")

    inflamed_mask = phenos_arr == INFLAMED_LABEL
    rho2, pval2, n2 = spearman_safe(
        tf_clean[inflamed_mask], tumor_primary[inflamed_mask]
    )
    logger.info(
        f"  ρ = {rho2:.4f}, p = {pval2:.3e}, n = {n2:,}"
    )

    # Diferencia entre fenotipos
    delta_rho = rho1 - rho2 if (np.isfinite(rho1) and np.isfinite(rho2)) else np.nan
    if np.isfinite(delta_rho):
        if abs(delta_rho) < 0.1:
            interp2 = "ρ_Desert ≈ ρ_Inflamed: la correlación TF-Tumor no es Desert-específica"
        else:
            interp2 = (
                f"ρ_Desert − ρ_Inflamed = {delta_rho:+.3f}: "
                "la correlación TF-Tumor es diferente entre fenotipos"
            )
    else:
        interp2 = "DATOS INSUFICIENTES"

    logger.info(f"  Δρ (Desert − Inflamed) = {delta_rho:+.4f}")
    logger.info(f"  → {interp2}")

    results.append({
        "test": "T2_rho_TFclean_vs_Tumor_intraInflamed",
        "variable_x": "MYC_TF_activity_clean",
        "variable_y": tumor_primary_name,
        "phenotype": INFLAMED_LABEL,
        "spearman_rho": rho2,
        "p_value": pval2,
        "n": n2,
        "interpretation": interp2,
        "circular": False,
        "delta_rho_vs_desert": delta_rho,
    })

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: ρ(MYC_Hallmark, tumor_primary) intra-Desert
    # Control: si ρ_Hallmark ≈ ρ_TF_clean → ambos miden proliferación
    # ═══════════════════════════════════════════════════════════════════════
    if hallmark is not None:
        logger.info("\n" + "=" * 70)
        logger.info("TEST 3: ρ(MYC_Hallmark_Combined, Tumor) intra-Desert [control MYC]")

        rho3, pval3, n3 = spearman_safe(
            hallmark[desert_mask], tumor_primary[desert_mask]
        )
        logger.info(
            f"  ρ = {rho3:.4f}, p = {pval3:.3e}, n = {n3:,}"
        )

        delta_rho_hallmark = (
            rho1 - rho3
            if (np.isfinite(rho1) and np.isfinite(rho3))
            else np.nan
        )
        if np.isfinite(delta_rho_hallmark):
            if abs(delta_rho_hallmark) < 0.1:
                interp3 = (
                    "ρ_TF_clean ≈ ρ_Hallmark: ambos miden el mismo fenómeno "
                    "(proliferación tumoral). TF_clean no añade información sobre MYC."
                )
            else:
                interp3 = (
                    f"ρ_TF_clean − ρ_Hallmark = {delta_rho_hallmark:+.3f}: "
                    "TF_clean captura algo diferente al Hallmark."
                )
        else:
            interp3 = "DATOS INSUFICIENTES"

        logger.info(f"  Δρ (TF_clean − Hallmark) = {delta_rho_hallmark:+.4f}")
        logger.info(f"  → {interp3}")

        results.append({
            "test": "T3_rho_Hallmark_vs_Tumor_intraDesert",
            "variable_x": "MYC_Hallmark_Combined",
            "variable_y": tumor_primary_name,
            "phenotype": DESERT_LABEL,
            "spearman_rho": rho3,
            "p_value": pval3,
            "n": n3,
            "interpretation": interp3,
            "circular": False,
            "delta_rho_TFclean_minus_Hallmark": delta_rho_hallmark,
        })
    else:
        logger.info("  TEST 3: omitido (MYC_Hallmark_Combined no disponible)")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: Cohen's d de tumor_primary (Desert vs Inflamed)
    # Si d_Tumor ≈ d_TF_clean (+0.347) → MYC_TF_clean mide fracción tumoral
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("TEST 4: d(Tumor_Desert vs Tumor_Inflamed) [escala del confounding]")

    d_tumor = cohens_d_pooled(
        tumor_primary[desert_mask],
        tumor_primary[inflamed_mask],
    )
    _, pval4 = mannwhitneyu(
        tumor_primary[desert_mask][np.isfinite(tumor_primary[desert_mask])],
        tumor_primary[inflamed_mask][np.isfinite(tumor_primary[inflamed_mask])],
        alternative="two-sided",
    )

    logger.info(
        f"  d(Tumor Desert vs Inflamed) = {d_tumor:.4f}, p = {pval4:.3e}"
    )
    logger.info(
        f"  d(MYC_TF_clean Desert vs Inflamed) = +0.3475 [del log_19_myc_tf_clean]"
    )

    if np.isfinite(d_tumor):
        ratio = 0.3475 / d_tumor if abs(d_tumor) > 0.05 else np.nan
        if np.isfinite(ratio):
            if 0.7 <= ratio <= 1.3:
                interp4 = (
                    f"d_TF_clean / d_Tumor = {ratio:.2f} ≈ 1: "
                    "el efecto Desert vs Inflamed de MYC_TF_clean es proporcional "
                    "al de la fracción tumoral. CONFOUNDING PROBABLE."
                )
            elif ratio > 1.3:
                interp4 = (
                    f"d_TF_clean / d_Tumor = {ratio:.2f} > 1: "
                    "MYC_TF_clean tiene un efecto mayor que la fracción tumoral. "
                    "Posible señal MYC adicional al confounding."
                )
            else:
                interp4 = (
                    f"d_TF_clean / d_Tumor = {ratio:.2f} < 0.7: "
                    "el efecto de MYC_TF_clean es menor que el de la fracción tumoral."
                )
        else:
            interp4 = f"d_Tumor ≈ 0: fracción tumoral no difiere entre fenotipos (d={d_tumor:.4f})"
    else:
        interp4 = "DATOS INSUFICIENTES"

    logger.info(f"  → {interp4}")

    results.append({
        "test": "T4_d_Tumor_Desert_vs_Inflamed",
        "variable_x": tumor_primary_name,
        "phenotype": f"{DESERT_LABEL}_vs_{INFLAMED_LABEL}",
        "cohens_d": d_tumor,
        "p_value": pval4,
        "reference_d_TFclean": 0.3475,
        "ratio_TFclean_over_Tumor": (
            round(0.3475 / d_tumor, 3)
            if (np.isfinite(d_tumor) and abs(d_tumor) > 0.05)
            else None
        ),
        "interpretation": interp4,
        "circular": False,
    })

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: ρ(MYC_TF_clean, TUMOR_SCORE_4G) intra-Desert con C2L también
    # Si C2L != expr génica → incluir ambas para robustez
    # ═══════════════════════════════════════════════════════════════════════
    if use_c2l:
        logger.info("\n" + "=" * 70)
        logger.info(
            "TEST 5: ρ(MYC_TF_clean, TUMOR_SCORE_4G) intra-Desert "
            "[expresión génica como segundo proxy]"
        )
        rho5, pval5, n5 = spearman_safe(
            tf_clean[desert_mask], tumor_expr[desert_mask]
        )
        logger.info(f"  ρ = {rho5:.4f}, p = {pval5:.3e}, n = {n5:,}")
        results.append({
            "test": "T5_rho_TFclean_vs_TumorExpr_intraDesert",
            "variable_x": "MYC_TF_activity_clean",
            "variable_y": "TUMOR_SCORE_4G (EPCAM/KRT18/KRT19/TOP2A)",
            "phenotype": DESERT_LABEL,
            "spearman_rho": rho5,
            "p_value": pval5,
            "n": n5,
            "interpretation": (
                "Segundo proxy independiente de fracción tumoral (expresión génica)"
            ),
            "circular": False,
        })

    return results


# ── Figura ────────────────────────────────────────────────────────────────────
def plot_confound(adata, results, tumor_primary_name, logger):
    """
    Figura de 2 paneles:
      A: scatter MYC_TF_clean vs tumor_marker intra-Desert (subsample)
      B: barras d(Desert vs Inflamed) para TF_clean vs tumor_marker
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        phenos_arr = adata.obs[PHENOTYPE_COL].values
        desert_mask  = phenos_arr == DESERT_LABEL
        inflamed_mask = phenos_arr == INFLAMED_LABEL
        tf_clean = adata.obs["MYC_TF_activity_clean"].values

        # Obtener tumor_primary para la figura
        tumor_c2l, _ = get_c2l_column(adata, "Tumor", logger)
        if tumor_c2l is None:
            tumor_c2l, _ = get_c2l_column(adata, "Normal_Epithelial", logger)
        tumor_expr_arr = get_gene_score(
            adata, TUMOR_MARKERS_INDEPENDENT, "TUMOR_SCORE_4G_fig", logger
        )
        tumor_plot = tumor_c2l if (tumor_c2l is not None) else tumor_expr_arr

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(
            "MYC TF Activity (clean) vs Tumor Fraction — Proliferation Confound Test",
            fontsize=12, fontweight="bold"
        )

        # Panel A: scatter intra-Desert (subsample max 3000)
        ax = axes[0]
        idx_des = np.where(desert_mask & np.isfinite(tumor_plot)
                           & np.isfinite(tf_clean))[0]
        rng = np.random.default_rng(SEED)
        idx_sample = rng.choice(idx_des, min(3000, len(idx_des)), replace=False)
        ax.scatter(
            tumor_plot[idx_sample],
            tf_clean[idx_sample],
            alpha=0.25, s=4, color="#3498DB", rasterized=True
        )
        rho_plot, _, _ = spearman_safe(
            tf_clean[idx_des], tumor_plot[idx_des]
        )
        ax.set_xlabel(tumor_primary_name, fontsize=10)
        ax.set_ylabel("MYC TF Activity (clean, z-score)", fontsize=10)
        ax.set_title(
            f"Intra-Desert scatter\nSpearman ρ = {rho_plot:.3f}",
            fontsize=10, fontweight="bold"
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Panel B: Cohen's d comparison
        ax2 = axes[1]
        # Calcular d para tumor_primary
        d_tumor_dv = cohens_d_pooled(
            tumor_plot[desert_mask], tumor_plot[inflamed_mask]
        )
        d_tf_dv = 0.3475  # del log

        bars_data = {
            "MYC TF\n(clean)": d_tf_dv,
            tumor_primary_name.replace("meanscell_abundance_w_sf_", "C2L_"): d_tumor_dv,
        }
        colors = ["#2C7BB6", "#D7191C"]
        bars = ax2.bar(
            list(bars_data.keys()),
            list(bars_data.values()),
            color=colors, edgecolor="black", linewidth=0.8, width=0.5
        )
        for bar, val in zip(bars, bars_data.values()):
            if np.isfinite(val):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.01 * np.sign(val),
                    f"{val:+.3f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=10, fontweight="bold"
                )
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_ylabel("Cohen's d (Desert vs Inflamed)", fontsize=10)
        ax2.set_title(
            "Effect size comparison\n(same Desert vs Inflamed contrast)",
            fontsize=10, fontweight="bold"
        )
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

        plt.tight_layout()
        fpath = RESULTS_DIR / "fig_myc_tf_vs_tumor_confound.png"
        plt.savefig(fpath, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"\n  Figura: {fpath}")

    except Exception as e:
        logger.warning(f"  Figura: {e}")


# ── Resumen final ─────────────────────────────────────────────────────────────
def verdict(results, logger):
    """
    Genera veredicto sobre la hipótesis de confounding.
    Lee los resultados de los tests y produce interpretación consolidada.
    """
    logger.info("\n" + "=" * 70)
    logger.info("VEREDICTO: ¿d=+0.347 es proliferación tumoral o señal MYC?")
    logger.info("=" * 70)

    t1 = next((r for r in results
               if r["test"] == "T1_rho_TFclean_vs_Tumor_intraDesert"), None)
    t2 = next((r for r in results
               if r["test"] == "T2_rho_TFclean_vs_Tumor_intraInflamed"), None)
    t3 = next((r for r in results
               if r["test"] == "T3_rho_Hallmark_vs_Tumor_intraDesert"), None)
    t4 = next((r for r in results
               if r["test"] == "T4_d_Tumor_Desert_vs_Inflamed"), None)

    rho1 = t1["spearman_rho"] if t1 else np.nan
    rho2 = t2["spearman_rho"] if t2 else np.nan
    d_tumor = t4["cohens_d"] if t4 else np.nan
    ratio = t4.get("ratio_TFclean_over_Tumor") if t4 else None

    points_confounding = 0
    points_signal = 0
    evidence = []

    if np.isfinite(rho1) and abs(rho1) >= 0.4:
        points_confounding += 2
        evidence.append(f"T1: ρ(TF, Tumor) intra-Desert = {rho1:.3f} ≥ 0.4 → confounding")
    elif np.isfinite(rho1) and abs(rho1) < 0.2:
        points_signal += 2
        evidence.append(f"T1: ρ(TF, Tumor) intra-Desert = {rho1:.3f} < 0.2 → sin confounding")
    else:
        evidence.append(f"T1: ρ(TF, Tumor) intra-Desert = {rho1:.3f} → ambiguo")

    if np.isfinite(rho1) and np.isfinite(rho2):
        delta = abs(rho1 - rho2)
        if delta < 0.1:
            points_confounding += 1
            evidence.append(f"T2: |Δρ Desert-Inflamed| = {delta:.3f} < 0.1 → no Desert-específico")
        else:
            points_signal += 1
            evidence.append(f"T2: |Δρ Desert-Inflamed| = {delta:.3f} ≥ 0.1 → Desert-específico")

    if t3 is not None and np.isfinite(t3.get("delta_rho_TFclean_minus_Hallmark", np.nan)):
        delta3 = abs(t3["delta_rho_TFclean_minus_Hallmark"])
        if delta3 < 0.1:
            points_confounding += 1
            evidence.append(
                f"T3: |ρ_TFclean − ρ_Hallmark| = {delta3:.3f} < 0.1 → TF_clean ≈ Hallmark"
            )
        else:
            points_signal += 1
            evidence.append(
                f"T3: |ρ_TFclean − ρ_Hallmark| = {delta3:.3f} ≥ 0.1 → TF_clean añade info"
            )

    if ratio is not None:
        if 0.7 <= ratio <= 1.3:
            points_confounding += 2
            evidence.append(
                f"T4: d_TF_clean/d_Tumor = {ratio:.2f} ≈ 1 → efectos proporcionales"
            )
        else:
            points_signal += 1
            evidence.append(
                f"T4: d_TF_clean/d_Tumor = {ratio:.2f} ≠ 1 → efectos distintos"
            )

    for e in evidence:
        logger.info(f"  {e}")

    logger.info(f"\n  Puntos confounding: {points_confounding}")
    logger.info(f"  Puntos señal MYC:   {points_signal}")

    if points_confounding >= points_signal + 2:
        final_verdict = "CONFOUNDING: d=+0.347 refleja fracción tumoral/proliferación, no señal MYC específica"
        recommendation = (
            "Reportar d=+0.347 como null honesto sobre MYC.\n"
            "El score TF mide proliferación tumoral en contexto Visium.\n"
            "Sin deconvolución de ciclo celular, no es posible separar\n"
            "actividad MYC de fracción tumoral con datos bulk-spot."
        )
    elif points_signal >= points_confounding + 2:
        final_verdict = "SEÑAL MYC: d=+0.347 tiene componente independiente de fracción tumoral"
        recommendation = (
            "Reportar d=+0.347 con caveat sobre confounding parcial.\n"
            "Señalar que el efecto es Desert-específico y más allá del Hallmark."
        )
    else:
        final_verdict = "AMBIGUO: evidencia insuficiente para separar señal de confounding"
        recommendation = (
            "Reportar como resultado exploratorio con incertidumbre explícita.\n"
            "d=+0.347 no puede atribuirse inequívocamente a actividad MYC."
        )

    logger.info(f"\n  VEREDICTO FINAL: {final_verdict}")
    logger.info(f"\n  RECOMENDACIÓN:\n  {recommendation}")

    return {
        "final_verdict": final_verdict,
        "recommendation": recommendation,
        "points_confounding": points_confounding,
        "points_signal": points_signal,
        "evidence": evidence,
        "d_TFclean_Desert_vs_Inflamed": 0.3475,
        "d_Tumor_Desert_vs_Inflamed": float(d_tumor) if np.isfinite(d_tumor) else None,
        "rho_T1": float(rho1) if np.isfinite(rho1) else None,
        "rho_T2": float(rho2) if np.isfinite(rho2) else None,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    logger = setup_logging()

    logger.info("=" * 70)
    logger.info("PROLIFERATION CONFOUND TEST — MYC TF Activity (clean)")
    logger.info("¿Es el d=+0.347 señal MYC o fracción tumoral?")
    logger.info(f"Output: {RESULTS_DIR}")
    logger.info("=" * 70)

    # ── Cargar datos ──────────────────────────────────────────────────────────
    logger.info("\n  Cargando datos...")
    adata = load_adata(logger)
    logger.info(f"  Spots: {adata.n_obs:,} | Genes .X: {adata.n_vars:,}")
    if adata.raw:
        logger.info(f"  Genes .raw: {adata.raw.n_vars:,}")

    dist = adata.obs[PHENOTYPE_COL].value_counts()
    logger.info(f"  Fenotipos:\n{dist.to_string()}")

    # ── Ejecutar tests ────────────────────────────────────────────────────────
    results = test_proliferation_confound(adata, logger)

    # ── FDR sobre p-values de tests (solo los que tienen p_value) ─────────────
    pvals = [
        r["p_value"] for r in results
        if "p_value" in r and r["p_value"] is not None and np.isfinite(r["p_value"])
    ]
    if pvals:
        _, qvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")
        q_iter = iter(qvals)
        for r in results:
            if "p_value" in r and r["p_value"] is not None and np.isfinite(r["p_value"]):
                r["q_value"] = float(next(q_iter))

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "myc_tf_proliferation_confound.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\n  CSV: {csv_path}")

    # ── Figura ────────────────────────────────────────────────────────────────
    tumor_c2l, c2l_col = get_c2l_column(adata, "Tumor", logger)
    if tumor_c2l is None:
        tumor_c2l, c2l_col = get_c2l_column(adata, "Normal_Epithelial", logger)
    tumor_name = c2l_col if c2l_col else "TUMOR_SCORE_4G"
    plot_confound(adata, results, tumor_name, logger)

    # ── Veredicto ─────────────────────────────────────────────────────────────
    vdict = verdict(results, logger)

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    output = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "d_MYC_TF_clean_Desert_vs_Inflamed": 0.3475,
        "d_MYC_TF_contaminado_Desert_vs_Inflamed": 0.6216,
        "d_MYC_Hallmark_Desert_vs_Inflamed": -0.06,
        "artefacto_circularidad": round(0.6216 - 0.3475, 4),
        "tests": results,
        "verdict": vdict,
    }
    json_path = RESULTS_DIR / "myc_tf_proliferation_confound.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"  JSON: {json_path}")

    logger.info(f"\n  Tiempo total: {time.time()-t0:.1f}s")
    logger.info("  STATUS: COMPLETADO")


if __name__ == "__main__":
    main()

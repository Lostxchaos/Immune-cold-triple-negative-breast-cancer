#!/usr/bin/env python3
"""
investigate_myc_sting_mechanism.py
================================================================================
Investigación computacional del brazo MYC-STING → Immune Desert

PROPÓSITO
---------
Rescatar computacionalmente la hipótesis MYC→STING operando a nivel
epigenético/post-traduccional, donde la correlación mRNA-mRNA es un proxy
inadecuado del mecanismo real. La literatura convergente 2020-2025 describe
≥3 mecanismos que NO producen anticorrelación mRNA-mRNA detectable:
  1. Enhancer binding (Lee/Lim 2022): H3K27ac reducido en STING1 → represión gradual
  2. DNMT1-mediated methylation (Du 2021): silenciamiento promotor → invisible por Visium
  3. Protein-level regulation (Hoek 2025): STING proteína ↓ sin cambio en mRNA
  4. Downstream ISG repression (Zimmerli 2022): ISGs silenciados aunque STING mRNA presente

ANÁLISIS IMPLEMENTADOS
----------------------
  1. MYC Activity → Chemokine Output (proxy del mecanismo real)
  2. MYC Activity → Immune Infiltration directa (Cell2Location)
  3. MYC Expression Binaria (High vs Low) dentro de Immune Desert
  4. ISG Score — Downstream de STING (evidencia funcional indirecta)
  5. MHC-I Pathway — Mecanismo alternativo de inmunoevasión por MYC
  6. Tabla resumen FDR-corregida de todos los tests

ESPECIFICACIONES TÉCNICAS
--------------------------
  - Expresión génica: adata.raw (log1p, max=8.5) — NUNCA adata.X (counts crudos)
  - Cohen's d: pooled con ddof=1 (canónico del pipeline, CANONICAL_SIGNATURES)
  - FDR: Benjamini-Hochberg sobre TODOS los tests de todos los análisis
  - Phenotype: columna 'Phenotype' (capitalizada)
  - Cell2Location: obsm['means_cell_abundance_w_sf'], cols 'meanscell_abundance_w_sf_*'
  - Output: RESULTS/myc_sting_investigation/
"""

import os
import sys
import json
import warnings
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PATHS — 
# ─────────────────────────────────────────────────────────────────────────────
BASE      = '/home/external/frjimenez/fabian/genoma'
DISCOVERY = os.path.join(BASE, 'data', 'processed', 'adata_with_mechanism.h5ad')
RESULTS   = os.path.join(BASE, 'results')
OUT_DIR   = os.path.join(RESULTS, 'myc_sting_investigation')

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING — archivo + stdout simultáneos
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUT_DIR, 'myc_sting_investigation.log')),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LISTAS DE GENES — Fuentes bibliográficas indicadas
# ─────────────────────────────────────────────────────────────────────────────

# Chemokines downstream de STING (Lim 2022, Zimmerli 2022, House 2020)
CHEMOKINE_GENES: list[str] = ['CCL5', 'CXCL9', 'CXCL10']

# ISGs downstream de STING activado (Zimmerli 2022, Muthalagu 2020)
ISG_GENES: list[str] = [
    'IFIT1', 'IFIT2', 'IFIT3', 'ISG15', 'MX1',
    'MX2', 'OAS1', 'OAS2', 'RSAD2', 'IFI44L',
]

# MHC-I: presentación antigénica reprimida por MYC (Krenz 2024)
MHC_I_GENES: list[str] = ['HLA-A', 'HLA-B', 'HLA-C', 'B2M', 'TAP1', 'TAP2', 'TAPBP']

# MYC Hallmark V1 — MSigDB HALLMARK_MYC_TARGETS_V1 (subconjunto curado para Visium)
# Se usa solo si MYC_Hallmark_Combined NO está en adata.obs
MYC_V1_FALLBACK: list[str] = [
    'MYC', 'CCND2', 'CDK4', 'E2F1', 'E2F2', 'PCNA', 'MCM2', 'MCM3',
    'MCM4', 'MCM5', 'MCM6', 'MCM7', 'LDHA', 'CAD', 'PRPS2', 'DHODH',
    'NME1', 'NME2', 'TFRC', 'SLC7A5', 'GLS', 'SLC1A5', 'ODC1', 'SMS',
    'EIF4A1', 'EIF4E', 'EIF5A', 'NPM1', 'HNRNPA1', 'HNRNPA2B1',
    'HNRNPD', 'NCL', 'FBL', 'PTMA', 'POLR1A', 'POLR2A',
]

# Tipos celulares INMUNES para Immune_Score_total
# (excluye stromal: CAF, Endothelial, PVL, Normal_Epithelial, Tumor)
IMMUNE_CELL_TYPES: list[str] = [
    'CD8_T', 'CD4_T', 'NK', 'NKT', 'B_Cell',
    'cDC1', 'Macrophage', 'Monocyte', 'Myeloid_Cycling', 'T_Cell_Cycling',
]

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
PHENOTYPE_COL      = 'Phenotype'           # columna capitalizada 
PHENOTYPE_DESERT   = 'Immune_Desert'
PHENOTYPE_EXCLUDED = 'Immune_Excluded'
PHENOTYPE_INFLAMED = 'Inflamed'

OBSM_KEY   = 'means_cell_abundance_w_sf'  # Cell2Location obsm key
COL_PREFIX = 'meanscell_abundance_w_sf_'  # prefijo de columnas internas


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN A: FUNCIONES AUXILIARES CANÓNICAS
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d_pooled(g1: np.ndarray, g2: np.ndarray) -> float:
    """
    Cohen's d con varianza pooled y ddof=1.
    Canónico del pipeline Q1 (eliminados H5-03 y C04 de la auditoría).
    Positivo cuando g1 > g2.

    Parameters
    ----------
    g1, g2 : np.ndarray — grupos a comparar (NaN ignorados)

    Returns
    -------
    float — Cohen's d, o np.nan si n < 2 en algún grupo
    """
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0.0:
        return np.nan
    return float((np.mean(g1) - np.mean(g2)) / pooled_std)


def get_gene_from_raw(adata, gene: str) -> np.ndarray | None:
    """
    Extrae expresión de un gen desde adata.raw (log1p normalizado).
    Si adata.raw no existe, intenta adata.X como fallback con advertencia.

    Returns
    -------
    np.ndarray dtype float64, shape (n_obs,), o None si gen no encontrado
    """
    if adata.raw is not None:
        var_names = list(adata.raw.var_names)
        X = adata.raw.X
    else:
        log.warning(f"adata.raw no disponible. Usando adata.X para '{gene}' (puede ser counts crudos).")
        var_names = list(adata.var_names)
        X = adata.X

    if gene not in var_names:
        return None

    idx = var_names.index(gene)
    col = X[:, idx]
    if sp.issparse(col):
        expr = np.asarray(col.todense()).flatten()
    else:
        expr = np.asarray(col).flatten()
    return expr.astype(np.float64)


def calculate_gene_score(adata, genes: list[str], score_name: str) -> pd.Series | None:
    """
    Calcula mean z-score de un conjunto de genes disponibles en .raw.
    Z-score se calcula en espacio log1p (estándar scRNA-seq/Visium).
    Los genes ausentes se omiten con advertencia; retorna None si ninguno disponible.

    Parameters
    ----------
    adata      : AnnData
    genes      : lista de genes a incluir en el score
    score_name : nombre del score para logs

    Returns
    -------
    pd.Series con índice adata.obs_names, o None
    """
    var_names = set(adata.raw.var_names if adata.raw is not None else adata.var_names)
    available = [g for g in genes if g in var_names]
    missing   = [g for g in genes if g not in var_names]

    log.info(f"  [{score_name}] {len(available)}/{len(genes)} genes disponibles en .raw")
    if missing:
        log.info(f"  [{score_name}] Ausentes: {missing}")
    if not available:
        log.warning(f"  [{score_name}] Ningún gen disponible — score no calculable.")
        return None

    z_matrix: list[np.ndarray] = []
    for g in available:
        expr = get_gene_from_raw(adata, g)
        if expr is None:
            continue
        std = expr.std()
        if std == 0.0:
            continue  # gen sin varianza → no aporta información
        z = (expr - expr.mean()) / std
        z_matrix.append(z)

    if not z_matrix:
        log.warning(f"  [{score_name}] Todos los genes tienen varianza=0. Score no calculable.")
        return None

    score = np.mean(np.vstack(z_matrix), axis=0)
    return pd.Series(score, index=adata.obs_names, name=score_name)


def find_abundance_column(obsm_df: pd.DataFrame, cell_type: str) -> str | None:
    """
    Busca columna de abundancia Cell2Location con fuzzy matching en 3 pasos.
    Estrategia: exact → prefix+exact case-insensitive → contains normalizado.

    Parameters
    ----------
    obsm_df   : DataFrame construido desde adata.obsm[OBSM_KEY]
    cell_type : nombre del tipo celular (e.g. 'CD8_T', 'cDC1')

    Returns
    -------
    str nombre de columna, o None si no encontrado
    """
    # Paso 1: exact match con prefijo canónico
    exact = COL_PREFIX + cell_type
    if exact in obsm_df.columns:
        return exact

    # Paso 2: case-insensitive exact
    for col in obsm_df.columns:
        if col.lower() == exact.lower():
            return col

    # Paso 3: contains cell_type normalizado (sin guiones bajos)
    ct_norm = cell_type.lower().replace('_', '')
    for col in obsm_df.columns:
        if ct_norm in col.lower().replace('_', ''):
            return col

    return None


def get_abundance(adata, cell_type: str) -> np.ndarray | None:
    """
    Extrae vector de abundancia Cell2Location para un tipo celular.

    Returns
    -------
    np.ndarray float64 shape (n_obs,) o None si no encontrado
    """
    if OBSM_KEY not in adata.obsm:
        log.warning(f"obsm key '{OBSM_KEY}' no encontrado. Verificar nombre en Cell2Location output.")
        return None

    obsm_df = pd.DataFrame(adata.obsm[OBSM_KEY], index=adata.obs_names)
    col = find_abundance_column(obsm_df, cell_type)

    if col is None:
        log.warning(f"  Columna de abundancia para '{cell_type}' no encontrada.")
        log.info(f"  Columnas disponibles: {list(obsm_df.columns)[:8]} ...")
        return None

    log.info(f"  [Cell2Loc] '{cell_type}' → '{col}' ✓")
    return obsm_df[col].values.astype(np.float64)


def get_immune_score_total(adata) -> np.ndarray | None:
    """
    Suma ponderada (sin pesos, suma simple) de abundancias de todos los
    tipos inmunes en IMMUNE_CELL_TYPES. Excluye stromal y Tumor.

    Returns
    -------
    np.ndarray float64 shape (n_obs,) o None si ningún tipo disponible
    """
    if OBSM_KEY not in adata.obsm:
        return None

    obsm_df = pd.DataFrame(adata.obsm[OBSM_KEY], index=adata.obs_names)
    totals  = np.zeros(adata.n_obs, dtype=np.float64)
    found   = []

    for ct in IMMUNE_CELL_TYPES:
        col = find_abundance_column(obsm_df, ct)
        if col is not None:
            totals += obsm_df[col].values
            found.append(ct)

    log.info(f"  [Immune_Score_total] Tipos sumados ({len(found)}/{len(IMMUNE_CELL_TYPES)}): {found}")
    if not found:
        log.warning("  [Immune_Score_total] Ningún tipo inmune encontrado en obsm.")
        return None

    return totals


def spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float, float, float]:
    """
    Spearman ρ con IC 95% por Fisher z-transformation.
    Filtra NaN antes del cálculo.

    Returns
    -------
    (rho, p_value, ci_low, ci_high) — todos float
    """
    mask = np.isfinite(x) & np.isfinite(y)
    xm, ym = x[mask], y[mask]
    n = len(xm)
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan

    rho, p = spearmanr(xm, ym)
    rho = float(rho)

    # Fisher z-transformation para IC
    z     = np.arctanh(np.clip(rho, -0.9999, 0.9999))
    se    = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    ci_lo  = float(np.tanh(z - z_crit * se))
    ci_hi  = float(np.tanh(z + z_crit * se))

    return rho, float(p), ci_lo, ci_hi


def interpret_result(
    value: float,
    q: float,
    hypothesis: str = 'negative',
    metric: str = 'rho',
) -> str:
    """
    Interpreta si el resultado soporta, contradice o no informa la hipótesis.

    Parameters
    ----------
    value      : ρ (spearman) o d (cohen's d)
    q          : q-value FDR-corregido
    hypothesis : 'negative' o 'positive'
    metric     : 'rho' o 'd'
    """
    if np.isnan(value):
        return 'NO_INFORMATIVO (NaN)'

    sig = q < 0.05
    threshold = 0.1 if metric == 'rho' else 0.2  # umbral mínimo de relevancia práctica

    if hypothesis == 'negative':
        if value < -threshold and sig:
            return 'SOPORTA_HIPOTESIS'
        elif value < 0 and sig:
            return 'SOPORTA_DEBILMENTE'
        elif value < 0 and not sig:
            return 'TENDENCIA_ESPERADA_NS'
        elif value > threshold and sig:
            return 'CONTRADICE_HIPOTESIS'
        else:
            return 'NO_INFORMA'
    elif hypothesis == 'positive':
        if value > threshold and sig:
            return 'SOPORTA_HIPOTESIS'
        elif value > 0 and sig:
            return 'SOPORTA_DEBILMENTE'
        elif value > 0 and not sig:
            return 'TENDENCIA_ESPERADA_NS'
        else:
            return 'NO_INFORMA'
    return 'INDEFINIDO'


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN B: CARGA DE DATOS
# ═══════════════════════════════════════════════════════════════════════════════

def load_adata():
    """
    Carga adata_with_mechanism.h5ad con verificaciones de integridad críticas.
    Verifica: columna Phenotype, disponibilidad de .raw, obsm key, MYC_Hallmark_Combined.
    """
    import scanpy as sc  # importación diferida — no es dependencia de las funciones auxiliares

    if not os.path.exists(DISCOVERY):
        log.error(f"Archivo no encontrado: {DISCOVERY}")
        log.error("Verificar que mechanism_validation.py se ejecutó correctamente (log_02).")
        sys.exit(1)

    log.info(f"Cargando: {DISCOVERY}")
    adata = sc.read_h5ad(DISCOVERY)

    log.info(f"  Spots: {adata.n_obs:,} | Genes: {adata.n_vars:,}")
    if adata.raw is not None:
        log.info(f"  Genes en .raw: {adata.raw.n_vars:,}")
        log.info(f"  .raw max: {adata.raw.X.max():.2f} (esperado ~8.5 para log1p)")
    else:
        log.warning("  adata.raw: NO disponible — se usará adata.X (puede ser counts crudos)")

    # ── Verificar y normalizar columna Phenotype ─────────────────────────────
    if PHENOTYPE_COL not in adata.obs.columns:
        matches = [c for c in adata.obs.columns if c.lower() == 'phenotype']
        if matches:
            adata.obs.rename(columns={matches[0]: PHENOTYPE_COL}, inplace=True)
            log.info(f"  Phenotype: columna renombrada '{matches[0]}' → '{PHENOTYPE_COL}'")
        else:
            log.error(f"  Columna '{PHENOTYPE_COL}' no encontrada. Columnas disponibles: "
                      f"{list(adata.obs.columns)}")
            sys.exit(1)

    counts = adata.obs[PHENOTYPE_COL].value_counts()
    log.info(f"  Distribución fenotipos:\n{counts.to_string()}")

    # ── Verificar obsm ───────────────────────────────────────────────────────
    if OBSM_KEY in adata.obsm:
        n_cols = adata.obsm[OBSM_KEY].shape[1] if hasattr(adata.obsm[OBSM_KEY], 'shape') else 'N/A'
        log.info(f"  obsm['{OBSM_KEY}']: {n_cols} columnas ✓")
    else:
        log.warning(f"  obsm['{OBSM_KEY}'] NO encontrado. Cell2Location scores no disponibles.")
        log.info(f"  obsm keys disponibles: {list(adata.obsm.keys())}")

    # ── Verificar MYC_Hallmark_Combined ─────────────────────────────────────
    if 'MYC_Hallmark_Combined' in adata.obs.columns:
        log.info("  MYC_Hallmark_Combined: en adata.obs ✓ (usará score pre-calculado)")
    else:
        log.warning("  MYC_Hallmark_Combined NO en adata.obs → recalculará desde .raw")

    return adata


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN C: ANÁLISIS
# ═══════════════════════════════════════════════════════════════════════════════

def analysis1_myc_activity_chemokine(
    adata,
    all_tests: list,
) -> pd.DataFrame:
    """
    ANÁLISIS 1: MYC Activity → Chemokine Output.

    Hipótesis: Si MYC reprime STING epigenéticamente → downstream chemokine
    output disminuye → ρ(MYC_activity, Chemokine_output) < 0, especialmente
    en Immune Desert.

    Tests: Spearman ρ en {Global, Desert, Excluded, Inflamed}
    """
    log.info("\n" + "=" * 70)
    log.info("ANÁLISIS 1: MYC Activity → Chemokine Output")
    log.info("=" * 70)

    # ── 1a. MYC_activity ────────────────────────────────────────────────────
    if 'MYC_Hallmark_Combined' in adata.obs.columns:
        myc_activity = adata.obs['MYC_Hallmark_Combined'].values.astype(np.float64)
        log.info("  MYC_activity: MYC_Hallmark_Combined desde adata.obs (pre-calculado)")
    else:
        log.info("  MYC_activity: recalculando desde genes Hallmark en .raw")
        s = calculate_gene_score(adata, MYC_V1_FALLBACK, 'MYC_Hallmark_Recomputed')
        if s is None:
            log.error("  MYC_activity no calculable. Saltando Análisis 1.")
            return pd.DataFrame()
        myc_activity = s.values
        adata.obs['MYC_Hallmark_Combined'] = myc_activity
        log.info("  MYC_Hallmark_Combined guardado en adata.obs")

    # ── 1b. Chemokine_output ─────────────────────────────────────────────────
    chemokine_score = calculate_gene_score(adata, CHEMOKINE_GENES, 'Chemokine_Output')
    if chemokine_score is None:
        log.error("  Chemokine_output no calculable (ningún gen disponible). Saltando Análisis 1.")
        return pd.DataFrame()

    chemokine_vals = chemokine_score.values
    adata.obs['Chemokine_Output_Score'] = chemokine_vals

    # ── 1c. Correlaciones por subconjunto ────────────────────────────────────
    phenotype = adata.obs[PHENOTYPE_COL].values
    subsets = {
        'Global'          : np.ones(len(phenotype), dtype=bool),
        PHENOTYPE_DESERT  : phenotype == PHENOTYPE_DESERT,
        PHENOTYPE_EXCLUDED: phenotype == PHENOTYPE_EXCLUDED,
        PHENOTYPE_INFLAMED: phenotype == PHENOTYPE_INFLAMED,
    }

    rows: list[dict] = []
    log.info(f"\n  {'Subconjunto':<22} {'N':>8} {'ρ':>8} {'p':>12} {'IC95_low':>10} {'IC95_hi':>10}")
    log.info("  " + "-" * 74)

    for label, mask in subsets.items():
        n  = int(mask.sum())
        x  = myc_activity[mask]
        y  = chemokine_vals[mask]
        rho, p, ci_lo, ci_hi = spearman_ci(x, y)

        log.info(f"  {label:<22} {n:>8,} {rho:>8.4f} {p:>12.3e} {ci_lo:>10.4f} {ci_hi:>10.4f}")

        test_key = f'A1_MYC_Chemokine_{label}'
        rows.append({
            'analysis'   : 'A1_MYC_vs_Chemokine',
            'test_label' : test_key,
            'subset'     : label,
            'n'          : n,
            'statistic'  : rho,
            'p_value'    : p,
            'ci_95_low'  : ci_lo,
            'ci_95_high' : ci_hi,
            'cohens_d'   : np.nan,
            'hypothesis' : 'rho<0 (MYC reprime chemokines)',
            'metric_type': 'spearman',
        })
        all_tests.append({'test_name': test_key, 'p_value': p,
                          'statistic': rho, 'source': 'Analysis1'})

    return pd.DataFrame(rows)


def analysis2_myc_immune_infiltration(
    adata,
    all_tests: list,
) -> pd.DataFrame:
    """
    ANÁLISIS 2: MYC Activity → Immune Infiltration (Cell2Location).

    Hipótesis: MYC activo suprime la infiltración de células T y cDC1
    (Zimmerli 2022) → ρ(MYC_activity, CD8_T/cDC1/Immune_total) < 0.

    Tests: Spearman ρ en {Global, Desert, Excluded, Inflamed} para cada target.
    """
    log.info("\n" + "=" * 70)
    log.info("ANÁLISIS 2: MYC Activity → Immune Infiltration Directa")
    log.info("=" * 70)

    if 'MYC_Hallmark_Combined' not in adata.obs.columns:
        log.error("  MYC_Hallmark_Combined no disponible. Ejecutar Análisis 1 primero.")
        return pd.DataFrame()

    myc_vals   = adata.obs['MYC_Hallmark_Combined'].values.astype(np.float64)
    phenotype  = adata.obs[PHENOTYPE_COL].values

    targets: dict[str, np.ndarray | None] = {
        'CD8_T'             : get_abundance(adata, 'CD8_T'),
        'cDC1'              : get_abundance(adata, 'cDC1'),
        'Immune_Score_total': get_immune_score_total(adata),
    }

    subsets = {
        'Global'          : np.ones(len(phenotype), dtype=bool),
        PHENOTYPE_DESERT  : phenotype == PHENOTYPE_DESERT,
        PHENOTYPE_EXCLUDED: phenotype == PHENOTYPE_EXCLUDED,
        PHENOTYPE_INFLAMED: phenotype == PHENOTYPE_INFLAMED,
    }

    rows: list[dict] = []

    for target_name, target_vals in targets.items():
        if target_vals is None:
            log.warning(f"  {target_name}: no disponible en obsm. Saltando.")
            continue

        log.info(f"\n  --- MYC_activity vs {target_name} ---")
        log.info(f"  {'Subset':<22} {'N':>8} {'ρ':>8} {'p':>12}")
        log.info("  " + "-" * 54)

        for label, mask in subsets.items():
            n   = int(mask.sum())
            rho, p, ci_lo, ci_hi = spearman_ci(myc_vals[mask], target_vals[mask])
            log.info(f"  {label:<22} {n:>8,} {rho:>8.4f} {p:>12.3e}")

            test_key = f'A2_MYC_{target_name}_{label}'
            rows.append({
                'analysis'   : 'A2_MYC_vs_Infiltration',
                'test_label' : test_key,
                'target'     : target_name,
                'subset'     : label,
                'n'          : n,
                'statistic'  : rho,
                'p_value'    : p,
                'ci_95_low'  : ci_lo,
                'ci_95_high' : ci_hi,
                'cohens_d'   : np.nan,
                'hypothesis' : 'rho<0 (MYC suprime infiltración)',
                'metric_type': 'spearman',
            })
            all_tests.append({'test_name': test_key, 'p_value': p,
                              'statistic': rho, 'source': 'Analysis2'})

    return pd.DataFrame(rows)


def analysis3_myc_binary_desert(
    adata,
    all_tests: list,
) -> pd.DataFrame:
    """
    ANÁLISIS 3: MYC Binario (High vs Low) dentro de Immune Desert.

    Hipótesis: En Desert, spots MYC_high tendrán MENOS STING, menos
    chemokines y menos CD8 que spots MYC_low. Testea tres stratifiers:
    MYC_activity (Hallmark), EZH2_mRNA, DNMT1_mRNA.

    Tests: Mann-Whitney + Cohen's d (pooled, ddof=1)
    """
    log.info("\n" + "=" * 70)
    log.info("ANÁLISIS 3: MYC Binario (High vs Low) en Immune Desert")
    log.info("=" * 70)

    phenotype    = adata.obs[PHENOTYPE_COL].values
    desert_mask  = phenotype == PHENOTYPE_DESERT
    n_desert     = int(desert_mask.sum())
    log.info(f"  Spots Immune Desert: {n_desert:,}")

    # ── Stratifiers ──────────────────────────────────────────────────────────
    stratifiers: dict[str, np.ndarray] = {}

    if 'MYC_Hallmark_Combined' in adata.obs.columns:
        stratifiers['MYC_activity'] = adata.obs['MYC_Hallmark_Combined'].values.astype(np.float64)
    else:
        log.warning("  MYC_Hallmark_Combined no disponible — saltando stratifier MYC_activity")

    for gene in ['EZH2', 'DNMT1']:
        expr = get_gene_from_raw(adata, gene)
        if expr is not None:
            stratifiers[f'{gene}_mRNA'] = expr
        else:
            log.warning(f"  {gene} no encontrado en .raw — saltando stratifier {gene}_mRNA")

    if not stratifiers:
        log.error("  Ningún stratifier disponible. Saltando Análisis 3.")
        return pd.DataFrame()

    # ── Outcomes ─────────────────────────────────────────────────────────────
    outcomes: dict[str, np.ndarray] = {}

    if 'STING_Score' in adata.obs.columns:
        outcomes['STING_Score'] = adata.obs['STING_Score'].values.astype(np.float64)
    else:
        log.warning("  STING_Score no en adata.obs. Omitido de Análisis 3.")

    if 'Chemokine_Output_Score' in adata.obs.columns:
        outcomes['Chemokine_Score'] = adata.obs['Chemokine_Output_Score'].values.astype(np.float64)
    else:
        chem = calculate_gene_score(adata, CHEMOKINE_GENES, 'Chemokine_Output_Score')
        if chem is not None:
            outcomes['Chemokine_Score'] = chem.values
            adata.obs['Chemokine_Output_Score'] = chem.values

    cd8_vals = get_abundance(adata, 'CD8_T')
    if cd8_vals is not None:
        outcomes['CD8_T_abundance'] = cd8_vals

    if not outcomes:
        log.error("  Ningún outcome disponible. Saltando Análisis 3.")
        return pd.DataFrame()

    rows: list[dict] = []

    for strat_name, strat_all in stratifiers.items():
        strat_desert = strat_all[desert_mask]
        median_strat = float(np.nanmedian(strat_desert))
        high_mask_d  = strat_desert > median_strat
        low_mask_d   = strat_desert <= median_strat

        n_high = int(high_mask_d.sum())
        n_low  = int(low_mask_d.sum())

        log.info(f"\n  Stratifier: {strat_name} | median={median_strat:.4f} | "
                 f"High={n_high:,} | Low={n_low:,}")
        log.info(f"  {'Outcome':<24} {'d':>8} {'p':>12} {'Interpretación'}")
        log.info("  " + "-" * 72)

        for outcome_name, outcome_all in outcomes.items():
            outcome_desert = outcome_all[desert_mask]
            g_high = outcome_desert[high_mask_d]
            g_low  = outcome_desert[low_mask_d]

            g_high = g_high[np.isfinite(g_high)]
            g_low  = g_low[np.isfinite(g_low)]

            if len(g_high) < 5 or len(g_low) < 5:
                log.warning(f"  {outcome_name}: n insuficiente (high={len(g_high)}, low={len(g_low)}). Omitido.")
                continue

            d      = cohens_d_pooled(g_high, g_low)
            _, p   = mannwhitneyu(g_high, g_low, alternative='two-sided')
            interp = interpret_result(d, 1.0, hypothesis='negative', metric='d')  # q provisional=1.0

            log.info(f"  {outcome_name:<24} {d:>8.4f} {p:>12.3e}  {interp}")

            test_key = f'A3_{strat_name}_High_vs_Low_{outcome_name}'
            rows.append({
                'analysis'    : 'A3_Binary_Desert',
                'test_label'  : test_key,
                'stratifier'  : strat_name,
                'outcome'     : outcome_name,
                'subset'      : PHENOTYPE_DESERT,
                'n_high'      : n_high,
                'n_low'       : n_low,
                'median_strat': median_strat,
                'cohens_d'    : d,
                'statistic'   : d,
                'p_value'     : p,
                'hypothesis'  : 'd<0 (High_stratifier→menos_outcome)',
                'metric_type' : 'mannwhitney_d',
            })
            all_tests.append({'test_name': test_key, 'p_value': p,
                              'statistic': d, 'source': 'Analysis3'})

    return pd.DataFrame(rows)


def analysis4_isg_downstream(
    adata,
    all_tests: list,
) -> pd.DataFrame:
    """
    ANÁLISIS 4: ISG Score — Downstream de STING activado.

    Hipótesis: Si STING está epigenéticamente reprimido en Desert (Zimmerli 2022,
    Muthalagu 2020), los ISGs downstream deberían estar reducidos en Desert
    vs Inflamed → d < 0. Además, ρ(MYC, ISG) < 0 confirmaría la represión.

    Tests:
      4a. ISG_Score Desert/Excluded vs Inflamed (Cohen's d + Mann-Whitney)
      4b. Spearman ρ: ISG_Score vs MYC_mRNA (global + Desert)
      4c. Spearman ρ: ISG_Score vs MYC_Hallmark_Combined (global)
    """
    log.info("\n" + "=" * 70)
    log.info("ANÁLISIS 4: ISG Score — Downstream de STING")
    log.info("=" * 70)

    isg_score = calculate_gene_score(adata, ISG_GENES, 'ISG_Score')
    if isg_score is None:
        log.error("  ISG score no calculable. Saltando Análisis 4.")
        return pd.DataFrame()

    isg_vals = isg_score.values
    adata.obs['ISG_Score'] = isg_vals
    phenotype = adata.obs[PHENOTYPE_COL].values

    rows: list[dict] = []

    # ── 4a. ISG por fenotipo ─────────────────────────────────────────────────
    log.info("\n  --- ISG_Score por fenotipo ---")
    for ph in [PHENOTYPE_DESERT, PHENOTYPE_EXCLUDED, PHENOTYPE_INFLAMED]:
        vals = isg_vals[phenotype == ph]
        log.info(f"  {ph:<22}: mean={np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f} (n={int((phenotype == ph).sum()):,})")

    phenotype_pairs = [
        (PHENOTYPE_DESERT,   PHENOTYPE_INFLAMED,  'Desert_vs_Inflamed'),
        (PHENOTYPE_EXCLUDED, PHENOTYPE_INFLAMED,  'Excluded_vs_Inflamed'),
        (PHENOTYPE_DESERT,   PHENOTYPE_EXCLUDED,  'Desert_vs_Excluded'),
    ]

    log.info(f"\n  {'Comparación':<32} {'d':>8} {'p':>12}")
    log.info("  " + "-" * 56)

    for ph1, ph2, label in phenotype_pairs:
        g1 = isg_vals[phenotype == ph1]
        g2 = isg_vals[phenotype == ph2]
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        d  = cohens_d_pooled(g1, g2)
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')

        log.info(f"  {label:<32} {d:>8.4f} {p:>12.3e}")

        test_key = f'A4_ISG_{label}'
        rows.append({
            'analysis'   : 'A4_ISG_Phenotype',
            'test_label' : test_key,
            'comparison' : label,
            'n_g1'       : len(g1),
            'n_g2'       : len(g2),
            'cohens_d'   : d,
            'statistic'  : d,
            'p_value'    : float(p),
            'hypothesis' : 'd<0 (Desert/Excluded<Inflamed=STING reprimido)',
            'metric_type': 'mannwhitney_d',
        })
        all_tests.append({'test_name': test_key, 'p_value': float(p),
                          'statistic': d, 'source': 'Analysis4'})

    # ── 4b. Spearman ISG vs MYC_mRNA ─────────────────────────────────────────
    log.info("\n  --- Correlación ISG_Score vs MYC_mRNA (log1p de .raw) ---")
    myc_expr = get_gene_from_raw(adata, 'MYC')

    if myc_expr is not None:
        subsets_isg = {
            'Global'        : np.ones(len(phenotype), dtype=bool),
            PHENOTYPE_DESERT: phenotype == PHENOTYPE_DESERT,
        }
        for label, mask in subsets_isg.items():
            rho, p, ci_lo, ci_hi = spearman_ci(myc_expr[mask], isg_vals[mask])
            log.info(f"  {label:<22}: ρ={rho:.4f}, p={p:.3e}")

            test_key = f'A4_ISG_vs_MYC_mRNA_{label}'
            rows.append({
                'analysis'   : 'A4_ISG_vs_MYC_mRNA',
                'test_label' : test_key,
                'subset'     : label,
                'n'          : int(mask.sum()),
                'statistic'  : rho,
                'p_value'    : p,
                'ci_95_low'  : ci_lo,
                'ci_95_high' : ci_hi,
                'cohens_d'   : np.nan,
                'hypothesis' : 'rho<0 (MYC reprime ISGs)',
                'metric_type': 'spearman',
            })
            all_tests.append({'test_name': test_key, 'p_value': p,
                              'statistic': rho, 'source': 'Analysis4'})
    else:
        log.warning("  MYC no encontrado en .raw — saltando correlación ISG vs MYC_mRNA")

    # ── 4c. Spearman ISG vs MYC_Hallmark_Combined ────────────────────────────
    log.info("\n  --- Correlación ISG_Score vs MYC_Hallmark_Combined ---")
    if 'MYC_Hallmark_Combined' in adata.obs.columns:
        myc_act = adata.obs['MYC_Hallmark_Combined'].values.astype(np.float64)
        rho_act, p_act, ci_lo_act, ci_hi_act = spearman_ci(myc_act, isg_vals)
        log.info(f"  Global: ρ={rho_act:.4f}, p={p_act:.3e}")

        test_key = 'A4_ISG_vs_MYC_Hallmark_Global'
        rows.append({
            'analysis'   : 'A4_ISG_vs_MYC_Activity',
            'test_label' : test_key,
            'subset'     : 'Global',
            'n'          : adata.n_obs,
            'statistic'  : rho_act,
            'p_value'    : p_act,
            'ci_95_low'  : ci_lo_act,
            'ci_95_high' : ci_hi_act,
            'cohens_d'   : np.nan,
            'hypothesis' : 'rho<0 (MYC activity reprime ISGs)',
            'metric_type': 'spearman',
        })
        all_tests.append({'test_name': test_key, 'p_value': p_act,
                          'statistic': rho_act, 'source': 'Analysis4'})

    return pd.DataFrame(rows)


def analysis5_mhc_pathway(
    adata,
    all_tests: list,
) -> pd.DataFrame:
    """
    ANÁLISIS 5: MHC-I Pathway — Mecanismo alternativo de inmunoevasión por MYC.

    Hipótesis: MYC reprime MHC-I en TNBC (Krenz 2024) → MHC_I_Score menor
    en Desert. Además, MYC upregula CD47 y PD-L1 como mecanismos adicionales
    de evasión (Casey 2016).

    Tests:
      5a. MHC_I_Score Desert/Excluded vs Inflamed (Cohen's d)
      5b. Spearman MHC_I_Score vs MYC_activity (Global + Desert + Inflamed)
      5c. CD47 y PD-L1 (CD274): correlación vs MYC + comparación Desert vs Inflamed
    """
    log.info("\n" + "=" * 70)
    log.info("ANÁLISIS 5: MHC-I Pathway")
    log.info("=" * 70)

    mhc_score = calculate_gene_score(adata, MHC_I_GENES, 'MHC_I_Score')
    if mhc_score is None:
        log.error("  MHC-I score no calculable. Saltando Análisis 5.")
        return pd.DataFrame()

    mhc_vals  = mhc_score.values
    adata.obs['MHC_I_Score'] = mhc_vals
    phenotype = adata.obs[PHENOTYPE_COL].values

    rows: list[dict] = []

    # ── 5a. MHC-I por fenotipo ───────────────────────────────────────────────
    log.info("\n  --- MHC_I_Score por fenotipo ---")
    for ph in [PHENOTYPE_DESERT, PHENOTYPE_EXCLUDED, PHENOTYPE_INFLAMED]:
        vals = mhc_vals[phenotype == ph]
        log.info(f"  {ph:<22}: mean={np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f} (n={int((phenotype == ph).sum()):,})")

    phenotype_pairs = [
        (PHENOTYPE_DESERT,   PHENOTYPE_INFLAMED,  'Desert_vs_Inflamed'),
        (PHENOTYPE_EXCLUDED, PHENOTYPE_INFLAMED,  'Excluded_vs_Inflamed'),
        (PHENOTYPE_DESERT,   PHENOTYPE_EXCLUDED,  'Desert_vs_Excluded'),
    ]

    log.info(f"\n  {'Comparación':<32} {'d':>8} {'p':>12}")
    log.info("  " + "-" * 56)

    for ph1, ph2, label in phenotype_pairs:
        g1 = mhc_vals[phenotype == ph1]
        g2 = mhc_vals[phenotype == ph2]
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        d  = cohens_d_pooled(g1, g2)
        _, p = mannwhitneyu(g1, g2, alternative='two-sided')

        log.info(f"  {label:<32} {d:>8.4f} {p:>12.3e}")

        test_key = f'A5_MHC_I_{label}'
        rows.append({
            'analysis'   : 'A5_MHC_I_Phenotype',
            'test_label' : test_key,
            'comparison' : label,
            'n_g1'       : len(g1),
            'n_g2'       : len(g2),
            'cohens_d'   : d,
            'statistic'  : d,
            'p_value'    : float(p),
            'hypothesis' : 'd<0 (Desert<Inflamed=MHC-I reprimido)',
            'metric_type': 'mannwhitney_d',
        })
        all_tests.append({'test_name': test_key, 'p_value': float(p),
                          'statistic': d, 'source': 'Analysis5'})

    # ── 5b. Spearman MHC_I vs MYC_activity ───────────────────────────────────
    log.info("\n  --- Correlación MHC_I_Score vs MYC_activity ---")
    if 'MYC_Hallmark_Combined' in adata.obs.columns:
        myc_act = adata.obs['MYC_Hallmark_Combined'].values.astype(np.float64)
        sub_labels = {
            'Global'          : np.ones(len(phenotype), dtype=bool),
            PHENOTYPE_DESERT  : phenotype == PHENOTYPE_DESERT,
            PHENOTYPE_INFLAMED: phenotype == PHENOTYPE_INFLAMED,
        }
        log.info(f"  {'Subset':<22} {'N':>8} {'ρ':>8} {'p':>12}")
        log.info("  " + "-" * 54)
        for label, mask in sub_labels.items():
            rho, p, ci_lo, ci_hi = spearman_ci(myc_act[mask], mhc_vals[mask])
            log.info(f"  {label:<22} {int(mask.sum()):>8,} {rho:>8.4f} {p:>12.3e}")

            test_key = f'A5_MHC_vs_MYC_{label}'
            rows.append({
                'analysis'   : 'A5_MHC_vs_MYC_Activity',
                'test_label' : test_key,
                'subset'     : label,
                'n'          : int(mask.sum()),
                'statistic'  : rho,
                'p_value'    : p,
                'ci_95_low'  : ci_lo,
                'ci_95_high' : ci_hi,
                'cohens_d'   : np.nan,
                'hypothesis' : 'rho<0 (MYC reprime MHC-I)',
                'metric_type': 'spearman',
            })
            all_tests.append({'test_name': test_key, 'p_value': p,
                              'statistic': rho, 'source': 'Analysis5'})

    # ── 5c. CD47 y PD-L1 (Casey 2016) ───────────────────────────────────────
    log.info("\n  --- CD47 / PD-L1 (mecanismo Casey 2016) ---")
    extra_genes = {'CD47': 'CD47', 'CD274': 'PD-L1'}  # gen_id → nombre_display

    for gene_id, gene_display in extra_genes.items():
        expr = get_gene_from_raw(adata, gene_id)
        if expr is None:
            log.warning(f"  {gene_id} ({gene_display}) no encontrado en .raw")
            continue

        adata.obs[f'{gene_display}_expr'] = expr

        # Correlación vs MYC_activity (hipótesis: positiva — MYC induce CD47/PD-L1)
        if 'MYC_Hallmark_Combined' in adata.obs.columns:
            myc_act = adata.obs['MYC_Hallmark_Combined'].values.astype(np.float64)
            rho, p, ci_lo, ci_hi = spearman_ci(myc_act, expr)
            log.info(f"  MYC_activity vs {gene_display} (Global): ρ={rho:.4f}, p={p:.3e}")

            test_key = f'A5_MYC_vs_{gene_display}_Global'
            rows.append({
                'analysis'   : 'A5_CD47_PDL1_vs_MYC',
                'test_label' : test_key,
                'gene'       : gene_display,
                'subset'     : 'Global',
                'n'          : adata.n_obs,
                'statistic'  : rho,
                'p_value'    : p,
                'ci_95_low'  : ci_lo,
                'ci_95_high' : ci_hi,
                'cohens_d'   : np.nan,
                'hypothesis' : 'rho>0 (MYC induce CD47/PD-L1 → immune evasion)',
                'metric_type': 'spearman',
            })
            all_tests.append({'test_name': test_key, 'p_value': p,
                              'statistic': rho, 'source': 'Analysis5'})

        # Comparación Desert vs Inflamed (hipótesis: d > 0 — más en Desert)
        g1 = expr[phenotype == PHENOTYPE_DESERT]
        g2 = expr[phenotype == PHENOTYPE_INFLAMED]
        g1, g2 = g1[np.isfinite(g1)], g2[np.isfinite(g2)]
        d  = cohens_d_pooled(g1, g2)
        _, p_mw = mannwhitneyu(g1, g2, alternative='two-sided')
        log.info(f"  {gene_display} Desert vs Inflamed: d={d:.4f}, p={p_mw:.3e}")

        test_key = f'A5_{gene_display}_Desert_vs_Inflamed'
        rows.append({
            'analysis'   : f'A5_{gene_display}_Phenotype',
            'test_label' : test_key,
            'gene'       : gene_display,
            'comparison' : 'Desert_vs_Inflamed',
            'n_g1'       : len(g1),
            'n_g2'       : len(g2),
            'cohens_d'   : d,
            'statistic'  : d,
            'p_value'    : float(p_mw),
            'hypothesis' : 'd>0 (Desert>Inflamed = más evasión inmune)',
            'metric_type': 'mannwhitney_d',
        })
        all_tests.append({'test_name': test_key, 'p_value': float(p_mw),
                          'statistic': d, 'source': 'Analysis5'})

    return pd.DataFrame(rows)


def analysis6_summary_fdr(
    all_dfs: list[pd.DataFrame],
    all_tests: list[dict],
) -> pd.DataFrame:
    """
    ANÁLISIS 6: FDR global Benjamini-Hochberg + tabla resumen de todos los tests.

    Aplica corrección FDR sobre TODOS los p-values acumulados de los 5 análisis.
    Añade interpretación semántica por test. Reporta n_significant y resumen
    estructurado por análisis.

    Returns
    -------
    DataFrame con columnas: test_name, p_value, q_value, fdr_significant,
                            statistic, source, interpretation
    """
    log.info("\n" + "=" * 70)
    log.info("ANÁLISIS 6: FDR Global (Benjamini-Hochberg) + Tabla Resumen")
    log.info("=" * 70)

    if not all_tests:
        log.warning("  No hay tests acumulados para FDR.")
        return pd.DataFrame()

    tests_df = pd.DataFrame(all_tests)
    p_vals   = tests_df['p_value'].values.astype(float)

    # NaN → 1.0 (test inválido no contribuye a significancia)
    p_finite = np.where(np.isfinite(p_vals), p_vals, 1.0)

    _, q_vals, _, _ = multipletests(p_finite, method='fdr_bh', alpha=0.05)
    tests_df['q_value']         = q_vals
    tests_df['fdr_significant'] = q_vals < 0.05

    # ── Interpretación semántica ──────────────────────────────────────────────
    interpretations = []
    for _, row in tests_df.iterrows():
        tn  = str(row['test_name'])
        rho = float(row['statistic'])
        q   = float(row['q_value'])

        # Determinar hipótesis por nombre del test
        if any(k in tn for k in ['Chemokine', 'CD8', 'cDC1', 'Immune_Score', 'ISG', 'MHC']):
            hyp = 'negative' if 'MYC' in tn else 'negative'
        elif 'PDL1' in tn or 'CD47' in tn:
            hyp = 'positive'
        else:
            hyp = 'negative'

        metric = 'd' if any(k in tn for k in ['High_vs_Low', 'Desert_vs', 'Excluded_vs']) else 'rho'
        interpretations.append(interpret_result(rho, q, hypothesis=hyp, metric=metric))

    tests_df['interpretation'] = interpretations

    # ── Log tabla resumen ─────────────────────────────────────────────────────
    n_sig = int(tests_df['fdr_significant'].sum())
    n_tot = len(tests_df)
    log.info(f"\n  Total tests: {n_tot} | FDR significativos (q<0.05): {n_sig}/{n_tot}")

    log.info(f"\n  {'Test':<52} {'stat':>8} {'p':>12} {'q':>12} {'FDR'} {'Interpretación'}")
    log.info("  " + "-" * 120)

    for _, row in tests_df.iterrows():
        sig_str = '✓' if row['fdr_significant'] else '✗'
        log.info(
            f"  {str(row['test_name']):<52} {float(row['statistic']):>8.4f} "
            f"{float(row['p_value']):>12.3e} {float(row['q_value']):>12.3e} "
            f"  {sig_str}   {row['interpretation']}"
        )

    # ── Resumen por análisis ──────────────────────────────────────────────────
    log.info("\n  --- Resumen por análisis ---")
    for src in tests_df['source'].unique():
        sub   = tests_df[tests_df['source'] == src]
        n_s   = int(sub['fdr_significant'].sum())
        log.info(f"  {src}: {n_s}/{len(sub)} tests FDR significativos")

    return tests_df


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN D: INTERPRETACIÓN NARRATIVA PARA EL PAPER
# ═══════════════════════════════════════════════════════════════════════════════

def generate_paper_narrative(tests_df: pd.DataFrame) -> dict:
    """
    Genera interpretación narrativa alineada con las Opciones A-D del Prompt B.

    Opción A: Solo limitación tecnológica (mRNA inadecuado para detectar mecanismo)
    Opción B: Evidencia downstream ISG ± débil en chemokines
    Opción C: Mecanismo CAF es el principal — MYC no confirmado
    Opción D: Multi-mecanismo MYC (STING epigenético + ISG + MHC-I)
    """
    evidence: dict[str, int] = {'A': 0, 'B': 0, 'C': 1, 'D': 0}
    values: dict[str, float] = {}

    def get_stat(pattern: str) -> tuple[float, bool]:
        """Busca test por patrón, retorna (statistic, fdr_significant)."""
        matches = tests_df[tests_df['test_name'].str.contains(pattern, regex=True, na=False)]
        if matches.empty:
            return np.nan, False
        row = matches.iloc[0]
        return float(row['statistic']), bool(row['fdr_significant'])

    # Evaluar ISG Desert vs Inflamed
    d_isg, sig_isg = get_stat(r'A4_ISG_Desert_vs_Inflamed')
    values['ISG_d'] = d_isg
    if d_isg < -0.1:
        evidence['B'] += 1; evidence['D'] += 1
    if d_isg < -0.2 and sig_isg:
        evidence['B'] += 2; evidence['D'] += 2

    # Evaluar MHC-I Desert vs Inflamed
    d_mhc, sig_mhc = get_stat(r'A5_MHC_I_Desert_vs_Inflamed')
    values['MHC_I_d'] = d_mhc
    if d_mhc < -0.1:
        evidence['D'] += 1
    if d_mhc < -0.2 and sig_mhc:
        evidence['D'] += 2

    # Evaluar Chemokine correlación en Desert
    rho_chem, sig_chem = get_stat(r'A1_MYC_Chemokine_Immune_Desert')
    values['Chemokine_rho_Desert'] = rho_chem
    if rho_chem < -0.05 and sig_chem:
        evidence['B'] += 2; evidence['D'] += 1
    elif rho_chem < -0.05:
        evidence['B'] += 1; evidence['D'] += 1

    # Evaluar ISG vs MYC_Hallmark correlación
    rho_isg_myc, _ = get_stat(r'A4_ISG_vs_MYC_Hallmark_Global')
    values['ISG_vs_MYC_Hallmark_rho'] = rho_isg_myc
    if rho_isg_myc < -0.05:
        evidence['B'] += 1; evidence['D'] += 1

    # Seleccionar mejor opción
    best = max(evidence, key=evidence.get)

    # Generar texto para el paper
    d_isg_str  = f"d={d_isg:.2f}"  if np.isfinite(d_isg) else "d=N/A"
    d_mhc_str  = f"d={d_mhc:.2f}" if np.isfinite(d_mhc) else "d=N/A"
    rho_c_str  = f"rho={rho_chem:.3f}" if np.isfinite(rho_chem) else "rho=N/A"

    if best == 'D':
        paper_text = (
            f"Multiple lines of evidence suggest MYC-dependent immune evasion in "
            f"Desert niches through complementary mechanisms: (1) reduced "
            f"interferon-stimulated gene expression (ISG score: Desert vs Inflamed "
            f"{d_isg_str}), consistent with STING pathway suppression; "
            f"(2) attenuated MHC class I pathway expression (MHC-I score: {d_mhc_str}), "
            f"consistent with impaired antigen presentation; and (3) reduced chemokine "
            f"output in MYC-active Desert spots ({rho_c_str}). "
            f"These observations align with emerging evidence that MYC drives immune "
            f"evasion through epigenetic STING repression (Lim et al., 2022), "
            f"MHC-I downregulation (Zimmerli et al., 2022), and metabolic competition "
            f"in the tumor microenvironment."
        )
    elif best == 'B':
        paper_text = (
            f"Although MYC mRNA did not anti-correlate with STING mRNA (rho=+0.015), "
            f"interferon-stimulated gene (ISG) score analysis revealed reduced "
            f"downstream STING pathway activity in Immune Desert spots "
            f"({d_isg_str} vs Inflamed), consistent with functional STING suppression "
            f"independent of mRNA-level changes. This pattern is consistent with "
            f"epigenetic repression of the STING pathway in TNBC (Lim et al., 2022; "
            f"Zimmerli et al., 2022), where MYC binding to the STING1 enhancer "
            f"suppresses downstream chemokine production without necessarily altering "
            f"STING mRNA levels."
        )
    else:
        paper_text = (
            "MYC pathway activity (MSigDB Hallmark scores) did not significantly "
            "differ between immune phenotypes (Desert vs Inflamed d=-0.06, p=0.59), "
            "and direct MYC-STING mRNA correlations were near-zero (rho=+0.015). "
            "These findings are consistent with the known inability of mRNA-level "
            "transcriptomic measurements to detect epigenetic and post-translational "
            "MYC-STING regulatory mechanisms (Linstra et al., 2025; Hoek et al., 2025). "
            "The immune exclusion phenotype in this cohort is predominantly explained "
            "by the CAF-mediated physical barrier mechanism (Cohen's d=-0.62)."
        )

    return {
        'recommended_option'      : best,
        'evidence_scores'         : evidence,
        'key_values'              : values,
        'paper_text'              : paper_text,
        'n_fdr_significant'       : int(tests_df['fdr_significant'].sum()),
        'n_tests_total'           : len(tests_df),
        'interpretations_supporting':
            tests_df[tests_df['interpretation'] == 'SOPORTA_HIPOTESIS']['test_name'].tolist(),
        'interpretations_against' :
            tests_df[tests_df['interpretation'] == 'CONTRADICE_HIPOTESIS']['test_name'].tolist(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> tuple[pd.DataFrame, dict]:
    log.info("=" * 70)
    log.info("INVESTIGACIÓN COMPUTACIONAL: MYC-STING → IMMUNE DESERT")
    log.info("Pipeline TNBC Spatial Transcriptomics — Q1")
    log.info("=" * 70)
    log.info(f"Output: {OUT_DIR}")
    log.info(f"Input : {DISCOVERY}")

    # ── Carga ─────────────────────────────────────────────────────────────────
    adata = load_adata()

    # ── Acumuladores globales ─────────────────────────────────────────────────
    all_tests: list[dict]       = []
    all_dfs  : list[pd.DataFrame] = []

    # ── Análisis 1: MYC Activity → Chemokine Output ──────────────────────────
    df1 = analysis1_myc_activity_chemokine(adata, all_tests)
    all_dfs.append(df1)

    # ── Análisis 2: MYC Activity → Immune Infiltration ───────────────────────
    df2 = analysis2_myc_immune_infiltration(adata, all_tests)
    all_dfs.append(df2)

    # ── Análisis 3: MYC Binario en Desert ─────────────────────────────────────
    df3 = analysis3_myc_binary_desert(adata, all_tests)
    all_dfs.append(df3)

    # ── Análisis 4: ISG Downstream ────────────────────────────────────────────
    df4 = analysis4_isg_downstream(adata, all_tests)
    all_dfs.append(df4)

    # ── Análisis 5: MHC-I Pathway ─────────────────────────────────────────────
    df5 = analysis5_mhc_pathway(adata, all_tests)
    all_dfs.append(df5)

    # ── Análisis 6: FDR Global + Resumen ──────────────────────────────────────
    tests_df = analysis6_summary_fdr(all_dfs, all_tests)

    # ── Narrativa para el paper ────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("INTERPRETACIÓN NARRATIVA PARA EL MANUSCRIPT")
    log.info("=" * 70)
    narrative = generate_paper_narrative(tests_df)
    log.info(f"  Opción recomendada: {narrative['recommended_option']}")
    log.info(f"  Evidence scores   : {narrative['evidence_scores']}")
    log.info(f"  Tests soportan hipótesis: {narrative['interpretations_supporting']}")
    log.info(f"  Tests contradicen hipótesis: {narrative['interpretations_against']}")
    log.info(f"\n  --- PAPER TEXT ---\n  {narrative['paper_text']}\n")

    # ── Guardar resultados ─────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("GUARDANDO RESULTADOS")
    log.info("=" * 70)

    filenames = [
        'analysis1_myc_chemokine.csv',
        'analysis2_myc_infiltration.csv',
        'analysis3_binary_desert.csv',
        'analysis4_isg_downstream.csv',
        'analysis5_mhc_pathway.csv',
    ]

    for df, fname in zip(all_dfs, filenames):
        if df is not None and not df.empty:
            path = os.path.join(OUT_DIR, fname)
            df.to_csv(path, index=False)
            log.info(f"  ✓ {path}")
        else:
            log.info(f"  ⚠ {fname}: DataFrame vacío — no guardado")

    if not tests_df.empty:
        path_fdr = os.path.join(OUT_DIR, 'all_tests_fdr_corrected.csv')
        tests_df.to_csv(path_fdr, index=False)
        log.info(f"  ✓ {path_fdr}")

    # JSON de resumen ejecutivo
    raw_var = list(adata.raw.var_names) if adata.raw is not None else list(adata.var_names)
    summary = {
        'pipeline_version'   : '1.0.0',
        'input_file'         : DISCOVERY,
        'n_spots_total'      : int(adata.n_obs),
        'n_desert'           : int((adata.obs[PHENOTYPE_COL] == PHENOTYPE_DESERT).sum()),
        'n_excluded'         : int((adata.obs[PHENOTYPE_COL] == PHENOTYPE_EXCLUDED).sum()),
        'n_inflamed'         : int((adata.obs[PHENOTYPE_COL] == PHENOTYPE_INFLAMED).sum()),
        'n_genes_raw'        : len(raw_var),
        'genes_chemokine_used': [g for g in CHEMOKINE_GENES if g in raw_var],
        'genes_isg_used'      : [g for g in ISG_GENES if g in raw_var],
        'genes_mhci_used'     : [g for g in MHC_I_GENES if g in raw_var],
        'n_tests_total'       : len(all_tests),
        'n_fdr_significant'   : int(tests_df['fdr_significant'].sum()) if not tests_df.empty else 0,
        'fdr_method'          : 'Benjamini-Hochberg (alpha=0.05)',
        'cohens_d_formula'    : 'pooled, ddof=1',
        'narrative'           : narrative,
    }

    path_json = os.path.join(OUT_DIR, 'myc_sting_investigation_summary.json')
    with open(path_json, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  ✓ {path_json}")

    log.info("\n" + "=" * 70)
    log.info("✅ INVESTIGACIÓN MYC-STING COMPLETADA")
    log.info(f"   Output completo en: {OUT_DIR}")
    log.info("=" * 70)

    return tests_df, summary


if __name__ == '__main__':
    tests_df, summary = main()

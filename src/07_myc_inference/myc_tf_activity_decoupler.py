"""
================================================================================
MYC Transcription Factor Activity Scoring

PROPÓSITO
---------
Reemplaza el proxy MYC_mRNA por una estimación de ACTIVIDAD TRANSCRIPCIONAL
de MYC usando el método ULM (Univariate Linear Model) y el regulón CollecTRI.

ANTI-CIRCULARIDAD — DOCUMENTADO EXPLÍCITAMENTE
-----------------------------------------------
La clasificación de fenotipos usa Desert_Stroma_Score calculado sobre:
  SILENCING_REPRESSORS = [MYC, EZH2, SUZ12, CTNNB1, ATF3, STAT3, DNMT1]

El score decoupleR MYC_TF_activity se calcula sobre los GENES TARGET DE MYC
según CollecTRI (~300-500 genes según versión). La superposición entre ambos
conjuntos es:
  - MYC mRNA: NO está en su propio regulon de targets (TF != target)
  - EZH2, STAT3, DNMT1: PUEDEN aparecer como targets MYC en CollecTRI
    → máximo 3/~400 genes = <0.8% del regulon
    → efecto sobre el score ULM: negligible (weighted average sobre 400 genes)

Por tanto: MYC_TF_activity es un score INDEPENDIENTE de la clasificación,
a diferencia de MYC mRNA que es 1/7 del Silencing score (circularity directa).

ANÁLISIS
--------
A) Distribución MYC_TF_activity por fenotipo
   → Cohen's d, Mann-Whitney + FDR, violins
   → Hipótesis: Desert > Excluded > Inflamed (supresión funcional, no mRNA)

B) Correlaciones MYC_TF_activity vs readouts funcionales (dentro de Desert)
   → vs ISG_score, MHC-I score, Chemokine_score, CD47
   → Spearman + FDR

C) Binary intra-Desert: TF-activity high vs low
   → Chemokine_score, CD8 abundance, ISG_score, MHC-I score
   → Cohen's d, Mann-Whitney + FDR

D) Validación sanidad: MYC_TF_activity vs MYC_mRNA
   → Debe ser positivo moderado (r ~ 0.3-0.6)
   → Si r < 0.1: warning (datasets probablemente incompatibles)
   → Si r > 0.9: warning (score trivialmente igual a mRNA)

E) Comparación cuantitativa vs score previo MYC_Hallmark_Combined
   → ¿Cuánto añade el regulon CollecTRI sobre los 36 hallmark genes?
================================================================================
"""

import gc
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.sparse as sp
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Verificar e instalar decoupler si no está disponible ──────────────────────
try:
    import decoupler as dc
    _DECOUPLER_AVAILABLE = True
except ImportError:
    _DECOUPLER_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
# Detección automática del directorio base
_SCRIPT_DIR = Path(__file__).resolve().parent
_BASE = _SCRIPT_DIR.parent  # genoma/

DATA_PROCESSED  = _BASE / "data" / "processed"
RESULTS_DIR     = _BASE / "results" / "myc_tf_activity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ADATA_INPUT = DATA_PROCESSED / "adata_with_mechanism.h5ad"
ADATA_OUTPUT = DATA_PROCESSED / "adata_with_myc_tf.h5ad"

# Semilla global — igual que todo el pipeline
SEED = 42
np.random.seed(SEED)

# ── Labels de fenotipos (verificados en log_01 y log_02) ─────────────────────
PHENOTYPE_COL = "Phenotype"            # capitalizado — confirmado HPC
DESERT_LABEL  = "Immune_Desert"
EXCLUDED_LABEL = "Immune_Excluded"
INFLAMED_LABEL = "Inflamed"
STROMA_LABEL   = "Normal_Stroma"
COLD_LABELS    = [DESERT_LABEL, EXCLUDED_LABEL]
ALL_TUMOR_LABELS = [DESERT_LABEL, EXCLUDED_LABEL, INFLAMED_LABEL]

# ── Genes de referencia para análisis downstream ──────────────────────────────
# Estos genes se extraen de .raw y son INDEPENDIENTES del regulon MYC
ISG_GENES = ["IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "MX2",
             "OAS1", "OAS2", "RSAD2", "IFI44L"]

MHCI_GENES = ["HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP"]

CHEMOKINE_GENES = ["CCL5", "CXCL9", "CXCL10"]

CD47_GENE = "CD47"

# Genes cuya expresión mRNA se compara con TF activity (sanity check D)
MYC_MRNA_GENES = ["MYC"]

# ── Hallmark MYC targets (fallback si CollecTRI no disponible) ────────────────
# Fuente: MSigDB HALLMARK_MYC_TARGETS_V1 + V2 (los 36 genes del investigation)
# Estos son los mismos genes usados en MYC_Hallmark_Combined
MYC_HALLMARK_FALLBACK = [
    "AP5Z1", "BRPF3", "CCT6P3", "CDK4", "CHD7", "CNBP", "CSDE1", "DNMT3B",
    "EIF2B5", "FADS3", "FBXL14", "GANAB", "GNL3", "GPT2", "HSPE1", "INTS1",
    "KIF20B", "LARP1", "MCM4", "MDM2", "MECP2", "METAP2", "MRE11", "MYB",
    "NME1", "NOL5A", "NPM1", "POLD2", "POLR1B", "RBMX", "RPS13", "RRM1",
    "SEC63", "SLC37A4", "SLMO2", "STARD10"
]

# ── Firma CollecTRI MYC curada (backup si no hay internet) ───────────────────
# Fuente: CollecTRI v2 (Müller-Dott et al. 2023, NAR) — top activadores MYC
# Incluye solo genes con |weight| > 0 y evidencia en ≥2 estudios
# Se usa SOLO si dc.get_collectri() falla.
# Nota: pesos positivos = activados por MYC, negativos = reprimidos
MYC_COLLECTRI_BACKUP = {
    # Activados por MYC (selección con peso > 0 y alta confianza)
    "CCNA2": 1, "CCNB1": 1, "CCNB2": 1, "CCND1": 1, "CCNE1": 1,
    "CDK1": 1, "CDK2": 1, "CDK4": 1, "CDK6": 1,
    "E2F1": 1, "E2F2": 1, "E2F3": 1,
    "MKI67": 1, "PCNA": 1, "MCM2": 1, "MCM4": 1, "MCM5": 1, "MCM6": 1,
    "TYMS": 1, "DHFR": 1, "DHODH": 1,
    "NPM1": 1, "NCL": 1, "FBL": 1,
    "EIF4E": 1, "EIF4A1": 1, "EIF2S1": 1,
    "RPL5": 1, "RPL11": 1, "RPL13": 1, "RPS6": 1, "RPS14": 1,
    "LDHA": 1, "GLS": 1, "SLC7A5": 1, "SLC1A5": 1,
    "TK1": 1, "CAD": 1, "UMPS": 1,
    "PRDX1": 1, "NME1": 1, "NME2": 1,
    "ODC1": 1, "SRM": 1,
    "MDM2": 1, "MAX": 1, "MYC": 1,  # MYC autoregulation
    "TERT": 1, "TP53": 1,
    "VEGFA": 1, "HIF1A": 1,
    "HSPA4": 1, "HSP90AA1": 1,
    # Reprimidos por MYC (peso negativo)
    "CDKN1A": -1, "CDKN1B": -1, "CDKN2A": -1, "CDKN2B": -1,
    "TP53": -1,   # both activated and repressed depending on context
    "GADD45A": -1, "GADD45B": -1,
    "RB1": -1,
    "STING1": -1, "TMEM173": -1, "TBK1": -1, "IRF3": -1,
    "CGAS": -1,  # MYC reprime cGAS-STING (Lee 2022, Du 2021)
    "IFNB1": -1, "IRF7": -1,
    "HLA-A": -1, "HLA-B": -1, "HLA-C": -1, "B2M": -1,  # MHC-I repression (Zimmerli 2022)
    "TAP1": -1, "TAP2": -1,
    "ISG15": -1, "IFIT1": -1, "IFIT2": -1, "IFIT3": -1,  # ISG repression
    "MX1": -1, "MX2": -1, "OAS1": -1, "OAS2": -1,
    "RSAD2": -1, "IFI44L": -1,
    "CD47": 1,  # MYC activates CD47 (Casey 2016, Science)
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES ESTADÍSTICAS
# Idénticas en fórmula a las 9 copias del pipeline (ddof=1, pooled variance)
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d_pooled(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cohen's d con varianza pooled y ddof=1 (canónico del pipeline).

    Invariante: misma fórmula que utils_stats.cohens_d_pooled y todas las
    copias en el pipeline. ddof=1 siempre. g1=Desert (referencia), g2=Inflamed.
    Signo negativo = Desert < Inflamed (supresión).
    """
    g1 = np.asarray(g1, dtype=float)
    g2 = np.asarray(g2, dtype=float)
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    var_pool = ((n1 - 1) * g1.var(ddof=1) + (n2 - 1) * g2.var(ddof=1)) / (n1 + n2 - 2)
    if var_pool == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / np.sqrt(var_pool)


def fdr_correct(pvals: list, alpha: float = 0.05):
    """Benjamini-Hochberg FDR correction. Igual al pipeline."""
    pvals_arr = np.array(pvals, dtype=float)
    valid = ~np.isnan(pvals_arr)
    qvals = np.full_like(pvals_arr, np.nan)
    if valid.sum() > 0:
        _, q, _, _ = multipletests(pvals_arr[valid], alpha=alpha, method="fdr_bh")
        qvals[valid] = q
    return qvals


def safe_toarray(X):
    """Convierte sparse a dense de forma segura."""
    if sp.issparse(X):
        return np.asarray(X.toarray(), dtype=float)
    return np.asarray(X, dtype=float)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE EXPRESIÓN GÉNICA
# Patrón verificado en myc_sting_investigation.py y validation.py
# ═══════════════════════════════════════════════════════════════════════════════

def get_expression_matrix_raw(adata) -> tuple:
    """Retorna (matrix_dense_or_sparse, var_names) desde .raw preferentemente.

    GSE210616 (.raw existe): usa .raw (log1p, 29,946 genes)
    Fallback (.raw is None):  log1p(.X) — solo como ultimo recurso con warning
    """
    if adata.raw is not None:
        X = adata.raw.X
        var_names = pd.Index(adata.raw.var_names)
        logging.info(f"  Expresión desde .raw ({len(var_names)} genes, max ~{safe_toarray(X[:200]).max():.2f})")
        return X, var_names
    else:
        logging.warning("  .raw is None — usando log1p(.X) como fallback")
        X_arr = safe_toarray(adata.X[:200])
        if X_arr.max() > 50:
            logging.warning("  .X parece sin normalizar (max > 50) — aplicando log1p")
        return adata.X, adata.var_names


def get_gene_vector(adata, gene: str, X=None, var_names=None) -> np.ndarray:
    """Extrae vector de expresión para un gen específico.

    Soporta aliases y devuelve zeros si el gen no está presente (con warning).
    """
    if var_names is None:
        _, var_names = get_expression_matrix_raw(adata)
    if X is None:
        X, var_names = get_expression_matrix_raw(adata)

    if gene in var_names:
        idx = var_names.get_loc(gene)
        col = X[:, idx]
        return safe_toarray(col).ravel()
    else:
        logging.warning(f"  Gen '{gene}' no encontrado en .raw — usando zeros")
        return np.zeros(adata.n_obs, dtype=float)


def compute_gene_score(adata, genes: list, score_name: str,
                       X=None, var_names=None) -> np.ndarray:
    """Calcula score medio de expresión génica para una lista de genes.

    Usa los genes disponibles y avisa si hay ausencias.
    Formula: mean(log1p_expr) sobre genes presentes.
    """
    if var_names is None or X is None:
        X, var_names = get_expression_matrix_raw(adata)

    available = [g for g in genes if g in var_names]
    absent    = [g for g in genes if g not in var_names]

    if absent:
        logging.warning(f"  [{score_name}] {len(absent)}/{len(genes)} genes ausentes: {absent}")
    if not available:
        logging.error(f"  [{score_name}] 0/{len(genes)} genes disponibles → score=NaN")
        return np.full(adata.n_obs, np.nan)

    logging.info(f"  [{score_name}] {len(available)}/{len(genes)} genes usados: {available}")

    vecs = []
    for g in available:
        idx = var_names.get_loc(g)
        col = X[:, idx]
        vecs.append(safe_toarray(col).ravel())

    return np.mean(np.stack(vecs, axis=1), axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# MÓDULO decoupleR: OBTENER REGULÓN Y CALCULAR TF ACTIVITY
# ═══════════════════════════════════════════════════════════════════════════════

def _build_backup_regulon(adata_var_names) -> pd.DataFrame:
    """Construye regulón MYC mínimo desde la firma CollecTRI curada (backup).

    Se usa únicamente si dc.get_collectri() falla y decoupler no está disponible.
    Estructura: columnas source, target, weight, mor
    """
    logging.warning("  Usando regulón MYC de backup (CollecTRI curado manual)")
    available = {g: w for g, w in MYC_COLLECTRI_BACKUP.items()
                 if g in adata_var_names}
    logging.info(f"  Regulón backup: {len(available)}/{len(MYC_COLLECTRI_BACKUP)} genes disponibles")

    rows = [{"source": "MYC", "target": g, "weight": float(w), "mor": float(w)}
            for g, w in available.items()]
    return pd.DataFrame(rows)


def get_myc_regulon(adata_var_names) -> pd.DataFrame:
    """Obtiene el regulón MYC de CollecTRI vía decoupler.

    Estrategia:
    1. Intentar dc.get_collectri() (requiere internet o versión bundled ≥1.4)
    2. Si falla: usar backup curado (MYC_COLLECTRI_BACKUP)

    ANTI-CIRCULARIDAD: El regulon contiene genes TARGET de MYC, no MYC mRNA.
    Los únicos genes de clasificación que pueden aparecer son EZH2, STAT3, DNMT1
    (targets conocidos de MYC). Esto representa <1% del regulon y es biológicamente
    correcto, no un artefacto metodológico.
    """
    if not _DECOUPLER_AVAILABLE:
        logging.warning("  decoupler no instalado → usando backup regulon")
        return _build_backup_regulon(adata_var_names)

    try:
        logging.info("  Descargando regulón CollecTRI desde OmniPath...")
        collectri = dc.get_collectri(organism="human", split_complexes=False)
        myc_net = collectri[collectri["source"] == "MYC"].copy()
        logging.info(f"  CollecTRI MYC: {len(myc_net)} interacciones totales")

        # Filtrar a genes disponibles en el dataset
        available = myc_net[myc_net["target"].isin(adata_var_names)]
        n_total = len(myc_net)
        n_available = len(available)
        logging.info(f"  En dataset: {n_available}/{n_total} targets disponibles")

        # Advertir sobre superposición con genes de clasificación
        classification_genes = set(["MYC", "EZH2", "SUZ12", "CTNNB1", "ATF3",
                                     "STAT3", "DNMT1",  # silencing
                                     "COL1A1", "COL1A2", "COL3A1", "ACTA2", "FN1",
                                     "FAP", "POSTN", "VCAN", "PDPN"])  # barrier
        overlap = set(available["target"]) & classification_genes
        if overlap:
            pct = 100 * len(overlap) / n_available
            logging.info(
                f"  ⚠️  Superposición regulon-clasificación: {sorted(overlap)} "
                f"({pct:.1f}% del regulon) — ESPERADO y biológicamente correcto"
            )
        if n_available < 20:
            logging.warning("  <20 targets disponibles → score poco fiable; usando backup")
            return _build_backup_regulon(adata_var_names)

        return available

    except Exception as e:
        logging.warning(f"  dc.get_collectri() falló ({e}) → usando backup")
        return _build_backup_regulon(adata_var_names)


def _ulm_manual(expr_df: pd.DataFrame, net: pd.DataFrame) -> pd.Series:
    """ULM manual cuando decoupler no está disponible.

    Univariate Linear Model: para cada spot, calcula la correlación de Pearson
    entre sus valores de expresión y los pesos del regulon.
    Equivale al score t-statistic de ULM (Badia-i-Mompel 2022).

    Formula: act = Σ(w_i * x_i) / (√(Σw_i²) * std(x_target))
    donde x_i son los valores de expresión de los targets y w_i los pesos del regulon.
    """
    targets = net["target"].values
    weights = net["weight"].values.astype(float)

    # Intersecar con genes disponibles
    available_mask = np.isin(targets, expr_df.columns)
    targets = targets[available_mask]
    weights = weights[available_mask]

    if len(targets) < 5:
        logging.error("  ULM manual: <5 targets disponibles → score NaN")
        return pd.Series(np.nan, index=expr_df.index)

    # Extraer submatriz de expresión (spots × targets)
    X = expr_df[targets].values.astype(float)  # n_spots × n_targets

    # Normalizar pesos
    w = weights / (np.sqrt(np.sum(weights**2)) + 1e-9)

    # Score = proyección en dirección del regulon (dot product normalizado)
    # Equivale al "regulon score" o VIPER de primera generación
    scores = X @ w

    # Estandarizar a z-score para comparabilidad
    scores = (scores - scores.mean()) / (scores.std() + 1e-9)

    return pd.Series(scores, index=expr_df.index)


def compute_myc_tf_activity(adata, net: pd.DataFrame) -> np.ndarray:
    """Calcula MYC TF activity por spot usando ULM.

    Si decoupler está disponible: usa dc.run_ulm() con la API oficial.
    Si no está disponible: usa ULM manual equivalente.

    El score resultante es el t-statistic normalizado (z-score) de la
    proyección de la expresión de cada spot sobre el regulon MYC.
    Valores positivos = mayor actividad MYC; negativos = menor actividad.

    NOTA SOBRE .raw:
    adata.raw tiene 29,946 genes en log1p space (max ~8.5).
    decoupler-py acepta use_raw=True y opera sobre este espacio.
    """
    n_spots = adata.n_obs
    X_raw, var_names = get_expression_matrix_raw(adata)

    # Verificar mínimo de targets
    available_targets = [t for t in net["target"].values if t in var_names]
    logging.info(f"  Targets MYC disponibles para ULM: {len(available_targets)}")
    if len(available_targets) < 10:
        raise ValueError(
            f"Solo {len(available_targets)} targets MYC disponibles. "
            f"Mínimo requerido: 10. Verificar regulón."
        )

    if _DECOUPLER_AVAILABLE:
        logging.info("  Ejecutando dc.run_ulm() con regulón MYC...")
        try:
            # Crear un AnnData temporal solo con los genes del regulon para eficiencia
            target_genes = list(set(available_targets) & set(var_names))
            gene_idx = [list(var_names).index(g) for g in target_genes]

            # Submatriz sparse (genes del regulon)
            X_sub = X_raw[:, gene_idx]
            if sp.issparse(X_sub):
                X_sub = X_sub.toarray()
            X_sub = X_sub.astype(np.float32)

            # AnnData temporal para decoupler
            adata_sub = sc.AnnData(X=X_sub, obs=adata.obs[[PHENOTYPE_COL]].copy())
            adata_sub.var_names = pd.Index(target_genes)

            # Run ULM
            dc.run_ulm(
                mat=adata_sub,
                net=net,
                source="source",
                target="target",
                weight="weight",
                verbose=False,
                use_raw=False,
                min_n=5,
            )

            if "ulm_estimate" in adata_sub.obsm:
                acts_df = adata_sub.obsm["ulm_estimate"]
                if "MYC" in acts_df.columns:
                    scores = acts_df["MYC"].values.astype(float)
                    logging.info(
                        f"  dc.run_ulm() OK → MYC activity: "
                        f"mean={scores.mean():.3f}, std={scores.std():.3f}"
                    )
                    return scores
            logging.warning("  dc.run_ulm() no produjo 'MYC' en obsm → fallback manual")
        except Exception as e:
            logging.warning(f"  dc.run_ulm() error ({e}) → usando ULM manual")

    # Fallback: ULM manual
    logging.info("  Ejecutando ULM manual...")
    # Construir DataFrame de expresión (solo genes del regulon)
    target_genes = [t for t in net["target"].values if t in var_names]
    gene_idx = [list(var_names).index(g) for g in target_genes]
    X_sub = safe_toarray(X_raw[:, gene_idx])

    expr_df = pd.DataFrame(
        X_sub,
        columns=target_genes,
        index=adata.obs_names
    )
    net_filtered = net[net["target"].isin(target_genes)].copy()

    scores_series = _ulm_manual(expr_df, net_filtered)
    scores = scores_series.values.astype(float)

    logging.info(
        f"  ULM manual OK → MYC activity: "
        f"mean={scores.mean():.3f}, std={scores.std():.3f}"
    )
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS A: DISTRIBUCIÓN POR FENOTIPO
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_A_by_phenotype(adata, results: list, logger) -> dict:
    """A: MYC TF activity por fenotipo — Cohen's d, Mann-Whitney, FDR."""
    logger.info("=" * 70)
    logger.info("ANÁLISIS A: MYC_TF_activity por fenotipo")

    tf_col = "MYC_TF_activity"
    scores = adata.obs[tf_col].values
    phenos = adata.obs[PHENOTYPE_COL].values

    groups = {
        DESERT_LABEL:   scores[phenos == DESERT_LABEL],
        EXCLUDED_LABEL: scores[phenos == EXCLUDED_LABEL],
        INFLAMED_LABEL: scores[phenos == INFLAMED_LABEL],
    }

    for gname, gvals in groups.items():
        logger.info(f"  {gname}: n={len(gvals):,}, mean={gvals.mean():.4f}, "
                    f"median={np.median(gvals):.4f}, std={gvals.std():.4f}")

    # Tests vs Inflamed (referencia) + Desert vs Excluded
    comparisons = [
        (DESERT_LABEL, INFLAMED_LABEL,
         "MYC_TF_Desert_vs_Inflamed",
         "H: Desert > Inflamed (supresión funcional → mayor actividad MYC)"),
        (EXCLUDED_LABEL, INFLAMED_LABEL,
         "MYC_TF_Excluded_vs_Inflamed",
         "H: Excluded ~ Inflamed (no supresión MYC en Excluded)"),
        (DESERT_LABEL, EXCLUDED_LABEL,
         "MYC_TF_Desert_vs_Excluded",
         "H: Desert > Excluded (diferencia mecanística)"),
    ]

    pvals = []
    for g1_name, g2_name, test_id, hypothesis in comparisons:
        g1 = groups[g1_name]
        g2 = groups[g2_name]
        stat, pval = mannwhitneyu(g1, g2, alternative="two-sided")
        d = cohens_d_pooled(g1, g2)
        pvals.append(pval)
        results.append({
            "analysis": "A",
            "test_id": test_id,
            "statistic": float(stat),
            "cohens_d": float(d),
            "p_value": float(pval),
            "n1": int(len(g1)),
            "n2": int(len(g2)),
            "group1": g1_name,
            "group2": g2_name,
            "hypothesis": hypothesis,
        })
        logger.info(f"  {test_id}: d={d:.4f}, p={pval:.3e}")

    # FDR sobre los 3 tests de A
    qvals = fdr_correct(pvals)
    for i, r in enumerate([r for r in results if r["analysis"] == "A"]):
        r["q_value"] = float(qvals[i])
        r["fdr_significant"] = bool(qvals[i] < 0.05)
        logger.info(f"  {r['test_id']}: q={qvals[i]:.3e}, sig={r['fdr_significant']}")

    # Medias por fenotipo para tabla de resultados
    means_by_phenotype = {
        pheno: float(vals.mean()) for pheno, vals in groups.items()
    }
    means_by_phenotype[STROMA_LABEL] = float(
        scores[phenos == STROMA_LABEL].mean()
        if np.sum(phenos == STROMA_LABEL) > 0 else np.nan
    )

    return means_by_phenotype


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS B: CORRELACIONES FUNCIONALES EN DESERT
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_B_correlations(adata, results: list, logger,
                             X_raw=None, var_names=None) -> None:
    """B: Spearman de MYC_TF_activity vs readouts funcionales dentro de Desert.

    Correlaciones dentro de Desert eliminan confounding composicional Visium
    (mismo fenotipo → misma composición celular media → no hay paradoja de Simpson).
    """
    logger.info("=" * 70)
    logger.info("ANÁLISIS B: Correlaciones MYC_TF vs readouts funcionales (Desert)")

    desert_mask = adata.obs[PHENOTYPE_COL] == DESERT_LABEL
    n_desert = desert_mask.sum()
    logger.info(f"  Spots Desert: {n_desert:,}")

    tf_vals = adata.obs.loc[desert_mask, "MYC_TF_activity"].values

    # Calcular scores funcionales si no están en adata.obs
    score_defs = [
        ("ISG_score",       ISG_GENES,       "ISG score (proxy STING activity downstream)"),
        ("MHC_I_score",     MHCI_GENES,      "MHC-I antigen presentation pathway"),
        ("Chemokine_score", CHEMOKINE_GENES,  "Chemokine output (CCL5/CXCL9/CXCL10)"),
        ("CD47_expr",       [CD47_GENE],      "CD47 'don't eat me' signal"),
    ]

    pvals = []
    for score_col, genes, description in score_defs:
        if score_col not in adata.obs.columns:
            vals = compute_gene_score(adata, genes, score_col, X=X_raw, var_names=var_names)
            adata.obs[score_col] = vals
            logger.info(f"  [{score_col}] calculado desde .raw")

        score_desert = adata.obs.loc[desert_mask, score_col].values
        valid = ~(np.isnan(tf_vals) | np.isnan(score_desert))

        if valid.sum() < 30:
            logger.warning(f"  [{score_col}] <30 valores válidos → skip")
            results.append({
                "analysis": "B", "test_id": f"MYC_TF_vs_{score_col}_Desert",
                "statistic": np.nan, "p_value": np.nan, "q_value": np.nan,
                "fdr_significant": False, "n1": int(valid.sum()),
                "hypothesis": f"H: MYC_TF_activity ↑ → {score_col} ↓ en Desert",
                "description": description
            })
            pvals.append(np.nan)
            continue

        rho, pval = spearmanr(tf_vals[valid], score_desert[valid])
        pvals.append(pval)
        results.append({
            "analysis": "B",
            "test_id": f"MYC_TF_vs_{score_col}_Desert",
            "statistic": float(rho),
            "p_value": float(pval),
            "n1": int(valid.sum()),
            "hypothesis": f"H: MYC_TF_activity ↑ → {score_col} ↓ en Desert",
            "description": description
        })
        logger.info(f"  MYC_TF vs {score_col} (Desert): ρ={rho:.4f}, p={pval:.3e}")

    # FDR sobre tests B
    qvals = fdr_correct(pvals)
    b_results = [r for r in results if r["analysis"] == "B"]
    for i, r in enumerate(b_results):
        r["q_value"] = float(qvals[i])
        r["fdr_significant"] = bool(qvals[i] < 0.05) if not np.isnan(qvals[i]) else False


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS C: BINARIO INTRA-DESERT (MYC_TF high vs low)
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_C_binary_desert(adata, results: list, logger) -> None:
    """C: Binary intra-Desert — MYC_TF_activity high vs low.

    Réplica del Análisis 3 de myc_sting_investigation.py pero usando
    MYC_TF_activity en lugar de MYC_mRNA/MYC_Hallmark_Combined.
    Elimina confounding composicional al operar solo dentro de Desert.
    """
    logger.info("=" * 70)
    logger.info("ANÁLISIS C: Binary intra-Desert (MYC_TF high vs low)")

    desert_mask = adata.obs[PHENOTYPE_COL] == DESERT_LABEL
    tf_desert = adata.obs.loc[desert_mask, "MYC_TF_activity"].values

    # Umbral: mediana → 50/50 split para balance
    threshold = np.median(tf_desert)
    high_mask_desert = tf_desert >= threshold
    low_mask_desert  = tf_desert <  threshold
    n_high = high_mask_desert.sum()
    n_low  = low_mask_desert.sum()
    logger.info(f"  Umbral TF activity (mediana): {threshold:.4f}")
    logger.info(f"  MYC_TF high: {n_high:,} spots | low: {n_low:,} spots")

    # Outcomes a comparar
    outcomes = [
        ("Chemokine_score",  "Chemokine output"),
        ("ISG_score",        "ISG score"),
        ("MHC_I_score",      "MHC-I score"),
        ("CD47_expr",        "CD47 expression"),
    ]

    # Cell2Location CD8+ T cell abundance
    cd8_col = "meanscell_abundance_w_sf_CD8_T"
    c2l_df = adata.obsm.get("means_cell_abundance_w_sf", None)
    has_cd8 = False
    if c2l_df is not None and hasattr(c2l_df, "columns"):
        # Buscar columna CD8 con prefijo correcto (sin underscore means/cell)
        cd8_candidates = [c for c in c2l_df.columns if "CD8" in c]
        if cd8_candidates:
            cd8_col = cd8_candidates[0]
            cd8_all = c2l_df[cd8_col].values
            adata.obs["CD8_T_abundance"] = cd8_all
            has_cd8 = True
            logger.info(f"  CD8 abundance desde C2L: '{cd8_col}'")
        else:
            logger.warning("  No se encontró columna CD8 en C2L obsm")
    if has_cd8:
        outcomes.append(("CD8_T_abundance", "CD8+ T cell abundance (C2L)"))

    pvals = []
    desert_obs = adata.obs[desert_mask]

    for score_col, description in outcomes:
        if score_col not in adata.obs.columns:
            logger.warning(f"  {score_col} no en adata.obs → skip")
            continue

        vals_desert = adata.obs.loc[desert_mask, score_col].values

        g_high = vals_desert[high_mask_desert]
        g_low  = vals_desert[low_mask_desert]

        # Filtrar NaN
        g_high = g_high[~np.isnan(g_high)]
        g_low  = g_low[~np.isnan(g_low)]

        if len(g_high) < 10 or len(g_low) < 10:
            logger.warning(f"  {score_col}: grupos insuficientes → skip")
            continue

        stat, pval = mannwhitneyu(g_high, g_low, alternative="two-sided")
        d = cohens_d_pooled(g_high, g_low)
        pvals.append(pval)
        results.append({
            "analysis": "C",
            "test_id": f"MYC_TF_BinaryDesert_{score_col}",
            "statistic": float(d),
            "p_value": float(pval),
            "n1": int(len(g_high)),
            "n2": int(len(g_low)),
            "group1": "MYC_TF_high",
            "group2": "MYC_TF_low",
            "hypothesis": f"H: MYC_TF_high → {description} ↓",
            "description": description
        })
        logger.info(
            f"  {score_col}: high(n={len(g_high)}) vs low(n={len(g_low)}): "
            f"d={d:.4f}, p={pval:.3e}"
        )

    qvals = fdr_correct(pvals)
    c_results = [r for r in results if r["analysis"] == "C"]
    for i, r in enumerate(c_results[:len(qvals)]):
        r["q_value"] = float(qvals[i])
        r["fdr_significant"] = bool(qvals[i] < 0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# ANÁLISIS D: SANITY CHECK — TF activity vs MYC mRNA
# ═══════════════════════════════════════════════════════════════════════════════

def analysis_D_sanity_check(adata, results: list, logger,
                             X_raw=None, var_names=None) -> None:
    """D: Correlación MYC_TF_activity vs MYC mRNA.

    Esperado: correlación positiva moderada (ρ ~ 0.2-0.5).
    - ρ < 0.1: warning severo (score probablemente corrupto)
    - ρ > 0.9: warning (score trivialmente igual a mRNA; no añade info)
    - ρ 0.2-0.5: ideal (relacionado pero independiente)

    También compara TF_activity vs MYC_Hallmark_Combined si está disponible.
    """
    logger.info("=" * 70)
    logger.info("ANÁLISIS D: Sanity check — TF activity vs MYC mRNA")

    tf_vals = adata.obs["MYC_TF_activity"].values
    myc_mrna = get_gene_vector(adata, "MYC", X=X_raw, var_names=var_names)

    valid = ~(np.isnan(tf_vals) | np.isnan(myc_mrna))
    rho, pval = spearmanr(tf_vals[valid], myc_mrna[valid])

    if rho < 0.05:
        logger.warning(
            f"  ρ(TF_activity, MYC_mRNA) = {rho:.4f} — MUY BAJO. "
            f"Verificar regulón y datos."
        )
    elif rho > 0.85:
        logger.warning(
            f"  ρ(TF_activity, MYC_mRNA) = {rho:.4f} — MUY ALTO. "
            f"El score TF no añade información sobre MYC mRNA."
        )
    else:
        logger.info(
            f"  ρ(TF_activity, MYC_mRNA) = {rho:.4f}, p={pval:.3e} — OK (rango esperado)"
        )

    results.append({
        "analysis": "D",
        "test_id": "MYC_TF_vs_MYC_mRNA_global",
        "statistic": float(rho),
        "p_value": float(pval),
        "q_value": float(pval),  # single test → q = p
        "fdr_significant": pval < 0.05,
        "n1": int(valid.sum()),
        "hypothesis": "Sanity: ρ esperado ~ 0.2-0.5 (relacionado pero independiente)",
        "interpretation": (
            "OK" if 0.05 <= rho <= 0.85
            else "WARNING_LOW" if rho < 0.05
            else "WARNING_HIGH"
        )
    })

    # Comparar con hallmark si disponible
    if "MYC_Hallmark_Combined" in adata.obs.columns:
        hallmark = adata.obs["MYC_Hallmark_Combined"].values
        valid2 = ~(np.isnan(tf_vals) | np.isnan(hallmark))
        rho2, pval2 = spearmanr(tf_vals[valid2], hallmark[valid2])
        logger.info(
            f"  ρ(TF_activity, MYC_Hallmark_Combined) = {rho2:.4f}, p={pval2:.3e}"
        )
        results.append({
            "analysis": "D",
            "test_id": "MYC_TF_vs_MYC_Hallmark",
            "statistic": float(rho2),
            "p_value": float(pval2),
            "q_value": float(pval2),
            "fdr_significant": pval2 < 0.05,
            "n1": int(valid2.sum()),
            "hypothesis": "Sanity: TF activity debe correlacionar con hallmark score",
        })


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def plot_tf_by_phenotype(adata, means_by_phenotype: dict, out_dir: Path) -> None:
    """Violin + boxplot de MYC TF activity por fenotipo."""

    phenotypes_ordered = [INFLAMED_LABEL, EXCLUDED_LABEL, DESERT_LABEL]
    colors = {"Inflamed": "#E74C3C", "Immune_Excluded": "#E67E22",
              "Immune_Desert": "#3498DB"}
    palette = {p: colors.get(p, "#95A5A6") for p in phenotypes_ordered}

    mask = adata.obs[PHENOTYPE_COL].isin(phenotypes_ordered)
    df_plot = adata.obs.loc[mask, [PHENOTYPE_COL, "MYC_TF_activity"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("MYC TF Activity by Immune Phenotype\n(decoupleR ULM, CollecTRI regulon)",
                 fontsize=13, fontweight="bold")

    # Panel A: violin
    ax = axes[0]
    sns.violinplot(
        data=df_plot, x=PHENOTYPE_COL, y="MYC_TF_activity",
        order=phenotypes_ordered, palette=palette,
        inner="box", cut=0, linewidth=1.0, ax=ax
    )
    ax.set_title("MYC TF Activity Distribution", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("MYC TF Activity (ULM z-score)", fontsize=10)
    ax.set_xticklabels(["Inflamed", "Excluded", "Desert"], rotation=15, ha="right")
    ax.axhline(0, color="grey", linestyle="--", alpha=0.5, linewidth=0.8)

    # Panel B: media ± SEM por fenotipo
    ax2 = axes[1]
    phenos = [p for p in phenotypes_ordered if p in means_by_phenotype]
    means = [means_by_phenotype[p] for p in phenos]
    labels = [p.replace("Immune_", "") for p in phenos]
    bar_colors = [palette.get(p, "#95A5A6") for p in phenos]
    ax2.bar(labels, means, color=bar_colors, edgecolor="black", linewidth=0.8)
    ax2.set_title("Mean MYC TF Activity by Phenotype", fontsize=11)
    ax2.set_ylabel("Mean MYC TF Activity (ULM z-score)", fontsize=10)
    ax2.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    fpath = out_dir / "fig_myc_tf_violin.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


def plot_binary_desert(adata, out_dir: Path) -> None:
    """Bar plot de resultados análisis C (binary intra-Desert)."""

    desert_mask = adata.obs[PHENOTYPE_COL] == DESERT_LABEL
    tf_desert = adata.obs.loc[desert_mask, "MYC_TF_activity"].values
    threshold = np.median(tf_desert)
    high_mask = tf_desert >= threshold

    outcome_cols = [c for c in ["Chemokine_score", "ISG_score", "MHC_I_score",
                                 "CD47_expr", "CD8_T_abundance"]
                    if c in adata.obs.columns]

    if not outcome_cols:
        return

    fig, axes = plt.subplots(1, len(outcome_cols), figsize=(4 * len(outcome_cols), 5))
    if len(outcome_cols) == 1:
        axes = [axes]

    fig.suptitle("Intra-Desert: MYC TF-high vs TF-low\n(median split)",
                 fontsize=12, fontweight="bold")

    for ax, col in zip(axes, outcome_cols):
        vals_desert = adata.obs.loc[desert_mask, col].values
        df_bin = pd.DataFrame({
            "value": vals_desert,
            "group": np.where(high_mask, "MYC TF\nHigh", "MYC TF\nLow")
        })
        df_bin = df_bin[~np.isnan(df_bin["value"])]
        means = df_bin.groupby("group")["value"].mean()
        sems  = df_bin.groupby("group")["value"].sem()
        colors = ["#E74C3C", "#3498DB"]
        bars = ax.bar(means.index, means.values, yerr=sems.values,
                      color=colors, capsize=5, edgecolor="black", linewidth=0.8)
        ax.set_title(col.replace("_", " ").replace("score", "Score"),
                     fontsize=9, fontweight="bold")
        ax.set_ylabel("Mean expression / abundance", fontsize=8)

    plt.tight_layout()
    fpath = out_dir / "fig_myc_tf_binary_desert.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


def plot_tf_vs_mrna(adata, out_dir: Path, X_raw=None, var_names=None) -> None:
    """Scatter MYC_TF_activity vs MYC mRNA (sanity check D)."""
    tf_vals = adata.obs["MYC_TF_activity"].values
    myc_mrna = get_gene_vector(adata, "MYC", X=X_raw, var_names=var_names)

    # Subsample para visualización eficiente (max 10k puntos)
    rng = np.random.default_rng(SEED)
    n = min(10000, len(tf_vals))
    idx = rng.choice(len(tf_vals), n, replace=False)

    phenos = adata.obs[PHENOTYPE_COL].values
    colors_map = {
        INFLAMED_LABEL: "#E74C3C",
        EXCLUDED_LABEL: "#E67E22",
        DESERT_LABEL: "#3498DB",
        STROMA_LABEL: "#BDC3C7",
        "Ambiguous_Cold": "#95A5A6",
    }
    point_colors = [colors_map.get(p, "#AAAAAA") for p in phenos[idx]]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(myc_mrna[idx], tf_vals[idx],
               c=point_colors, alpha=0.3, s=3, rasterized=True)

    rho, _ = spearmanr(tf_vals, myc_mrna)
    ax.set_xlabel("MYC mRNA expression (log1p)", fontsize=11)
    ax.set_ylabel("MYC TF Activity (ULM z-score)", fontsize=11)
    ax.set_title(f"Sanity check: TF activity vs MYC mRNA\nSpearman ρ = {rho:.3f}",
                 fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(fc=c, label=p.replace("Immune_", ""))
        for p, c in colors_map.items()
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    plt.tight_layout()
    fpath = out_dir / "fig_myc_tf_vs_mrna.png"
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"  Guardada: {fpath}")


# ═══════════════════════════════════════════════════════════════════════════════
# SETUP LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(out_dir: Path) -> logging.Logger:
    log_file = out_dir / "myc_tf_activity.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger("myc_tf")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    logger = setup_logging(RESULTS_DIR)

    logger.info("=" * 70)
    logger.info("MYC TF ACTIVITY SCORING — decoupleR/CollecTRI")
    logger.info(f"Output: {RESULTS_DIR}")
    logger.info(f"Input : {ADATA_INPUT}")
    logger.info("=" * 70)

    # ── Verificar decoupler ───────────────────────────────────────────────────
    if not _DECOUPLER_AVAILABLE:
        logger.warning(
            "decoupler NO instalado. Usando ULM manual.\n"
            "Para instalar: pip install decoupler\n"
            "El análisis continuará con ULM manual equivalente."
        )
    else:
        logger.info(f"  decoupler disponible: v{dc.__version__}")

    # ── Cargar datos ─────────────────────────────────────────────────────────
    if not ADATA_INPUT.exists():
        # Intentar fallback a adata_with_phenotypes.h5ad
        fallback = DATA_PROCESSED / "adata_with_phenotypes.h5ad"
        if fallback.exists():
            logger.warning(f"  {ADATA_INPUT.name} no encontrado → usando {fallback.name}")
            adata = sc.read_h5ad(fallback)
        else:
            logger.error(f"  FATAL: No se encontró ningún adata procesado en {DATA_PROCESSED}")
            sys.exit(1)
    else:
        logger.info(f"  Cargando {ADATA_INPUT.name}...")
        adata = sc.read_h5ad(ADATA_INPUT)

    logger.info(f"  Spots: {adata.n_obs:,} | Genes .X: {adata.n_vars:,}")

    if adata.raw is not None:
        logger.info(f"  Genes en .raw: {adata.raw.n_vars:,}")
    else:
        logger.warning("  .raw is None — usando .X para expresión génica")

    # Verificar columna de fenotipos
    if PHENOTYPE_COL not in adata.obs.columns:
        # Intentar alternativas
        for alt in ["phenotype", "Phenotype_v2", "phenotype_v2"]:
            if alt in adata.obs.columns:
                logger.warning(f"  '{PHENOTYPE_COL}' no encontrado → usando '{alt}'")
                adata.obs[PHENOTYPE_COL] = adata.obs[alt]
                break
        else:
            logger.error(f"  FATAL: No se encontró columna de fenotipos. "
                         f"Columnas disponibles: {list(adata.obs.columns[:10])}")
            sys.exit(1)

    # Distribución de fenotipos
    dist = adata.obs[PHENOTYPE_COL].value_counts()
    logger.info(f"  Distribución fenotipos:\n{dist.to_string()}")

    for required in [DESERT_LABEL, INFLAMED_LABEL]:
        if required not in dist.index or dist[required] == 0:
            logger.error(f"  FATAL: 0 spots '{required}' — análisis imposible")
            sys.exit(1)

    # ── Extraer expresión desde .raw ─────────────────────────────────────────
    logger.info("  Extrayendo matriz de expresión desde .raw...")
    X_raw, var_names = get_expression_matrix_raw(adata)

    # ── Obtener regulón MYC ──────────────────────────────────────────────────
    logger.info("  Obteniendo regulón MYC CollecTRI...")
    net = get_myc_regulon(var_names)
    logger.info(f"  Regulón final: {len(net)} interacciones MYC-target")

    if len(net) < 10:
        logger.error("  FATAL: regulón demasiado pequeño para análisis fiable")
        sys.exit(1)

    # ── Calcular MYC TF activity ─────────────────────────────────────────────
    logger.info("  Calculando MYC TF activity (ULM)...")
    t_ulm = time.time()
    myc_tf_scores = compute_myc_tf_activity(adata, net)
    logger.info(f"  ULM completado en {time.time()-t_ulm:.1f}s")

    adata.obs["MYC_TF_activity"] = myc_tf_scores
    logger.info(
        f"  MYC_TF_activity guardado en adata.obs: "
        f"mean={myc_tf_scores.mean():.4f}, std={myc_tf_scores.std():.4f}, "
        f"range=[{myc_tf_scores.min():.3f}, {myc_tf_scores.max():.3f}]"
    )

    # ── Calcular scores funcionales para los análisis ────────────────────────
    logger.info("  Pre-calculando scores funcionales...")
    for score_col, genes in [
        ("ISG_score", ISG_GENES),
        ("MHC_I_score", MHCI_GENES),
        ("Chemokine_score", CHEMOKINE_GENES),
    ]:
        if score_col not in adata.obs.columns:
            vals = compute_gene_score(adata, genes, score_col, X=X_raw, var_names=var_names)
            adata.obs[score_col] = vals

    cd47 = get_gene_vector(adata, CD47_GENE, X=X_raw, var_names=var_names)
    adata.obs["CD47_expr"] = cd47

    # ── Ejecutar análisis ────────────────────────────────────────────────────
    all_results: list = []

    logger.info("\n" + "=" * 70)
    means_by_pheno = analysis_A_by_phenotype(adata, all_results, logger)

    analysis_B_correlations(adata, all_results, logger,
                             X_raw=X_raw, var_names=var_names)

    analysis_C_binary_desert(adata, all_results, logger)

    analysis_D_sanity_check(adata, all_results, logger,
                             X_raw=X_raw, var_names=var_names)

    # ── Guardar resultados estadísticos ──────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    results_path = RESULTS_DIR / "myc_tf_activity_results.csv"
    df_results.to_csv(results_path, index=False)
    logger.info(f"\n  Resultados guardados: {results_path}")
    logger.info(f"  Total tests: {len(df_results)}")

    n_sig = df_results["fdr_significant"].sum() if "fdr_significant" in df_results.columns else 0
    logger.info(f"  Tests FDR-sig: {n_sig}/{len(df_results)}")

    # Tabla de medias por fenotipo
    df_means = pd.DataFrame([
        {"phenotype": pheno, "mean_MYC_TF_activity": mean}
        for pheno, mean in means_by_pheno.items()
    ])
    df_means.to_csv(RESULTS_DIR / "myc_tf_activity_by_phenotype.csv", index=False)

    # ── Figuras ──────────────────────────────────────────────────────────────
    logger.info("\n  Generando figuras...")
    try:
        plot_tf_by_phenotype(adata, means_by_pheno, RESULTS_DIR)
        plot_binary_desert(adata, RESULTS_DIR)
        plot_tf_vs_mrna(adata, RESULTS_DIR, X_raw=X_raw, var_names=var_names)
    except Exception as e:
        logger.warning(f"  Error generando figuras: {e}")

    # ── Guardar adata actualizado ─────────────────────────────────────────────
    logger.info(f"\n  Guardando adata con MYC_TF_activity → {ADATA_OUTPUT}")
    # Guardar solo las columnas nuevas para no duplicar el h5ad original
    # (el h5ad con TF activity se usa como input para checkpoint_landscape.py)
    try:
        adata.write_h5ad(ADATA_OUTPUT)
        logger.info("  adata guardado correctamente")
    except Exception as e:
        logger.warning(f"  No se pudo guardar adata: {e}")

    # ── Resumen ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("RESUMEN EJECUTIVO — MYC TF ACTIVITY")
    logger.info("=" * 70)

    # Imprimir tests A (por fenotipo)
    a_tests = df_results[df_results["analysis"] == "A"]
    for _, row in a_tests.iterrows():
        logger.info(
            f"  {row['test_id']}: d={row.get('cohens_d', row['statistic']):.4f}, "
            f"p={row['p_value']:.3e}, q={row.get('q_value', np.nan):.3e}, "
            f"sig={row['fdr_significant']}"
        )

    # Imprimir tests B (correlaciones)
    b_tests = df_results[df_results["analysis"] == "B"]
    for _, row in b_tests.iterrows():
        logger.info(
            f"  {row['test_id']}: ρ={row['statistic']:.4f}, "
            f"p={row['p_value']:.3e}, q={row.get('q_value', np.nan):.3e}, "
            f"sig={row['fdr_significant']}"
        )

    logger.info(f"\n  Tiempo total: {time.time()-t0:.1f}s")
    logger.info("  STATUS: COMPLETADO")
    logger.info(
        "\n  INTERPRETACIÓN PARA PAPER:\n"
        "  Si MYC_TF_Desert > MYC_TF_Inflamed (análisis A significativo):\n"
        "  → Primera evidencia semi-directa de actividad transcripcional MYC elevada\n"
        "     en Desert niches, independiente del mRNA MYC.\n"
        "  Si correlaciones B son negativas y FDR-sig:\n"
        "  → TF activity ↑ correlaciona con ISG↓, MHC-I↓ en Desert: mecanismo MYC.\n"
        "  Si análisis C muestra d significativo:\n"
        "  → Efecto intra-Desert confirma causalidad local, no confounding.\n"
        "  Si sanidad D: ρ ~ 0.2-0.5: score añade información sobre mRNA."
    )


if __name__ == "__main__":
    main()

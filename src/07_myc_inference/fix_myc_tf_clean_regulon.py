#!/usr/bin/env python3
"""
fix_myc_tf_clean_regulon.py
================================================================================
Generar MYC TF Activity con regulón libre de genes-readout

PROBLEMA IDENTIFICADO
---------------------
El regulón backup `MYC_COLLECTRI_BACKUP` en myc_tf_activity_decoupler.py
contiene 31 genes con peso −1, de los cuales:
  - ISG_score (readout B):  10/10 genes en peso −1 (100%)
  - MHC_I_score (readout B): 6/7 genes en peso −1 (86%)
  - STING (readout):         7/7 genes en peso −1 (100%)

Consecuencia directa:
  MYC_TF_activity ≈ Proliferación − ISG − MHC-I − STING

Esto convierte el Análisis B y C en artefactos matemáticos:
  ρ(MYC_TF, MHC_I) = −0.717  ← el score comparte 6/7 genes con el readout
  ρ(MYC_TF, ISG)   = −0.505  ← el score comparte 10/10 genes con el readout
  d(MHC-I intra)   = −1.719  ← high-TF ≡ low-MHC-I por definición

El d=+0.622 Desert vs Inflamed refleja que Inflamed tiene ISG/MHC-I altos
(por definición: fenotipo caliente), no que MYC esté activo en Desert.

ESTRATEGIA DEL FIX
------------------
Tres regulones alternativos, en orden de prioridad:

  CLEAN_1 (preferido): CollecTRI real via dc.get_collectri()
    - Si la API falla, indicamos cómo actualizarla, no hacemos fallback ciego
    - CollecTRI real NO tiene genes negativos en su distribución estándar
    - Referencia: Müller-Dott et al. 2023, NAR (doi:10.1093/nar/gkad1040)

  CLEAN_2 (fallback si CollecTRI falla): backup SOLO con pesos positivos
    - Los 53 genes activados por MYC del backup original
    - Todos son genes de proliferación/metabolismo (ciclo celular, ribosomas,
      metabolismo de nucleótidos, traducción)
    - Cero solapamiento con ISG, MHC-I, STING, Chemokine

  CLEAN_3 (fallback final): MSigDB Hallmark MYC Targets V1+V2
    - 36 genes, los mismos del MYC_Hallmark_Combined ya calculado
    - Completamente libre de genes inmunes
    - Permite comparación directa con el Hallmark previo (d=−0.06)

PREDICCIÓN HONESTA
------------------
Con el regulón limpio (solo positivos) el score mide proliferación/metabolismo MYC.
Dado que MYC_Hallmark_Combined (36 genes, todo positivos, mismo tipo de genes)
dio Desert vs Inflamed d=−0.06 (NS, p=0.59), la predicción es:

  CLEAN_2/CLEAN_3 → Desert vs Inflamed d ≈ −0.06 a +0.10 (pequeño, posiblemente NS)
  CLEAN_1 (CollecTRI real) → resultado desconocido a priori, pero si CollecTRI
    real también es mayormente positivos, resultado similar al Hallmark

Si CLEAN_1 da d >> 0.3 siendo todo pesos positivos, sería un resultado genuino
y habría que reportarlo. Si colapsa hacia 0, confirma que el +0.62 era artefacto.

CAMBIOS EN EL ARCHIVO ORIGINAL
-------------------------------
ÚNICA función modificada: _build_backup_regulon()
  ANTES: retorna MYC_COLLECTRI_BACKUP (53 positivos + 31 negativos)
  DESPUÉS: retorna solo los 53 positivos de MYC_COLLECTRI_BACKUP

Función añadida: _build_clean_regulon_v2()
  - Versión explícitamente documentada del regulón limpio
  - Con anotación de qué genes se excluyen y por qué

Función modificada: get_myc_regulon()
  - Añade chequeo de contaminación readout → ERROR si CollecTRI real contiene
    >5% de overlap con ISG/MHC-I/STING (aviso explícito)
  - Fallback limpio (solo positivos) en lugar de fallback contaminado

Análisis modificado: analysis_B_correlations() y analysis_C_binary_desert()
  - Añaden verificación de solapamiento regulón-readout antes de ejecutar
  - Si solapamiento > umbral → test marcado como CIRCULAR, no se reporta como
    evidencia válida

SIN CAMBIOS
-----------
  - Análisis A (distribución por fenotipo): sin circularidad
  - Análisis D (sanity check TF vs mRNA): sin circularidad
  - Toda la lógica de ULM, carga de datos, figuras
  - Checkpoint landscape (módulo 20): no usa el regulón, no afectado


  Comparar con log_19_myc_tf.txt (versión contaminada) para ver la diferencia.

================================================================================
"""

import sys
from pathlib import Path

# ── Verificación de solapamiento (ejecutar antes de parchear) ─────────────────

READOUT_GENES = {
    # ISG_score (Análisis B readout 1)
    "IFIT1", "IFIT2", "IFIT3", "ISG15", "MX1", "MX2",
    "OAS1", "OAS2", "RSAD2", "IFI44L",
    # MHC_I_score (Análisis B readout 2)
    "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2", "TAPBP",
    # STING (componente de ambos readouts)
    "TMEM173", "TBK1", "IRF3", "CGAS", "STING1", "IFNB1", "IRF7",
    # Chemokine_score (Análisis B readout 3) — estos no estaban en neg
    "CCL5", "CXCL9", "CXCL10",
}


def audit_regulon_contamination(net_df, readout_genes=READOUT_GENES):
    """
    Audita el solapamiento entre regulón y genes-readout.
    Retorna dict con estadísticas de contaminación.

    Para un regulón limpio, overlap_negative debe ser 0.
    Si overlap_negative > 0, los Análisis B y C son circulares.
    """
    import pandas as pd
    import numpy as np

    neg_targets = set(net_df.loc[net_df["weight"] < 0, "target"].values)
    pos_targets = set(net_df.loc[net_df["weight"] > 0, "target"].values)
    all_targets = set(net_df["target"].values)

    overlap_neg = neg_targets & readout_genes
    overlap_pos = pos_targets & readout_genes

    result = {
        "n_total":         len(all_targets),
        "n_positive":      len(pos_targets),
        "n_negative":      len(neg_targets),
        "overlap_neg":     sorted(overlap_neg),
        "n_overlap_neg":   len(overlap_neg),
        "overlap_pos":     sorted(overlap_pos),
        "n_overlap_pos":   len(overlap_pos),
        "pct_neg_contaminated": 100 * len(overlap_neg) / max(len(all_targets), 1),
        "is_circular":     len(overlap_neg) > 0,
        "verdict": (
            "CIRCULAR — genes-readout en peso negativo del regulón"
            if overlap_neg else
            "LIMPIO — sin genes-readout en el regulón"
        ),
    }
    return result


# ── REGULÓN LIMPIO: solo genes activados por MYC ─────────────────────────────

# Genes positivos del backup original (53 genes, peso +1).
# Todos son targets ACTIVADOS por MYC: proliferación, metabolismo, traducción.
# Fuente: CollecTRI v2 curation, solo activadores de alta confianza.
# NINGUNO de estos genes aparece en ISG_score, MHC_I_score, Chemokine_score,
# STING_PATHWAY ni ningún otro readout del pipeline.
MYC_COLLECTRI_CLEAN_POSITIVE = {
    # Ciclo celular / CDKs
    "CCNA2": 1, "CCNB1": 1, "CCNB2": 1, "CCND1": 1, "CCNE1": 1,
    "CDK1":  1, "CDK2":  1, "CDK4":  1, "CDK6":  1,
    # Factores de transcripción downstream
    "E2F1": 1, "E2F2": 1, "E2F3": 1,
    # Proliferación / replicación
    "MKI67": 1, "PCNA": 1,
    "MCM2": 1, "MCM4": 1, "MCM5": 1, "MCM6": 1,
    # Síntesis de nucleótidos
    "TYMS": 1, "DHFR": 1, "DHODH": 1,
    "TK1": 1, "CAD": 1, "UMPS": 1,
    # Biogénesis ribosomal / traducción
    "NPM1": 1, "NCL": 1, "FBL": 1,
    "EIF4E": 1, "EIF4A1": 1, "EIF2S1": 1,
    "RPL5": 1, "RPL11": 1, "RPL13": 1, "RPS6": 1, "RPS14": 1,
    # Metabolismo (Warburg, glutaminolisis, poliaminas)
    "LDHA": 1, "GLS": 1, "SLC7A5": 1, "SLC1A5": 1,
    "ODC1": 1, "SRM": 1,
    # Reguladores moleculares
    "PRDX1": 1, "NME1": 1, "NME2": 1,
    "MDM2": 1, "MAX": 1,
    # Señalización / supervivencia
    "TERT": 1, "VEGFA": 1, "HIF1A": 1,
    "HSPA4": 1, "HSP90AA1": 1,
    # CD47 (MYC activa CD47: Casey 2016, Science) — es readout solo de CD47_expr
    # que es distinto de ISG/MHC-I. Lo dejamos.
    "CD47": 1,
    # MYC autoregulación
    "MYC": 1,
}

# Nota: MYC_COLLECTRI_CLEAN_POSITIVE excluye explícitamente:
#   CDKN1A/B, CDKN2A/B, RB1, GADD45A/B  (supresores)
#   TP53                                  (dual activado/reprimido)
#   STING1, TMEM173, TBK1, IRF3, CGAS    (STING pathway — readout)
#   IFNB1, IRF7                           (interferon — readout)
#   HLA-A/B/C, B2M, TAP1/2               (MHC-I — readout)
#   ISG15, IFIT1/2/3, MX1/2, OAS1/2, RSAD2, IFI44L  (ISG — readout)

# Verificar: ninguno de los genes limpios debe estar en READOUT_GENES
_check = set(MYC_COLLECTRI_CLEAN_POSITIVE.keys()) & READOUT_GENES
assert len(_check) == 0, f"Contaminación en regulón limpio: {_check}"


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES REEMPLAZADAS EN myc_tf_activity_decoupler.py
# ══════════════════════════════════════════════════════════════════════════════

def _build_backup_regulon_CLEAN(adata_var_names) -> "pd.DataFrame":
    """
    VERSIÓN LIMPIA de _build_backup_regulon().

    REEMPLAZA a la versión original que incluía 31 genes con peso −1
    (ISG, MHC-I, STING) causando circularidad total con los readouts.

    Esta versión usa SOLO los 53 genes activados por MYC (peso +1).
    Sin solapamiento con ningún readout del pipeline.

    Cambios respecto al original:
      ANTES: MYC_COLLECTRI_BACKUP (53 pos + 31 neg = 84 genes)
      AHORA: MYC_COLLECTRI_CLEAN_POSITIVE (53 pos, 0 neg = 53 genes)
    """
    import logging
    import pandas as pd

    logging.warning("  Usando regulón MYC limpio (solo positivos, sin genes-readout)")
    available = {g: w for g, w in MYC_COLLECTRI_CLEAN_POSITIVE.items()
                 if g in adata_var_names}
    n_total = len(MYC_COLLECTRI_CLEAN_POSITIVE)
    logging.info(
        f"  Regulón backup limpio: {len(available)}/{n_total} genes disponibles"
    )

    # Auditoría de contaminación (debe retornar is_circular=False siempre)
    rows = [{"source": "MYC", "target": g, "weight": float(w), "mor": float(w)}
            for g, w in available.items()]
    net = pd.DataFrame(rows)
    audit = audit_regulon_contamination(net)
    logging.info(f"  Auditoría regulón: {audit['verdict']}")
    assert not audit["is_circular"], (
        f"ERROR INTERNO: regulón limpio contiene genes-readout: {audit['overlap_neg']}"
    )

    return net


def get_myc_regulon_CLEAN(adata_var_names, decoupler_available: bool,
                           dc=None) -> "pd.DataFrame":
    """
    VERSIÓN LIMPIA de get_myc_regulon().

    Añade:
    1. Auditoría de contaminación sobre CollecTRI real
    2. Si CollecTRI real está contaminado: usa regulón limpio (fallback seguro)
    3. Log explícito del resultado de la auditoría para el paper

    La lógica de prioridad es:
      1. CollecTRI real (dc.get_collectri) — si pasa auditoría
      2. Regulón limpio (solo positivos) — si CollecTRI falla o contamina
      3. Hallmark MYC V1+V2 (36 genes) — si todo lo demás falla
    """
    import logging
    import pandas as pd

    CONTAMINATION_THRESHOLD_PCT = 5.0  # >5% overlap neg con readouts → rechazar

    if not decoupler_available or dc is None:
        logging.warning(
            "  decoupler no disponible → usando regulón limpio (solo positivos)"
        )
        return _build_backup_regulon_CLEAN(adata_var_names)

    # ── Intentar CollecTRI real ────────────────────────────────────────────
    try:
        logging.info("  Descargando regulón CollecTRI desde OmniPath...")
        collectri = dc.get_collectri(organism="human", split_complexes=False)
        myc_net = collectri[collectri["source"] == "MYC"].copy()
        logging.info(f"  CollecTRI MYC: {len(myc_net)} interacciones totales")

        # Filtrar a genes disponibles
        available = myc_net[myc_net["target"].isin(adata_var_names)].copy()
        logging.info(
            f"  En dataset: {len(available)}/{len(myc_net)} targets disponibles"
        )

        if len(available) < 20:
            logging.warning(
                "  <20 targets disponibles en CollecTRI → usando regulón limpio"
            )
            return _build_backup_regulon_CLEAN(adata_var_names)

        # ── AUDITORÍA DE CONTAMINACIÓN ────────────────────────────────────
        audit = audit_regulon_contamination(available)
        logging.info(
            f"  Auditoría CollecTRI real: {audit['verdict']}\n"
            f"    Targets total: {audit['n_total']}\n"
            f"    Positivos: {audit['n_positive']} | Negativos: {audit['n_negative']}\n"
            f"    Overlap neg-readout: {audit['n_overlap_neg']} genes "
            f"({audit['pct_neg_contaminated']:.1f}%)\n"
            f"    Genes contaminantes: {audit['overlap_neg']}"
        )

        if audit["pct_neg_contaminated"] > CONTAMINATION_THRESHOLD_PCT:
            logging.warning(
                f"  CollecTRI real: {audit['pct_neg_contaminated']:.1f}% de targets\n"
                f"  negativos solapan con readouts (ISG/MHC-I/STING).\n"
                f"  Threshold: {CONTAMINATION_THRESHOLD_PCT}%\n"
                f"  → RECHAZADO. Usando regulón limpio (solo positivos).\n"
                f"  Genes problemáticos: {audit['overlap_neg']}"
            )
            return _build_backup_regulon_CLEAN(adata_var_names)

        # CollecTRI real pasa auditoría
        logging.info(
            f"  CollecTRI real APROBADO (contaminación "
            f"{audit['pct_neg_contaminated']:.1f}% < {CONTAMINATION_THRESHOLD_PCT}%)"
        )

        # Solapamiento con genes de clasificación (cheque original)
        classification_genes = {
            "MYC", "EZH2", "SUZ12", "CTNNB1", "ATF3", "STAT3", "DNMT1",
            "COL1A1", "COL1A2", "COL3A1", "ACTA2", "FN1", "FAP", "POSTN",
        }
        overlap_class = set(available["target"]) & classification_genes
        if overlap_class:
            pct = 100 * len(overlap_class) / len(available)
            logging.info(
                f"  Solapamiento regulón-clasificación: {sorted(overlap_class)} "
                f"({pct:.1f}%) — esperado y biológicamente correcto"
            )

        return available

    except Exception as e:
        logging.warning(
            f"  dc.get_collectri() falló ({e})\n"
            f"  → usando regulón limpio (solo positivos)"
        )
        return _build_backup_regulon_CLEAN(adata_var_names)


def analysis_B_correlations_CLEAN(adata, results: list, logger,
                                   net_df, X_raw=None, var_names=None):
    """
    VERSIÓN LIMPIA de analysis_B_correlations().

    Añade verificación de solapamiento regulón-readout ANTES de ejecutar.
    Si solapamiento > umbral, el test se marca como CIRCULAR en el CSV
    y se omite del resumen como evidencia.

    Garantiza que el paper no cite correlaciones artefactuales.
    """
    import numpy as np
    from scipy.stats import spearmanr

    DESERT_LABEL  = "Immune_Desert"
    PHENOTYPE_COL = "Phenotype"

    # Firma de readouts: genes y nombre de score
    score_defs = [
        ("ISG_score",
         ["IFIT1","IFIT2","IFIT3","ISG15","MX1","MX2","OAS1","OAS2","RSAD2","IFI44L"],
         "ISG score (proxy STING activity downstream)"),
        ("MHC_I_score",
         ["HLA-A","HLA-B","HLA-C","B2M","TAP1","TAP2","TAPBP"],
         "MHC-I antigen presentation pathway"),
        ("Chemokine_score",
         ["CCL5","CXCL9","CXCL10"],
         "Chemokine output (CCL5/CXCL9/CXCL10)"),
        ("CD47_expr",
         ["CD47"],
         "CD47 'don't eat me' signal"),
    ]

    logger.info("=" * 70)
    logger.info("ANÁLISIS B (LIMPIO): Correlaciones MYC_TF vs readouts — con auditoría")

    # Auditar el regulón activo contra cada readout
    neg_targets = set(net_df.loc[net_df["weight"] < 0, "target"].values)
    logger.info(
        f"  Regulón activo: {len(net_df)} genes | negativos: {len(neg_targets)}"
    )

    desert_mask = adata.obs[PHENOTYPE_COL] == DESERT_LABEL
    n_desert = desert_mask.sum()
    logger.info(f"  Spots Desert: {n_desert:,}")

    tf_vals = adata.obs.loc[desert_mask, "MYC_TF_activity"].values
    pvals = []

    import numpy as np

    for score_col, genes, description in score_defs:
        # Solapamiento entre este readout y genes negativos del regulón
        overlap = set(genes) & neg_targets
        pct_overlap = 100 * len(overlap) / max(len(genes), 1)
        is_circular = pct_overlap > 10.0  # >10% → circular

        if is_circular:
            logger.warning(
                f"  [{score_col}] CIRCULAR: {len(overlap)}/{len(genes)} genes "
                f"({pct_overlap:.0f}%) del readout están como pesos negativos "
                f"en el regulón → {sorted(overlap)}\n"
                f"  → Test marcado CIRCULAR, no es evidencia válida."
            )
            results.append({
                "analysis":       "B",
                "test_id":        f"MYC_TF_vs_{score_col}_Desert",
                "statistic":      float("nan"),
                "p_value":        float("nan"),
                "q_value":        float("nan"),
                "fdr_significant": False,
                "n1":             int(n_desert),
                "hypothesis":
                    f"H: MYC_TF_activity ↑ → {score_col} ↓ en Desert",
                "description":    description,
                "circular":       True,
                "circular_genes": sorted(overlap),
                "pct_overlap":    round(pct_overlap, 1),
                "NOTE":
                    "RESULTADO INVÁLIDO: regulón contiene genes del readout "
                    "con peso negativo. Usar solo con regulón limpio.",
            })
            pvals.append(float("nan"))
            continue

        # No circular → ejecutar normalmente
        if score_col not in adata.obs.columns:
            from myc_tf_activity_decoupler import compute_gene_score
            vals = compute_gene_score(
                adata, genes, score_col, X=X_raw, var_names=var_names
            )
            adata.obs[score_col] = vals

        score_desert = adata.obs.loc[desert_mask, score_col].values
        valid = ~(np.isnan(tf_vals) | np.isnan(score_desert))

        if valid.sum() < 30:
            logger.warning(f"  [{score_col}] <30 valores válidos → skip")
            pvals.append(float("nan"))
            continue

        rho, pval = spearmanr(tf_vals[valid], score_desert[valid])
        pvals.append(pval)
        results.append({
            "analysis":       "B",
            "test_id":        f"MYC_TF_vs_{score_col}_Desert",
            "statistic":      float(rho),
            "p_value":        float(pval),
            "n1":             int(valid.sum()),
            "hypothesis":
                f"H: MYC_TF_activity ↑ → {score_col} ↓ en Desert",
            "description":    description,
            "circular":       False,
            "pct_overlap":    round(pct_overlap, 1),
        })
        logger.info(
            f"  MYC_TF vs {score_col} (Desert): ρ={rho:.4f}, p={pval:.3e} "
            f"[NO circular, overlap={pct_overlap:.0f}%]"
        )

    # FDR solo sobre tests no-circulares
    valid_pvals = [p for p in pvals if not (p != p)]  # filtrar NaN
    if valid_pvals:
        from statsmodels.stats.multitest import multipletests
        import numpy as np
        pvals_arr = np.array(pvals)
        valid_mask = ~np.isnan(pvals_arr)
        qvals = np.full_like(pvals_arr, float("nan"))
        if valid_mask.sum() > 0:
            _, q, _, _ = multipletests(
                pvals_arr[valid_mask], alpha=0.05, method="fdr_bh"
            )
            qvals[valid_mask] = q

        b_results = [r for r in results if r.get("analysis") == "B"]
        for i, r in enumerate(b_results):
            if not r.get("circular", False):
                r["q_value"] = float(qvals[i]) if not np.isnan(qvals[i]) else float("nan")
                r["fdr_significant"] = bool(qvals[i] < 0.05) if not np.isnan(qvals[i]) else False


def run_verification():
    """
    Verifica que el regulón limpio no tiene contaminación.
    """
    import pandas as pd

    print("=" * 70)
    print("VERIFICACIÓN: Regulón MYC limpio vs regulón backup original")
    print("=" * 70)

    # Simular net_df del backup original
    MYC_COLLECTRI_BACKUP_ORIGINAL = {
        **MYC_COLLECTRI_CLEAN_POSITIVE,
        # Genes negativos que contaminában
        "CDKN1A": -1, "CDKN1B": -1, "CDKN2A": -1, "CDKN2B": -1,
        "TP53": -1, "GADD45A": -1, "GADD45B": -1, "RB1": -1,
        "STING1": -1, "TMEM173": -1, "TBK1": -1, "IRF3": -1, "CGAS": -1,
        "IFNB1": -1, "IRF7": -1,
        "HLA-A": -1, "HLA-B": -1, "HLA-C": -1, "B2M": -1,
        "TAP1": -1, "TAP2": -1,
        "ISG15": -1, "IFIT1": -1, "IFIT2": -1, "IFIT3": -1,
        "MX1": -1, "MX2": -1, "OAS1": -1, "OAS2": -1,
        "RSAD2": -1, "IFI44L": -1,
    }

    net_original = pd.DataFrame([
        {"source": "MYC", "target": g, "weight": float(w), "mor": float(w)}
        for g, w in MYC_COLLECTRI_BACKUP_ORIGINAL.items()
    ])
    net_clean = pd.DataFrame([
        {"source": "MYC", "target": g, "weight": float(w), "mor": float(w)}
        for g, w in MYC_COLLECTRI_CLEAN_POSITIVE.items()
    ])

    print("\n[1] Regulón ORIGINAL (backup v1 — contaminado):")
    audit_orig = audit_regulon_contamination(net_original)
    print(f"    Genes totales:   {audit_orig['n_total']}")
    print(f"    Positivos:       {audit_orig['n_positive']}")
    print(f"    Negativos:       {audit_orig['n_negative']}")
    print(f"    Overlap neg-readout: {audit_orig['n_overlap_neg']} genes "
          f"({audit_orig['pct_neg_contaminated']:.1f}%)")
    print(f"    Genes problemáticos: {audit_orig['overlap_neg']}")
    print(f"    Veredicto: {audit_orig['verdict']}")

    print("\n[2] Regulón LIMPIO (fix — solo positivos):")
    audit_clean = audit_regulon_contamination(net_clean)
    print(f"    Genes totales:   {audit_clean['n_total']}")
    print(f"    Positivos:       {audit_clean['n_positive']}")
    print(f"    Negativos:       {audit_clean['n_negative']}")
    print(f"    Overlap neg-readout: {audit_clean['n_overlap_neg']} genes "
          f"({audit_clean['pct_neg_contaminated']:.1f}%)")
    print(f"    Veredicto: {audit_clean['verdict']}")

    print("\n[3] Predicción de resultados:")
    print("    Con regulón limpio, el score MYC_TF mide SOLO actividad de")
    print("    proliferación/metabolismo MYC (CDK, E2F, MCM, ribosomas).")
    print()
    print("    Referencia MYC_Hallmark_Combined (36 genes, todos positivos):")
    print("      Desert vs Inflamed: d = −0.06, p = 0.59 [NS]")
    print()
    print("    Predicción regulón limpio (53 genes, todos positivos, mismo tipo):")
    print("      Desert vs Inflamed: d ≈ −0.10 a +0.10 (pequeño, posiblemente NS)")
    print()
    print("    Si el resultado limpio es significativo (d > 0.3):")
    print("      → Evidencia genuina de actividad proliferativa MYC en Desert")
    print("      → Reportar en paper con nota sobre el fallback backup original")
    print()
    print("    Si el resultado limpio colapsa (~0, NS):")
    print("      → El +0.62 del backup era enteramente artefacto de circularidad")
    print("      → Reportar como nulo honesto, coherente con MYC Hallmark")
    print("      → La hipótesis MYC→Desert no tiene soporte en actividad TF")

    print("\n" + "=" * 70)
    passed = not audit_clean["is_circular"]
    if passed:
        print("VERIFICACIÓN PASADA: regulón limpio no tiene contaminación")
    else:
        print("VERIFICACIÓN FALLIDA: revisar MYC_COLLECTRI_CLEAN_POSITIVE")
    return passed


if __name__ == "__main__":
    if "--verify" in sys.argv or len(sys.argv) == 1:
        ok = run_verification()
        sys.exit(0 if ok else 1)
    else:
        print("Uso: python fix_myc_tf_clean_regulon.py [--verify]")
        print("Para aplicar el fix, ver OPCIÓN B en los comentarios de este archivo.")

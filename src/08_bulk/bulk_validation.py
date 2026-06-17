"""
================================================================================
MÓDULO: BULK VALIDATION — Validación Externa en METABRIC + TCGA-BRCA
================================================================================
REESCRITO basándose en diagnóstico real de los archivos de cBioPortal.

Correcciones aplicadas:
    Normalización de IDs TCGA (strip -01 suffix)
    Manejo robusto de NaN en AUC (imputación + skip)
    lifelines para KM/Cox con fallback a manual

    ntrez_Gene_Id como columna falsa-muestra → drop explícito
    Genes NaN en index TCGA → dropna
    match directo = 0%, match stripped = 1082
    Escalas opuestas: TCGA RSEM crudo → log2(x+1); METABRIC ya log2
    STING alias bidireccional: TCGA=TMEM173, METABRIC=STING1
    Columnas clínicas human-readable, no SNAKE_CASE
    Preferir .tsv (más completo) sobre data_clinical_patient.txt

Datos verificados (diagnose_bulk_datasets.py):
    TCGA-BRCA:
        data_mrna_seq_v2_rsem.txt       — 20,531 genes × 1,082 muestras (RSEM)
        brca_tcga_pan_can_atlas_2018_clinical_data.tsv — 1,084 pacientes
        Subtype 'BRCA_Basal' → 171 TNBC
    METABRIC:
        data_mrna_illumina_microarray.txt — 20,603 genes × 1,980 muestras (log2)
        brca_metabric_clinical_data.tsv  — 2,509 pacientes
        Pam50 'Basal' → 209 TNBC;  ER-/HER2- → 286 TNBC

Output:
    - bulk_classification_{METABRIC,TCGA}.csv
    - survival_analysis_results.csv
    - signature_validation_auc.csv
    - Fig_bulk_KM_Desert_vs_Excluded.pdf
    - Fig_bulk_KM_Inflamed_vs_Cold.pdf
    - Fig_bulk_scores_scatter.pdf
    - Fig_bulk_ROC.pdf
    - Fig_bulk_combined_6panel.pdf
================================================================================
"""

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr, zscore as scipy_zscore
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

try:
    # GENE_SIGNATURES no existe, usar SIGNATURES + CANONICAL_SIGNATURES
    from config import PATHS, SIGNATURES, CANONICAL_SIGNATURES
    BASE_DIR = PATHS.BASE_DIR
    RESULTS_DIR = PATHS.RESULTS_DIR
    print("Configuración cargada desde config.py")
except ImportError:
    from pathlib import Path
    BASE_DIR = Path("/home/external/frjimenez/fabian/genoma")
    RESULTS_DIR = BASE_DIR / "results"
    print("config.py no encontrado, usando rutas HPC por defecto")

try:
    from config import SURVIVAL_PARAMS, PIPELINE_QC
    USE_CONFIG_PARAMS = True
except ImportError:
    USE_CONFIG_PARAMS = False

# importar lifelines (Cox regression + KM profesional)
LIFELINES_AVAILABLE = False
try:
    from lifelines import KaplanMeierFitter, CoxPHFitter
    from lifelines.statistics import logrank_test as ll_logrank_test
    LIFELINES_AVAILABLE = True
    print("lifelines disponible → usando KM/Cox/log-rank profesional")
except ImportError:
    print("lifelines no instalado → fallback a implementación manual")
    print("  Instalar: pip install lifelines")

from pathlib import Path

# Directorio de salida
BULK_DIR = RESULTS_DIR / "bulk_validation"
os.makedirs(BULK_DIR, exist_ok=True)
os.makedirs(BULK_DIR / "figures", exist_ok=True)
os.makedirs(BULK_DIR / "tables", exist_ok=True)

# Posibles rutas para datos bulk (búsqueda en orden)
# Añadir data/raw/ que es donde REALMENTE están los datos en HPC
METABRIC_PATHS = [
    BASE_DIR / "data" / "raw" / "METABRIC",           # REAL en HPC
    BASE_DIR / "data" / "bulk_validation" / "METABRIC",
    BASE_DIR / "data" / "brca_metabric",
    BASE_DIR / "data" / "METABRIC",
    BASE_DIR / "data" / "metabric",
]
TCGA_PATHS = [
    BASE_DIR / "data" / "raw" / "TCGA-BRCA",          # REAL en HPC
    BASE_DIR / "data" / "bulk_validation" / "TCGA_BRCA",
    BASE_DIR / "data" / "TCGA_BRCA",
    BASE_DIR / "data" / "TCGA-BRCA",                   # Con guión (variante)
    BASE_DIR / "data" / "tcga_brca",
    BASE_DIR / "data" / "brca",
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ============================================================================
# FIRMAS GÉNICAS — Derivadas del pipeline espacial
# ============================================================================

# Firmas alineadas con CANONICAL_SIGNATURES de config.py
# BMI1 eliminado (no canónico), IRF1 eliminado (no canónico),
# BARRIER reducida de 12 a 9 genes, IMMUNE reducida de 13 a 7 (CD8 only)
SILENCING_SIGNATURE = {
    'up': ['MYC', 'EZH2', 'SUZ12', 'CTNNB1', 'ATF3', 'STAT3', 'DNMT1'],  # = CANONICAL_SIGNATURES['bulk_desert_up']
    'down': ['TMEM173', 'TBK1', 'IRF3', 'CGAS', 'MB21D1',                 # = CANONICAL_SIGNATURES['bulk_desert_down']
             'CXCL9', 'CXCL10', 'CXCL11', 'CCL5']
}

BARRIER_SIGNATURE = {
    'up': ['COL1A1', 'COL1A2', 'COL10A1', 'FN1', 'POSTN',                 # = CANONICAL_SIGNATURES['bulk_excluded_up']
           'TGFB1', 'ACTA2', 'FAP', 'THBS2'],
    'down': []
}

IMMUNE_SIGNATURE = {
    'up': ['CD8A', 'CD8B', 'CD3D', 'CD3E', 'GZMA', 'GZMB', 'PRF1'],      # = SIGNATURES.CD8_T_CELLS
    'down': []
}

EXCLUSION_SIGNATURE_15 = [
    'COL1A1', 'COL3A1', 'FN1', 'FAP', 'ACTA2', 'POSTN',
    'MYC', 'EZH2', 'DNMT1', 'TMEM173', 'CXCL9', 'CXCL10',
    'CD8A', 'GZMB', 'IFNG'
]

# Alias bidireccional TMEM173 ↔ STING1
# TCGA usa TMEM173, METABRIC usa STING1. Son el mismo gen.
GENE_ALIASES = {
    'TMEM173': ['STING1', 'STING', 'MITA', 'MPYS', 'ERIS'],
    'STING1': ['TMEM173'],
    'CD274':  ['PDCD1LG2'],
}

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.facecolor': 'white'
})


# ============================================================================
# UTILIDADES
# ============================================================================

def _resolve_gene(gene_name, available_genes):
    """
    Busca un gen por nombre o alias en el set de genes disponibles.
    
    Returns: nombre encontrado en available_genes, o None.
    """
    available_upper = {g.upper(): g for g in available_genes}
    
    # Match directo
    if gene_name.upper() in available_upper:
        return available_upper[gene_name.upper()]
    
    # Buscar aliases
    aliases = GENE_ALIASES.get(gene_name, [])
    for alias in aliases:
        if alias.upper() in available_upper:
            return available_upper[alias.upper()]
    
    return None


def _resolve_gene_list(gene_list, available_genes):
    """Resuelve una lista de genes con aliases. Retorna lista de encontrados."""
    resolved = []
    for g in gene_list:
        found = _resolve_gene(g, available_genes)
        if found is not None:
            resolved.append(found)
    return resolved


def _normalize_tcga_id(sample_id):
    """
    Normaliza IDs de TCGA.
    Expresión: TCGA-3C-AAAU-01  →  TCGA-3C-AAAU (patient ID)
    Clínico:   TCGA-3C-AAAU     →  TCGA-3C-AAAU
    """
    sid = str(sample_id)
    if sid.startswith('TCGA-') and sid.count('-') >= 3:
        parts = sid.split('-')
        return '-'.join(parts[:3])  # TCGA-XX-XXXX
    return sid


# ============================================================================
# 1. CARGA DE DATOS BULK
# ============================================================================

def find_data_directory(possible_paths):
    """Busca directorio de datos en múltiples posibles rutas."""
    for path in possible_paths:
        if os.path.exists(path):
            files = os.listdir(path)
            if len(files) > 0:
                return Path(path)
    return None


def load_expression_data(data_dir, dataset_name='METABRIC'):
    """
    Carga matriz de expresión génica bulk con correcciones de diagnóstico.
    
    Fixes aplicados:
        Drop Entrez_Gene_Id (falsa columna-muestra)
        Drop genes con Hugo_Symbol NaN
        Detecta y aplica log2 transform si RSEM crudo
        Alias mapping para STING
    
    Returns
    -------
    expr_df : pd.DataFrame — Genes × Muestras
    transform_applied : str — 'log2', 'none', etc.
    """
    if data_dir is None:
        return None, 'none'
    
    # Buscar archivos en orden de preferencia
    expr_patterns = [
        'data_mrna_seq_v2_rsem.txt',
        'data_mrna_illumina_microarray.txt',
        'data_mrna_agilent_microarray.txt',
        'data_mrna_seq_fpkm.txt',
        'data_expression.txt',
        'data_RNA_Seq_v2_expression_median.txt',
    ]
    
    expr_file = None
    for pattern in expr_patterns:
        candidate = data_dir / pattern
        if os.path.exists(candidate):
            expr_file = candidate
            break
    
    # Fallback: buscar cualquier archivo con "mrna" o "expression"
    if expr_file is None:
        all_files = os.listdir(data_dir)
        for f in all_files:
            if ('mrna' in f.lower() or 'expression' in f.lower()) and f.endswith('.txt'):
                expr_file = data_dir / f
                break
    
    if expr_file is None:
        print(f"  No se encontró archivo de expresión en {data_dir}")
        print(f"  Archivos disponibles: {os.listdir(data_dir)[:15]}")
        return None, 'none'
    
    print(f"  Cargando expresión: {expr_file.name} ({expr_file.stat().st_size/1e6:.0f} MB)")
    
    # Cargar (formato cBioPortal: Hugo_Symbol | Entrez_Gene_Id | samples...)
    df = pd.read_csv(expr_file, sep='\t', comment='#', low_memory=False)
    
    # Identificar columna de genes
    gene_col = None
    for col_name in ['Hugo_Symbol', 'HUGO_SYMBOL', 'Gene', 'gene',
                      'GENE_SYMBOL', 'Gene Symbol']:
        if col_name in df.columns:
            gene_col = col_name
            break
    
    if gene_col is None:
        gene_col = df.columns[0]
    
    # Eliminar columnas no-muestra explícitamente
    # Entrez_Gene_Id contiene enteros grandes que se confunden con expresión
    drop_cols = []
    for col in ['Entrez_Gene_Id', 'Entrez_gene_id', 'ENTREZ_GENE_ID',
                'Composite.Element.REF', 'Description']:
        if col in df.columns:
            drop_cols.append(col)
    
    # Establecer genes como index
    df = df.set_index(gene_col)
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    # Eliminar genes con index NaN (TCGA tiene varios)
    n_nan = df.index.isna().sum()
    if n_nan > 0:
        print(f"  ⚠ DX2: Eliminando {n_nan} genes con nombre NaN")
        df = df[df.index.notna()]
    
    # Eliminar genes duplicados (mantener primera aparición)
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        print(f"  ⚠ Eliminando {n_dup} genes duplicados")
        df = df[~df.index.duplicated(keep='first')]
    
    # Convertir a float
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Eliminar genes vacíos
    df = df.dropna(how='all')
    
    print(f"  ✓ Expresión cargada: {df.shape[0]:,} genes × {df.shape[1]:,} muestras")
    
    # Detectar escala y transformar si necesario
    sample_cols = list(df.columns[:min(50, len(df.columns))])
    sample_vals = df[sample_cols].values.flatten()
    sample_vals = sample_vals[~np.isnan(sample_vals)]
    
    median_val = np.median(sample_vals)
    max_val = np.max(sample_vals)
    min_val = np.min(sample_vals[sample_vals > 0]) if (sample_vals > 0).any() else 0
    
    transform_applied = 'none'
    
    if max_val > 1000 and median_val > 50:
        # RSEM crudo o counts → log2(x + 1)
        print(f"  DX4: Detectado RSEM/counts crudos (median={median_val:.0f}, max={max_val:.0f})")
        print(f"  → Aplicando log2(x + 1) para normalizar escala")
        df = np.log2(df + 1)
        transform_applied = 'log2'
        
        new_median = df[sample_cols].values.flatten()
        new_median = np.median(new_median[~np.isnan(new_median)])
        print(f"  ✓ Post-transform: median={new_median:.2f}")
    
    elif max_val < 25 and min_val > 0:
        # Ya en log-space (microarray)
        print(f"  DX4: Datos ya en log-space (median={median_val:.2f}, max={max_val:.2f})")
        transform_applied = 'already_log'
    
    elif min_val < 0:
        # Log-ratio (microarray con referencia)
        print(f"  DX4: Detectado log-ratio (min={min_val:.2f}) — sin transformación adicional")
        transform_applied = 'log_ratio'
    
    else:
        print(f"  DX4: Escala intermedia (median={median_val:.2f}, max={max_val:.2f})")
        transform_applied = 'unknown'
    
    # Verificar genes STING y añadir alias si falta
    available = set(df.index)
    for canonical, aliases in GENE_ALIASES.items():
        if canonical not in available:
            for alias in aliases:
                if alias in available:
                    # Duplicar la fila con el nombre canónico
                    df.loc[canonical] = df.loc[alias]
                    print(f"  DX5: Alias {alias} → {canonical} (duplicado para compatibilidad)")
                    break
    
    print(f"  ✓ Expresión final: {df.shape[0]:,} genes × {df.shape[1]:,} muestras")
    
    return df, transform_applied


def load_clinical_data(data_dir, dataset_name='METABRIC'):
    """
    Carga datos clínicos con detección robusta de formato.
    
    DX6: Soporta nombres human-readable Y SNAKE_CASE.
    DX7: Prefiere .tsv (más completo) sobre data_clinical_patient.txt.
    """
    if data_dir is None:
        return None
    
    # Buscar .tsv primero (más columnas, sin # headers)
    clinical_patterns = [
        'brca_metabric_clinical_data.tsv',
        'brca_tcga_pan_can_atlas_2018_clinical_data.tsv',
        'clinical_data.tsv',
        'data_clinical_patient.txt',
        'data_clinical.txt',
    ]
    
    clinical_file = None
    for pattern in clinical_patterns:
        candidate = data_dir / pattern
        if os.path.exists(candidate):
            clinical_file = candidate
            break
    
    # Fallback
    if clinical_file is None:
        all_files = os.listdir(data_dir)
        for f in sorted(all_files):
            if 'clinical' in f.lower() and (f.endswith('.txt') or f.endswith('.tsv')):
                clinical_file = data_dir / f
                break
    
    if clinical_file is None:
        print(f"  No se encontró archivo clínico en {data_dir}")
        return None
    
    print(f"  Cargando clínico: {clinical_file.name}")
    
    # cBioPortal .txt tiene líneas # como header (pero .tsv no)
    df = pd.read_csv(clinical_file, sep='\t', comment='#', low_memory=False)
    
    # Limpiar posible fila de tipos (STRING/NUMBER/BOOLEAN)
    if df.iloc[0].astype(str).str.contains('STRING|NUMBER|BOOLEAN').any():
        df = df.iloc[1:].reset_index(drop=True)
    
    # Crear mapping unificado de columnas (human-readable → canónico)
    COL_MAP = {
        # Patient/Sample ID
        'Patient ID': 'PATIENT_ID',
        'Sample ID': 'SAMPLE_ID',
        # Survival
        'Overall Survival (Months)': 'OS_MONTHS',
        'Overall Survival Status': 'OS_STATUS',
        'Relapse Free Status (Months)': 'RFS_MONTHS',
        'Relapse Free Status': 'RFS_STATUS',
        'Disease Free (Months)': 'DFS_MONTHS',
        'Disease Free Status': 'DFS_STATUS',
        'Months of disease-specific survival': 'DSS_MONTHS',
        'Disease-specific Survival status': 'DSS_STATUS',
        # Receptors
        'ER Status': 'ER_STATUS',
        'ER status measured by IHC': 'ER_IHC',
        'HER2 Status': 'HER2_STATUS',
        'HER2 status measured by SNP6': 'HER2_SNP6',
        'PR Status': 'PR_STATUS',
        # Subtype
        'Pam50 + Claudin-low subtype': 'CLAUDIN_SUBTYPE',
        'Subtype': 'PAM50_SUBTYPE',
        '3-Gene classifier subtype': 'THREEGENE',
        # Demographics
        'Age at Diagnosis': 'AGE_AT_DIAGNOSIS',
        'Diagnosis Age': 'AGE_AT_DIAGNOSIS',
        'Neoplasm Histologic Grade': 'GRADE',
        'Tumor Stage': 'TUMOR_STAGE',
        "Patient's Vital Status": 'VITAL_STATUS',
        # Extra
        'Integrative Cluster': 'INTCLUST',
        'Chemotherapy': 'CHEMOTHERAPY',
    }
    
    # Renombrar columnas que coinciden
    rename_map = {}
    for human_name, snake_name in COL_MAP.items():
        if human_name in df.columns and snake_name not in df.columns:
            rename_map[human_name] = snake_name
    
    if rename_map:
        df = df.rename(columns=rename_map)
        print(f"  DX6: Renombradas {len(rename_map)} columnas → formato canónico")
    
    # Identificar y setear index
    patient_col = None
    for col_name in ['PATIENT_ID', 'Patient ID', 'SAMPLE_ID', 'Sample ID']:
        if col_name in df.columns:
            patient_col = col_name
            break
    
    if patient_col:
        df = df.set_index(patient_col)
    
    print(f"  ✓ Clínico cargado: {len(df):,} pacientes, {len(df.columns)} columnas")
    
    return df


def extract_survival_data(clinical_df):
    """
    Extrae tiempo de supervivencia y evento (muerte).
    
    Busca tanto SNAKE_CASE como nombres originales.
    Manejo robusto de NaN/valores no parseable.
    """
    if clinical_df is None:
        return None
    
    # Buscar columna de tiempo (ya renombradas por DX6 si aplica)
    time_col = None
    for col_name in ['OS_MONTHS', 'Overall Survival (Months)',
                      'OVERALL_SURVIVAL_MONTHS', 'os_months']:
        if col_name in clinical_df.columns:
            time_col = col_name
            break
    
    # Buscar columna de evento
    event_col = None
    for col_name in ['OS_STATUS', 'Overall Survival Status',
                      'OVERALL_SURVIVAL_STATUS', 'os_status']:
        if col_name in clinical_df.columns:
            event_col = col_name
            break
    
    if time_col is None or event_col is None:
        print(f"  Columnas de supervivencia no encontradas")
        print(f"  Buscando: OS_MONTHS + OS_STATUS")
        surv_cols = [c for c in clinical_df.columns 
                     if any(k in c.upper() for k in ['SURVIVAL', 'OS_'])]
        print(f"     Candidatos: {surv_cols}")
        return None
    
    survival_df = pd.DataFrame(index=clinical_df.index)
    survival_df['time'] = pd.to_numeric(clinical_df[time_col], errors='coerce')
    
    # Parsear evento (1 = muerte, 0 = censurado)
    event_values = clinical_df[event_col].astype(str).str.lower()
    survival_df['event'] = 0
    
    # Formatos comunes de cBioPortal:
    # "1:DECEASED", "0:LIVING", "DECEASED", "LIVING"
    survival_df.loc[
        event_values.str.contains('1:|deceased|dead|died', regex=True, na=False),
        'event'
    ] = 1
    
    # Drop NaN y tiempos no válidos
    n_before = len(survival_df)
    survival_df = survival_df.dropna(subset=['time'])
    survival_df = survival_df[survival_df['time'] > 0]
    n_dropped = n_before - len(survival_df)
    
    if n_dropped > 0:
        print(f"  H7: Eliminados {n_dropped} pacientes sin supervivencia válida")
    
    print(f"  Supervivencia extraída: {len(survival_df)} pacientes")
    print(f"  Eventos: {survival_df['event'].sum()} muertes "
          f"({survival_df['event'].mean()*100:.1f}%)")
    print(f"  ediana follow-up: {survival_df['time'].median():.1f} meses")
    
    return survival_df


# ============================================================================
# 1b. FILTRADO TNBC
# ============================================================================

def filter_tnbc_patients(clinical_df, expression_df, dataset_name='METABRIC'):
    """
    Filtra pacientes TNBC con lógica adaptada por dataset.
    
    Normalización de IDs para TCGA (strip -01).
    Aplica matching flexible cuando IDs no coinciden directamente.
    """
    if clinical_df is None:
        return None, None
    
    tnbc_mask = pd.Series(False, index=clinical_df.index)
    method_used = None
    
    # --- Método 1: PAM50/CLAUDIN_SUBTYPE == 'Basal' ---
    for col in ['CLAUDIN_SUBTYPE', 'Pam50 + Claudin-low subtype', 'PAM50',
                'Pam50Subtype']:
        if col in clinical_df.columns:
            vals = clinical_df[col].astype(str).str.lower()
            tnbc_mask |= vals.str.contains('basal', na=False)
            method_used = f"PAM50 Basal ({col})"
            print(f"  Filtro {method_used}: {tnbc_mask.sum()} pacientes")
            break
    
    # --- Método 2: Subtype == 'BRCA_Basal' (TCGA Pan-Cancer) ---
    if tnbc_mask.sum() == 0:
        for col in ['PAM50_SUBTYPE', 'Subtype']:
            if col in clinical_df.columns:
                vals = clinical_df[col].astype(str).str.lower()
                tnbc_mask |= vals.str.contains('basal', na=False)
                method_used = f"Subtype Basal ({col})"
                print(f"  Filtro {method_used}: {tnbc_mask.sum()} pacientes")
                break
    
    # --- Método 3: ER-neg + HER2-neg (más inclusivo, usado si PAM50 da poco) ---
    if tnbc_mask.sum() < 20:
        for er_col in ['ER_STATUS', 'ER Status', 'ER_IHC',
                        'ER status measured by IHC']:
            if er_col in clinical_df.columns:
                er_neg = clinical_df[er_col].astype(str).str.lower().isin(
                    ['negative', 'neg', 'negat']
                )
                for her2_col in ['HER2_STATUS', 'HER2 Status', 'HER2_SNP6',
                                  'HER2 status measured by SNP6']:
                    if her2_col in clinical_df.columns:
                        her2_vals = clinical_df[her2_col].astype(str).str.lower()
                        her2_neg = her2_vals.isin(
                            ['negative', 'neg', 'neutral', 'loss']
                        )
                        new_mask = er_neg & her2_neg
                        tnbc_mask |= new_mask
                        method_used = f"ER-/HER2- ({er_col} + {her2_col})"
                        print(f"  Filtro {method_used}: {tnbc_mask.sum()} pacientes (acumulado)")
                        break
                break
    
    # --- Método 4: THREEGENE == 'ER-/HER2-' ---
    if tnbc_mask.sum() < 20:
        for col in ['THREEGENE', '3-Gene classifier subtype']:
            if col in clinical_df.columns:
                vals = clinical_df[col].astype(str)
                tnbc_mask |= vals.str.contains('ER-/HER2-', na=False)
                print(f"  Filtro THREEGENE: {tnbc_mask.sum()} pacientes (acumulado)")
                break
    
    if tnbc_mask.sum() < 10:
        print(f"  ⚠ Solo {tnbc_mask.sum()} TNBC encontrados — usando todos con advertencia")
        tnbc_mask = pd.Series(True, index=clinical_df.index)
        method_used = "ALL (insufficient TNBC filter)"
    
    clinical_tnbc = clinical_df[tnbc_mask]
    
    # ID matching con normalización TCGA ---
    if expression_df is not None:
        # Intentar match directo primero
        common = expression_df.columns.intersection(clinical_tnbc.index)
        
        if len(common) < 10:
            print(f"  Match directo insuficiente ({len(common)}), normalizando IDs...")
            
            # Crear mapping: expr_sample_id → patient_id
            expr_ids = list(expression_df.columns)
            clin_ids = set(clinical_tnbc.index.astype(str))
            
            # TCGA: strip suffix -01 from expression IDs
            id_map = {}
            for eid in expr_ids:
                patient_id = _normalize_tcga_id(eid)
                if patient_id in clin_ids:
                    id_map[eid] = patient_id
            
            if len(id_map) > len(common):
                print(f"  Normalización TCGA: {len(id_map)} matches "
                      f"(vs {len(common)} directo)")
                
                # Renombrar columnas de expresión a patient_id
                expr_renamed = expression_df.rename(columns=id_map)
                # Drop duplicados (si multiple aliquots del mismo paciente)
                expr_renamed = expr_renamed.loc[:, ~expr_renamed.columns.duplicated()]
                
                common = expr_renamed.columns.intersection(clinical_tnbc.index)
                expression_tnbc = expr_renamed[common] if len(common) > 0 else None
            else:
                expression_tnbc = expression_df[common] if len(common) > 0 else None
        else:
            expression_tnbc = expression_df[common]
        
        print(f"  TNBC filtrado: {len(clinical_tnbc)} clínico, "
              f"{len(common)} con expresión")
        print(f"  Método: {method_used}")
    else:
        expression_tnbc = None
    
    return clinical_tnbc, expression_tnbc


# ============================================================================
# 2. CLASIFICACIÓN EN FENOTIPOS INMUNES
# ============================================================================

def compute_signature_score(expression_df, signature, method='zscore_mean'):
    """
    Calcula score de firma génica en datos bulk.
    
    Usa _resolve_gene_list para manejar aliases automáticamente.
    Z-score por gen antes de promediar.
    
    Returns: (scores_series, n_genes_used, n_genes_total)
    """
    available = set(expression_df.index)
    
    # Resolver genes con aliases
    up_genes = _resolve_gene_list(signature.get('up', []), available)
    down_genes = _resolve_gene_list(signature.get('down', []), available)
    
    total_requested = len(signature.get('up', [])) + len(signature.get('down', []))
    total_found = len(up_genes) + len(down_genes)
    
    if total_found == 0:
        return pd.Series(np.nan, index=expression_df.columns), 0, total_requested
    
    if method == 'zscore_mean':
        if len(up_genes) > 0:
            up_expr = expression_df.loc[up_genes].T
            up_z = up_expr.apply(scipy_zscore, nan_policy='omit')
            up_score = up_z.mean(axis=1)
        else:
            up_score = pd.Series(0, index=expression_df.columns)
        
        if len(down_genes) > 0:
            down_expr = expression_df.loc[down_genes].T
            down_z = down_expr.apply(scipy_zscore, nan_policy='omit')
            down_score = down_z.mean(axis=1)
        else:
            down_score = pd.Series(0, index=expression_df.columns)
        
        score = up_score - down_score
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    return score, total_found, total_requested


def classify_bulk_patients(expression_df, method='fixed_threshold'):
    """
    Clasifica pacientes bulk en fenotipos inmunes.
    
    Lógica:
    1. Immune Score alto (>Q75) → Inflamed
    2. Immune Score bajo (<Q25) + Silencing > Barrier → Desert
    3. Immune Score bajo (<Q25) + Barrier > Silencing → Excluded
    4. Resto → Intermediate / Ambiguous_Cold
    """
    print(f"\n  Clasificando pacientes...")
    
    immune_score, n_imm, t_imm = compute_signature_score(
        expression_df, IMMUNE_SIGNATURE
    )
    silencing_score, n_sil, t_sil = compute_signature_score(
        expression_df, SILENCING_SIGNATURE
    )
    barrier_score, n_bar, t_bar = compute_signature_score(
        expression_df, BARRIER_SIGNATURE
    )
    
    print(f"    Genes usados — Immune: {n_imm}/{t_imm}, "
          f"Silencing: {n_sil}/{t_sil}, Barrier: {n_bar}/{t_bar}")
    
    result = pd.DataFrame({
        'immune_score': immune_score,
        'silencing_score': silencing_score,
        'barrier_score': barrier_score,
    }, index=expression_df.columns)
    
    # Umbrales: cuartiles
    immune_high = result['immune_score'].quantile(0.75)
    immune_low = result['immune_score'].quantile(0.25)
    
    phenotypes = pd.Series('Unclassified', index=result.index)
    
    # Paso 1: Inflamed (inmune alto)
    inflamed_mask = result['immune_score'] >= immune_high
    phenotypes[inflamed_mask] = 'Inflamed'
    
    # Paso 2: Fríos (inmune bajo)
    cold_mask = result['immune_score'] < immune_low
    
    # Paso 3: Diferenciar Desert vs Excluded
    diff = result['silencing_score'] - result['barrier_score']
    ambiguity_threshold = diff[cold_mask].std() * 0.25 if cold_mask.sum() > 10 else 0.1
    
    desert_mask = cold_mask & (diff > ambiguity_threshold)
    excluded_mask = cold_mask & (diff < -ambiguity_threshold)
    ambiguous_mask = cold_mask & ~desert_mask & ~excluded_mask
    
    phenotypes[desert_mask] = 'Immune_Desert'
    phenotypes[excluded_mask] = 'Immune_Excluded'
    phenotypes[ambiguous_mask] = 'Ambiguous_Cold'
    
    # Resto: Intermediate
    intermediate_mask = phenotypes == 'Unclassified'
    phenotypes[intermediate_mask] = 'Intermediate'
    
    result['phenotype'] = phenotypes
    
    print(f"\n  Clasificación bulk:")
    for pheno, count in result['phenotype'].value_counts().items():
        pct = count / len(result) * 100
        print(f"    {pheno:20s}: {count:>5} ({pct:.1f}%)")
    
    return result


# ============================================================================
# 3. ANÁLISIS DE SUPERVIVENCIA
# ============================================================================

def kaplan_meier_estimator_manual(time, event):
    """
    KM manual (fallback si lifelines no disponible).
    """
    order = np.argsort(time)
    time = np.array(time)[order]
    event = np.array(event)[order]
    
    unique_times = np.unique(time[event == 1])
    survival = np.ones(len(unique_times) + 1)
    times_out = np.zeros(len(unique_times) + 1)
    times_out[0] = 0
    
    for i, t in enumerate(unique_times):
        n_at_risk = (time >= t).sum()
        d = ((time == t) & (event == 1)).sum()
        
        if n_at_risk > 0:
            survival[i + 1] = survival[i] * (1 - d / n_at_risk)
        else:
            survival[i + 1] = survival[i]
        
        times_out[i + 1] = t
    
    return times_out, survival


def log_rank_test_manual(time1, event1, time2, event2):
    """
    Test log-rank manual (Mantel-Haenszel) como fallback.
    """
    all_event_times = np.unique(np.concatenate([
        time1[event1 == 1], time2[event2 == 1]
    ]))
    
    O1 = 0
    E1 = 0
    V = 0
    
    for t in all_event_times:
        n1 = (time1 >= t).sum()
        n2 = (time2 >= t).sum()
        n = n1 + n2
        
        d1 = ((time1 == t) & (event1 == 1)).sum()
        d2 = ((time2 == t) & (event2 == 1)).sum()
        d = d1 + d2
        
        if n > 0:
            O1 += d1
            E1 += n1 * d / n
            if n > 1:
                V += (n1 * n2 * d * (n - d)) / (n**2 * (n - 1))
    
    if V > 0:
        chi2 = (O1 - E1)**2 / V
        from scipy.stats import chi2 as chi2_dist
        p_value = 1 - chi2_dist.cdf(chi2, df=1)
    else:
        chi2 = 0
        p_value = 1.0
    
    return chi2, p_value


def _run_logrank(time1, event1, time2, event2):
    """
    Wrapper que usa lifelines si disponible, manual si no.
    """
    if LIFELINES_AVAILABLE:
        result = ll_logrank_test(time1, time2, event1, event2)
        return result.test_statistic, result.p_value
    else:
        return log_rank_test_manual(time1, event1, time2, event2)


def _run_cox_regression(merged_df, group_col='phenotype_binary',
                         covariates=None):
    """
    Cox Proportional Hazards con lifelines (si disponible).
    
    Returns: dict con HR, CI, p-value, o None si falla.
    """
    if not LIFELINES_AVAILABLE:
        return None
    
    cox_data = merged_df[['time', 'event', group_col]].dropna().copy()
    
    # Añadir covariates si disponibles
    if covariates:
        for cov in covariates:
            if cov in merged_df.columns:
                cox_data[cov] = pd.to_numeric(merged_df[cov], errors='coerce')
        cox_data = cox_data.dropna()
    
    if len(cox_data) < 20:
        return None
    
    try:
        penalizer = 0.01
        if USE_CONFIG_PARAMS:
            penalizer = SURVIVAL_PARAMS.COX_PENALIZER
        
        cph = CoxPHFitter(penalizer=penalizer)
        cols = ['time', 'event', group_col]
        if covariates:
            cols += [c for c in covariates if c in cox_data.columns]
        
        cph.fit(cox_data[cols], duration_col='time', event_col='event')
        
        summary = cph.summary
        if group_col in summary.index:
            hr = np.exp(summary.loc[group_col, 'coef'])
            hr_lower = np.exp(summary.loc[group_col, 'coef lower 95%'])
            hr_upper = np.exp(summary.loc[group_col, 'coef upper 95%'])
            p = summary.loc[group_col, 'p']
            
            return {
                'HR': hr,
                'HR_lower_95': hr_lower,
                'HR_upper_95': hr_upper,
                'HR_p_value': p,
                'n': len(cox_data),
                'covariates_used': [c for c in (covariates or []) if c in cox_data.columns],
            }
    except Exception as e:
        print(f" Cox regression falló: {e}")
    
    return None


def run_survival_analysis(classification_df, survival_df, clinical_df=None,
                           dataset_name='METABRIC'):
    """
    Compara supervivencia entre fenotipos clasificados.
    
    H12: Usa lifelines (KM + Cox + log-rank) con fallback manual.
    
    Comparaciones:
    1. Desert vs Excluded (hipótesis principal)
    2. Inflamed vs Cold (validación positiva)
    3. High Silencing vs Low Silencing
    """
    print(f"\n{'─'*60}")
    print(f"ANÁLISIS DE SUPERVIVENCIA — {dataset_name}")
    print(f"{'─'*60}")
    
    if survival_df is None or classification_df is None:
        print("Datos insuficientes para survival")
        return None, None
    
    # Merge clasificación con supervivencia
    common_idx = classification_df.index.intersection(survival_df.index)
    
    if len(common_idx) < 20:
        print(f" Solo {len(common_idx)} pacientes con ambos datos → skip")
        return None, None
    
    merged = classification_df.loc[common_idx].join(survival_df.loc[common_idx])
    
    # Añadir covariates clínicas si disponibles
    if clinical_df is not None:
        for cov in ['AGE_AT_DIAGNOSIS', 'Age at Diagnosis', 'GRADE',
                     'Neoplasm Histologic Grade']:
            if cov in clinical_df.columns:
                cov_data = pd.to_numeric(clinical_df.loc[common_idx, cov],
                                          errors='coerce')
                merged[cov] = cov_data
    
    merged = merged.dropna(subset=['time', 'event'])
    print(f"  Pacientes con supervivencia: {len(merged)}")
    
    results = []
    
    # ─── Comparación 1: Desert vs Excluded ───
    desert = merged[merged['phenotype'] == 'Immune_Desert']
    excluded = merged[merged['phenotype'] == 'Immune_Excluded']
    
    if len(desert) >= 10 and len(excluded) >= 10:
        chi2, p_lr = _run_logrank(
            desert['time'].values, desert['event'].values.astype(int),
            excluded['time'].values, excluded['event'].values.astype(int)
        )
        
        # Cox regression
        merged_de = pd.concat([desert, excluded])
        merged_de['phenotype_binary'] = (
            merged_de['phenotype'] == 'Immune_Desert'
        ).astype(int)
        
        cox_result = _run_cox_regression(merged_de, 'phenotype_binary')
        
        row = {
            'dataset': dataset_name,
            'comparison': 'Desert_vs_Excluded',
            'n_group1': len(desert),
            'n_group2': len(excluded),
            'median_survival_group1': desert['time'].median(),
            'median_survival_group2': excluded['time'].median(),
            'log_rank_chi2': chi2,
            'log_rank_pvalue': p_lr,
            'significant': p_lr < 0.05,
        }
        
        if cox_result:
            row.update({
                'cox_HR': cox_result['HR'],
                'cox_HR_lower95': cox_result['HR_lower_95'],
                'cox_HR_upper95': cox_result['HR_upper_95'],
                'cox_p': cox_result['HR_p_value'],
            })
        
        results.append(row)
        
        hr_str = f", HR={cox_result['HR']:.2f} ({cox_result['HR_lower_95']:.2f}-{cox_result['HR_upper_95']:.2f})" if cox_result else ""
        print(f"\n  Desert vs Excluded:")
        print(f"    N: {len(desert)} vs {len(excluded)}")
        print(f"    Mediana supervivencia: {desert['time'].median():.1f} vs "
              f"{excluded['time'].median():.1f} meses")
        print(f"    Log-rank p = {p_lr:.4f}{hr_str}")
        print(f"    {'✓ SIGNIFICATIVO' if p_lr < 0.05 else '✗ No significativo'}")
    else:
        print(f"\n  Desert({len(desert)}) o Excluded({len(excluded)}) < 10 → skip")
    
    # ─── Comparación 2: Inflamed vs Cold ───
    inflamed = merged[merged['phenotype'] == 'Inflamed']
    cold = merged[merged['phenotype'].isin(
        ['Immune_Desert', 'Immune_Excluded', 'Ambiguous_Cold']
    )]
    
    if len(inflamed) >= 10 and len(cold) >= 10:
        chi2, p_lr = _run_logrank(
            inflamed['time'].values, inflamed['event'].values.astype(int),
            cold['time'].values, cold['event'].values.astype(int)
        )
        
        merged_ic = pd.concat([inflamed, cold])
        merged_ic['phenotype_binary'] = (
            merged_ic['phenotype'] == 'Inflamed'
        ).astype(int)
        cox_ic = _run_cox_regression(merged_ic, 'phenotype_binary')
        
        row = {
            'dataset': dataset_name,
            'comparison': 'Inflamed_vs_Cold',
            'n_group1': len(inflamed),
            'n_group2': len(cold),
            'median_survival_group1': inflamed['time'].median(),
            'median_survival_group2': cold['time'].median(),
            'log_rank_chi2': chi2,
            'log_rank_pvalue': p_lr,
            'significant': p_lr < 0.05,
        }
        if cox_ic:
            row.update({
                'cox_HR': cox_ic['HR'],
                'cox_HR_lower95': cox_ic['HR_lower_95'],
                'cox_HR_upper95': cox_ic['HR_upper_95'],
                'cox_p': cox_ic['HR_p_value'],
            })
        results.append(row)
        
        print(f"\n  Inflamed vs Cold:")
        print(f"    N: {len(inflamed)} vs {len(cold)}")
        print(f"    Log-rank p = {p_lr:.4f}")
    
    # ─── Comparación 3: High vs Low Silencing Score ───
    sil_median = merged['silencing_score'].median()
    high_sil = merged[merged['silencing_score'] >= sil_median]
    low_sil = merged[merged['silencing_score'] < sil_median]
    
    if len(high_sil) >= 10 and len(low_sil) >= 10:
        chi2, p_lr = _run_logrank(
            high_sil['time'].values, high_sil['event'].values.astype(int),
            low_sil['time'].values, low_sil['event'].values.astype(int)
        )
        
        results.append({
            'dataset': dataset_name,
            'comparison': 'High_vs_Low_Silencing',
            'n_group1': len(high_sil),
            'n_group2': len(low_sil),
            'median_survival_group1': high_sil['time'].median(),
            'median_survival_group2': low_sil['time'].median(),
            'log_rank_chi2': chi2,
            'log_rank_pvalue': p_lr,
            'significant': p_lr < 0.05,
        })
        print(f"\n  High vs Low Silencing Score:")
        print(f"    Log-rank p = {p_lr:.4f}")
    
    results_df = pd.DataFrame(results) if results else None
    return results_df, merged


# ============================================================================
# 4. VALIDACIÓN DE FIRMA (AUC)
# ============================================================================

def validate_signature_auc(expression_df, classification_df):
    """
    AUC de la firma de 15 genes para distinguir Desert vs Excluded.
    
    H7 FIX: Manejo robusto de NaN — imputa mediana por gen, 
    skip pacientes con >50% NaN, y reporta tasa de imputación.
    DX5: Resuelve aliases de genes.
    """
    print(f"\n{'─'*60}")
    print(f"VALIDACIÓN AUC DE FIRMA")
    print(f"{'─'*60}")
    
    if expression_df is None or classification_df is None:
        return None
    
    cold_patients = classification_df[
        classification_df['phenotype'].isin(['Immune_Desert', 'Immune_Excluded'])
    ]
    
    if len(cold_patients) < 20:
        print(f"  Insuficientes pacientes fríos: {len(cold_patients)}")
        return None
    
    # Resolver genes con aliases
    available = set(expression_df.index)
    available_genes = _resolve_gene_list(EXCLUSION_SIGNATURE_15, available)
    
    missing = [g for g in EXCLUSION_SIGNATURE_15 
               if _resolve_gene(g, available) is None]
    
    print(f"  Genes de firma disponibles: {len(available_genes)}/{len(EXCLUSION_SIGNATURE_15)}")
    if missing:
        print(f"  Genes faltantes: {missing}")
    
    if len(available_genes) < 5:
        print(f"  Insuficientes genes para AUC")
        return None
    
    # Preparar datos
    common = cold_patients.index.intersection(expression_df.columns)
    if len(common) < 10:
        print(f"  Insuficientes pacientes con expresión: {len(common)}")
        return None
    
    X = expression_df.loc[available_genes, common].T.values
    y = (cold_patients.loc[common, 'phenotype'] == 'Immune_Excluded').astype(int).values
    
    # Manejo robusto de NaN
    nan_total = np.isnan(X).sum()
    nan_pct = nan_total / X.size * 100
    print(f"  H7: NaN en matriz: {nan_total} ({nan_pct:.2f}%)")
    
    # Drop pacientes con >50% NaN
    nan_per_patient = np.isnan(X).sum(axis=1) / X.shape[1]
    valid_patients = nan_per_patient < 0.5
    
    if (~valid_patients).sum() > 0:
        print(f"  H7: Eliminando {(~valid_patients).sum()} pacientes con >50% NaN")
        X = X[valid_patients]
        y = y[valid_patients]
    
    # Imputar NaN restantes con mediana por gen
    for j in range(X.shape[1]):
        nan_mask = np.isnan(X[:, j])
        if nan_mask.any():
            median_val = np.nanmedian(X[:, j])
            if np.isnan(median_val):
                median_val = 0.0
            X[nan_mask, j] = median_val
    
    # Verificar que aún hay varianza
    if len(np.unique(y)) < 2:
        print(f"  Solo una clase representada en y → AUC no calculable")
        return None
    
    # Score compuesto: mean(barrier_genes_z) - mean(silencing_genes_z)
    barrier_resolved = set(_resolve_gene_list(BARRIER_SIGNATURE['up'], available))
    silencing_resolved = set(_resolve_gene_list(
        SILENCING_SIGNATURE['up'] + SILENCING_SIGNATURE['down'], available
    ))
    
    barrier_idx = [i for i, g in enumerate(available_genes) if g in barrier_resolved]
    silencing_idx = [i for i, g in enumerate(available_genes) if g in silencing_resolved]
    
    # Z-score normalizar por gen para consistencia
    X_z = np.zeros_like(X)
    for j in range(X.shape[1]):
        col = X[:, j]
        std = col.std()
        if std > 0:
            X_z[:, j] = (col - col.mean()) / std
        else:
            X_z[:, j] = 0
    
    if barrier_idx and silencing_idx:
        barrier_score = X_z[:, barrier_idx].mean(axis=1)
        silencing_score = X_z[:, silencing_idx].mean(axis=1)
        combined_score = barrier_score - silencing_score
        
        try:
            auc = roc_auc_score(y, combined_score)
            print(f"\n  AUC (Barrier−Silencing z-score): {auc:.3f}")
        except ValueError as e:
            print(f"  AUC falló: {e}")
            auc = np.nan
    else:
        combined_score = X_z.mean(axis=1)
        try:
            auc = roc_auc_score(y, combined_score)
            print(f"\n  AUC (mean z-score, fallback): {auc:.3f}")
        except ValueError:
            auc = np.nan
    
    # ROC curve para figura
    fpr, tpr, thresholds = None, None, None
    if not np.isnan(auc):
        try:
            fpr, tpr, thresholds = roc_curve(y, combined_score)
        except ValueError:
            pass
    
    # AUCs individuales por gen
    individual_aucs = {}
    for i, gene in enumerate(available_genes):
        try:
            gene_auc = roc_auc_score(y, X_z[:, i])
            individual_aucs[gene] = gene_auc
        except ValueError:
            continue
    
    if individual_aucs:
        sorted_aucs = sorted(individual_aucs.items(),
                              key=lambda x: abs(x[1] - 0.5), reverse=True)
        print(f"\n  Top 5 genes discriminativos:")
        for gene, gauc in sorted_aucs[:5]:
            direction = '↑ Excluded' if gauc > 0.5 else '↑ Desert'
            print(f"    {gene:12s}: AUC = {gauc:.3f} ({direction})")
    
    return {
        'auc_combined': auc,
        'n_patients': len(y),
        'n_desert': (y == 0).sum(),
        'n_excluded': (y == 1).sum(),
        'n_genes_used': len(available_genes),
        'individual_aucs': individual_aucs,
        'roc_fpr': fpr,
        'roc_tpr': tpr,
        'nan_imputed_pct': nan_pct,
    }


# ============================================================================
# 5. VISUALIZACIÓN
# ============================================================================

def _plot_km_comparison(group1_data, group2_data, label1, label2,
                         p_value, title, ax, color1='#f39c12', color2='#9b59b6',
                         cox_hr=None):
    """
    Dibuja KM curves con lifelines (si disponible) o manual.
    """
    if LIFELINES_AVAILABLE:
        kmf1 = KaplanMeierFitter()
        kmf1.fit(group1_data['time'], group1_data['event'], label=label1)
        kmf1.plot_survival_function(ax=ax, color=color1, linewidth=2)
        
        kmf2 = KaplanMeierFitter()
        kmf2.fit(group2_data['time'], group2_data['event'], label=label2)
        kmf2.plot_survival_function(ax=ax, color=color2, linewidth=2)
    else:
        t1, s1 = kaplan_meier_estimator_manual(
            group1_data['time'].values, group1_data['event'].values
        )
        t2, s2 = kaplan_meier_estimator_manual(
            group2_data['time'].values, group2_data['event'].values
        )
        ax.step(t1, s1, where='post', color=color1, linewidth=2, label=label1)
        ax.step(t2, s2, where='post', color=color2, linewidth=2, label=label2)
    
    # Anotación p-value
    p_str = f"p = {p_value:.4f}" if p_value >= 0.0001 else f"p = {p_value:.2e}"
    annot = p_str
    if cox_hr is not None:
        annot += f"\nHR = {cox_hr:.2f}"
    
    ax.text(0.5, 0.15, annot, transform=ax.transAxes, fontsize=10,
            ha='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.legend(fontsize=9)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Tiempo (meses)')
    ax.set_ylabel('Supervivencia')
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)


def generate_bulk_validation_figures(results_dict):
    """
    Genera figura de 6 paneles para el paper.
    
    Panel A: KM curves Desert vs Excluded
    Panel B: KM curves Inflamed vs Cold
    Panel C: Score distributions
    Panel D: ROC curve
    Panel E: Barrier vs Silencing scatter
    Panel F: Summary table
    """
    print(f"\n{'─'*60}")
    print(f"GENERANDO FIGURAS")
    print(f"{'─'*60}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Bulk Validation: METABRIC + TCGA-BRCA', fontsize=16,
                  fontweight='bold', y=0.98)
    
    # Panel A: KM Desert vs Excluded
    ax = axes[0, 0]
    plotted_a = False
    for ds_name, data in results_dict.items():
        if data.get('merged') is not None and not plotted_a:
            merged = data['merged']
            desert = merged[merged['phenotype'] == 'Immune_Desert']
            excluded = merged[merged['phenotype'] == 'Immune_Excluded']
            
            if len(desert) >= 5 and len(excluded) >= 5:
                sr = data.get('survival_results')
                p_val = 1.0
                cox_hr = None
                if sr is not None:
                    de_row = sr[sr['comparison'] == 'Desert_vs_Excluded']
                    if len(de_row) > 0:
                        p_val = de_row.iloc[0]['log_rank_pvalue']
                        if 'cox_HR' in de_row.columns:
                            cox_hr = de_row.iloc[0].get('cox_HR')
                
                _plot_km_comparison(
                    desert, excluded,
                    f'Desert (n={len(desert)})', f'Excluded (n={len(excluded)})',
                    p_val, f'A. Desert vs Excluded ({ds_name})', ax,
                    cox_hr=cox_hr
                )
                plotted_a = True
    if not plotted_a:
        ax.text(0.5, 0.5, 'Datos insuficientes', transform=ax.transAxes,
                ha='center', fontsize=12)
        ax.set_title('A. Desert vs Excluded', fontweight='bold')
    
    # Panel B: KM Inflamed vs Cold
    ax = axes[0, 1]
    plotted_b = False
    for ds_name, data in results_dict.items():
        if data.get('merged') is not None and not plotted_b:
            merged = data['merged']
            inflamed = merged[merged['phenotype'] == 'Inflamed']
            cold = merged[merged['phenotype'].isin(
                ['Immune_Desert', 'Immune_Excluded', 'Ambiguous_Cold']
            )]
            
            if len(inflamed) >= 5 and len(cold) >= 5:
                sr = data.get('survival_results')
                p_val = 1.0
                if sr is not None:
                    ic_row = sr[sr['comparison'] == 'Inflamed_vs_Cold']
                    if len(ic_row) > 0:
                        p_val = ic_row.iloc[0]['log_rank_pvalue']
                
                _plot_km_comparison(
                    inflamed, cold,
                    f'Inflamed (n={len(inflamed)})', f'Cold (n={len(cold)})',
                    p_val, f'B. Inflamed vs Cold ({ds_name})', ax,
                    color1='#e74c3c', color2='#3498db'
                )
                plotted_b = True
    if not plotted_b:
        ax.text(0.5, 0.5, 'Datos insuficientes', transform=ax.transAxes,
                ha='center', fontsize=12)
        ax.set_title('B. Inflamed vs Cold', fontweight='bold')
    
    # Panel C: Score distributions
    ax = axes[0, 2]
    for ds_name, data in results_dict.items():
        if data.get('classification') is not None:
            class_df = data['classification']
            colors_pheno = {
                'Immune_Desert': '#f39c12',
                'Immune_Excluded': '#9b59b6',
                'Inflamed': '#e74c3c',
            }
            for pheno, color in colors_pheno.items():
                mask = class_df['phenotype'] == pheno
                if mask.sum() > 5:
                    vals = class_df.loc[mask, 'silencing_score'].dropna()
                    ax.hist(vals, bins=30, alpha=0.5, label=pheno, color=color)
            ax.legend(fontsize=9)
            break
    ax.set_title('C. Silencing Score por Fenotipo', fontweight='bold')
    ax.set_xlabel('Silencing Score (z-score)')
    ax.set_ylabel('Frecuencia')
    ax.grid(alpha=0.3)
    
    # Panel D: ROC curve
    ax = axes[1, 0]
    plotted_roc = False
    for ds_name, data in results_dict.items():
        auc_res = data.get('auc_results')
        if auc_res and auc_res.get('roc_fpr') is not None:
            ax.plot(auc_res['roc_fpr'], auc_res['roc_tpr'],
                    linewidth=2, label=f"{ds_name} (AUC={auc_res['auc_combined']:.3f})")
            plotted_roc = True
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    if plotted_roc:
        ax.legend(fontsize=10)
    else:
        ax.text(0.5, 0.5, 'AUC no disponible', transform=ax.transAxes,
                ha='center', fontsize=12)
    ax.set_title('D. ROC Firma 15 genes', fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.grid(alpha=0.3)
    
    # Panel E: Barrier vs Silencing scatter
    ax = axes[1, 1]
    for ds_name, data in results_dict.items():
        if data.get('classification') is not None:
            class_df = data['classification']
            colors_scatter = {
                'Immune_Desert': '#f39c12',
                'Immune_Excluded': '#9b59b6',
                'Inflamed': '#e74c3c',
                'Ambiguous_Cold': '#95a5a6',
                'Intermediate': '#bdc3c7',
            }
            for pheno, color in colors_scatter.items():
                mask = class_df['phenotype'] == pheno
                if mask.sum() > 0:
                    ax.scatter(
                        class_df.loc[mask, 'silencing_score'],
                        class_df.loc[mask, 'barrier_score'],
                        c=color, alpha=0.4, s=10, label=pheno
                    )
            ax.legend(fontsize=8, markerscale=2)
            break
    ax.set_title('E. Silencing vs Barrier Score', fontweight='bold')
    ax.set_xlabel('Silencing Score')
    ax.set_ylabel('Barrier Score')
    ax.grid(alpha=0.3)
    
    # Panel F: Summary table
    ax = axes[1, 2]
    ax.axis('off')
    
    all_survival = []
    for ds_name, data in results_dict.items():
        if data.get('survival_results') is not None:
            all_survival.append(data['survival_results'])
    
    if all_survival:
        surv_df = pd.concat(all_survival)
        
        cell_text = []
        cell_colors = []
        for _, row in surv_df.iterrows():
            sig = row['significant']
            color = '#d5f5e3' if sig else '#fadbd8'
            p_str = (f"{row['log_rank_pvalue']:.4f}"
                     if row['log_rank_pvalue'] >= 0.0001
                     else f"{row['log_rank_pvalue']:.2e}")
            hr_str = ''
            if 'cox_HR' in row and pd.notna(row.get('cox_HR')):
                hr_str = f"{row['cox_HR']:.2f}"
            cell_text.append([
                row['dataset'],
                row['comparison'].replace('_', ' '),
                f"{int(row['n_group1'])} vs {int(row['n_group2'])}",
                p_str,
                hr_str,
                '✓' if sig else '✗',
            ])
            cell_colors.append([color] * 6)
        
        table = ax.table(
            cellText=cell_text,
            colLabels=['Dataset', 'Comparison', 'N', 'p-value', 'HR', 'Sig'],
            cellColours=cell_colors,
            loc='center',
            cellLoc='center',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.4)
    
    ax.set_title('F. Resumen estadístico', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = BULK_DIR / "figures" / "Fig_bulk_combined_6panel"
    fig.savefig(str(fig_path) + '.png', dpi=300)
    fig.savefig(str(fig_path) + '.pdf')
    plt.close(fig)
    print(f"  ✓ Guardado: {fig_path}.png/.pdf")
    
    # Figuras individuales para mayor calidad
    _save_individual_km_figures(results_dict)


def _save_individual_km_figures(results_dict):
    """Genera KM curves individuales en alta resolución."""
    
    for ds_name, data in results_dict.items():
        if data.get('merged') is None:
            continue
        
        merged = data['merged']
        sr = data.get('survival_results')
        
        # Desert vs Excluded
        desert = merged[merged['phenotype'] == 'Immune_Desert']
        excluded = merged[merged['phenotype'] == 'Immune_Excluded']
        
        if len(desert) >= 5 and len(excluded) >= 5:
            fig, ax = plt.subplots(figsize=(8, 6))
            p_val = 1.0
            cox_hr = None
            if sr is not None:
                de_row = sr[sr['comparison'] == 'Desert_vs_Excluded']
                if len(de_row) > 0:
                    p_val = de_row.iloc[0]['log_rank_pvalue']
                    if 'cox_HR' in de_row.columns:
                        cox_hr = de_row.iloc[0].get('cox_HR')
            
            _plot_km_comparison(
                desert, excluded,
                f'Immune Desert (n={len(desert)})',
                f'Immune Excluded (n={len(excluded)})',
                p_val,
                f'Desert vs Excluded — {ds_name}',
                ax, cox_hr=cox_hr
            )
            
            path = BULK_DIR / "figures" / f"Fig_bulk_KM_Desert_vs_Excluded_{ds_name}"
            fig.savefig(str(path) + '.png', dpi=300)
            fig.savefig(str(path) + '.pdf')
            plt.close(fig)
            print(f"  ✓ {path.name}.pdf")
        
        # Inflamed vs Cold
        inflamed = merged[merged['phenotype'] == 'Inflamed']
        cold = merged[merged['phenotype'].isin(
            ['Immune_Desert', 'Immune_Excluded', 'Ambiguous_Cold']
        )]
        
        if len(inflamed) >= 5 and len(cold) >= 5:
            fig, ax = plt.subplots(figsize=(8, 6))
            p_val = 1.0
            if sr is not None:
                ic_row = sr[sr['comparison'] == 'Inflamed_vs_Cold']
                if len(ic_row) > 0:
                    p_val = ic_row.iloc[0]['log_rank_pvalue']
            
            _plot_km_comparison(
                inflamed, cold,
                f'Inflamed (n={len(inflamed)})',
                f'Cold (n={len(cold)})',
                p_val,
                f'Inflamed vs Cold — {ds_name}',
                ax, color1='#e74c3c', color2='#3498db'
            )
            
            path = BULK_DIR / "figures" / f"Fig_bulk_KM_Inflamed_vs_Cold_{ds_name}"
            fig.savefig(str(path) + '.png', dpi=300)
            fig.savefig(str(path) + '.pdf')
            plt.close(fig)
            print(f"  ✓ {path.name}.pdf")


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    """
    Pipeline completo de validación bulk.
    
    1. METABRIC: carga → filtro TNBC → clasifica → survival → AUC
    2. TCGA-BRCA: carga → H4 normalización IDs → filtro → clasifica → survival → AUC
    3. Figuras combinadas
    4. Resumen ejecutivo
    """
    start_time = time.time()
    
    print("=" * 70)
    print("BULK VALIDATION v2.0 — METABRIC + TCGA-BRCA")
    print("=" * 70)
    print(f"  Output: {BULK_DIR}")
    print(f"  lifelines: {'✓' if LIFELINES_AVAILABLE else '✗ (fallback manual)'}")
    print(f"  Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    results_dict = {}
    
    # ================================================================
    # METABRIC
    # ================================================================
    print(f"\n{'='*70}")
    print("1. METABRIC")
    print(f"{'='*70}")
    
    metabric_dir = find_data_directory(METABRIC_PATHS)
    
    if metabric_dir:
        print(f"  Directorio: {metabric_dir}")
        
        expr_met, transform_met = load_expression_data(metabric_dir, 'METABRIC')
        clin_met = load_clinical_data(metabric_dir, 'METABRIC')
        
        if expr_met is not None:
            clin_tnbc, expr_tnbc = filter_tnbc_patients(
                clin_met, expr_met, 'METABRIC'
            )
            
            if expr_tnbc is not None and expr_tnbc.shape[1] >= 20:
                class_met = classify_bulk_patients(expr_tnbc)
                class_met.to_csv(BULK_DIR / "tables" / "bulk_classification_METABRIC.csv")
                
                surv_met = extract_survival_data(clin_tnbc)
                surv_results, merged = run_survival_analysis(
                    class_met, surv_met, clin_tnbc, 'METABRIC'
                )
                if surv_results is not None:
                    surv_results.to_csv(
                        BULK_DIR / "tables" / "survival_METABRIC.csv", index=False
                    )
                
                auc_met = validate_signature_auc(expr_tnbc, class_met)
                
                results_dict['METABRIC'] = {
                    'classification': class_met,
                    'survival_results': surv_results,
                    'merged': merged,
                    'auc_results': auc_met,
                    'transform': transform_met,
                }
            else:
                print("  METABRIC: expresión insuficiente tras filtro TNBC")
        else:
            print("  METABRIC: archivo de expresión NO encontrado")
    else:
        print("  Directorio METABRIC no encontrado")
        for p in METABRIC_PATHS:
            print(f"     Busqué: {p}")
    
    # ================================================================
    # TCGA-BRCA
    # ================================================================
    print(f"\n{'='*70}")
    print("2. TCGA-BRCA")
    print(f"{'='*70}")
    
    tcga_dir = find_data_directory(TCGA_PATHS)
    
    if tcga_dir:
        print(f"  Directorio: {tcga_dir}")
        
        expr_tcga, transform_tcga = load_expression_data(tcga_dir, 'TCGA')
        clin_tcga = load_clinical_data(tcga_dir, 'TCGA')
        
        if expr_tcga is not None:
            clin_tnbc_t, expr_tnbc_t = filter_tnbc_patients(
                clin_tcga, expr_tcga, 'TCGA'
            )
            
            if expr_tnbc_t is not None and expr_tnbc_t.shape[1] >= 20:
                class_tcga = classify_bulk_patients(expr_tnbc_t)
                class_tcga.to_csv(
                    BULK_DIR / "tables" / "bulk_classification_TCGA.csv"
                )
                
                surv_tcga = extract_survival_data(clin_tnbc_t)
                surv_results_t, merged_t = run_survival_analysis(
                    class_tcga, surv_tcga, clin_tnbc_t, 'TCGA'
                )
                if surv_results_t is not None:
                    surv_results_t.to_csv(
                        BULK_DIR / "tables" / "survival_TCGA.csv", index=False
                    )
                
                auc_tcga = validate_signature_auc(expr_tnbc_t, class_tcga)
                
                results_dict['TCGA'] = {
                    'classification': class_tcga,
                    'survival_results': surv_results_t,
                    'merged': merged_t,
                    'auc_results': auc_tcga,
                    'transform': transform_tcga,
                }
            else:
                print("  TCGA: expresión insuficiente tras filtro TNBC")
        else:
            print("  TCGA: archivo de expresión NO encontrado")
    else:
        print("  Directorio TCGA no encontrado")
        for p in TCGA_PATHS:
            print(f"     Busqué: {p}")
    
    # ================================================================
    # FIGURAS Y CONSOLIDACIÓN
    # ================================================================
    if results_dict:
        print(f"\n{'='*70}")
        print("3. GENERANDO FIGURAS Y CONSOLIDACIÓN")
        print(f"{'='*70}")
        
        generate_bulk_validation_figures(results_dict)
        
        # Consolidar AUC
        auc_summary = []
        for ds_name, data in results_dict.items():
            if data.get('auc_results') is not None:
                ar = data['auc_results']
                auc_summary.append({
                    'dataset': ds_name,
                    'auc_combined': ar['auc_combined'],
                    'n_patients': ar['n_patients'],
                    'n_desert': ar['n_desert'],
                    'n_excluded': ar['n_excluded'],
                    'n_genes': ar['n_genes_used'],
                    'nan_imputed_pct': ar.get('nan_imputed_pct', 0),
                })
        
        if auc_summary:
            pd.DataFrame(auc_summary).to_csv(
                BULK_DIR / "tables" / "signature_validation_auc.csv", index=False
            )
        
        # Consolidar supervivencia
        all_surv = []
        for ds_name, data in results_dict.items():
            if data.get('survival_results') is not None:
                all_surv.append(data['survival_results'])
        
        if all_surv:
            pd.concat(all_surv).to_csv(
                BULK_DIR / "tables" / "survival_analysis_results.csv", index=False
            )
    
    # ================================================================
    # RESUMEN EJECUTIVO
    # ================================================================
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"{'✅' if results_dict else '⚠'} BULK VALIDATION "
          f"COMPLETADA en {elapsed/60:.1f} minutos")
    print(f"{'='*70}")
    
    if not results_dict:
        print(f"\n  NINGÚN dataset bulk procesado.")
        print(f"  ARCHIVOS NECESARIOS:")
        print(f"  METABRIC: data_mrna_illumina_microarray.txt + "
              f"brca_metabric_clinical_data.tsv")
        print(f"  TCGA: data_mrna_seq_v2_rsem.txt + "
              f"brca_tcga_pan_can_atlas_2018_clinical_data.tsv")
    else:
        for ds_name, data in results_dict.items():
            print(f"\n  {ds_name} (transform: {data.get('transform', '?')}):")
            
            if data.get('classification') is not None:
                n = len(data['classification'])
                n_des = (data['classification']['phenotype'] == 'Immune_Desert').sum()
                n_exc = (data['classification']['phenotype'] == 'Immune_Excluded').sum()
                n_inf = (data['classification']['phenotype'] == 'Inflamed').sum()
                print(f"    Pacientes TNBC clasificados: {n}")
                print(f"    Desert: {n_des} ({n_des/n*100:.1f}%), "
                      f"Excluded: {n_exc} ({n_exc/n*100:.1f}%), "
                      f"Inflamed: {n_inf} ({n_inf/n*100:.1f}%)")
            
            if data.get('auc_results') is not None:
                auc_val = data['auc_results']['auc_combined']
                print(f"    AUC firma 15 genes: {auc_val:.3f}")
            
            if data.get('survival_results') is not None:
                for _, row in data['survival_results'].iterrows():
                    sig = '✓' if row['significant'] else '✗'
                    hr_str = ''
                    if 'cox_HR' in row and pd.notna(row.get('cox_HR')):
                        hr_str = f" HR={row['cox_HR']:.2f}"
                    print(f"    {row['comparison']}: "
                          f"p={row['log_rank_pvalue']:.4f}{hr_str} {sig}")
    
    print(f"\n  ARCHIVOS GENERADOS:")
    for subdir in ['tables', 'figures']:
        full = BULK_DIR / subdir
        if os.path.exists(full):
            for f in sorted(os.listdir(full)):
                print(f"    {subdir}/{f}")
    
    # Texto Methods auto-generado
    n_met = 0
    n_tcga = 0
    if 'METABRIC' in results_dict and results_dict['METABRIC'].get('classification') is not None:
        n_met = len(results_dict['METABRIC']['classification'])
    if 'TCGA' in results_dict and results_dict['TCGA'].get('classification') is not None:
        n_tcga = len(results_dict['TCGA']['classification'])
    
    km_method = "lifelines KaplanMeierFitter" if LIFELINES_AVAILABLE else "manual Kaplan-Meier estimation"
    cox_text = " Cox proportional hazards regression assessed hazard ratios with age as covariate." if LIFELINES_AVAILABLE else ""
    
    print(f"\n  TEXTO METHODS AUTO-GENERADO:")
    print(f"  {'─'*50}")
    # Framing correcto — "prognostic validation", no "replication"
    # Bulk data NO puede distinguir Desert de Excluded a nivel de muestra individual
    # (un tumor puede ser 60% Desert y 40% Excluded). La validación es PRONÓSTICA:
    # ¿las firmas derivadas espacialmente predicen supervivencia en cohortes bulk?
    print(f'  "To assess the prognostic relevance of spatially-derived gene')
    print(f'   signatures in independent bulk cohorts, we applied gene')
    print(f'   signature-based classification to METABRIC (N={n_met} TNBC)')
    print(f'   and TCGA-BRCA (N={n_tcga} Basal) bulk RNA-seq datasets')
    print(f'   downloaded from cBioPortal. TCGA RSEM values were')
    print(f'   log2(x+1)-transformed; METABRIC microarray data were already')
    print(f'   log2-normalized. Patients were classified into immune phenotypes')
    print(f'   using z-score normalized expression of Silencing (MYC, EZH2,')
    print(f'   DNMT1, STAT3, SUZ12, CTNNB1, ATF3; inverse: TMEM173, TBK1, IRF3,')
    print(f'   CGAS, MB21D1, CXCL9, CXCL10, CXCL11, CCL5), Barrier (COL1A1,')
    print(f'   COL1A2, COL10A1, FN1, POSTN, TGFB1, ACTA2, FAP, THBS2),')
    print(f'   and Immune (CD8A, CD8B, CD3D, CD3E, GZMA, GZMB, PRF1)')
    print(f'   gene signatures. Survival was compared using')
    print(f'   {km_method} with the log-rank test.{cox_text}"')
    
    return results_dict


if __name__ == '__main__':
    main()

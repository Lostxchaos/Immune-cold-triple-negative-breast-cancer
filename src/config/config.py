"""
================================================================================
CONFIGURACION CENTRALIZADA - TNBC SPATIAL TRANSCRIPTOMICS PIPELINE
================================================================================

Este modulo centraliza toda la configuracion del pipeline para facilitar
la reproducibilidad y el ajuste de parametros.

ACTUALIZACION 1:
- Agregados genes validados: SPP1, STAT3, DNMT1
- Nuevos umbrales de presencia celular segun estandares Q1
- Firma TAM_POLARIZATION para ratio CXCL9:SPP1 
- Parametros de sensibilidad para analisis de robustez

ACTUALIZACION 2:
- CANONICAL_SIGNATURES como fuente única de verdad
- PHENOTYPE_COLORS canónico centralizado
================================================================================
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch


# ============================================================================
# DETECCION DE ENTORNO
# ============================================================================

def detect_environment() -> str:
    """Detecta si estamos en HPC o en maquina local"""
    hpc_indicators = [
        'dgx' in os.environ.get('HOSTNAME', '').lower(),
        'sc-lab' in os.environ.get('HOSTNAME', '').lower(),
        '/home/external' in os.path.expanduser('~'),
    ]
    
    if any(hpc_indicators):
        return 'HPC'
    return 'LOCAL'

ENVIRONMENT = detect_environment()


# ============================================================================
# CONFIGURACION DE RUTAS
# ============================================================================

@dataclass
class PathConfig:
    """Configuracion de rutas del proyecto"""
    
    # IMPORTANTE: Modificar esta ruta segun tu sistema
    BASE_DIR: Path = Path("/home/external/frjimenez/fabian/genoma")
    
    def __post_init__(self):
        """Recalcular rutas dependientes del BASE_DIR"""
        # Datos crudos Visium
        self.VISIUM_GSE210616 = self.BASE_DIR / "data/raw/GSE210616_RAW"
        self.VISIUM_GSE213688 = self.BASE_DIR / "data/raw/GSE213688_RAW"
        
        # Referencia scRNA-seq
        self.SCRNA_GSE176078 = self.BASE_DIR / "data/raw/GSE176078_scRNA"
        
        # Datos procesados
        self.PROCESSED_DIR = self.BASE_DIR / "data/processed"
        
        # Modelos Cell2Location
        self.MODELS_DIR = self.BASE_DIR / "models"
        
        # Resultados
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.FIGURES_DIR = self.RESULTS_DIR / "figures"
        self.TABLES_DIR = self.RESULTS_DIR / "tables"
    
    def create_directories(self):
        """Crea todos los directorios necesarios"""
        dirs_to_create = [
            self.PROCESSED_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.FIGURES_DIR,
            self.TABLES_DIR,
        ]
        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Directorios creados en: {self.BASE_DIR}")
    
    def update_base_dir(self, new_base: Path):
        """Actualiza el directorio base y recalcula todas las rutas"""
        self.BASE_DIR = Path(new_base)
        self.__post_init__()

PATHS = PathConfig()


# ============================================================================
# FIRMAS GENICAS DEL MECANISMO (ACTUALIZADAS v2.0)
# ============================================================================

@dataclass
class GeneSignatures:
    """
    Firmas genicas del mecanismo de Silencio de Senalizacion
    
    HIPOTESIS CENTRAL (validada por literatura 2020-2024):
    MYC/EZH2/STAT3 (represores) -> suprimen TMEM173 (STING) via DNMT1
    -> bloquean CXCL9/10/11 -> cDC1 no reclutadas -> Desierto Inmune
    """
    
    # =========================================================================
    # REPRESORES DEL EJE STING (alta expresion en Desert)
    # =========================================================================
    SILENCING_REPRESSORS: List[str] = field(default_factory=lambda: [
        'MYC',       # Oncogen que reprime STING directamente (Lee 2022)
        'EZH2',      # Metiltransferasa H3K27me3 (silenciamiento epigenetico)
        'SUZ12',     # Componente del complejo PRC2 con EZH2
        'CTNNB1',    # Beta-catenina (via WNT, inmunosupresora)
        'ATF3',      # Factor de transcripcion inmunosupresor
        'STAT3',     # NUEVO: Coopera con MYC para represion STING (Snoeren 2025)
        'DNMT1',     # NUEVO: MYC activa DNMT1 -> metilacion STING (Wu 2021)
    ])
    
    # =========================================================================
    # VIA STING (baja expresion en Desert)
    # =========================================================================
    STING_PATHWAY: List[str] = field(default_factory=lambda: [
        'TMEM173',   # STING (Stimulator of Interferon Genes) - TARGET CENTRAL
        'TBK1',      # Quinasa downstream de STING
        'IRF3',      # Factor de transcripcion de interferones
        'CGAS',      # Sensor de DNA citosolico (upstream de STING)
        'MB21D1',    # Alias de cGAS en algunas anotaciones
        # NOTA PAPER: MB21D1 está AUSENTE en Discovery (GSE210616) .raw Y .X.
        # STING1 (alias TMEM173) también ausente. STING_Score se calcula con
        # 4/5 genes. Methods debe decir "4 of 5 STING pathway genes were
        # detected in the Discovery dataset (MB21D1 absent)."
        # Validation (GSE213688): 0/5 STING genes presentes — score=0.
    ])
    
    # =========================================================================
    # QUIMIOQUINAS DE RECLUTAMIENTO (ausentes en Desert)
    # =========================================================================
    CHEMOKINE_SIGNALS: List[str] = field(default_factory=lambda: [
        'CXCL9',     # Quimioquina de celulas T - CRITICO para ratio
                     # NOTA FIX-29: CXCL9 también en TAM_POLARIZATION (intencional:
                     # rol dual como quimioquina de reclutamiento y marcador TAM anti-tumoral,
                     # Bill 2023). Si ambas firmas se usan en el mismo modelo, considerar colinealidad.
        'CXCL10',    # Quimioquina de celulas T (IP-10)
        'CXCL11',    # Quimioquina de celulas T (I-TAC)
        'CCL5',      # RANTES - reclutamiento de T y NK
        'HLA-A',     # Presentacion de antigeno MHC-I
        'B2M',       # Beta-2-microglobulina (componente MHC-I)
    ])
    
    # =========================================================================
    # POLARIZACION DE TAMs (NUEVO - Bill et al. Science 2023)
    # El ratio CXCL9:SPP1 define polaridad mejor que M1/M2 clasico
    # =========================================================================
    TAM_POLARIZATION: List[str] = field(default_factory=lambda: [
        'SPP1',      # Osteopontina - TAMs pro-tumorales (alto en Desert)
        'CXCL9',     # TAMs anti-tumorales (bajo en Desert)
        'CD163',     # Marcador M2-like
        'MRC1',      # CD206 - marcador M2-like
        'CD68',      # Pan-macrofago
        'APOE',      # TAMs lipid-associated
    ])
    
    # =========================================================================
    # BARRERA FISICA (alta en Excluded, baja en Desert)
    # Basado en Hammerl et al. Nat Commun 2021
    # =========================================================================
    PHYSICAL_BARRIER: List[str] = field(default_factory=lambda: [
        'COL1A1',    # Colageno tipo I
        'COL1A2',    # Colageno tipo I
        'COL10A1',   # Colageno tipo X (Hammerl signature)
        'FN1',       # Fibronectina
        'POSTN',     # Periostina (matriz extracelular)
        'TGFB1',     # TGF-beta (pro-fibrotico)
        'ACTA2',     # Alpha-SMA (fibroblastos activados/myCAFs)
        'FAP',       # Fibroblast Activation Protein (Hammerl signature)
        'THBS2',     # Thrombospondin-2 (Hammerl signature)
    ])
    
    # =========================================================================
    # ESTROMA DE DESIERTO (vasos disfuncionales - dPVL)
    # NOTA: Estos marcadores son exploratorios, no usar para clasificacion
    # =========================================================================
    DESERT_STROMA: List[str] = field(default_factory=lambda: [
        'RGS5',      # Pericitos inmaduros
        'MCAM',      # CD146 (marcador de pericitos)
        'VWF',       # Factor von Willebrand (endotelio)
        'PECAM1',    # CD31 (endotelio)
    ])
    
    # =========================================================================
    # MARCADORES TUMORALES
    # =========================================================================
    TUMOR_MARKERS: List[str] = field(default_factory=lambda: [
        'EPCAM',     # Marcador epitelial
        'KRT8',      # Queratina 8
        'KRT18',     # Queratina 18
        'KRT19',     # Queratina 19
        'MKI67',     # Proliferacion (Ki-67)
    ])
    
    # =========================================================================
    # CELULAS T CD8+
    # =========================================================================
    CD8_T_CELLS: List[str] = field(default_factory=lambda: [
        'CD8A',
        'CD8B',
        'CD3D',
        'CD3E',
        'GZMA',      # Granzima A
        'GZMB',      # Granzima B
        'PRF1',      # Perforina
    ])
    
    # =========================================================================
    # CELULAS DENDRITICAS CONVENCIONALES TIPO 1 (cDC1)
    # =========================================================================
    CDC1_MARKERS: List[str] = field(default_factory=lambda: [
        'CLEC9A',    # Marcador especifico de cDC1
        'XCR1',      # Receptor de quimioquina
        'BATF3',     # Factor de transcripcion
        'IRF8',      # Factor de transcripcion
    ])
    
    # =========================================================================
    # FIRMA DESIERTO INMUNE (Hammerl et al. Nat Commun 2021)
    # 28% de TNBC - caracterizado por S100A7, WNT/PPARG
    # =========================================================================
    IMMUNE_DESERT_SIGNATURE: List[str] = field(default_factory=lambda: [
        'S100A7',    # Calgranulin - marcador de desierto
        'S100A8',    # Calgranulin
        'S100A9',    # Calgranulin
        'PPARG',     # Via metabolica
        'WNT5A',     # Via WNT
    ])
    
    # =========================================================================
    # FIRMA EXCLUSION INMUNE (Hammerl et al. Nat Commun 2021)
    # 26% de TNBC - caracterizado por TGFb activation, ECM
    # =========================================================================
    IMMUNE_EXCLUDED_SIGNATURE: List[str] = field(default_factory=lambda: [
        'THBS2',     # Thrombospondin-2
        'FAP',       # Fibroblast Activation Protein
        'COL10A1',   # Colageno tipo X
        'COL1A1',    # Colageno tipo I
        'TGFB1',     # TGF-beta
    ])

SIGNATURES = GeneSignatures()


# ============================================================================
# FIX: CANONICAL_SIGNATURES — FUENTE ÚNICA DE VERDAD
# ============================================================================
# Estas son las firmas EXACTAS usadas en Discovery (GSE210616).
# TODOS los módulos (validation.py, bulk_validation.py, robustness_stress_tests.py,
# mechanism_validation_additions.py) DEBEN importar de aquí.
# NO definir firmas locales en ningún otro módulo.
# ============================================================================

CANONICAL_SIGNATURES = {
    # ----- Firmas para clasificación de fenotipos (phenotype_classifier.py) -----
    'silencing_repressors': SIGNATURES.SILENCING_REPRESSORS,
    'sting_pathway':        SIGNATURES.STING_PATHWAY,
    'chemokine_signals':    SIGNATURES.CHEMOKINE_SIGNALS,
    'physical_barrier':     SIGNATURES.PHYSICAL_BARRIER,
    'tumor_markers':        SIGNATURES.TUMOR_MARKERS,
    'cd8_t_cells':          SIGNATURES.CD8_T_CELLS,
    'desert_stroma':        SIGNATURES.DESERT_STROMA,
    'tam_polarization':     SIGNATURES.TAM_POLARIZATION,

    # ----- Firmas Hammerl et al. (referencia externa) -----
    'immune_desert_signature':   SIGNATURES.IMMUNE_DESERT_SIGNATURE,
    'immune_excluded_signature': SIGNATURES.IMMUNE_EXCLUDED_SIGNATURE,

    # ----- Marcadores cDC1 (para validación post-hoc) -----
    'cdc1_markers': SIGNATURES.CDC1_MARKERS,

    # ----- Mapping para Bulk Validation (genes up/down por fenotipo) -----
    # Estos son los genes que se esperan UP o DOWN en Desert vs Excluded
    # para clasificar muestras bulk como Desert-like o Excluded-like.
    'bulk_desert_up': [
        # Represores activos en Desert
        'MYC', 'EZH2', 'SUZ12', 'CTNNB1', 'ATF3', 'STAT3', 'DNMT1',
    ],
    'bulk_desert_down': [
        # Vías suprimidas en Desert (STING + quimioquinas)
        'TMEM173', 'TBK1', 'IRF3', 'CGAS', 'MB21D1',
        'CXCL9', 'CXCL10', 'CXCL11', 'CCL5',
    ],
    'bulk_excluded_up': [
        # Barrera física activa en Excluded
        'COL1A1', 'COL1A2', 'COL10A1', 'FN1', 'POSTN',
        'TGFB1', 'ACTA2', 'FAP', 'THBS2',
    ],
    'bulk_excluded_down': [
        # En Excluded la maquinaria STING puede estar activa pero bloqueada
        # por barrera física — no necesariamente down-regulada
    ],
}


# ============================================================================
# FIX: PHENOTYPE_COLORS — PALETA CANÓNICA
# ============================================================================
# Esquema coherente para TODAS las figuras del paper.
# Desert=rojo (caliente/hostil), Excluded=azul (frío/barrera),
# Inflamed=verde (activo), Normal_Stroma=gris, Ambiguous=púrpura.
#
# TODOS los módulos de visualización (visualization.py, validation.py)
# DEBEN importar de aquí. NO definir colores locales.
# ============================================================================

PHENOTYPE_COLORS = {
    'Immune_Desert':   '#d62728',   # Rojo — silenciamiento activo
    'Immune_Excluded': '#2C7BB6',   # Azul — barrera física
    'Inflamed':        '#2ca02c',   # Verde — respuesta inmune activa
    'Normal_Stroma':   '#7f7f7f',   # Gris — tejido normal
    'Ambiguous_Cold':  '#9467bd',   # Púrpura — no clasificable
    'Unclassified':    '#bcbd22',   # Amarillo/oliva — sin clasificar
}

DATASET_COLORS = {
    'GSE210616': '#1f77b4',  # Azul — Discovery
    'GSE213688': '#ff7f0e',  # Naranja — Validation
}


# ============================================================================
# PARAMETROS DE PRESENCIA CELULAR 
# ============================================================================

@dataclass
class CellPresenceParams:
    """
    Parametros para evaluar presencia/ausencia celular con Cell2Location
    
    IMPORTANTE - INTERPRETACION CORRECTA DE CUANTILES:
    - q05 = "al menos esta cantidad esta presente con alta confianza"
    - q05 = 0 NO PRUEBA AUSENCIA, solo "insuficiente evidencia para presencia sustancial"
    
    CRITERIOS COMBINADOS:
    Un tipo celular tiene "evidencia limitada de presencia" cuando:
    1. q05_cell_abundance ≈ 0
    2. q05_nUMI_factors < 50 (umbral recomendado)
    3. q95 bajo (incertidumbre acotada)
    
    LENGUAJE APROPIADO:
    - CORRECTO: "evidencia limitada de presencia", "abundancia estimada baja"
    - INCORRECTO: "ausencia", "no hay celulas", "celulas ausentes"
    """
    
    # Umbral de nUMI para evidencia limitada (Cell2Location)
    NUMI_THRESHOLD: int = 50
    
    # Umbral de q05 para abundancia muy baja
    Q05_ABUNDANCE_THRESHOLD: float = 0.1
    
    # Umbral de q95 para incertidumbre acotada
    Q95_UPPER_THRESHOLD: float = 1.0
    
    # Numero minimo de spots para analisis estadistico
    MIN_SPOTS_FOR_STATS: int = 10
    
    # Efecto minimo (Cohen's d) para significancia biologica
    MIN_COHENS_D: float = 0.5

CELL_PRESENCE_PARAMS = CellPresenceParams()


# ============================================================================
# PARAMETROS CELL2LOCATION
# ============================================================================

@dataclass
class Cell2LocationParams:
    """
    Parametros para deconvolucion con Cell2Location
    
    NOTA METODOLOGICA:
    Probar detection_alpha = 20 (alta variabilidad tecnica) y 200 (baja)
    para evaluar robustez segun Li et al. Nat Commun 2023
    """
    
    # Parametros del modelo
    N_CELLS_PER_LOCATION: int = 30
    DETECTION_ALPHA: float = 200.0
    
    # Valores alternativos para sensibilidad
    DETECTION_ALPHA_ALTERNATIVES: List[float] = field(default_factory=lambda: [20.0, 200.0])
    
    # Entrenamiento del modelo de referencia
    REF_MAX_EPOCHS: int = 400
    REF_BATCH_SIZE: int = 2500
    REF_TRAIN_SIZE: float = 1.0
    
    # Entrenamiento del modelo espacial
    SPATIAL_MAX_EPOCHS: int = 30000
    SPATIAL_BATCH_SIZE: Optional[int] = None  # Auto-detect
    
    # Control de exportacion de cuantiles
    EXPORT_QUANTILES: bool = True
    QUANTILES: List[float] = field(default_factory=lambda: [0.05, 0.5, 0.95])
    
    # Hardware
    USE_GPU: bool = True
    
    @staticmethod
    def get_device():
        """Detecta y retorna el dispositivo disponible"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"[OK] GPU detectada: {torch.cuda.get_device_name(0)}")
            print(f"     Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            device = torch.device('cpu')
            print("[WARN] GPU no disponible, usando CPU")
        return device

CELL2LOC_PARAMS = Cell2LocationParams()


# ============================================================================
# PARAMETROS DE CLASIFICACION DE FENOTIPOS
# ============================================================================

@dataclass
class PhenotypeClassificationParams:
    """
    Parametros para la clasificacion mecanistica de fenotipos
    
    LOGICA JERARQUICA:
    1. Filtrar tejido normal (Tumor_Score < umbral)
    2. Identificar Inflamed (CD8_Score > umbral)
    3. Para tumores frios: comparar Silence_Score vs Barrier_Score
       - Si Silence > Barrier -> Immune_Desert
       - Si Barrier > Silence -> Immune_Excluded
       - Si |Silence - Barrier| < umbral -> Ambiguous_Cold
    """
    
    # Umbrales de expresion (percentiles)
    TUMOR_PERCENTILE: int = 60    # Spots con tumor > p60 de Tumor_Score
    CD8_PERCENTILE: int = 75       # Inflamed si CD8 > p75
    
    # Umbral de ambiguedad para tumores frios
    COLD_AMBIGUITY_THRESHOLD: float = 0.1
    
    # Metodo de calculo de scores
    SCORE_METHOD: str = 'mean'
    
    # Normalizacion de scores
    NORMALIZE_SCORES: bool = True  # z-score por muestra

PHENOTYPE_PARAMS = PhenotypeClassificationParams()


# ============================================================================
# PARAMETROS DE ANALISIS DE SENSIBILIDAD (NUEVO)
# ============================================================================

@dataclass
class SensitivityAnalysisParams:
    """
    Parametros para analisis de sensibilidad de umbrales
    
    CRITERIO DE ROBUSTEZ (estandar Q1):
    - >80% de configuraciones deben mantener p < 0.05 = ROBUSTO
    - >90% = ALTA CONFIANZA
    - <80% = Requiere revision de parametros
    """
    
    # Variaciones de percentiles (±20% del valor base)
    TUMOR_PERCENTILE_RANGE: List[int] = field(default_factory=lambda: [48, 60, 72])
    CD8_PERCENTILE_RANGE: List[int] = field(default_factory=lambda: [60, 75, 90])
    
    # Variaciones de umbral de ambiguedad
    AMBIGUITY_THRESHOLD_RANGE: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.15, 0.2])
    
    # Variaciones de umbrales de presencia celular
    NUMI_THRESHOLD_RANGE: List[int] = field(default_factory=lambda: [25, 50, 75, 100])
    Q95_THRESHOLD_RANGE: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.5, 2.0])
    
    # Criterio de robustez
    ROBUSTNESS_THRESHOLD: float = 0.80  # 80% de configuraciones significativas
    HIGH_CONFIDENCE_THRESHOLD: float = 0.90  # 90% para alta confianza

SENSITIVITY_PARAMS = SensitivityAnalysisParams()


# ============================================================================
# PARAMETROS DE CONTROL DE CALIDAD
# ============================================================================

@dataclass
class QualityControlParams:
    """
    Parametros para filtrado de calidad de datos Visium
    
    Basado en estandares Nature Communications (De Zuani et al. 2024)
    """
    
    # Filtros por spot
    MIN_COUNTS_PER_SPOT: int = 800   # Actualizado de 500 a 800 (estandar Q1)
    MAX_COUNTS_PER_SPOT: int = 50000
    MIN_GENES_PER_SPOT: int = 250    # Actualizado de 200 a 250 (estandar Q1)
    MAX_MT_PERCENT: float = 20.0
    
    # Complejidad minima (genes/UMI ratio)
    MIN_COMPLEXITY: float = 0.80
    
    # Filtros por gen
    MIN_SPOTS_PER_GENE: int = 3
    
    # Normalizacion
    TARGET_SUM: float = 1e4

QC_PARAMS = QualityControlParams()


# ============================================================================
# PARAMETROS DE INTEGRACION (HARMONY)
# ============================================================================

@dataclass
class IntegrationParams:
    """Parametros para integracion con Harmony"""
    
    BATCH_KEY: str = 'batch'
    N_PCS: int = 50
    THETA: float = 2.0
    LAMBDA_VALUE: float = 1.0
    MAX_ITER: int = 20

HARMONY_PARAMS = IntegrationParams()


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def save_config_to_json(output_path: Optional[Path] = None) -> Path:
    """Guarda la configuracion actual en JSON para reproducibilidad"""
    if output_path is None:
        output_path = PATHS.RESULTS_DIR / 'pipeline_config.json'
    
    config = {
        'version': '2.5',
        'environment': ENVIRONMENT,
        'signatures': {
            'silencing_repressors': list(SIGNATURES.SILENCING_REPRESSORS),
            'sting_pathway': list(SIGNATURES.STING_PATHWAY),
            'chemokine_signals': list(SIGNATURES.CHEMOKINE_SIGNALS),
            'tam_polarization': list(SIGNATURES.TAM_POLARIZATION),
            'physical_barrier': list(SIGNATURES.PHYSICAL_BARRIER),
            'tumor_markers': list(SIGNATURES.TUMOR_MARKERS),
            'cd8_t_cells': list(SIGNATURES.CD8_T_CELLS),
            'immune_desert_signature': list(SIGNATURES.IMMUNE_DESERT_SIGNATURE),
            'immune_excluded_signature': list(SIGNATURES.IMMUNE_EXCLUDED_SIGNATURE),
        },
        # FIX AUDIT v2.5: Exportar CANONICAL_SIGNATURES para trazabilidad
        'canonical_signatures': {k: list(v) for k, v in CANONICAL_SIGNATURES.items()},
        'cell_presence_params': {
            'numi_threshold': CELL_PRESENCE_PARAMS.NUMI_THRESHOLD,
            'q05_abundance_threshold': CELL_PRESENCE_PARAMS.Q05_ABUNDANCE_THRESHOLD,
            'q95_upper_threshold': CELL_PRESENCE_PARAMS.Q95_UPPER_THRESHOLD,
            'min_cohens_d': CELL_PRESENCE_PARAMS.MIN_COHENS_D,
        },
        'cell2location_params': {
            'n_cells_per_location': CELL2LOC_PARAMS.N_CELLS_PER_LOCATION,
            'detection_alpha': CELL2LOC_PARAMS.DETECTION_ALPHA,
            'ref_max_epochs': CELL2LOC_PARAMS.REF_MAX_EPOCHS,
            'spatial_max_epochs': CELL2LOC_PARAMS.SPATIAL_MAX_EPOCHS,
            'export_quantiles': CELL2LOC_PARAMS.EXPORT_QUANTILES,
        },
        'phenotype_thresholds': {
            'tumor_percentile': PHENOTYPE_PARAMS.TUMOR_PERCENTILE,
            'cd8_percentile': PHENOTYPE_PARAMS.CD8_PERCENTILE,
            'cold_ambiguity': PHENOTYPE_PARAMS.COLD_AMBIGUITY_THRESHOLD,
        },
        'phenotype_colors': PHENOTYPE_COLORS,
        'sensitivity_params': {
            'robustness_threshold': SENSITIVITY_PARAMS.ROBUSTNESS_THRESHOLD,
            'high_confidence_threshold': SENSITIVITY_PARAMS.HIGH_CONFIDENCE_THRESHOLD,
        },
        'qc_params': {
            'min_counts': QC_PARAMS.MIN_COUNTS_PER_SPOT,
            'min_genes': QC_PARAMS.MIN_GENES_PER_SPOT,
            'max_mt_percent': QC_PARAMS.MAX_MT_PERCENT,
            'min_complexity': QC_PARAMS.MIN_COMPLEXITY,
        },
    }
    
    PATHS.create_directories()
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return output_path


def print_config_summary():
    """Imprime resumen de la configuracion"""
    print("=" * 80)
    print("CONFIGURACION DEL PIPELINE TNBC SPATIAL TRANSCRIPTOMICS")
    print("=" * 80)
    print(f"\nEntorno detectado: {ENVIRONMENT}")
    print(f"Directorio base: {PATHS.BASE_DIR}")
    print(f"\nFirmas genicas del mecanismo:")
    print(f"  - Represores (MYC/EZH2/STAT3/DNMT1): {len(SIGNATURES.SILENCING_REPRESSORS)} genes")
    print(f"  - Via STING: {len(SIGNATURES.STING_PATHWAY)} genes")
    print(f"  - Quimioquinas: {len(SIGNATURES.CHEMOKINE_SIGNALS)} genes")
    print(f"  - Polarizacion TAM (NUEVO): {len(SIGNATURES.TAM_POLARIZATION)} genes")
    print(f"  - Barrera fisica: {len(SIGNATURES.PHYSICAL_BARRIER)} genes")
    print(f"\nFirmas canónicas (CANONICAL_SIGNATURES): {len(CANONICAL_SIGNATURES)} categorías")
    print(f"Colores de fenotipos (PHENOTYPE_COLORS): {len(PHENOTYPE_COLORS)} fenotipos")
    print(f"\nParametros de presencia celular (estandar Q1):")
    print(f"  - Umbral nUMI: {CELL_PRESENCE_PARAMS.NUMI_THRESHOLD}")
    print(f"  - Cohen's d minimo: {CELL_PRESENCE_PARAMS.MIN_COHENS_D}")
    print(f"\nParametros Cell2Location:")
    print(f"  - Epocas modelo espacial: {CELL2LOC_PARAMS.SPATIAL_MAX_EPOCHS}")
    print(f"  - Cuantiles a exportar: {CELL2LOC_PARAMS.EXPORT_QUANTILES}")
    print(f"\nParametros de sensibilidad:")
    print(f"  - Umbral de robustez: {SENSITIVITY_PARAMS.ROBUSTNESS_THRESHOLD*100:.0f}%")
    print("=" * 80)


if __name__ == '__main__':
    print_config_summary()
    config_path = save_config_to_json()
    print(f"\n[OK] Configuracion guardada en: {config_path}")

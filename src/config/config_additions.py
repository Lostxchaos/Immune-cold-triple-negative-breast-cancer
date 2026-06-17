"""
================================================================================
ADICIONES A config.py — Parámetros para módulos nuevos
================================================================================
Añadir estos bloques al final de config.py existente.
Son nuevas clases y constantes que los módulos nuevos importan.

NO sobreescribe nada existente (PathConfig, GeneSignatures, Cell2LocationParams
se mantienen intactos).
================================================================================
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict


# ============================================================================
# PATHS PARA MÓDULOS NUEVOS
# ============================================================================

@dataclass
class BulkValidationPaths:
    """Rutas para bulk_validation.py"""
    # Directorio base de datos bulk
    BULK_DATA_DIR: Path = Path("/home/external/frjimenez/fabian/genoma/data/bulk_validation")
    
    # METABRIC (descargar manualmente de cBioPortal)
    METABRIC_DIR: Path = BULK_DATA_DIR / "METABRIC"
    METABRIC_EXPRESSION: Path = METABRIC_DIR / "data_mrna_illumina_microarray.txt"
    METABRIC_CLINICAL: Path = METABRIC_DIR / "data_clinical_patient.txt"
    
    # TCGA-BRCA (descargar manualmente de GDC/cBioPortal)
    TCGA_DIR: Path = BULK_DATA_DIR / "TCGA_BRCA"
    TCGA_EXPRESSION: Path = TCGA_DIR / "data_mrna_seq_v2_rsem.txt"
    TCGA_CLINICAL: Path = TCGA_DIR / "data_clinical_patient.txt"
    
    # Output
    BULK_RESULTS_DIR: Path = Path("/home/external/frjimenez/fabian/genoma/results/bulk_validation")


@dataclass
class GeodesicBenchmarkParams:
    """Parámetros para geodesic_benchmark.py"""
    K_VALUES: List[int] = field(default_factory=lambda: [4, 6, 8, 10])
    ABUNDANCE_PERCENTILE: float = 75.0
    MIN_THRESHOLD: float = 0.1
    SOURCE_CELL_TYPE: str = 'cDC1'
    TARGET_CELL_TYPE: str = 'Tumor'
    # Criterio de insensitividad: CV del ratio < 15%
    MAX_CV_PERCENT: float = 15.0


@dataclass
class RobustnessParams:
    """Parámetros para robustness_stress_tests.py"""
    N_PERMUTATIONS_SPOT: int = 1000
    N_PERMUTATIONS_PATIENT: int = 500
    NOISE_SIGMAS: List[float] = field(default_factory=lambda: [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    DROPOUT_FRACTIONS: List[float] = field(default_factory=lambda: [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    KMEANS_K_VALUES: List[int] = field(default_factory=lambda: [3, 4, 5])
    LEIDEN_RESOLUTIONS: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7, 1.0])
    MIN_ARI_ALTERNATIVE: float = 0.3
    MIN_COHENS_D_ROBUST: float = 0.5
    NOISE_D_THRESHOLD_SIGMA: float = 0.3  # |d|≥0.5 debe mantenerse hasta σ≥0.3


@dataclass
class ExtendedSensitivityParams:
    """Parámetros para sensitivity_analysis v4.0 (rangos extremos)"""
    TUMOR_PERCENTILES: List[int] = field(
        default_factory=lambda: [40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
    )
    IMMUNE_PERCENTILES: List[int] = field(
        default_factory=lambda: [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    )
    AMBIGUITY_MARGINS: List[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    )
    # Total combinaciones: 11 × 10 × 7 = 770
    
    # Parámetros de referencia (default del pipeline)
    # FIX: Acción — 65→60 para coincidir con PHENOTYPE_PARAMS.TUMOR_PERCENTILE
    REF_TUMOR_PCT: int = 60
    REF_IMMUNE_PCT: int = 75
    REF_AMBIGUITY: float = 0.10
    
    # Umbrales de robustez
    KAPPA_SUBSTANTIAL: float = 0.80
    KAPPA_MODERATE: float = 0.60
    D_MEDIUM: float = 0.50


@dataclass
class MechanismValidationV3Params:
    """Parámetros para mechanism_validation v3.0 (adiciones)"""
    # Bloque 1: Desert-only correlations
    MIN_SPOTS_CORRELATION: int = 50
    DESERT_LABEL: str = 'Immune_Desert'
    
    # Bloque 2: MYC Hallmark
    MIN_GENES_HALLMARK: int = 10
    
    # Bloque 3: Macrophage polarization
    MACROPHAGE_PERCENTILE: float = 75.0
    MIN_SPOTS_POLARIZATION: int = 30
    PSEUDOCOUNT_RATIO: float = 0.1
    
    # FDR correction flags
    APPLY_FDR_MACROPHAGE: bool = True
    FDR_METHOD: str = 'fdr_bh'  # Benjamini-Hochberg


@dataclass
class SurvivalAnalysisParams:
    """
    Parámetros para análisis de supervivencia con lifelines.
    H12: Migración de Kaplan-Meier manual → lifelines estandarizado.
    """
    # lifelines KaplanMeierFitter
    KM_CONFIDENCE_LEVEL: float = 0.95
    KM_TIMELINE_MAX_MONTHS: int = 120   # Limitar eje X
    
    # lifelines CoxPHFitter
    COX_PENALIZER: float = 0.01         # Regularización L2 para estabilidad
    COX_STEP_SIZE: float = 0.5          # Paso del solver Newton
    COX_COVARIATES: List[str] = field(
        default_factory=lambda: ['age', 'grade', 'stage']
    )
    
    # Umbrales de significancia
    LOGRANK_ALPHA: float = 0.05
    HR_CLINICAL_THRESHOLD: float = 1.5   # HR > 1.5 = clínicamente relevante


@dataclass  
class ZScoreValidationParams:
    """
    H14: Parámetros para validación z-score fijos vs percentiles.
    Genera Supplementary Figure de concordancia.
    """
    Z_THRESHOLDS: Dict[str, tuple] = field(
        default_factory=lambda: {
            'conservative': (1.0, 1.0),
            'default': (0.5, 0.5),
            'relaxed': (0.0, 0.0),
            'stringent': (1.5, 1.5),
        }
    )
    MIN_KAPPA_CONCORDANCE: float = 0.60  # "moderate agreement"


@dataclass
class PipelineQCParams:
    """
    Parámetros globales de quality control para publicación.
    """
    # Mínimo de genes por firma para considerar score válido
    MIN_GENES_SIGNATURE: int = 3
    
    # FDR global
    GLOBAL_FDR_METHOD: str = 'fdr_bh'   # Benjamini-Hochberg
    GLOBAL_FDR_ALPHA: float = 0.05
    
    # Umbrales de robustez
    MIN_COHENS_D_REPORTABLE: float = 0.30   # Efecto pequeño-medio
    MIN_SPOTS_PER_GROUP: int = 50            # Mínimo para tests estadísticos


# ============================================================================
# INSTANCIAS GLOBALES (para importar desde otros módulos)
# ============================================================================

BULK_PATHS = BulkValidationPaths()
GEODESIC_PARAMS = GeodesicBenchmarkParams()
ROBUSTNESS_PARAMS = RobustnessParams()
EXTENDED_SENSITIVITY = ExtendedSensitivityParams()
MECHANISM_V3_PARAMS = MechanismValidationV3Params()
SURVIVAL_PARAMS = SurvivalAnalysisParams()
ZSCORE_VALIDATION = ZScoreValidationParams()
PIPELINE_QC = PipelineQCParams()


# ============================================================================
# GUÍA DE INTEGRACIÓN
# ============================================================================
"""
INSTRUCCIONES:

1. Abrir config.py en el HPC
2. Copiar TODO este archivo al FINAL de config.py
3. Verificar que no hay conflictos de nombres con clases existentes

Los módulos nuevos importan así:

  from config import BULK_PATHS, GEODESIC_PARAMS, ROBUSTNESS_PARAMS
  from config import EXTENDED_SENSITIVITY, MECHANISM_V3_PARAMS
  from config import SURVIVAL_PARAMS, ZSCORE_VALIDATION, PIPELINE_QC

Si prefieres NO modificar config.py, los módulos nuevos funcionan
con defaults internos (todos los parámetros tienen valores por defecto).
Pero centralizar en config.py es mejor práctica para publicación.
"""

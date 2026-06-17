"""
================================================================================
MODULO ESTADÍSTICO CENTRALIZADO — utils_stats.py
================================================================================
FIX : Unificar Cohen's d en función canónica.
FIX : FDR Benjamini-Hochberg centralizado.

Este módulo es la ÚNICA fuente de verdad para:
  - Cohen's d (pooled, ddof=1)
  - Corrección FDR (Benjamini-Hochberg)
  - Mann-Whitney U con validación de n mínimo
  - Bootstrap CI con seed fijo para reproducibilidad

NINGÚN otro módulo debe definir su propia versión de estas funciones.
================================================================================
"""

import numpy as np
from scipy.stats import mannwhitneyu
from typing import Callable, Optional, Tuple


# ============================================================================
# COHEN'S D — FÓRMULA CANÓNICA (ddof=1, pooled ponderada)
# ============================================================================

def cohens_d_pooled(g1, g2) -> float:
    """
    Calcula Cohen's d con desviación estándar pooled ponderada (ddof=1).

    Esta es la ÚNICA implementación válida para todo el pipeline.
    Fórmula canónica (Lakens, 2013):

        d = (mean1 - mean2) / s_pooled

    donde:
        s_pooled = sqrt(((n1-1)*s1^2 + (n2-1)*s2^2) / (n1+n2-2))

    Parameters
    ----------
    g1, g2 : array-like
        Dos grupos de valores numéricos.

    Returns
    -------
    float
        Cohen's d. Positivo si g1 > g2, negativo si g1 < g2.
        Retorna 0.0 si la desviación pooled es ~0.
    """
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)

    # FIX AUDIT v2.5: Filtrar NaN/Inf
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0

    # ddof=1 (Bessel's correction) — estándar para muestras
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)

    # Pooled SD ponderada por grados de libertad
    s_pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if s_pooled < 1e-10:
        return 0.0

    return float((np.mean(g1) - np.mean(g2)) / s_pooled)


# ============================================================================
# FDR — BENJAMINI-HOCHBERG
# ============================================================================

def apply_fdr(pvalues, method: str = 'fdr_bh', alpha: float = 0.05):
    """
    Aplica corrección de tests múltiples (por defecto Benjamini-Hochberg).

    Wrapper de statsmodels.stats.multitest.multipletests.

    Parameters
    ----------
    pvalues : array-like
        Vector de p-values crudos.
    method : str
        Método de corrección. Default 'fdr_bh' (Benjamini-Hochberg).
        Otros: 'bonferroni', 'holm', 'fdr_by'.
    alpha : float
        Nivel de significancia (default 0.05).

    Returns
    -------
    reject : np.ndarray[bool]
        Máscara de hipótesis rechazadas tras FDR.
    q_values : np.ndarray[float]
        P-values ajustados (q-values).
    """
    from statsmodels.stats.multitest import multipletests

    pvalues = np.asarray(pvalues, dtype=np.float64)

    # Manejar NaN: reemplazar temporalmente con 1.0 (no significativo)
    nan_mask = np.isnan(pvalues)
    pvalues_clean = pvalues.copy()
    pvalues_clean[nan_mask] = 1.0

    # Clamp a [0, 1] por seguridad
    pvalues_clean = np.clip(pvalues_clean, 0.0, 1.0)

    reject, q_values, _, _ = multipletests(pvalues_clean, alpha=alpha, method=method)

    # Restaurar NaN en posiciones originales
    q_values[nan_mask] = np.nan
    reject[nan_mask] = False

    return reject, q_values


# ============================================================================
# MANN-WHITNEY U — CON VALIDACIÓN
# ============================================================================

def safe_mannwhitney(g1, g2, min_n: int = 10, alternative: str = 'two-sided'):
    """
    Mann-Whitney U test con validación de tamaño muestral mínimo.

    Parameters
    ----------
    g1, g2 : array-like
        Dos grupos de valores numéricos.
    min_n : int
        Tamaño mínimo por grupo. Si algún grupo es menor, retorna NaN.
    alternative : str
        Hipótesis alternativa: 'two-sided', 'less', 'greater'.

    Returns
    -------
    stat : float
        Estadístico U. NaN si n insuficiente.
    pval : float
        P-value. NaN si n insuficiente.
    """
    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)

    # Filtrar NaN/Inf
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    if len(g1) < min_n or len(g2) < min_n:
        return np.nan, np.nan

    # Verificar varianza (Mann-Whitney requiere al menos algo de variación)
    if np.std(g1) < 1e-10 and np.std(g2) < 1e-10:
        return np.nan, 1.0

    stat, pval = mannwhitneyu(g1, g2, alternative=alternative)
    return float(stat), float(pval)


# ============================================================================
# BOOTSTRAP CI — CON SEED FIJO
# ============================================================================

def bootstrap_ci(
    g1,
    g2,
    stat_func: Optional[Callable] = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval para una función de dos grupos.

    FIX AUDIT v2.5: Acción ⓻ — Seed fijo para reproducibilidad.

    Parameters
    ----------
    g1, g2 : array-like
        Dos grupos de valores numéricos.
    stat_func : callable, optional
        Función que toma (g1, g2) y retorna un escalar.
        Default: cohens_d_pooled.
    n_boot : int
        Número de iteraciones bootstrap.
    ci : float
        Nivel de confianza (default 0.95 → IC 95%).
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    observed : float
        Valor observado de la estadística.
    ci_lower : float
        Límite inferior del IC.
    ci_upper : float
        Límite superior del IC.
    """
    if stat_func is None:
        stat_func = cohens_d_pooled

    g1 = np.asarray(g1, dtype=np.float64)
    g2 = np.asarray(g2, dtype=np.float64)

    # Filtrar NaN/Inf
    g1 = g1[np.isfinite(g1)]
    g2 = g2[np.isfinite(g2)]

    observed = stat_func(g1, g2)

    rng = np.random.RandomState(seed)
    boot_stats = np.empty(n_boot)

    for i in range(n_boot):
        b1 = rng.choice(g1, size=len(g1), replace=True)
        b2 = rng.choice(g2, size=len(g2), replace=True)
        boot_stats[i] = stat_func(b1, b2)

    alpha = 1 - ci
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return observed, ci_lower, ci_upper


# ============================================================================
# SELF-TEST (ejecutar con: python utils_stats.py)
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("SELF-TEST: utils_stats.py")
    print("=" * 60)

    # Test 1: Cohen's d conocido
    np.random.seed(42)
    a = np.random.normal(0, 1, 100)
    b = np.random.normal(0.8, 1, 100)
    d = cohens_d_pooled(a, b)
    print(f"\n[1] Cohen's d (esperado ~-0.8): {d:.3f}", end="  ")
    print("✅" if -1.2 < d < -0.4 else "❌")

    # Test 2: Cohen's d con NaN/Inf
    c = np.array([1, 2, 3, np.nan, np.inf, 4, 5])
    e = np.array([5, 6, 7, 8, -np.inf, 9, 10])
    d2 = cohens_d_pooled(c, e)
    print(f"[2] Cohen's d con NaN/Inf: {d2:.3f}", end="  ")
    print("✅" if np.isfinite(d2) and d2 < 0 else "❌")

    # Test 3: FDR
    pvals = np.array([0.001, 0.01, 0.04, 0.08, 0.5, 0.9])
    reject, qvals = apply_fdr(pvals)
    print(f"[3] FDR: {sum(reject)} rechazados de {len(pvals)}", end="  ")
    print("✅" if sum(reject) >= 1 and all(q >= p for p, q in zip(pvals, qvals)) else "❌")

    # Test 4: safe_mannwhitney
    stat, pval = safe_mannwhitney(a, b)
    print(f"[4] Mann-Whitney: stat={stat:.1f}, p={pval:.4f}", end="  ")
    print("✅" if pval < 0.05 else "❌")

    stat_small, pval_small = safe_mannwhitney([1, 2, 3], [4, 5, 6], min_n=10)
    print(f"[5] MW n<min: stat={stat_small}, p={pval_small}", end="  ")
    print("✅" if np.isnan(pval_small) else "❌")

    # Test 5: Bootstrap CI
    obs, lo, hi = bootstrap_ci(a, b, n_boot=500, seed=42)
    print(f"[6] Bootstrap: d={obs:.3f}, CI=[{lo:.3f}, {hi:.3f}]", end="  ")
    print("✅" if lo < obs < hi or lo < hi else "❌")

    print("\n[OK] Self-test completado.")

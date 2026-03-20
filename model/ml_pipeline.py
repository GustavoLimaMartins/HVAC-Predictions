"""
MLPipeline  Pipeline Sequencial de Treinamento e Inferência
=============================================================

Fluxo de 5 estágios + 2 pré-etapas:

    PRÉ-ETAPA 1  _filter_outliers()   — remove ruído e extremos de consumo_kwh
    PRÉ-ETAPA 2  SegmentedMLPipeline  — um modelo por tipo de máquina HVAC

    1. CONFIG      MLPipelineConfig  — hiperparâmetros, split, modelos
    2. PREPROCESS  _preprocess()     — ModelSchema + to_numpy
    3. SEARCH      _run_search()     — ParameterSampler + KFold CV
    4. EVALUATE    _compute_metrics()— MAE / RMSE / R² / WMAPE
    5. PERSIST     save()            — joblib + JSON

Inferência:
    ``predict()`` delega toda a conversão de inputs ao módulo
    ``tools/normalizer.py`` (classe ``MLNormalizer``), garantindo que a
    normalização em inferência seja **idêntica** ao fluxo de treino sem
    duplicar lógica.

Uso mínimo:

    >>> cfg  = MLPipelineConfig(n_iter=30, cv=5, test_size=0.3)
    >>> best = MLPipeline.compare(df_raw, config=cfg)
    >>> best.save("model/artifacts/best.joblib")

Uso com candidatos personalizados:

    >>> cfg = MLPipelineConfig(
    ...     candidates=[
    ...         LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    ...         (XGBRegressor(random_state=42, n_jobs=-1), {"n_estimators": [500, 1000]}),
    ...     ],
    ...     n_iter=20, cv=3, test_size=0.2,
    ... )
    >>> best = MLPipeline.compare(df_raw, config=cfg)
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import warnings
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.pre_process.schema import ModelSchema
from tools.normalizer import MLNormalizer


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════

_logger = logging.getLogger(__name__)
_SEP    = "─" * 58

_TARGET: str = "consumo_kwh"

_SCHEMA_FIELDS: list[str] = [
    "hora", "data", "consumo_kwh", "machine_type",
    "estacao", "grupo_regional",
    "Temperatura_C", "Temperatura_Percebida_C",
    "Umidade_Relativa_%", "Precipitacao_mm",
    "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
    "Irradiancia_Direta_Wm2", "Irradiancia_Difusa_Wm2",
    "consumo_lag_1h", "consumo_lag_24h", "consumo_rolling_mean_3h"
]

# Colunas declaradas como categórico nativo no LightGBM (via categorical_feature).
# No XGBoost são tratadas como inteiro numérico.
# (hora e mes agora usam Target Encoding — não precisam de tratamento categórico.)
_CAT_COLS: list[str] = ["grupo_regional"]

# Suprime o warning de feature names quando numpy é passado ao LightGBM
_FN_WARNING = "X does not have valid feature names"


# ══════════════════════════════════════════════════════════════════════════════
#  GRADES PADRÃO DE HIPERPARÂMETROS
# ══════════════════════════════════════════════════════════════════════════════

_PARAM_GRIDS: dict[type, dict[str, list]] = {
    LGBMRegressor: {
        "n_estimators":      [300, 500, 700, 1000],
        "learning_rate":     [0.01, 0.05, 0.1],
        "num_leaves":        [31, 63, 127],
        "max_depth":         [-1, 10, 20, 30],
        "min_child_samples": [20, 50, 100],
        "subsample":         [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.5, 0.7, 0.9, 1.0],
        "reg_alpha":         [0.0, 0.1, 0.5],
        "reg_lambda":        [0.0, 0.1, 0.5, 1.0],
    },
    XGBRegressor: {
        "n_estimators":     [300, 500, 700, 1000],
        "learning_rate":    [0.01, 0.05, 0.1],
        "max_depth":        [3, 5, 7, 10],
        "min_child_weight": [1, 3, 5, 10],
        "subsample":        [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.5, 0.7, 0.9, 1.0],
        "reg_alpha":        [0.0, 0.1, 0.5],
        "reg_lambda":       [0.5, 1.0, 2.0],
        "gamma":            [0.0, 0.1, 0.5],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO — estágio 1
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLPipelineConfig:
    """
    Parâmetros centralizados do pipeline.

    Todos os hiperparâmetros de busca, split e candidatos vivem aqui.
    Passe uma instância para MLPipeline.compare() ou SegmentedMLPipeline
    para controle total.

    Attributes:
        candidates   : Estimadores a comparar. Aceita estimador puro ou
                       tupla (estimador, param_grid). Se None, usa LGBM + XGB.
        n_iter       : Combinações aleatórias por modelo.
        cv           : Folds de validação cruzada.
        test_size    : Fração do dataset reservada para teste (0-1).
        random_state : Semente global de reprodutibilidade.
        artifacts_dir: Diretório de saída para .joblib e .json.
        noise_floor  : Pré-etapa 1 — limiar mínimo de consumo_kwh (kWh).
                       Registros abaixo desse valor são classificados como
                       ruído (standby / falha de sensor) e removidos.
        iqr_factor   : Pré-etapa 1 — fator k da Tukey fence modificada.
                       Remove consumo_kwh fora de [Q1 − k·IQR, Q3 + k·IQR].
                       Use 3.0 (extremos) em vez de 1.5 (suave) para preservar
                       variações sazonais legítimas de demanda energética.
        min_segment_size: Pré-etapa 1 — tamanho mínimo de amostra por segmento
                   para aplicar limiares dinâmicos. Abaixo disso, usa
                   fallback global para reduzir instabilidade estatística.
        noise_quantile: Pré-etapa 1 — quantil baixo usado para definir o
                   limiar de ruído dinâmico por segmento.
        upper_quantile_cap: Pré-etapa 1 — teto opcional por quantil para
               aumentar sensibilidade a extremos superiores (cauda alta).
         segment_params: Pré-etapa 1 — sobrescritas por segmento (chave =
             tipo_maquina normalizado). Permite definir por tipo:
             noise_floor, iqr_factor, min_segment_size,
             noise_quantile, upper_quantile_cap.
    """
    candidates:    list[Any]  | None = None
    n_iter:        int               = 20
    cv:            int               = 5
    test_size:     float             = 0.2
    random_state:  int               = 42
    artifacts_dir: Path              = field(default_factory=lambda: Path("model/artifacts"))
    # pré-etapa 1 — filtro de outliers de consumo_kwh
    noise_floor:   float             = 0.5   # kWh mínimo em operação real (abaixo = ruído)
    iqr_factor:    float             = 3.0   # k da Tukey fence (3.0 = apenas extremos)
    min_segment_size: int            = 40    # mínimo por tipo para limiar dinâmico estável
    noise_quantile: float            = 0.05  # quantil baixo para ruído adaptativo por segmento
    upper_quantile_cap: float | None = None  # cap superior opcional (None = desabilitado)
    segment_params: dict[str, dict[str, float | int | None]] | None = None

    def resolve_candidates(self) -> list[tuple[BaseEstimator, dict | None]]:
        """Normaliza candidates para lista de (estimador, param_grid | None)."""
        raw = self.candidates or [
            LGBMRegressor(random_state=self.random_state, n_jobs=-1, verbose=-1),
            XGBRegressor(random_state=self.random_state, n_jobs=-1),
        ]
        return [c if isinstance(c, tuple) else (c, None) for c in raw]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE LOG
# ══════════════════════════════════════════════════════════════════════════════

def _log_block(title: str) -> None:
    _logger.info(_SEP)
    _logger.info("  %s", title)
    _logger.info(_SEP)


def _log_metrics(metrics: dict[str, float], header: str = "") -> None:
    if header:
        _log_block(header)
    _logger.info("  %-12s %.4f",  "MAE:",      metrics["MAE"])
    _logger.info("  %-12s %.4f",  "RMSE:",     metrics["RMSE"])
    _logger.info("  %-12s %.4f",  "R2:",       metrics["R2"])
    _logger.info("  %-12s %.2f%%", "MAPE:",     metrics["MAPE"])
    _logger.info("  %-12s %.2f%%", "WMAPE:",    metrics["WMAPE"])
    _logger.info("  %-12s %.2f%%", "Acuracia:", metrics["Acuracia"])


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE MÉTRICA — estágio 4
# ══════════════════════════════════════════════════════════════════════════════

def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE = sum(|y-yhat|) / sum(|y|) * 100  (imune a zeros no denominador)."""
    denom = np.abs(y_true).sum()
    return float("nan") if denom == 0 else float(np.abs(y_true - y_pred).sum() / denom * 100)


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE = mean(|y-yhat| / max(|y|, eps)) * 100."""
    eps = 1e-8
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Retorna MAE, RMSE, R2, MAPE, WMAPE e Acuracia."""
    mape = _mape(y_true, y_pred)
    wmape = _wmape(y_true, y_pred)
    return {
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2":       float(r2_score(y_true, y_pred)),
        "MAPE":     mape,
        "WMAPE":    wmape,
        "Acuracia": 100.0 - mape,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PRÉ-ETAPA 1 — FILTRO DE OUTLIERS DE CONSUMO
# ══════════════════════════════════════════════════════════════════════════════

def _filter_outliers(
    df: pl.DataFrame,
    noise_floor: float,
    iqr_factor: float,
    min_segment_size: int,
    noise_quantile: float,
    upper_quantile_cap: float | None = None,
    segment_params: dict[str, dict[str, float | int | None]] | None = None,
    log_details: bool = True,
) -> tuple[pl.DataFrame, dict[str, dict[str, float | int | bool]]]:
    """
    Remove instâncias de ruído (consumo ≈ 0) e distorções (outliers extremos)
    de 'consumo_kwh' antes do treinamento, com parametrização dinâmica por
    machine_type/tipo_maquina.

    Filtro de ruído — dinâmico por segmento
    ────────────────────────────────────────
    Para cada segmento de máquina, o limiar inferior de ruído é estimado por:

        floor_seg = clip(Q_noise(seg), floor_global*0.25, floor_global*4.0)

    onde Q_noise(seg) é o quantil baixo (noise_quantile) do consumo_kwh no
    segmento. Isso permite adaptar o piso mínimo para tecnologias com cargas
    naturalmente menores/maiores.

    Filtro de distorções — Tukey fence modificada
    ─────────────────────────────────────────────
    Picos anômalos de demanda e erros de medição de alta magnitude são
    identificados pela cerca de Tukey com fator k:

        Q₁  = percentil 25 de consumo_kwh
        Q₃  = percentil 75 de consumo_kwh
        IQR = Q₃ − Q₁

        Limite inferior : max(Q₁ − k · IQR,  noise_floor)
        Limite superior : Q₃ + k · IQR

    O valor padrão k = 3.0 (fence "extrema") preserva variações sazonais e
    picos de demanda legítimos de ar-condicionado, removendo apenas anomalias
    estatísticas severas. O valor k = 1.5 (padrão Tukey) seria excessivamente
    restritivo para séries de consumo energético, cuja distribuição é assimétrica
    à direita (skew positivo) por natureza.

    Args:
        df               : DataFrame com coluna 'consumo_kwh'.
        noise_floor      : Piso global base de consumo (kWh).
        iqr_factor       : Fator k da Tukey fence.
        min_segment_size : Tamanho mínimo por segmento para dinâmica estável.
        noise_quantile   : Quantil baixo base para piso dinâmico.
        upper_quantile_cap: Quantil superior base opcional para cauda alta.
        segment_params   : Sobrescritas por segmento.
        log_details      : Se True, emite logs por segmento e consolidado.

    Returns:
        (df_filtrado, thresholds_by_segment).
    """
    n_before = len(df)
    if n_before == 0:
        return df, {}

    # Diagnóstico: machine_type ausente/vazio gera segmento em branco nos logs.
    if "machine_type" in df.columns:
        n_missing_mt = int(
            df.select(
                (
                    pl.col("machine_type").is_null()
                    | (pl.col("machine_type").cast(pl.String).str.strip_chars() == "")
                ).sum()
            ).item()
        )
        if n_missing_mt > 0:
            _logger.warning(
                "machine_type com valor nulo/vazio em %d registro(s); usando rótulo 'DESCONHECIDO'.",
                n_missing_mt,
            )

    # Resolve o segmentador de forma robusta (tipo_maquina > machine_type > global).
    if "tipo_maquina" in df.columns:
        seg_series = df["tipo_maquina"].cast(pl.String).alias("_outlier_segment")
    elif "machine_type" in df.columns:
        seg_series = (
            ModelSchema(df.select("machine_type"), ["machine_type"])
            .adjust_machine_type()
            .df["tipo_maquina"]
            .cast(pl.String)
            .alias("_outlier_segment")
        )
    else:
        seg_series = pl.Series(["__GLOBAL__"] * n_before, dtype=pl.String).alias("_outlier_segment")

    df_work = df.with_row_index("__row_idx__").with_columns(seg_series)
    df_work = df_work.with_columns(
        pl.when(
            pl.col("_outlier_segment").is_null()
            | (pl.col("_outlier_segment").cast(pl.String).str.strip_chars() == "")
        )
        .then(pl.lit("DESCONHECIDO"))
        .otherwise(pl.col("_outlier_segment").cast(pl.String))
        .alias("_outlier_segment")
    )
    segments = sorted(df_work["_outlier_segment"].unique().to_list())

    floor_base = max(float(noise_floor), 0.0)
    q_noise = float(min(max(noise_quantile, 0.0), 0.25))
    q_upper = None if upper_quantile_cap is None else float(min(max(upper_quantile_cap, 0.90), 0.9999))

    def _seg_overrides(seg_name: str) -> dict[str, float | int | None]:
        if not segment_params:
            return {}
        keys = [
            seg_name,
            seg_name.upper(),
            seg_name.replace("-", " "),
            seg_name.replace("-", " ").upper(),
            seg_name.replace(" ", "_"),
            seg_name.replace(" ", "_").upper(),
        ]
        for k in keys:
            if k in segment_params:
                return segment_params[k]
        return {}

    filtered_chunks: list[pl.DataFrame] = []
    thresholds_by_segment: dict[str, dict[str, float | int | bool]] = {}

    n_noise_total = 0
    n_ext_total = 0

    for seg in segments:
        part = df_work.filter(pl.col("_outlier_segment") == seg)
        n_seg_before = len(part)
        s = part[_TARGET]

        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = max(q3 - q1, 0.0)

        # Estatísticas auxiliares para parametrização dinâmica por segmento
        q10 = float(s.quantile(0.10))
        q50 = float(s.quantile(0.50))
        q90 = float(s.quantile(0.90))
        q99 = float(s.quantile(0.99))

        seg_iqr_factor = float(iqr_factor)
        seg_noise_q = float(q_noise)
        seg_upper_q = q_upper
        seg_min_size = int(min_segment_size)
        seg_floor_base = float(floor_base)
        tail_ratio = float("nan")

        # Dinâmica orientada por tipo de máquina + formato da distribuição
        if n_seg_before >= max(min_segment_size, 10):
            spread_mid = max(q90 - q50, 1e-9)
            spread_low = max(q50 - q10, 1e-9)
            spread_total = max(q90 - q10, 1e-9)
            tail_ratio = max(q99 - q90, 0.0) / spread_mid

            seg_name = str(seg).upper()
            # Segmentos tipicamente mais estáveis em carga -> mais sensível
            if "HI-WALL" in seg_name or "JANELA" in seg_name or "(ACJ)" in seg_name:
                seg_iqr_factor *= 0.90
                seg_noise_q = min(seg_noise_q + 0.02, 0.20)

            # Segmentos com maior variabilidade operacional -> ruído menos agressivo
            if "INVERTER" in seg_name or "ROOFTOP" in seg_name:
                seg_noise_q = max(seg_noise_q - 0.01, 0.01)

            # Cauda alta pronunciada => estreita cerca superior
            if tail_ratio >= 1.50:
                seg_iqr_factor = min(seg_iqr_factor, 1.25)
                seg_upper_q = min(seg_upper_q if seg_upper_q is not None else 0.99, 0.985)
            elif tail_ratio >= 1.00:
                seg_iqr_factor = min(seg_iqr_factor, 1.40)
                seg_upper_q = min(seg_upper_q if seg_upper_q is not None else 0.995, 0.99)
            elif tail_ratio >= 0.60:
                seg_iqr_factor = min(seg_iqr_factor, 1.60)
                if seg_upper_q is None:
                    seg_upper_q = 0.995

            # Distribuição compacta => pode elevar piso de ruído de forma segura
            if spread_total <= max(q50, 1e-6) * 0.35 and spread_mid / spread_low < 1.8:
                seg_noise_q = min(seg_noise_q + 0.02, 0.20)

        seg_noise_q = float(min(max(seg_noise_q, 0.0), 0.25))

        # Overrides explícitos por segmento (prioridade máxima)
        ov = _seg_overrides(str(seg))
        if ov:
            if ov.get("noise_floor") is not None:
                seg_floor_base = max(float(ov["noise_floor"]), 0.0)
            if ov.get("iqr_factor") is not None:
                seg_iqr_factor = float(max(float(ov["iqr_factor"]), 0.1))
            if ov.get("min_segment_size") is not None:
                seg_min_size = max(int(ov["min_segment_size"]), 1)
            if ov.get("noise_quantile") is not None:
                seg_noise_q = float(min(max(float(ov["noise_quantile"]), 0.0), 0.25))
            if "upper_quantile_cap" in ov:
                qv = ov.get("upper_quantile_cap")
                seg_upper_q = None if qv is None else float(min(max(float(qv), 0.90), 0.9999))

        use_fallback = n_seg_before < seg_min_size
        if use_fallback:
            seg_floor = seg_floor_base
        else:
            seg_q_low = float(s.quantile(seg_noise_q))
            lo_cap = seg_floor_base * 0.25
            hi_cap = seg_floor_base * 4.0 if seg_floor_base > 0 else max(seg_q_low, 0.0)
            seg_floor = min(max(seg_q_low, lo_cap), hi_cap)
            # noise_floor (global ou override) deve ser piso mínimo real
            seg_floor = max(seg_floor, seg_floor_base)

        lo = max(float(q1 - seg_iqr_factor * iqr), seg_floor)
        hi_iqr = float(q3 + seg_iqr_factor * iqr)
        hi = hi_iqr
        hi_qcap = None
        if seg_upper_q is not None and n_seg_before >= max(seg_min_size, 10):
            hi_qcap = float(s.quantile(seg_upper_q))
            hi = min(hi, hi_qcap)
        if hi < lo:
            hi = lo

        part_after_floor = part.filter(pl.col(_TARGET) >= seg_floor)
        part_after_iqr = part_after_floor.filter(pl.col(_TARGET).is_between(lo, hi))

        kept_min = float(part_after_iqr[_TARGET].min()) if len(part_after_iqr) else float("nan")
        kept_max = float(part_after_iqr[_TARGET].max()) if len(part_after_iqr) else float("nan")

        n_noise = n_seg_before - len(part_after_floor)
        n_ext = len(part_after_floor) - len(part_after_iqr)
        n_noise_total += n_noise
        n_ext_total += n_ext

        thresholds_by_segment[str(seg)] = {
            "n_before": n_seg_before,
            "noise_floor": round(float(seg_floor), 6),
            "q1": round(float(q1), 6),
            "q3": round(float(q3), 6),
            "iqr": round(float(iqr), 6),
            "lo": round(float(lo), 6),
            "hi": round(float(hi), 6),
            "hi_iqr": round(float(hi_iqr), 6),
            "hi_qcap": round(float(hi_qcap), 6) if hi_qcap is not None else None,
            "kept_min": round(float(kept_min), 6) if kept_min == kept_min else None,
            "kept_max": round(float(kept_max), 6) if kept_max == kept_max else None,
            "iqr_factor_used": round(float(seg_iqr_factor), 4),
            "noise_quantile_used": round(float(seg_noise_q), 4),
            "upper_quantile_cap_used": round(float(seg_upper_q), 4) if seg_upper_q is not None else None,
            "min_segment_size_used": int(seg_min_size),
            "tail_ratio": round(float(tail_ratio), 4) if tail_ratio == tail_ratio else None,
            "fallback_global_floor": bool(use_fallback),
            "manual_override": bool(ov),
        }

        if log_details:
            _logger.info(
                "  Outliers[%s]: removidos=%d (ruido=%d extremos=%d) | faixa=[%.3f, %.3f] kWh"
                " | pos=[%.3f, %.3f] kWh | k=%.2f q_noise=%.3f%s%s",
                seg,
                n_seg_before - len(part_after_iqr),
                n_noise,
                n_ext,
                lo,
                hi,
                kept_min,
                kept_max,
                seg_iqr_factor,
                seg_noise_q,
                " [fallback]" if use_fallback else "",
                f" [qcap={seg_upper_q:.4f}]" if hi_qcap is not None and seg_upper_q is not None else "",
            )

        filtered_chunks.append(part_after_iqr)

    if filtered_chunks:
        df_filtered = (
            pl.concat(filtered_chunks)
            .sort("__row_idx__")
            .drop(["__row_idx__", "_outlier_segment"])
        )
    else:
        df_filtered = df_work.head(0).drop(["__row_idx__", "_outlier_segment"])

    n_total = n_before - len(df_filtered)
    if log_details:
        _logger.info(
            "  Outliers (geral): %d removidos (%.2f%%) — ruido=%d  extremos=%d",
            n_total,
            n_total / n_before * 100 if n_before else 0,
            n_noise_total,
            n_ext_total,
        )
    return df_filtered, thresholds_by_segment


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER DE PERSISTÊNCIA — estágio 5
# ══════════════════════════════════════════════════════════════════════════════

def _save_report_json(path: Path, data: dict) -> None:
    """Serializa qualquer dicionário de relatório em JSON indentado."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, default=str)
    _logger.info("Relatorio salvo em: %s", path)


def _save_params_json(
    artifacts_dir: Path,
    model_name: str,
    best_params: dict,
    mape_cv: float,
    mape_test: float,
) -> None:
    """Salva os melhores hiperparametros em <artifacts_dir>/<model>_best_params.json."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{model_name}_best_params.json"
    record = {
        "model":         model_name,
        "best_mape_cv":   round(mape_cv,   6),
        "best_mape_test": round(mape_test, 6),
        # Compatibilidade retroativa (nomes legados)
        "best_mae_cv":   round(mape_cv,   6),
        "best_mae_test": round(mape_test, 6),
        "best_params":   best_params,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, default=str)
    _logger.info("Hiperparametros salvos em: %s", path)


def _plot_local_cleaning_impact(df: pl.DataFrame, cfg: MLPipelineConfig) -> None:
    """Visualiza o impacto local (por tipo de máquina) da limpeza de outliers."""
    if "machine_type" not in df.columns:
        _logger.warning("Visualização local ignorada: coluna 'machine_type' ausente.")
        return

    _norm_before = (
        ModelSchema(df.select("machine_type"), ["machine_type"])
        .adjust_machine_type()
        .df["tipo_maquina"]
        .alias("_norm_type")
    )
    df_before = df.with_columns(_norm_before)

    df_after, _ = _filter_outliers(
        df,
        noise_floor=cfg.noise_floor,
        iqr_factor=cfg.iqr_factor,
        min_segment_size=cfg.min_segment_size,
        noise_quantile=cfg.noise_quantile,
        upper_quantile_cap=cfg.upper_quantile_cap,
        segment_params=cfg.segment_params,
        log_details=False,
    )
    _norm_after = (
        ModelSchema(df_after.select("machine_type"), ["machine_type"])
        .adjust_machine_type()
        .df["tipo_maquina"]
        .alias("_norm_type")
    )
    df_after = df_after.with_columns(_norm_after)

    before_counts = (
        df_before.group_by("_norm_type").len().rename({"len": "n_before"})
    )
    after_counts = (
        df_after.group_by("_norm_type").len().rename({"len": "n_after"})
    )
    summary = (
        before_counts
        .join(after_counts, on="_norm_type", how="left")
        .with_columns(pl.col("n_after").fill_null(0).cast(pl.Int64))
        .with_columns([
            (pl.col("n_before") - pl.col("n_after")).alias("removed"),
            (pl.when(pl.col("n_before") > 0)
             .then((pl.col("n_before") - pl.col("n_after")) / pl.col("n_before") * 100)
             .otherwise(0.0)
             .alias("removed_pct")),
        ])
        .sort("removed_pct", descending=True)
    )

    segs = summary["_norm_type"].to_list()
    n_before = summary["n_before"].to_list()
    n_after = summary["n_after"].to_list()
    removed_pct = summary["removed_pct"].to_list()

    x = np.arange(len(segs))
    width = 0.38
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    axes[0].bar(x - width / 2, n_before, width=width, label="Antes", color="#4C78A8")
    axes[0].bar(x + width / 2, n_after, width=width, label="Depois", color="#59A14F")
    axes[0].set_title("Impacto local da limpeza por segmento — contagem (consumo_kwh)")
    axes[0].set_ylabel("Quantidade de registros")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(segs, rotation=35, ha="right")
    axes[0].legend()

    axes[1].bar(x, removed_pct, color="#E15759")
    axes[1].set_title("Percentual removido por segmento")
    axes[1].set_ylabel("Removido (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(segs, rotation=35, ha="right")

    total_before = sum(n_before)
    total_after = sum(n_after)
    total_removed = total_before - total_after
    total_pct = (total_removed / total_before * 100) if total_before else 0.0
    fig.suptitle(
        f"Limpeza de outliers local | removidos={total_removed}/{total_before} ({total_pct:.2f}%)",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class MLPipeline:
    """
    Pipeline sequencial de ML para predicao de consumo HVAC (kWh).

    Estagios internos
    -----------------
    1. _preprocess  -> _build_schema (TE hora/mes) + to_numpy
    2. _run_search  -> ParameterSampler + KFold CV + refit
    3. predict      -> _build_schema (TE inference) + alinha features

    Ponto de entrada principal: ``MLPipeline.compare(df, config=cfg)``.

    Attributes:
        model            : Estimador sklearn treinado.
        feature_columns_ : Lista de features na ordem do treinamento.
        metrics_         : MAE / RMSE / R2 / WMAPE / Acuracia no teste.
    """

    def __init__(
        self,
        model: BaseEstimator | None = None,
        config: MLPipelineConfig | None = None,
    ) -> None:
        """
        Args:
            model : Estimador sklearn instanciado. Se None, usa LGBMRegressor.
            config: Configuracao do pipeline. Se None, usa MLPipelineConfig().
        """
        self.config = config or MLPipelineConfig()

        self.model: BaseEstimator = model or LGBMRegressor(
            n_estimators=300,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        self.feature_columns_: list[str] = []
        self.metrics_:         dict[str, float] = {}
        self.report_:          dict              = {}
        self.compare_results_: list              = []  # [(MLPipeline, name)] ordenado por WMAPE
        self._te_map:          dict[str, dict] | None = None
        self._clipping_limits: dict | None            = None  # ✅ Novo: persistir limites
        self._is_fitted:       bool              = False

    # -- Target Encoding + Schema ML ------------------------------------------

    def _build_schema(
        self,
        df: pl.DataFrame,
        *,
        inference: bool = False,
    ) -> pl.DataFrame:
        """
        Constrói o schema ML com Target Encoding para ``hora`` e ``mes``.

        Em modo **treino** (``inference=False``), calcula as médias
        suavizadas de ``consumo_kwh`` por valor de hora/mês e armazena
        o mapa em ``self._te_map`` para reutilização em inferência.

        Em modo **inferência** (``inference=True``), aplica o mapa
        pré-computado; valores não vistos recebem a média global como
        fallback.

        Args:
            df        : DataFrame com as colunas de ``_SCHEMA_FIELDS``.
            inference : Se True, usa ``self._te_map`` previamente calculado.

        Returns:
            pl.DataFrame pronto para conversão numpy.
        """
        te_map = self._te_map if inference else None
        schema = (
            ModelSchema(df, _SCHEMA_FIELDS)
            .add_date_features()
            .adjust_machine_type()
            .make_target_encoding_columns(
                ["hora", "mes"], encoding_map=te_map,
            )
            .make_categorical_columns(["grupo_regional"])
            .make_one_hot_encode_columns(["tipo_maquina", "estacao", "periodo_dia"])
        )
        
        # IMPORTANTE: make_clipping_min_max_columns é aplicado APENAS em dados de treino
        if not inference:
            schema.make_clipping_min_max_columns([
                "Temperatura_C", "Temperatura_Percebida_C",
                "Umidade_Relativa_%", "Precipitacao_mm",
                "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
                "Irradiancia_Direta_Wm2", "Irradiancia_Difusa_Wm2",
                "consumo_lag_1h", "consumo_lag_24h", "consumo_rolling_mean_3h"
            ])
            # ✅ Persistir limites de clipping para inferência
            self._clipping_limits = schema.clipping_limits_
        if not inference:
            self._te_map = schema.target_encoding_map_
        return schema.df

    # -- estagio 2 -- pre-processamento ---------------------------------------

    def _preprocess(
        self,
        df: pl.DataFrame,
        log_label: str = "PRE-PROCESSAMENTO",
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Pré-etapa 1 + ModelSchema: filtra outliers de consumo_kwh e
        aplica o pipeline de features para treino.

        Etapas:
            1. _filter_outliers()  — remove ruído e distorções de consumo_kwh
            2. _build_schema()     — features de data, TE(hora/mes), OHE, Clipping+MinMax
            3. _to_numpy()         — converte para float32 numpy

        Returns:
            (X, y, feature_columns) — prontos para train_test_split.
        """
        _log_block(log_label)

        # pré-etapa 1 — remove ruído e distorções de consumo_kwh
        n_raw = len(df)
        df, thresholds_by_segment = _filter_outliers(
            df,
            noise_floor=self.config.noise_floor,
            iqr_factor=self.config.iqr_factor,
            min_segment_size=self.config.min_segment_size,
            noise_quantile=self.config.noise_quantile,
            upper_quantile_cap=self.config.upper_quantile_cap,
            segment_params=self.config.segment_params,
        )

        if len(df) == 0:
            raise ValueError(
                "Sem dados após _filter_outliers(). "
                "Revise noise_floor/iqr_factor/segment_params para este recorte."
            )

        # captura estatísticas de consumo_kwh após filtragem (dados limpos)
        _c = df[_TARGET]
        self.report_.update({
            "n_raw":          n_raw,
            "n_after_filter": len(df),
            "n_filtered":     n_raw - len(df),
            "filtered_pct":   round((n_raw - len(df)) / n_raw * 100, 2) if n_raw else 0.0,
            "consumo_kwh_stats": {
                "min":    round(float(_c.min()),          4),
                "max":    round(float(_c.max()),          4),
                "mean":   round(float(_c.mean()),         4),
                "std":    round(float(_c.std()),          4),
                "q25":    round(float(_c.quantile(0.25)), 4),
                "median": round(float(_c.median()),       4),
                "q75":    round(float(_c.quantile(0.75)), 4),
            },
            "outlier_thresholds_by_segment": thresholds_by_segment,
        })

        _logger.info("Aplicando ModelSchema (Target Encoding: hora, mes)...")
        df_out = self._build_schema(df)
        _logger.info(
            "Schema concluido: %d registros x %d colunas",
            df_out.shape[0], df_out.shape[1],
        )

        feature_columns = [c for c in df_out.columns if c != _TARGET]
        X = self._to_numpy(df_out.select(feature_columns))
        y = df_out[_TARGET].to_numpy()
        return X, y, feature_columns

    def _to_numpy(self, df: pl.DataFrame) -> np.ndarray:
        """Converte DataFrame para float32. Colunas Categorical -> UInt32 codes."""
        exprs = [
            pl.col(c).to_physical().cast(pl.Float32)
            if df.schema[c] == pl.Categorical
            else pl.col(c).cast(pl.Float32)
            for c in df.columns
        ]
        return df.select(exprs).to_numpy()

    def _cat_fit_params(self, feature_columns: list[str]) -> dict:
        """Retorna {"categorical_feature": [indices]} para LightGBM; {} para os demais."""
        if not isinstance(self.model, LGBMRegressor):
            return {}
        cat_idx = [i for i, c in enumerate(feature_columns) if c in _CAT_COLS]
        return {"categorical_feature": cat_idx} if cat_idx else {}

    def _align_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Alinha colunas de inferencia ao vetor do treinamento (preenche ausentes com 0)."""
        missing = set(self.feature_columns_) - set(df.columns)
        if missing:
            df = df.with_columns([pl.lit(0, dtype=pl.UInt8).alias(c) for c in missing])
        return df.select(self.feature_columns_)

    # -- estagio 3 -- busca de hiperparametros --------------------------------

    def _run_search(
        self,
        X_train: np.ndarray,
        X_test:  np.ndarray,
        y_train: np.ndarray,
        y_test:  np.ndarray,
        feature_columns: list[str],
        param_grid: dict[str, list] | None = None,
        save_params: bool = True,
    ) -> "MLPipeline":
        """
        Executa ParameterSampler + KFold CV e re-treina o melhor estimador.

        O param_grid e resolvido na ordem: argumento explicito -> _PARAM_GRIDS
        -> ValueError. O melhor conjunto de parametros e selecionado pelo menor
        MAPE medio de CV e persistido em artifacts/.

        Args:
            X_train, X_test : Arrays de features.
            y_train, y_test : Arrays de target.
            feature_columns : Nomes das features (mesma ordem das colunas de X).
            param_grid      : Grade customizada; None usa _PARAM_GRIDS.

        Returns:
            Self atualizado com o melhor estimador e metricas.
        """
        cfg        = self.config
        model_name = type(self.model).__name__

        # -- resolve grade ----------------------------------------------------
        if param_grid is None:
            if type(self.model) not in _PARAM_GRIDS:
                raise ValueError(
                    f"param_grid obrigatorio para '{model_name}'. "
                    f"Grades embutidas: {', '.join(t.__name__ for t in _PARAM_GRIDS)}."
                )
            param_grid = _PARAM_GRIDS[type(self.model)]

        sampler    = list(ParameterSampler(param_grid, n_iter=cfg.n_iter, random_state=cfg.random_state))
        base_model = clone(self.model)
        cat_params = self._cat_fit_params(feature_columns)
        kf         = KFold(n_splits=cfg.cv, shuffle=False)
        width      = len(str(len(sampler)))

        _log_block(f"BUSCA  [{model_name}]")
        _logger.info(
            "%d combinacoes x %d folds = %d fits",
            cfg.n_iter, cfg.cv, cfg.n_iter * cfg.cv,
        )

        # -- loop de busca ----------------------------------------------------
        best_score:  float = -np.inf
        best_params: dict  = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_FN_WARNING, category=UserWarning)

            for i, params in enumerate(sampler, 1):
                estimator   = clone(base_model).set_params(**params)
                fold_scores = [
                    -_mape(
                        y_train[va],
                        clone(estimator)
                            .fit(X_train[tr], y_train[tr], **cat_params)
                            .predict(X_train[va]),
                    )
                    for tr, va in kf.split(X_train)
                ]
                score = float(np.mean(fold_scores))
                is_new_best = score > best_score
                if is_new_best:
                    best_score  = score
                    best_params = params

                _logger.info(
                    "  [%s/%d] MAPE=%.4f%% | best=%.4f%%%s",
                    f"{i:{width}d}", len(sampler),
                    -score, -best_score,
                    "  ← NEW BEST" if is_new_best else "",
                )

        # -- re-treina com os melhores parametros -----------------------------
        best_estimator = clone(base_model).set_params(**best_params)
        best_estimator.fit(X_train, y_train, **cat_params)

        self.model            = best_estimator
        self.feature_columns_ = feature_columns
        self._is_fitted       = True

        # -- avalia no teste --------------------------------------------------
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_FN_WARNING, category=UserWarning)
            y_pred = self.model.predict(X_test)

        test_pct      = len(X_test) / (len(X_train) + len(X_test)) * 100
        self.metrics_ = _compute_metrics(y_test, y_pred)

        self.report_.update({
            "model_name":  model_name,
            "n_train":     len(X_train),
            "n_test":      len(X_test),
            "n_features":  int(X_train.shape[1]),
            "best_params": best_params,
            "mape_cv":     round(-best_score, 6),
            "metrics":     self.metrics_,
        })

        _log_block(f"RESULTADO  [{model_name}]")
        _logger.info("Melhores hiperparametros:")
        for k, v in best_params.items():
            _logger.info("  %-22s %s", k + ":", v)
        _logger.info("Avaliacao no teste (%.0f%%):", test_pct)
        _log_metrics(self.metrics_)

        # -- persiste hiperparametros -----------------------------------------
        if save_params:
            _save_params_json(
                cfg.artifacts_dir, model_name,
                best_params, -best_score, self.metrics_["MAPE"],
            )
        return self

    # -- API publica ----------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> "MLPipeline":
        """
        Treina sem busca de hiperparametros (usa os parametros atuais do modelo).

        Util para re-treinar um modelo ja configurado com os melhores parametros
        carregados de um JSON, ou para um treino rapido de baseline.

        Args:
            df: DataFrame no schema inicial (14 colunas brutas).

        Returns:
            Self.
        """
        X, y, feature_columns = self._preprocess(df)
        self.feature_columns_  = feature_columns

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )

        model_name = type(self.model).__name__
        _log_block(f"TREINAMENTO  [{model_name}]")
        _logger.info("amostras=%d | features=%d", X_train.shape[0], X_train.shape[1])

        self.model.fit(X_train, y_train, **self._cat_fit_params(feature_columns))
        self._is_fitted = True

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_FN_WARNING, category=UserWarning)
            y_pred = self.model.predict(X_test)

        self.metrics_ = _compute_metrics(y_test, y_pred)
        _log_metrics(self.metrics_, header=f"AVALIACAO  (test={self.config.test_size:.0%})")
        return self

    def tune(
        self,
        df: pl.DataFrame,
        param_grid: dict[str, list] | None = None,
    ) -> "MLPipeline":
        """
        Pre-processa e executa a busca de hiperparametros para este pipeline.

        Parametros de busca (n_iter, cv, test_size, random_state) sao lidos
        de ``self.config``. Para comparar multiplos modelos no mesmo split,
        prefira ``MLPipeline.compare()``.

        Args:
            df        : DataFrame no schema inicial (14 colunas brutas).
            param_grid: Grade customizada. None usa _PARAM_GRIDS.

        Returns:
            Self.
        """
        X, y, feature_columns = self._preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        return self._run_search(X_train, X_test, y_train, y_test, feature_columns, param_grid)

    @classmethod
    def compare(
        cls,
        df: pl.DataFrame,
        config: MLPipelineConfig | None = None,
    ) -> "MLPipeline":
        """
        Compara multiplos modelos num split compartilhado e retorna o melhor.

        O pre-processamento e executado uma unica vez. Todos os candidatos
        recebem os mesmos X_train / X_test, garantindo comparacao justa.
        O vencedor e o pipeline com menor MAPE no conjunto de teste.

        Args:
            df    : DataFrame no schema inicial (14 colunas brutas).
            config: Configuracao completa (candidatos, n_iter, cv, ...).
                    Se None, usa MLPipelineConfig() com LGBM + XGB padrao.

        Returns:
            MLPipeline com o melhor modelo treinado.
        """
        cfg        = config or MLPipelineConfig()
        candidates = cfg.resolve_candidates()

        # estagio 2 -- pre-processamento unico --------------------------------
        helper = cls(config=cfg)
        X, y, feature_columns = helper._preprocess(df, log_label="PRE-PROCESSAMENTO  (compartilhado)")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg.test_size, random_state=cfg.random_state,
        )
        _logger.info("Split: treino=%d | teste=%d", len(X_train), len(X_test))

        # estagio 3 -- busca por candidato ------------------------------------
        results: list[tuple[MLPipeline, str]] = []
        for model, param_grid in candidates:
            pipeline = cls(model=clone(model), config=cfg)
            pipeline._run_search(X_train, X_test, y_train, y_test, feature_columns, param_grid, save_params=False)
            results.append((pipeline, type(model).__name__))

        # estagio 4 -- tabela comparativa -------------------------------------
        results.sort(key=lambda r: r[0].metrics_["MAPE"])
        col_w = 22
        _log_block("COMPARACAO DE MODELOS")
        _logger.info(
            "  %-*s %10s %10s %8s %9s %9s %11s",
            col_w, "Modelo", "MAE", "RMSE", "R2", "MAPE", "WMAPE", "Acuracia",
        )
        _logger.info("  " + "-" * (col_w + 63))
        for pipe, name in results:
            m = pipe.metrics_
            _logger.info(
                "  %-*s %10.4f %10.4f %8.4f %8.2f%% %8.2f%% %10.2f%%",
                col_w, name, m["MAE"], m["RMSE"], m["R2"], m["MAPE"], m["WMAPE"], m["Acuracia"],
            )

        best_pipeline, best_name = results[0]
        _logger.info("")
        _logger.info(
            "  + Melhor modelo: %s  (MAPE=%.2f%%  Acuracia=%.2f%%)",
            best_name,
            best_pipeline.metrics_["MAPE"],
            best_pipeline.metrics_["Acuracia"],
        )

        # propaga estatísticas de pré-processamento do helper ao pipeline vencedor
        for key in ("n_raw", "n_after_filter", "n_filtered", "filtered_pct", "consumo_kwh_stats"):
            if key in helper.report_:
                best_pipeline.report_[key] = helper.report_[key]

        # expoe todos os resultados no pipeline vencedor (usado por SegmentedMLPipeline)
        best_pipeline.compare_results_ = results  # ordenado por WMAPE (melhor primeiro)
        best_pipeline._te_map          = helper._te_map  # Target Encoding map (hora/mes)
        best_pipeline._clipping_limits = helper._clipping_limits  # Clipping limits do treino

        # Persistência mínima fica no fluxo segmentado (SegmentedMLPipeline.save).
        # Aqui apenas retornamos o melhor pipeline já treinado/comparado.
        return best_pipeline

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Prediz consumo_kwh para novos dados no schema inicial.

        Toda a conversão de inputs é delegada a ``MLNormalizer.transform()``
        (módulo ``tools/normalizer.py``), que reproduz fielmente o fluxo de
        ``_build_schema(inference=True)`` + alinhamento + sanitização +
        conversão para float32 numpy.

        A coluna 'consumo_kwh' não deve estar presente — uma dummy é
        inserida internamente pelo normalizer.

        Args:
            df: DataFrame no schema inicial sem a coluna target.

        Returns:
            np.ndarray com os valores preditos (kWh).

        Raises:
            RuntimeError: Modelo não treinado.
            ValueError  : Coluna target presente no input.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() ou tune() antes de predict().")
        if _TARGET in df.columns:
            raise ValueError(f"Remova a coluna '{_TARGET}' do DataFrame de entrada.")

        normalizer = MLNormalizer(
            feature_columns=self.feature_columns_,
            te_map=self._te_map,
            clipping_limits=self._clipping_limits,
        )
        X = normalizer.transform(df)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=_FN_WARNING, category=UserWarning)
            return self.model.predict(X)

    # -- estagio 5 -- persistencia --------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Salva o pipeline completo (modelo + feature_columns_ + metrics_) com joblib.

        Args:
            path: Caminho do arquivo de saida (.joblib).

        Raises:
            RuntimeError: Modelo nao treinado.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() ou tune() antes de save().")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        _logger.info("Pipeline salvo em: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPipeline":
        """Carrega um pipeline salvo do disco."""
        pipeline: MLPipeline = joblib.load(path)
        _logger.info("Pipeline carregado de: %s", path)
        return pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  PRÉ-ETAPA 2 — PIPELINE SEGMENTADO POR TIPO DE MÁQUINA
# ══════════════════════════════════════════════════════════════════════════════

class SegmentedMLPipeline:
    """
    Pré-etapa 2: treina um MLPipeline independente por valor único de
    'machine_type', capturando padrões específicos de cada tecnologia HVAC
    (Split, VRF, Chiller, self-contained, etc.).

    Cada segmento recebe automaticamente:
        - Filtro de outliers de consumo_kwh (pré-etapa 1)
        - Busca independente de hiperparâmetros (ParameterSampler + KFold)
        - Avaliação isolada de métricas

    Ponto de entrada: ``SegmentedMLPipeline(config=cfg).fit(df_raw)``

    Attributes:
        segments_ : Dicionário {machine_type → MLPipeline treinado}.
        metrics_  : Dicionário {machine_type → métricas do segmento}.
    """

    def __init__(self, config: MLPipelineConfig | None = None) -> None:
        self.config     = config or MLPipelineConfig()
        self.segments_:    dict[str, MLPipeline]         = {}
        self.metrics_:     dict[str, dict[str, float]]   = {}
        self.all_results_: dict[str, list]               = {}  # mt → [(MLPipeline, name), ...]
        self.skipped_segments_: dict[str, str]           = {}
        self._is_fitted: bool                            = False

    # -- treinamento ----------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> "SegmentedMLPipeline":
        """
        Treina um MLPipeline por valor único de 'machine_type'.

        A pré-etapa 1 (filtro de outliers) é aplicada automaticamente dentro
        de cada MLPipeline.compare() via _preprocess().

        Args:
            df: DataFrame no schema inicial (14 colunas brutas).

        Returns:
            Self.
        """
        # Normaliza machine_type ANTES da segmentação para consolidar rótulos
        # equivalentes (ex: 'split cassete' e 'split-cassete' → 'SPLIT CASSETE').
        _norm_series = (
            ModelSchema(df.select("machine_type"), ["machine_type"])
            .adjust_machine_type()
            .df["tipo_maquina"]
            .alias("_norm_type")
        )
        df = df.with_columns(_norm_series).with_columns(
            pl.when(
                pl.col("_norm_type").is_null()
                | (pl.col("_norm_type").cast(pl.String).str.strip_chars() == "")
            )
            .then(pl.lit("DESCONHECIDO"))
            .otherwise(pl.col("_norm_type").cast(pl.String))
            .alias("_norm_type")
        )

        machine_types = sorted(df["_norm_type"].unique().to_list())
        _log_block(f"SEGMENTACAO  [pré-etapa 2 — {len(machine_types)} tipos de máquina]")
        for mt in machine_types:
            n = int((df["_norm_type"] == mt).sum())
            _logger.info("  → '%s'  (%d registros)", mt, n)

        skipped: dict[str, str] = {}
        for mt in machine_types:
            # diretório exclusivo por segmento: artifacts_dir/segmented/<tipo>/
            safe    = mt.replace("/", "_").replace(" ", "_")
            seg_dir = self.config.artifacts_dir / "ml_hvac" / safe
            seg_cfg = _dc_replace(self.config, artifacts_dir=seg_dir)

            df_seg             = df.filter(pl.col("_norm_type") == mt).drop("_norm_type")
            _log_block(f"SEGMENTO  '{mt}'")
            try:
                pipeline = MLPipeline.compare(df_seg, config=seg_cfg)
            except ValueError as exc:
                reason = str(exc)
                skipped[mt] = reason
                _logger.warning("Segmento '%s' ignorado: %s", mt, reason)
                continue
            self.segments_[mt]    = pipeline
            self.metrics_[mt]     = pipeline.metrics_
            self.all_results_[mt] = pipeline.compare_results_

        self.skipped_segments_ = skipped
        if skipped:
            _log_block(f"SEGMENTOS ML PULADOS  [n={len(skipped)}]")
            for mt, reason in skipped.items():
                _logger.warning("  - %s | motivo: %s", mt, reason)
        else:
            _logger.info("Nenhum segmento foi pulado na etapa segmentada ML.")

        if not self.segments_:
            raise RuntimeError(
                "Nenhum segmento treinável após filtro de outliers e schema. "
                "Revise parâmetros de limpeza (noise_floor/iqr_factor/segment_params)."
            )

        self._is_fitted = True
        self._log_summary()
        self._save_consolidated_report()
        return self

    def _log_summary(self) -> None:
        """Exibe tabela comparativa de métricas por segmento."""
        col_w = 24
        _log_block("RESUMO SEGMENTADO")
        _logger.info(
            "  %-*s %10s %10s %8s %9s %9s %11s",
            col_w, "Segmento", "MAE", "RMSE", "R2", "MAPE", "WMAPE", "Acuracia",
        )
        _logger.info("  " + "-" * (col_w + 63))
        for mt, m in self.metrics_.items():
            _logger.info(
                "  %-*s %10.4f %10.4f %8.4f %8.2f%% %8.2f%% %10.2f%%",
                col_w, mt, m["MAE"], m["RMSE"], m["R2"], m["MAPE"], m["WMAPE"], m["Acuracia"],
            )

    def _save_consolidated_report(self) -> None:
        """
        Gera o raio-x completo da modelagem segmentada em
        ``<artifacts_dir>/ml_model_report.json``.

        Estrutura do arquivo:
        ─────────────────────
        timestamp    : ISO-8601
        n_segments   : quantidade de segmentos treinados
        consolidated : totais de instâncias (n_train, n_test)
                       + métricas ponderadas por n_test do melhor modelo
                       (MAE, RMSE, WMAPE, Acuracia)
        segmented    : por machine_type —
                         best_model : nome do modelo vencedor (menor WMAPE)
                         n_train    : amostras de treino
                         n_test     : amostras de teste
                         models     : {model_name → {MAE, RMSE, R2, WMAPE, Acuracia}}
        """
        segments_data: dict = {}
        n_train_total = n_test_total = 0

        for mt, pipeline in self.segments_.items():
            r    = pipeline.report_
            n_t  = r.get("n_test", 0)
            candidates = self.all_results_.get(mt, [(pipeline, r.get("model_name", ""))])
            segments_data[mt] = {
                "best_model": r.get("model_name", ""),
                "n_train":    r.get("n_train", 0),
                "n_test":     n_t,
                "n_features": len(pipeline.feature_columns_),
                "feature_columns": pipeline.feature_columns_,
                "best_params": r.get("best_params", {}),
                "models": {
                    name: {k: round(v, 6) for k, v in p.metrics_.items()}
                    for p, name in candidates
                },
            }
            n_train_total += r.get("n_train", 0)
            n_test_total  += n_t

        def _wavg(key: str) -> float:
            if n_test_total == 0:
                return float("nan")
            return round(
                sum(
                    p.report_.get("metrics", {}).get(key, 0) * p.report_.get("n_test", 0)
                    for p in self.segments_.values()
                ) / n_test_total,
                6,
            )

        report = {
            "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
            "n_segments": len(self.segments_),
            "consolidated": {
                "n_train":  n_train_total,
                "n_test":   n_test_total,
                "MAE":      _wavg("MAE"),
                "RMSE":     _wavg("RMSE"),
                "MAPE":     _wavg("MAPE"),
                "WMAPE":    _wavg("WMAPE"),
                "Acuracia": _wavg("Acuracia"),
            },
            "segmented": segments_data,
        }

        _save_report_json(self.config.artifacts_dir / "ml_hvac" / "ml_model_report.json", report)

    # -- inferência -----------------------------------------------------------

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Roteia cada linha ao pipeline do respectivo 'machine_type'.

        Args:
            df: DataFrame no schema inicial sem a coluna target.

        Returns:
            np.ndarray com os valores preditos (kWh), na ordem original.

        Raises:
            RuntimeError: Modelo não treinado.
            ValueError  : Coluna target presente, ou machine_type desconhecido.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() antes de predict().")
        if _TARGET in df.columns:
            raise ValueError(f"Remova a coluna '{_TARGET}' do DataFrame de entrada.")

        # Normaliza machine_type para rotear ao segmento correto (espelha fit())
        _norm_series = (
            ModelSchema(df.select("machine_type"), ["machine_type"])
            .adjust_machine_type()
            .df["tipo_maquina"]
            .alias("_norm_type")
        )
        df_with_norm = df.with_columns(_norm_series)

        present = set(df_with_norm["_norm_type"].unique().to_list())
        unknown = present - set(self.segments_.keys())
        if unknown:
            raise ValueError(
                f"Tipos de maquina nao vistos no treinamento: {sorted(unknown)}. "
                f"Disponiveis: {sorted(self.segments_.keys())}"
            )

        result = np.zeros(len(df), dtype=np.float32)
        df_idx = df_with_norm.with_row_index("__row_idx__")

        for mt, pipeline in self.segments_.items():
            rows = df_idx.filter(pl.col("_norm_type") == mt)
            if len(rows) == 0:
                continue
            indices         = rows["__row_idx__"].to_numpy()
            result[indices] = pipeline.predict(rows.drop(["__row_idx__", "_norm_type"]))

        return result

    # -- persistência ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Salva o pipeline vencedor de cada segmento e um manifesto JSON.

        Estrutura gerada (mínima para inferência + interpretabilidade):
            {path}/
                manifest.json                  — mapeamento segmento → arquivo + métricas
                {segment}/
                    best_pipeline.joblib        — MLPipeline vencedor do segmento

        Args:
            path: Diretório base de saída, e.g. ``artifacts_dir / "ml_hvac"``.

        Raises:
            RuntimeError: Modelo não treinado.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() antes de save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        segment_files: dict[str, str] = {}
        for mt, pipeline in self.segments_.items():
            safe    = mt.replace("/", "_").replace(" ", "_")
            seg_dir = path / safe
            seg_dir.mkdir(parents=True, exist_ok=True)
            fname   = f"{safe}/best_pipeline.joblib"
            pipeline.save(seg_dir / "best_pipeline.joblib")
            segment_files[mt] = fname

        manifest = {"segment_files": segment_files, "metrics": self.metrics_}
        with (path / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

        _logger.info("SegmentedMLPipeline salvo em: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SegmentedMLPipeline":
        """
        Carrega um SegmentedMLPipeline salvo do disco.

        Args:
            path: Diretório criado por save().

        Returns:
            SegmentedMLPipeline pronto para predict().
        """
        path = Path(path)
        with (path / "manifest.json").open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        segmented = cls()
        for mt, fname in manifest["segment_files"].items():
            segmented.segments_[mt] = MLPipeline.load(path / fname)
        segmented.metrics_   = manifest["metrics"]
        segmented._is_fitted = True

        _logger.info("SegmentedMLPipeline carregado de: %s", path)
        return segmented


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO DIRETA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parquet_path = Path(r"use_case\files\final_dataframe.parquet")
    if not parquet_path.exists():
        _logger.error("Arquivo nao encontrado: %s", parquet_path)
        sys.exit(1)

    df_raw = pl.read_parquet(parquet_path)
    _logger.info("%d registros carregados | colunas: %d", df_raw.shape[0], df_raw.shape[1])

    df_schema = ModelSchema(df_raw, _SCHEMA_FIELDS).build()

    _log_block("SCHEMA — raio-x do dataframe normalizado")
    _logger.info("  Registros         : %d", df_schema.shape[0])
    _logger.info("  Colunas totais    : %d", df_schema.shape[1])

    ohe_prefixes = ("tipo_maquina_", "estacao_", "periodo_dia_")
    ohe_cols     = [c for c in df_schema.columns if c.startswith(ohe_prefixes)]
    _logger.info("  Colunas OHE       : %d", len(ohe_cols))
    for col in ohe_cols:
        _logger.info("    %s", col)

    _logger.info("  Nota: hora e mes serao codificados via Target Encoding no pipeline ML")
    _logger.info("  'data' removida   : %s", str("data"         not in df_schema.columns))
    _logger.info("  'machine_type'    : %s", str("machine_type" not in df_schema.columns))

    cont_cols = [
        "Temperatura_C", "Temperatura_Percebida_C",
        "Umidade_Relativa_%", "Precipitacao_mm",
        "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
        "Irradiancia_Direta_Wm2", "Irradiancia_Difusa_Wm2",
        'consumo_lag_1h', 'consumo_rolling_mean_3h', 'consumo_lag_24h'
    ]
    _logger.info("  Features continuas normalizadas (min / max apos clipping+minmax):")
    for col in cont_cols:
        s = df_schema[col]
        _logger.info(
            "    %-30s  min=%.4f  max=%.4f  mean=%.4f",
            col, float(s.min()), float(s.max()), float(s.mean()),
        )

    bool_cols = ["is_feriado", "is_vespera_feriado", "is_dia_util"]
    _logger.info("  Features booleanas (taxa de True):")
    for col in bool_cols:
        if col in df_schema.columns:
            rate = float(df_schema[col].mean()) * 100
            _logger.info("    %-26s  %.1f%%", col, rate)

    cat_cols_present = [c for c in df_schema.columns
                        if df_schema.schema[c] == pl.Categorical]
    _logger.info("  Colunas Categorical : %s", cat_cols_present)
    _logger.info("  Target presente     : %s", str(_TARGET in df_schema.columns))

    # -- 1. Configuracao ------------------------------------------------------
    cfg = MLPipelineConfig(
        candidates=[
            LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            XGBRegressor(random_state=42, n_jobs=-1),
        ],
        n_iter=20,
        cv=5,
        test_size=0.15,
        random_state=42,
        noise_floor=0.59,
        iqr_factor=1.75,
        min_segment_size=35,
        noise_quantile=0.08,
        upper_quantile_cap=0.95,
        segment_params={
            # Chave deve ser o tipo normalizado (exibido no log Outliers[...])
            # Exemplos de sobrescrita por segmento:
            "AR CONDICIONADO DE JANELA (ACJ)": {
                "iqr_factor": 2.0,
                "noise_floor": 0.59,
                "upper_quantile_cap": 0.99,
                "noise_quantile": 0.08,
            },
            "SPLIT CASSETE": {
                "iqr_factor": 2.0,
                "noise_floor": 1.59,
                "upper_quantile_cap": 0.99,
                "noise_quantile": 0.08,
            },
            "SPLIT DUTO": {
                "iqr_factor": 2.0,
                "noise_floor": 1.69,
                "upper_quantile_cap": 0.99,
                "noise_quantile": 0.08,
            },
            "SPLIT HI-WALL": {
                "iqr_factor": 2.0,
                "noise_floor": 0.79,
                "upper_quantile_cap": 0.99,
                "noise_quantile": 0.08,
            },
            "SPLIT PISO-TETO": {
                "iqr_factor": 2.0,
                "noise_floor": 1.59,
                "upper_quantile_cap": 0.99,
                "noise_quantile": 0.08,
            },
            "SPLITÃO": {
                "iqr_factor": 1.5,
                "noise_floor": 5.49,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.20,
            },
            "SPLITÃO INVERTER": {
                "iqr_factor": 1.5,
                "noise_floor": 3.49,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.20,
            },
            "SPLITÃO ROOFTOP": {
                "iqr_factor": 1.5,
                "noise_floor": 5.99,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.20
            },
            "SPLITÃO SELF CONTAINED": {
                "iqr_factor": 1.5,
                "noise_floor": 5.49,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.20
            }
        },
    )

    # Visualização local do impacto da limpeza (por tipo de máquina)
    _plot_local_cleaning_impact(df_raw, cfg)

    # -- 2-3. Pre-processamento (pré-etapa 1 integrada) -> Busca -> Avaliacao --
    # Avulso: salva em ml_hvac/global para não misturar com artefatos segmentados.
    best = MLPipeline.compare(
        df_raw,
        config=_dc_replace(cfg, artifacts_dir=cfg.artifacts_dir / "ml_hvac" / "global"),
    )

    # -- 3. Persistencia (modelo global) --------------------------------------
    best.save(cfg.artifacts_dir / "ml_hvac" / "global" / "best_pipeline.joblib")

    # -- 4. Pipeline segmentado (pré-etapa 2) ---------------------------------
    seg = SegmentedMLPipeline(config=cfg)
    seg.fit(df_raw)

    # -- 5. Persistencia (segmentado) -----------------------------------------
    seg.save(cfg.artifacts_dir / "ml_hvac")

    # -- 6. Demo: predição das 10 primeiras linhas (global vs segmentado) -----
    _log_block("DEMO — Predição das 10 primeiras linhas do dataset")

    # Aplica o mesmo filtro de outliers usado no treino para que o demo
    # contenha apenas registros representativos (sem ruído / extremos).
    df_clean, _ = _filter_outliers(
        df_raw,
        noise_floor=cfg.noise_floor,
        iqr_factor=cfg.iqr_factor,
        min_segment_size=cfg.min_segment_size,
        noise_quantile=cfg.noise_quantile,
        upper_quantile_cap=cfg.upper_quantile_cap,
        segment_params=cfg.segment_params,
    )

    df_demo = df_clean.head(10)
    y_real  = df_demo[_TARGET].to_numpy()

    # Prepara input sem target
    df_input = df_demo.drop(_TARGET) if _TARGET in df_demo.columns else df_demo

    # Predição GLOBAL (modelo vencedor)
    y_global = best.predict(df_input)

    # Predição SEGMENTADA (cada linha roteada ao modelo do seu machine_type)
    y_seg = seg.predict(df_input)

    # Normaliza machine_type para exibição
    _mt_display = (
        ModelSchema(df_demo.select("machine_type"), ["machine_type"])
        .adjust_machine_type()
        .df["tipo_maquina"]
        .to_list()
    )

    col_w_mt = max(len("Segmento"), max(len(m) for m in _mt_display))
    _logger.info("")
    _logger.info(
        "  %-4s  %-*s  %12s  %12s  %12s",
        "#", col_w_mt, "Segmento", "Real (kWh)", "Global (kWh)", "Segm. (kWh)",
    )
    _logger.info("  " + "-" * (4 + 2 + col_w_mt + 2 + 12 + 2 + 12 + 2 + 12))

    for i in range(len(df_demo)):
        _logger.info(
            "  %-4d  %-*s  %12.4f  %12.4f  %12.4f",
            i + 1,
            col_w_mt, _mt_display[i],
            y_real[i],
            y_global[i],
            y_seg[i],
        )

    # Métricas rápidas
    from sklearn.metrics import mean_absolute_error as _mae, r2_score as _r2

    _logger.info("")
    _logger.info(
        "  %-*s  %12s  %12.4f  %12.4f",
        4 + 2 + col_w_mt, "MAE (10 linhas):", "",
        _mae(y_real, y_global),
        _mae(y_real, y_seg),
    )
    _logger.info(
        "  %-*s  %12s  %12.4f  %12.4f",
        4 + 2 + col_w_mt, "R²  (10 linhas):", "",
        _r2(y_real, y_global),
        _r2(y_real, y_seg),
    )
    _logger.info("")
    _logger.info("Demo de predição finalizada ✔")

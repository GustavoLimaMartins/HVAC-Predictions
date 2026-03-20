"""
DLPipeline — Wide & Deep Neural Network para predição de consumo HVAC
======================================================================

Fluxo de 5 estágios + 2 pré-etapas:

    PRÉ-ETAPA 1  _filter_outliers()    — remove ruído (consumo ≈ 0) e extremos
                                          via Tukey fence modificada, aplicada
                                          por segmento de machine_type para
                                          respeitar distribuições heterogêneas
                                          entre tecnologias HVAC.
    PRÉ-ETAPA 2  SegmentedDLPipeline  — um DLPipeline por tipo de máquina HVAC

    1. CONFIG      DLPipelineConfig   — arquitetura, treino, callbacks
    2. PREPROCESS  FeatureDeriver + ModelSchema — derivação de features + schema
    3. TRAIN       fit()              — Wide & Deep + EarlyStopping
    4. EVALUATE    _compute_metrics() — MAE / RMSE / R² / WMAPE
    5. PERSIST     save() / load()    — SavedModel + JSON

Inferência:
    ``predict()`` delega toda a conversão de inputs ao módulo
    ``tools/normalizer.py`` (classe ``DLNormalizer``), garantindo que a
    normalização em inferência seja **idêntica** ao fluxo de treino sem
    duplicar lógica.

REFATORAÇÃO (Message 15):
    Eliminado DLSchema — classe que reimplementava logic. de FeatureDeriver
    + ModelSchema. Novo fluxo de _preprocess():
    
    1. _assign_grupo_regional_knn() → geo lookup via BallTree Haversine
    2. ModelSchema.add_date_features() → period_dia + features temporais
    3. ModelSchema.adjust_machine_type() + OHE + Clipping+MinMax
    4. Extrai embeddings Int32 (hora, mes, grupo_regional, periodo_dia)
    5. Retorna X_emb dict + X_dense array (idêntico ao anterior)

Arquitetura Wide & Deep:

    grupo_regional  →  Embedding(n_groups, embedding_dim)  →  Flatten  ─┐
                                                                         ├──► Concat
    features_densas ─────────────────────────────────────────────────────┘      │
                                                                                 ▼
                                                             Dense(128) → BatchNorm → Dropout(20%)
                                                                                 │
                                                             Dense(64)  → BatchNorm → Dropout(20%)
                                                                                 │
                                                             Dense(1, linear) = consumo_kwh

Schema DL vs. Schema ML:

    hora, mes       → Int32 puro para Entity Embedding  [era OHE / Categorical no ML]
    grupo_regional  → Int32 puro para Entity Embedding  [era Clipping + MinMax]
    periodo_dia     → Int32 (0-3) para Entity Embedding [era OHE no ML]
    demais          → sem alteração

Instalação:
    pip install tensorflow

Uso mínimo:

    >>> cfg  = DLPipelineConfig(epochs=100, batch_size=512)
    >>> pipe = DLPipeline(config=cfg).fit(df_raw)
    >>> pipe.save("model/artifacts/dl_hvac")
"""

from __future__ import annotations

import os

# Suprime mensagens de inicialização do TensorFlow antes do primeiro import:
#   0 = DEBUG+INFO+WARNING+ERROR  (tudo visível)
#   1 = WARNING+ERROR             (sem INFO)
#   2 = ERROR                     (sem WARNING)
#   3 = silêncio total            (apenas erros fatais)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import datetime
import datetime
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    import tensorflow as tf

    # AdamW migrou de experimental para o namespace principal no TF 2.12.
    # Tenta o caminho novo → experimental → fallback para Adam.
    _AdamW = (
        getattr(tf.keras.optimizers, "AdamW", None)
        or getattr(getattr(tf.keras.optimizers, "experimental", None), "AdamW", None)
        or tf.keras.optimizers.Adam
    )
    from tensorflow.keras import Input, Model                          # type: ignore
    from tensorflow.keras.layers import (                              # type: ignore
        BatchNormalization, Concatenate, Dense, Dropout, Embedding, Flatten,
    )
except ImportError as _e:
    raise ImportError(
        "TensorFlow não encontrado. Instale com:\n  pip install tensorflow"
    ) from _e

# Rebaixa o logger raiz do TensorFlow para ERROR após o import,
# eliminando os avisos de GPU/Windows e de oneDNN em runtime.
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model.pre_process.schema import ModelSchema
from tools.normalizer import DLNormalizer, _assign_grupo_regional_knn, compute_normalization_stats


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
    "consumo_lag_1h", "consumo_lag_24h", "consumo_rolling_mean_3h",
]

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
    _logger.info("  %-12s %.4f",   "MAE:",      metrics["MAE"])
    _logger.info("  %-12s %.4f",   "RMSE:",     metrics["RMSE"])
    _logger.info("  %-12s %.4f",   "R2:",        metrics["R2"])
    _logger.info("  %-12s %.2f%%", "WMAPE:",     metrics["WMAPE"])
    _logger.info("  %-12s %.2f%%", "Acuracia:",  metrics["Acuracia"])


def _sanitize_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Substitui NaN e ±inf por 0.0 em todas as colunas float do DataFrame.

    Necessário porque `make_clipping_min_max_columns` produz NaN quando
    todos os valores de uma coluna no segmento são idênticos
    (upper − lower = 0 → divisão por zero). NaN propagados para o modelo
    geram predições NaN que derrubam o cálculo de métricas.

    Emite WARNING com a lista de colunas afetadas para rastreabilidade.
    """
    float_dtypes = (pl.Float32, pl.Float64)
    nan_cols = [
        c for c in df.columns
        if df[c].dtype in float_dtypes
        and (df[c].is_nan().any() or df[c].is_infinite().any())
    ]
    if nan_cols:
        _logger.warning(
            "NaN/inf em %d coluna(s) apos schema (divisao por zero na normalizacao) "
            "— substituindo por 0.0: %s",
            len(nan_cols), nan_cols,
        )
        for c in nan_cols:
            np_dtype = np.float32 if df[c].dtype == pl.Float32 else np.float64
            arr = df[c].to_numpy().copy().astype(np_dtype)
            arr = np.where(np.isfinite(arr), arr, np_dtype(0.0))
            df  = df.with_columns(pl.Series(c, arr))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS DE MÉTRICA — estágio 4
# ══════════════════════════════════════════════════════════════════════════════

def _wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """WMAPE = sum(|y-yhat|) / sum(|y|) * 100  (imune a zeros no denominador)."""
    denom = np.abs(y_true).sum()
    return float("nan") if denom == 0 else float(np.abs(y_true - y_pred).sum() / denom * 100)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Retorna MAE, RMSE, R2, WMAPE e Acuracia."""
    wmape = _wmape(y_true, y_pred)
    return {
        "MAE":      float(mean_absolute_error(y_true, y_pred)),
        "RMSE":     float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2":       float(r2_score(y_true, y_pred)),
        "WMAPE":    wmape,
        "Acuracia": 100.0 - wmape,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DL SCHEMA — estágio 2
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO — estágio 1
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DLPipelineConfig:
    """
    Parâmetros centralizados da DLPipeline.

    Arquitetura
    -----------
    embedding_dim      : Dimensão do vetor de Embedding para grupo_regional.
    embedding_dim_hora : Dimensão do Entity Embedding para hora (0–23).
    embedding_dim_mes  : Dimensão do Entity Embedding para mes (1–12).
    hidden_units       : Neurônios por camada Dense oculta [camada1, camada2, ...].
    dropout_rate       : Dropout inicial (primeira camada oculta).
    min_dropout_rate   : Dropout final (última camada oculta), para decaimento
                         regressivo entre camadas.

    Treino
    ------
    epochs          : Número máximo de épocas.
    batch_size      : Tamanho do lote (256, 512 ou 1024 recomendado).
    learning_rate   : Taxa inicial do AdamW.
    loss            : Função de perda ("huber" ou "mae").

    Callbacks
    ---------
    patience_stop   : Épocas sem melhora para EarlyStopping.
    patience_lr     : Épocas sem melhora para ReduceLROnPlateau.
    reduce_lr_factor: Fator de redução do LR (0.5 → LR / 2).
    min_lr          : Learning rate mínima permitida.

    Split
    -----
    test_size       : Fração reservada para avaliação final (0–1).
    val_size        : Fração do conjunto de treino usada para validação.
    random_state    : Semente de reprodutibilidade.
    artifacts_dir   : Diretório de saída para SavedModel e JSON.

    Pré-etapa 1 — filtro de outliers de consumo_kwh
    ------------------------------------------------
    noise_floor     : Limiar mínimo de consumo (kWh). Registros abaixo desse
                      valor são classificados como ruído (standby / falha de
                      sensor) e removidos antes do treinamento.
    iqr_factor      : Fator k da Tukey fence modificada. Remove consumo_kwh
                      fora de [Q1 − k·IQR, Q3 + k·IQR]. Use 3.0 (extremos)
                      em vez de 1.5 (suave) para preservar variações sazonais
                      legítimas de demanda energética. O filtro é aplicado
                      por segmento de machine_type, pois chillers, splits e
                      self-containeds têm distribuições de consumo em ordens
                      de magnitude distintas.
    min_segment_size: Tamanho mínimo por segmento para aplicar lógica
                      dinâmica estável; abaixo disso usa fallback local.
    noise_quantile  : Quantil baixo base para estimar piso dinâmico de ruído.
    upper_quantile_cap: Quantil superior opcional para limitar cauda alta.
    segment_params  : Sobrescritas por segmento (tipo_maquina normalizado),
                      com suporte a: noise_floor, iqr_factor, min_segment_size,
                      noise_quantile e upper_quantile_cap.
    """
    # arquitetura
    embedding_dim:      int       = 8    # grupo_regional
    embedding_dim_hora: int       = 4    # hora (0-23)
    embedding_dim_mes:     int       = 3    # mes (1-12)
    embedding_dim_periodo: int       = 2    # periodo_dia (4 faixas horárias)
    hidden_units:          list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate:       float     = 0.30
    min_dropout_rate:   float     = 0.10

    # treino
    epochs:           int       = 200
    batch_size:       int       = 512
    learning_rate:    float     = 0.001
    loss:             str       = "huber"

    # callbacks
    patience_stop:    int       = 15
    patience_lr:      int       = 5
    reduce_lr_factor: float     = 0.5
    min_lr:           float     = 1e-6

    # split
    test_size:        float     = 0.2
    val_size:         float     = 0.1
    random_state:     int       = 42

    # artefatos
    artifacts_dir:    Path      = field(default_factory=lambda: Path("model/artifacts"))

    # pré-etapa 1 — filtro de outliers de consumo_kwh
    noise_floor:      float     = 0.5   # kWh mínimo em operação real (abaixo = ruído)
    iqr_factor:       float     = 3.0   # k da Tukey fence (3.0 = apenas extremos)
    min_segment_size: int       = 40
    noise_quantile:   float     = 0.05
    upper_quantile_cap: float | None = None
    segment_params:   dict[str, dict[str, float | int | None]] | None = None


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
    return_bounds: bool = False,
) -> pl.DataFrame | tuple[pl.DataFrame, dict[str, dict[str, float]]]:
    """
    Remove instâncias de ruído (consumo ≈ 0) e distorções (outliers extremos)
    de 'consumo_kwh' ANTES do treinamento com parametrização dinâmica por
    segmento (machine_type/tipo_maquina), equivalente ao pipeline de ML.

    Filtro de ruído — baixa carga / standby
    ────────────────────────────────────────
    Em sistemas HVAC, leituras próximas de zero indicam equipamento em modo
    standby, sensor desligado ou falha de medição. São removidos registros com:

        consumo_kwh  <  noise_floor

    Filtro de distorções — Tukey fence modificada por segmento
    ───────────────────────────────────────────────────────────
    A cerca de Tukey com fator k é calculada por segmento. Além disso,
    k, quantil de ruído e cap superior podem ser ajustados dinamicamente
    por tipo e formato da distribuição, com opção de override manual.

        Q₁  = percentil 25 de consumo_kwh  (no segmento)
        Q₃  = percentil 75 de consumo_kwh  (no segmento)
        IQR = Q₃ − Q₁

        Limite inferior : max(Q₁ − k · IQR,  noise_floor)
        Limite superior : Q₃ + k · IQR

    O valor padrão k = 3.0 (fence "extrema") preserva variações sazonais e
    picos de demanda legítimos de ar-condicionado, removendo apenas anomalias
    estatísticas severas. O valor k = 1.5 (padrão Tukey) seria excessivamente
    restritivo para séries de consumo energético, cuja distribuição é assimétrica
    à direita (skew positivo) por natureza.

    Args:
        df               : DataFrame com colunas 'consumo_kwh' e 'machine_type'.
        noise_floor      : Limiar mínimo global de consumo (kWh).
        iqr_factor       : Fator k global da Tukey fence.
        min_segment_size : Tamanho mínimo para dinâmica por segmento.
        noise_quantile   : Quantil baixo base para piso dinâmico.
        upper_quantile_cap: Quantil superior base opcional para cauda alta.
        segment_params   : Overrides por segmento.
        return_bounds    : Se True, retorna também dict com faixas dinâmicas
                           por segmento: {tipo: {lo, hi, q1, q3, k, floor}}.

    Returns:
        DataFrame filtrado.  Se ``return_bounds=True``, retorna
        ``(df_filtrado, bounds_dict)``.
    """
    n_before = len(df)

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

    # normaliza machine_type → nomes canônicos para segmentação coerente
    _norm = (
        ModelSchema(df.select("machine_type"), ["machine_type"])
        .adjust_machine_type()
        .df["tipo_maquina"]
        .alias("_norm_type")
    )
    df = df.with_row_index("__row_idx__").with_columns(_norm).with_columns(
        pl.when(
            pl.col("_norm_type").is_null()
            | (pl.col("_norm_type").cast(pl.String).str.strip_chars() == "")
        )
        .then(pl.lit("DESCONHECIDO"))
        .otherwise(pl.col("_norm_type").cast(pl.String))
        .alias("_norm_type")
    )

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

    segments_out: list[pl.DataFrame] = []
    n_noise_total = 0
    n_ext_total = 0
    _seg_bounds: dict[str, dict[str, float]] = {}

    for mt in sorted(df["_norm_type"].unique().to_list()):
        seg = df.filter(pl.col("_norm_type") == mt)
        n_seg = len(seg)
        if n_seg == 0:
            continue

        s = seg[_TARGET]
        q1 = float(s.quantile(0.25))
        q3 = float(s.quantile(0.75))
        iqr = max(q3 - q1, 0.0)

        # ajustes dinâmicos (mesma lógica do ML)
        q10 = float(s.quantile(0.10))
        q50 = float(s.quantile(0.50))
        q90 = float(s.quantile(0.90))
        q99 = float(s.quantile(0.99))

        seg_iqr_factor = float(iqr_factor)
        seg_noise_q = float(q_noise)
        seg_upper_q = q_upper
        seg_min_size = int(min_segment_size)
        seg_floor_base = float(floor_base)

        if n_seg >= max(min_segment_size, 10):
            spread_mid = max(q90 - q50, 1e-9)
            spread_low = max(q50 - q10, 1e-9)
            spread_total = max(q90 - q10, 1e-9)
            tail_ratio = max(q99 - q90, 0.0) / spread_mid

            seg_name = str(mt).upper()
            if "HI-WALL" in seg_name or "JANELA" in seg_name or "(ACJ)" in seg_name:
                seg_iqr_factor *= 0.90
                seg_noise_q = min(seg_noise_q + 0.02, 0.20)
            if "INVERTER" in seg_name or "ROOFTOP" in seg_name:
                seg_noise_q = max(seg_noise_q - 0.01, 0.01)

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

            if spread_total <= max(q50, 1e-6) * 0.35 and spread_mid / spread_low < 1.8:
                seg_noise_q = min(seg_noise_q + 0.02, 0.20)

        seg_noise_q = float(min(max(seg_noise_q, 0.0), 0.25))

        # overrides explícitos por segmento
        ov = _seg_overrides(str(mt))
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

        use_fallback = n_seg < seg_min_size
        if use_fallback:
            seg_floor = seg_floor_base
        else:
            seg_q_low = float(s.quantile(seg_noise_q))
            lo_cap = seg_floor_base * 0.25
            hi_cap = seg_floor_base * 4.0 if seg_floor_base > 0 else max(seg_q_low, 0.0)
            seg_floor = min(max(seg_q_low, lo_cap), hi_cap)
            seg_floor = max(seg_floor, seg_floor_base)

        lo = max(float(q1 - seg_iqr_factor * iqr), seg_floor)
        hi_iqr = float(q3 + seg_iqr_factor * iqr)
        hi = hi_iqr
        if seg_upper_q is not None and n_seg >= max(seg_min_size, 10):
            hi = min(hi, float(s.quantile(seg_upper_q)))
        if hi < lo:
            hi = lo

        _seg_bounds[str(mt)] = {"lo": lo, "hi": hi, "q1": q1, "q3": q3,
                                 "k": seg_iqr_factor, "floor": seg_floor}

        seg_after_floor = seg.filter(pl.col(_TARGET) >= seg_floor)
        seg_clean = seg_after_floor.filter(pl.col(_TARGET).is_between(lo, hi))

        n_noise = n_seg - len(seg_after_floor)
        n_ext = len(seg_after_floor) - len(seg_clean)
        n_noise_total += n_noise
        n_ext_total += n_ext

        kept_min = float(seg_clean[_TARGET].min()) if len(seg_clean) else float("nan")
        kept_max = float(seg_clean[_TARGET].max()) if len(seg_clean) else float("nan")
        _logger.info(
            "  Outliers[%s]: removidos=%d (ruido=%d extremos=%d) | faixa=[%.3f, %.3f] kWh"
            " | pos=[%.3f, %.3f] kWh | k=%.2f q_noise=%.3f%s",
            mt,
            n_seg - len(seg_clean),
            n_noise,
            n_ext,
            lo,
            hi,
            kept_min,
            kept_max,
            seg_iqr_factor,
            seg_noise_q,
            f" [qcap={seg_upper_q:.4f}]" if seg_upper_q is not None else "",
        )

        segments_out.append(seg_clean)

    df_clean = (
        pl.concat(segments_out).sort("__row_idx__").drop(["__row_idx__", "_norm_type"])
        if segments_out
        else df.head(0).drop(["__row_idx__", "_norm_type"])
    )

    n_total = n_before - len(df_clean)
    _logger.info(
        "  Outliers (geral): %d removidos (%.2f%%) — ruido=%d  extremos=%d",
        n_total,
        n_total / n_before * 100 if n_before else 0,
        n_noise_total,
        n_ext_total,
    )
    if return_bounds:
        return df_clean, _seg_bounds
    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
#  ARQUITETURA — estágio 3
# ══════════════════════════════════════════════════════════════════════════════

def _build_wide_deep_model(
    n_dense_features: int,
    n_groups: int,
    n_horas: int,
    n_meses: int,
    n_periodos: int,
    cfg: DLPipelineConfig,
) -> "tf.keras.Model":
    """
    Constrói o modelo Wide & Deep com Entity Embeddings.

    Fluxo A — Entity Embeddings (grupo_regional + hora + mes + periodo_dia)
    ───────────────────────────────────────────────────────────────────────
    grupo_regional  →  Embedding(n_groups,   embedding_dim)          → Flatten
    hora            →  Embedding(n_horas,    embedding_dim_hora)     → Flatten
    mes             →  Embedding(n_meses,    embedding_dim_mes)      → Flatten
    periodo_dia     →  Embedding(n_periodos, embedding_dim_periodo)  → Flatten

    Fluxo B — Dense (features contínuas + OHE)
    ───────────────────────────────────────────
    Input(shape=(n_dense_features,), float32)
        → passado diretamente à concatenação

    Saída
    ─────
    Concat(emb_grupo, emb_hora, emb_mes, emb_periodo, dense_features)
        → [Dense(u, relu) → BatchNorm → Dropout]  ×  len(hidden_units)
        → Dense(1, linear) = consumo_kwh

    Entity Embeddings (Guo & Berkhahn, 2016) permitem que a rede aprenda
    representações vetoriais densas para variáveis categóricas, capturando
    relações não-lineares entre horas do dia / meses do ano e o consumo
    sem impor premissa de circularidade (sin/cos).

    Args:
        n_dense_features : Número de features no Fluxo B.
        n_groups         : input_dim do Embedding de grupo_regional.
        n_horas          : input_dim do Embedding de hora (tipicamente 24).
        n_meses          : input_dim do Embedding de mes (tipicamente 13).
        n_periodos       : input_dim do Embedding de periodo_dia (tipicamente 4).
        cfg              : Configuração do pipeline.

    Returns:
        tf.keras.Model pronto para compilação.
    """
    # ── Fluxo A — Entity Embeddings ──────────────────────────────────────
    input_grupo = Input(shape=(1,), name="grupo_regional", dtype="int32")
    emb_grupo = Embedding(
        input_dim=n_groups,
        output_dim=cfg.embedding_dim,
        name="group_embedding",
    )(input_grupo)
    emb_grupo = Flatten(name="group_embedding_flat")(emb_grupo)

    input_hora = Input(shape=(1,), name="hora", dtype="int32")
    emb_hora = Embedding(
        input_dim=n_horas,
        output_dim=cfg.embedding_dim_hora,
        name="hora_embedding",
    )(input_hora)
    emb_hora = Flatten(name="hora_embedding_flat")(emb_hora)

    input_mes = Input(shape=(1,), name="mes", dtype="int32")
    emb_mes = Embedding(
        input_dim=n_meses,
        output_dim=cfg.embedding_dim_mes,
        name="mes_embedding",
    )(input_mes)
    emb_mes = Flatten(name="mes_embedding_flat")(emb_mes)

    input_periodo = Input(shape=(1,), name="periodo_dia", dtype="int32")
    emb_periodo = Embedding(
        input_dim=n_periodos,
        output_dim=cfg.embedding_dim_periodo,
        name="periodo_embedding",
    )(input_periodo)
    emb_periodo = Flatten(name="periodo_embedding_flat")(emb_periodo)

    # ── Fluxo B — features densas (clima + OHE + binárias) ──────────────
    input_dense = Input(shape=(n_dense_features,), name="dense_features", dtype="float32")

    # ── Concatenação e camadas ocultas ──────────────────────────────────
    x = Concatenate(name="wide_deep_concat")([
        emb_grupo, emb_hora, emb_mes, emb_periodo, input_dense,
    ])

    # Dropout regressivo: da primeira para a última camada oculta
    if len(cfg.hidden_units) == 1:
        dropout_schedule = [float(cfg.dropout_rate)]
    else:
        dropout_schedule = np.linspace(
            float(cfg.dropout_rate),
            float(cfg.min_dropout_rate),
            num=len(cfg.hidden_units),
        ).tolist()

    for i, units in enumerate(cfg.hidden_units):
        x = Dense(units, activation="relu", name=f"hidden_{i + 1}")(x)
        x = BatchNormalization(name=f"bn_{i + 1}")(x)
        x = Dropout(dropout_schedule[i], name=f"dropout_{i + 1}")(x)

    output = Dense(1, activation="linear", name="output")(x)
    return Model(
        inputs=[input_grupo, input_hora, input_mes, input_periodo, input_dense],
        outputs=output,
        name="wide_deep_hvac",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class DLPipeline:
    """
    Pipeline sequencial Wide & Deep para predição de consumo HVAC (kWh).

    Ponto de entrada: ``DLPipeline(config=cfg).fit(df_raw)``.

    Attributes:
        model_           : Modelo Keras treinado.
        feature_columns_ : Nomes das features densas (Fluxo B), em ordem.
        n_groups_        : input_dim do Embedding (max(grupo_regional) + 1).
        n_horas_         : input_dim do Embedding de hora (24).
        n_meses_         : input_dim do Embedding de mes (13).
        history_         : Histórico de loss/mae por época.
        metrics_         : MAE / RMSE / R2 / WMAPE / Acuracia no conjunto de teste.
    """

    def __init__(self, config: DLPipelineConfig | None = None) -> None:
        self.config            = config or DLPipelineConfig()
        self.model_:           "tf.keras.Model | None" = None
        self.feature_columns_: list[str]               = []
        self.n_groups_:        int                     = 0
        self.n_horas_:         int                     = 24
        self.n_meses_:         int                     = 13
        self.n_periodos_:      int                     = 4
        self.history_:         dict                    = {}
        self.metrics_:         dict[str, float]        = {}
        self.train_info_:      dict                    = {}
        self._is_fitted:       bool                    = False
        self._normalization_stats_: dict | None       = None  # Estatísticas de normalização
        self._te_map:          dict | None            = None  # Target Encoding map (compatibilidade futura)
        self._clipping_limits: dict | None            = None  # ✅ Novo: persistir limites de clipping

    # -- estágio 2 — pré-processamento ----------------------------------------

    def _preprocess(
        self,
        df: pl.DataFrame,
        log_label: str = "PRE-PROCESSAMENTO",
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, list[str], dict[str, int]]:
        """
        Pré-processa dados usando FeatureDeriver + ModelSchema (sem DLSchema).

        Refactoring: Elimina DLSchema (que reimplementava lógica) e usa
        FeatureDeriver para derivação automática + ModelSchema direto.

        Returns:
            X_emb          : dict {nome → int32 array (n, 1)} para cada
                             Embedding (grupo_regional, hora, mes)
            X_dense        : float32 array (n, d) — features para Fluxo B
            y              : float32 array (n,)   — target consumo_kwh
            feature_columns: nomes das colunas em X_dense (na ordem)
            emb_sizes      : dict {nome → input_dim} de cada Embedding
        """
        _log_block(log_label)

        # pré-etapa 1 — remove ruído e distorções de consumo_kwh
        _n_raw = len(df)
        df     = _filter_outliers(
            df,
            noise_floor=self.config.noise_floor,
            iqr_factor=self.config.iqr_factor,
            min_segment_size=self.config.min_segment_size,
            noise_quantile=self.config.noise_quantile,
            upper_quantile_cap=self.config.upper_quantile_cap,
            segment_params=self.config.segment_params,
        )
        _n_after = len(df)
        self._preprocess_info: dict = {
            "n_raw":          _n_raw,
            "n_after_filter": _n_after,
            "n_removed":      _n_raw - _n_after,
            "pct_removed":    round((_n_raw - _n_after) / _n_raw * 100, 2) if _n_raw else 0.0,
        }

        # pré-etapa 2 — derivação de features + schema de transformação
        _logger.info("Derivando grupo_regional via BallTree KNN...")
        if "latitude" in df.columns and "longitude" in df.columns:
            df = _assign_grupo_regional_knn(df)
        
        # pré-etapa 2b — aplicar ModelSchema para transformações ML
        _logger.info("Aplicando ModelSchema (derivação + OHE + Clipping+MinMax, etc)...")
        
        # Instancia schema manualmente para controlar a ordem
        schema = ModelSchema.__new__(ModelSchema)
        schema.df = df.clone()
        schema._schema_fields = _SCHEMA_FIELDS
        schema.clipping_limits_ = {}  # ✅ Inicializar atributo que foi bypassado pelo __new__()
        
        schema.add_date_features()
        
        # Depois aplica outras transformações
        schema.adjust_machine_type()
        schema.make_categorical_columns(["grupo_regional"])
        schema.make_one_hot_encode_columns(["tipo_maquina", "estacao", "periodo_dia"])
        
        # IMPORTANTE: make_clipping_min_max_columns é aplicado APENAS em dados de treino
        schema.make_clipping_min_max_columns([
            "Temperatura_C", "Temperatura_Percebida_C",
            "Umidade_Relativa_%", "Precipitacao_mm",
            "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
            "Irradiancia_Direta_Wm2", "Irradiancia_Difusa_Wm2",
            "consumo_lag_1h", "consumo_lag_24h", "consumo_rolling_mean_3h"
        ])
        
        # ✅ Novo: persistir clipping_limits para reutilização em inferência
        self._clipping_limits = schema.clipping_limits_
        
        df_ml = schema.df
        
        hora_arr  = df_ml["hora"].to_numpy().astype(np.int32)
        mes_arr   = df_ml["mes"].to_numpy().astype(np.int32)
        grupo_arr = df_ml["grupo_regional"].to_numpy().astype(np.int32)
        
        # Madrugada (0-6), Manhã (7-11), Tarde (12-18), Noite (19-23)
        periodo_arr = np.where(
            hora_arr <= 6, 0,
            np.where(hora_arr <= 11, 1,
                     np.where(hora_arr <= 18, 2, 3)),
        ).astype(np.int32)
        
        # pré-etapa 2d — remove artefatos incompatíveis com DL (OHE/Categorical)
        periodo_ohe = [c for c in df_ml.columns if c.startswith("periodo_dia_")]
        drop_cols   = periodo_ohe + [c for c in ("mes", "grupo_regional") if c in df_ml.columns]
        df_dl       = df_ml.drop(drop_cols)

        # pré-etapa 2e — adiciona Entity Embeddings como Int32
        df_dl = df_dl.with_columns([
            pl.Series("hora",           hora_arr),
            pl.Series("mes",            mes_arr),
            pl.Series("grupo_regional", grupo_arr),
            pl.Series("periodo_dia",    periodo_arr),
        ])
        
        # pré-etapa 2f — sanitiza NaN/inf → 0.0
        df_dl = _sanitize_features(df_dl)
        
        _logger.info(
            "Schema concluido: %d registros x %d colunas",
            df_dl.shape[0], df_dl.shape[1],
        )

        # Em alguns segmentos, o filtro de outliers pode remover 100% das linhas.
        # Evita erro em arr.max() com array vazio e delega decisão ao chamador.
        if df_dl.height == 0:
            raise ValueError(
                "Sem dados apos pre-processamento (outlier filter + schema). "
                "Ajuste noise_floor/iqr_factor/segment_params para este segmento."
            )

        # Entity Embeddings → X_emb (int32, um Input por coluna)
        emb_cols = ["grupo_regional", "hora", "mes", "periodo_dia"]
        X_emb: dict[str, np.ndarray] = {}
        emb_sizes: dict[str, int] = {}
        for col in emb_cols:
            arr = df_dl[col].to_numpy().astype(np.int32).reshape(-1, 1)
            X_emb[col] = arr
            emb_sizes[col] = int(arr.max()) + 1

        # Demais features → X_dense (float32, Fluxo B)
        dense_cols = [c for c in df_dl.columns if c not in (_TARGET, *emb_cols)]
        X_dense = df_dl.select(
            [pl.col(c).cast(pl.Float32) for c in dense_cols]
        ).to_numpy()

        y = df_dl[_TARGET].to_numpy().astype(np.float32)
        return X_emb, X_dense, y, dense_cols, emb_sizes

    # -- estágio 3 — construção e compilação do modelo ------------------------

    def _build_and_compile(self, n_dense_features: int) -> "tf.keras.Model":
        """Constrói e compila o modelo Wide & Deep com Entity Embeddings."""
        cfg   = self.config
        model = _build_wide_deep_model(
            n_dense_features,
            n_groups=self.n_groups_,
            n_horas=self.n_horas_,
            n_meses=self.n_meses_,
            n_periodos=self.n_periodos_,
            cfg=cfg,
        )
        model.compile(
            optimizer=_AdamW(learning_rate=cfg.learning_rate),
            loss=cfg.loss,
            metrics=["mae"],
        )
        return model

    # -- API pública -----------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> "DLPipeline":
        """
        Pre-processa os dados e treina o modelo Wide & Deep.

        Etapas:
            1. FeatureDeriver.derive() + ModelSchema.build() → X_emb (dict), X_dense, y
            2. train_test_split (teste) → train_test_split (validação)
            3. _build_and_compile() → modelo Keras
            4. model.fit() com EarlyStopping + ReduceLROnPlateau
            5. Avaliação no conjunto de teste

        Args:
            df: DataFrame no schema inicial (14 colunas brutas).

        Returns:
            Self.
        """
        cfg = self.config
        X_emb, X_dense, y, feature_columns, emb_sizes = self._preprocess(df)

        self.feature_columns_ = feature_columns
        self.n_groups_        = emb_sizes["grupo_regional"]
        self.n_horas_         = emb_sizes.get("hora", 24)
        self.n_meses_         = emb_sizes.get("mes", 13)
        self.n_periodos_      = emb_sizes.get("periodo_dia", 4)

        # -- helper para split sincronizado de múltiplos arrays ---------------
        def _split(*arrays, **kwargs):
            """train_test_split para N arrays."""
            return train_test_split(*arrays, **kwargs)

        # Concatena embeddings + dense para split sincronizado
        emb_keys = sorted(X_emb.keys())
        all_arrays = [X_emb[k] for k in emb_keys] + [X_dense, y]

        # -- split treino / teste ---------------------------------------------
        split_te = _split(
            *all_arrays,
            test_size=cfg.test_size,
            random_state=cfg.random_state,
        )
        tr_arrays = split_te[0::2]  # índices pares
        te_arrays = split_te[1::2]  # índices ímpares

        # -- split treino / validação (para EarlyStopping) --------------------
        split_va = _split(
            *tr_arrays,
            test_size=cfg.val_size,
            random_state=cfg.random_state,
        )
        tr_arrays = split_va[0::2]
        va_arrays = split_va[1::2]

        # Reconstrói dicts por nome
        def _to_emb_dict(arrays: list) -> dict[str, np.ndarray]:
            return {k: arrays[i] for i, k in enumerate(emb_keys)}

        emb_tr, X_dense_tr, y_tr = _to_emb_dict(tr_arrays[:-2]), tr_arrays[-2], tr_arrays[-1]
        emb_va, X_dense_va, y_va = _to_emb_dict(va_arrays[:-2]), va_arrays[-2], va_arrays[-1]
        emb_te, X_dense_te, y_te = _to_emb_dict(te_arrays[:-2]), te_arrays[-2], te_arrays[-1]

        self.train_info_ = {
            **self._preprocess_info,
            "n_train":            len(y_tr),
            "n_val":              len(y_va),
            "n_test":             len(y_te),
            "n_features_dense":   X_dense.shape[1],
            "n_groups_embedding": self.n_groups_,
            "n_horas_embedding":    self.n_horas_,
            "n_meses_embedding":    self.n_meses_,
            "n_periodos_embedding": self.n_periodos_,
        }

        _log_block("TREINAMENTO  [Wide & Deep + Entity Embeddings]")
        _logger.info(
            "treino=%d | val=%d | teste=%d",
            len(y_tr), len(y_va), len(y_te),
        )
        _logger.info(
            "features_densas=%d | embeddings: grupo(%d→%dd) hora(%d→%dd) mes(%d→%dd) periodo(%d→%dd)",
            X_dense.shape[1],
            self.n_groups_,   cfg.embedding_dim,
            self.n_horas_,    cfg.embedding_dim_hora,
            self.n_meses_,    cfg.embedding_dim_mes,
            self.n_periodos_, cfg.embedding_dim_periodo,
        )
        _logger.info(
            "hidden=%s | dropout=%.0f%%→%.0f%% | loss=%s | lr=%.4f | batch=%d",
            " -> ".join(str(u) for u in cfg.hidden_units),
            cfg.dropout_rate * 100,
            cfg.min_dropout_rate * 100,
            cfg.loss, cfg.learning_rate, cfg.batch_size,
        )

        # -- compilação -------------------------------------------------------
        self.model_ = self._build_and_compile(X_dense.shape[1])

        # -- callbacks --------------------------------------------------------
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=cfg.patience_stop,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=cfg.reduce_lr_factor,
                patience=cfg.patience_lr,
                min_lr=cfg.min_lr,
                verbose=1,
            ),
        ]

        # -- treino -----------------------------------------------------------
        history = self.model_.fit(
            x={**emb_tr, "dense_features": X_dense_tr},
            y=y_tr,
            validation_data=(
                {**emb_va, "dense_features": X_dense_va},
                y_va,
            ),
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            callbacks=callbacks,
            verbose=1,
        )
        self.history_   = {k: [float(v) for v in vals]
                           for k, vals in history.history.items()}
        
        # Calcula estatísticas de normalização a partir dos dados de treino
        self._normalization_stats_ = compute_normalization_stats(
            X_dense_tr,
            feature_names=feature_columns,
        )
        
        self._is_fitted = True

        # -- avaliação no teste -----------------------------------------------
        y_pred = self.model_.predict(
            {**emb_te, "dense_features": X_dense_te},
            verbose=0,
        ).flatten()

        n_total = sum(len(X_emb[k]) for k in emb_keys[:1])  # todos têm mesmo len
        test_pct = len(y_te) / n_total * 100 if n_total else 0
        self.metrics_ = _compute_metrics(y_te, y_pred)
        _log_metrics(self.metrics_, header=f"AVALIACAO  (test={test_pct:.0f}%)")
        return self

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Prediz consumo_kwh para novos dados no schema inicial.

        Toda a conversão de inputs é delegada a ``DLNormalizer.transform()``
        (módulo ``tools/normalizer.py``), que reproduz fielmente o fluxo de
        ``DLSchema.build()`` + sanitização + separação Embeddings/Dense,
        usando os metadados do artefato treinado.

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
            raise RuntimeError("Modelo nao treinado. Execute fit() antes de predict().")
        if _TARGET in df.columns:
            raise ValueError(f"Remova a coluna '{_TARGET}' do DataFrame de entrada.")

        normalizer = DLNormalizer(
            feature_columns=self.feature_columns_,
            n_groups=self.n_groups_,
            n_horas=self.n_horas_,
            n_meses=self.n_meses_,
            n_periodos=self.n_periodos_,
            clipping_limits=self._clipping_limits,
        )
        inputs = normalizer.transform(df)

        return self.model_.predict(inputs, verbose=0).flatten()

    # -- estágio 5 — persistência ---------------------------------------------

    def save(self, path: str | Path) -> None:
        """
        Salva o pipeline em um diretório.

        Estrutura gerada:
            {path}/
                keras_model.keras        — modelo Keras formato nativo
                meta.json                — feature_columns, n_groups, config, metrics
                metadata_norm.json       — parâmetros de normalização para inferência

        Args:
            path: Diretorio de saida (criado automaticamente).

        Raises:
            RuntimeError: Modelo nao treinado.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() antes de save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # modelo Keras — formato nativo .keras (recomendado a partir do Keras 3)
        tf.keras.models.save_model(self.model_, str(path / "keras_model.keras"))

        # metadados — converte Path para str para serialização JSON
        cfg_dict = {
            k: str(v) if isinstance(v, Path) else v
            for k, v in asdict(self.config).items()
        }
        meta = {
            "feature_columns": self.feature_columns_,
            "n_groups":        self.n_groups_,
            "n_horas":         self.n_horas_,
            "n_meses":         self.n_meses_,
            "n_periodos":      self.n_periodos_,
            "metrics":         self.metrics_,
            "train_info":      self.train_info_,
            "config":          cfg_dict,
        }
        with (path / "meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        # Exporta metadata de normalização com estatísticas de treino
        _logger.info("Exportando parametros de normalização...")
        metadata_norm = {
            "version": "1.0",
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "dense_features": self._normalization_stats_ or {},
            "target_encoding": self._te_map or {},  # ✅ Compatibilidade futura para TE
            "clipping_limits": self._clipping_limits or {},  # ✅ Novo: persistir limites de clipping
            "embeddings": {
                "grupo_regional": {
                    "input_dim": self.n_groups_,
                    "value_range": [0, self.n_groups_ - 1],
                },
                "hora": {
                    "input_dim": self.n_horas_,
                    "value_range": [0, self.n_horas_ - 1],
                },
                "mes": {
                    "input_dim": self.n_meses_,
                    "value_range": [1, self.n_meses_ - 1],
                },
                "periodo_dia": {
                    "input_dim": self.n_periodos_,
                    "value_range": [0, self.n_periodos_ - 1],
                },
            },
            "metrics": self.metrics_,
            "train_info": self.train_info_,
        }
        with (path / "metadata_norm.json").open("w", encoding="utf-8") as fh:
            json.dump(metadata_norm, fh, indent=2, default=str)

        _logger.info("Pipeline salvo em: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "DLPipeline":
        """
        Carrega um pipeline salvo do disco.

        Args:
            path: Diretorio criado por save().

        Returns:
            DLPipeline pronto para predict().
        """
        path = Path(path)
        with (path / "meta.json").open(encoding="utf-8") as fh:
            meta = json.load(fh)

        cfg_raw                  = meta["config"]
        cfg_raw["artifacts_dir"] = Path(cfg_raw.get("artifacts_dir", "model/artifacts"))
        config = DLPipelineConfig(**cfg_raw)

        pipeline                   = cls(config=config)
        pipeline.model_            = tf.keras.models.load_model(str(path / "keras_model.keras"))
        pipeline.feature_columns_  = meta["feature_columns"]
        pipeline.n_groups_         = meta["n_groups"]
        pipeline.n_horas_          = meta.get("n_horas", 24)
        pipeline.n_meses_          = meta.get("n_meses", 13)
        pipeline.n_periodos_       = meta.get("n_periodos", 4)
        pipeline.metrics_          = meta["metrics"]
        pipeline.train_info_       = meta.get("train_info", {})
        pipeline._is_fitted        = True
        
        # ✅ Carregar Target Encoding map se disponível (compatibilidade futura)
        with (path / "metadata_norm.json").open(encoding="utf-8") as fh:
            metadata_norm = json.load(fh)
            pipeline._clipping_limits = metadata_norm.get("clipping_limits") or None  # ✅ Novo: carregar clipping limits

        _logger.info("Pipeline carregado de: %s", path)
        return pipeline


# ══════════════════════════════════════════════════════════════════════════════
#  PRÉ-ETAPA 2 — PIPELINE SEGMENTADO POR TIPO DE MÁQUINA
# ══════════════════════════════════════════════════════════════════════════════

class SegmentedDLPipeline:
    """
    Pré-etapa 2: treina um DLPipeline independente por valor único de
    'machine_type', capturando padrões específicos de cada tecnologia HVAC
    (Split Hi-Wall, Splitão, Chiller, Self-Contained, VRF, etc.).

    Justificativa
    ─────────────
    Cada tecnologia HVAC possui perfil energético distinto:
      • Split Hi-Wall  → 1–5 kWh   (residencial / pequeno comercial)
      • Splitão Rooftop→ 5–25 kWh  (médio comercial)
      • Chiller Água   → 20–150 kWh (grandes instalações)
    Um modelo único tende a subajustar nos extremos da distribuição.
    Modelos segmentados capturam padrões intragrupo com maior precisão.

    A pré-etapa 1 (_filter_outliers) é aplicada automaticamente dentro de
    cada DLPipeline.fit() via _preprocess(), usando os thresholds do config.

    Ponto de entrada: ``SegmentedDLPipeline(config=cfg).fit(df_raw)``

    Attributes:
        segments_ : Dicionário {machine_type → DLPipeline treinado}.
        metrics_  : Dicionário {machine_type → métricas do segmento}.
    """

    def __init__(self, config: DLPipelineConfig | None = None) -> None:
        self.config      = config or DLPipelineConfig()
        self.segments_:  dict[str, DLPipeline]        = {}
        self.metrics_:   dict[str, dict[str, float]]  = {}
        self.skipped_segments_: dict[str, str]        = {}
        self._is_fitted: bool                         = False

    # -- treinamento ----------------------------------------------------------

    def fit(self, df: pl.DataFrame) -> "SegmentedDLPipeline":
        """
        Treina um DLPipeline por valor único de 'machine_type'.

        A pré-etapa 1 (filtro de outliers por segmento) é aplicada
        automaticamente dentro de cada DLPipeline via _preprocess().

        Args:
            df: DataFrame no schema inicial (14 colunas brutas).

        Returns:
            Self.
        """
        # Normaliza machine_type ANTES da segmentação para consolidar rótulos
        # equivalentes (ex: 'split cassete' e 'split-cassete' → 'SPLIT CASSETE').
        # O df original (com machine_type bruto) é preservado e repassado ao
        # DLPipeline, que executa ModelSchema.adjust_machine_type() internamente.
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
        _log_block(
            f"SEGMENTACAO  [pré-etapa 2 — {len(machine_types)} tipos de máquina]"
        )
        for mt in machine_types:
            n = int((df["_norm_type"] == mt).sum())
            _logger.info("  → '%s'  (%d registros)", mt, n)

        skipped: dict[str, str] = {}
        for mt in machine_types:
            df_seg             = df.filter(pl.col("_norm_type") == mt).drop("_norm_type")
            _log_block(f"SEGMENTO DL  '{mt}'")
            pipeline           = DLPipeline(config=self.config)
            try:
                pipeline.fit(df_seg)
            except ValueError as exc:
                reason = str(exc)
                skipped[mt] = reason
                _logger.warning(
                    "Segmento '%s' ignorado: %s",
                    mt,
                    reason,
                )
                continue
            self.segments_[mt] = pipeline
            self.metrics_[mt]  = pipeline.metrics_

        self.skipped_segments_ = skipped
        if skipped:
            _log_block(f"SEGMENTOS DL PULADOS  [n={len(skipped)}]")
            for mt, reason in skipped.items():
                _logger.warning("  - %s | motivo: %s", mt, reason)
        else:
            _logger.info("Nenhum segmento foi pulado na etapa segmentada DL.")

        if not self.segments_:
            raise RuntimeError(
                "Nenhum segmento treinável após filtro de outliers e schema. "
                "Revise parâmetros de limpeza (noise_floor/iqr_factor/segment_params)."
            )

        self._is_fitted = True
        self._log_summary()
        return self

    def _log_summary(self) -> None:
        """Exibe tabela comparativa de métricas por segmento."""
        col_w = 30
        _log_block("RESUMO SEGMENTADO  [DL]")
        _logger.info(
            "  %-*s %10s %10s %8s %9s %11s",
            col_w, "Segmento", "MAE", "RMSE", "R2", "WMAPE", "Acuracia",
        )
        _logger.info("  " + "-" * (col_w + 52))
        for mt, m in self.metrics_.items():
            _logger.info(
                "  %-*s %10.4f %10.4f %8.4f %8.2f%% %10.2f%%",
                col_w, mt, m["MAE"], m["RMSE"], m["R2"], m["WMAPE"], m["Acuracia"],
            )

    # -- relatório consolidado ------------------------------------------------

    def _save_report(self, path: Path) -> None:
        """
        Gera e persiste o raio-x da modelagem segmentada em JSON.

        Arquivo: {path}/dl_model_report.json

        Estrutura
        ---------
        timestamp    : data/hora de geração (ISO-8601)
        n_segments   : quantidade de modelos segmentados treinados
        consolidated : totais de instâncias (n_train, n_val, n_test)
                       + métricas ponderadas por n_test
                       (MAE, RMSE, WMAPE, Acuracia)
        segmented    : por machine_type — n_train, n_val, n_test,
                       MAE, RMSE, R2, WMAPE, Acuracia

        Args:
            path: Diretório onde dl_model_report.json será gravado.
        """
        segments_data: dict = {}
        n_test_total = n_train_total = n_val_total = 0

        for mt, pipeline in self.segments_.items():
            ti  = pipeline.train_info_
            met = pipeline.metrics_
            n_t = ti.get("n_test", 0)
            segments_data[mt] = {
                "n_train": ti.get("n_train", 0),
                "n_val":   ti.get("n_val",   0),
                "n_test":  n_t,
                **{k: round(v, 6) for k, v in met.items()},
            }
            n_test_total  += n_t
            n_train_total += ti.get("n_train", 0)
            n_val_total   += ti.get("n_val",   0)

        def _wavg(key: str) -> float:
            if n_test_total == 0:
                return float("nan")
            return round(
                sum(
                    p.metrics_[key] * p.train_info_.get("n_test", 0)
                    for p in self.segments_.values()
                ) / n_test_total,
                6,
            )

        report = {
            "timestamp":  datetime.datetime.now().isoformat(timespec="seconds"),
            "n_segments": len(self.segments_),
            "consolidated": {
                "n_train":  n_train_total,
                "n_val":    n_val_total,
                "n_test":   n_test_total,
                "MAE":      _wavg("MAE"),
                "RMSE":     _wavg("RMSE"),
                "WMAPE":    _wavg("WMAPE"),
                "Acuracia": _wavg("Acuracia"),
            },
            "segmented": segments_data,
        }

        path.mkdir(parents=True, exist_ok=True)
        report_path = path / "dl_model_report.json"
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        _logger.info("Relatorio consolidado salvo em: %s", report_path)

    # -- inferência -----------------------------------------------------------

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Roteia cada linha ao DLPipeline do respectivo 'machine_type'.

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
        df_with_norm = df.with_columns(_norm_series).with_columns(
            pl.when(
                pl.col("_norm_type").is_null()
                | (pl.col("_norm_type").cast(pl.String).str.strip_chars() == "")
            )
            .then(pl.lit("DESCONHECIDO"))
            .otherwise(pl.col("_norm_type").cast(pl.String))
            .alias("_norm_type")
        )

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
        Salva todos os pipelines de segmento e um manifesto JSON.

        Estrutura gerada:
            {path}/
                manifest.json           — mapeamento segmento → diretório + métricas
                segment_{mt}/
                    keras_model.keras
                    meta.json

        Args:
            path: Diretório de saída (criado automaticamente).

        Raises:
            RuntimeError: Modelo não treinado.
        """
        if not self._is_fitted:
            raise RuntimeError("Modelo nao treinado. Execute fit() antes de save().")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        segment_dirs: dict[str, str] = {}
        for mt, pipeline in self.segments_.items():
            safe  = mt.replace("/", "_").replace(" ", "_")
            dname = f"segment_{safe}"
            pipeline.save(path / dname)
            segment_dirs[mt] = dname

        manifest = {"segment_dirs": segment_dirs, "metrics": self.metrics_}
        with (path / "manifest.json").open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, default=str)

        self._save_report(path)
        _logger.info("SegmentedDLPipeline salvo em: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "SegmentedDLPipeline":
        """
        Carrega um SegmentedDLPipeline salvo do disco.

        Args:
            path: Diretório criado por save().

        Returns:
            SegmentedDLPipeline pronto para predict().
        """
        path = Path(path)
        with (path / "manifest.json").open(encoding="utf-8") as fh:
            manifest = json.load(fh)

        segmented = cls()
        for mt, dname in manifest["segment_dirs"].items():
            segmented.segments_[mt] = DLPipeline.load(path / dname)
        segmented.metrics_   = manifest["metrics"]
        segmented._is_fitted = True

        _logger.info("SegmentedDLPipeline carregado de: %s", path)
        return segmented


# ══════════════════════════════════════════════════════════════════════════════
#  CARREGAMENTO E VALIDACAO DE DADOS
# ══════════════════════════════════════════════════════════════════════════════

def _load_raw_df(csv_path: Path) -> pl.DataFrame:
    """
    Carrega o Parquet e aplica o fluxo completo de normalização do ModelSchema
    para validar e inspecionar os dados de entrada.

    Espelha a execução direta de ``model/pre_process/schema.py``:

        1. pl.read_parquet           — leitura do arquivo
        2. ModelSchema.__init__      — verifica colunas obrigatórias
        3. ModelSchema.build()       — pipeline completo de normalização
                                       (adjust_machine_type, features de data,
                                        OHE, Clipping+MinMax)
        4. Log de inspecão          — schema final, OHE geradas, estatísticas
                                       de consumo_kwh e tipos de máquina

    A funão retorna o DataFrame **bruto** (pré-build), pois o DLSchema
    re-executa o ModelSchema internamente com as transformações específicas
    da arquitetura Wide & Deep (Entity Embeddings Int32 para grupo_regional,
    hora e mes).

    Args:
        parquet_path: Caminho para o arquivo Parquet de entrada.

    Returns:
        pl.DataFrame no schema bruto, pronto para ser passado ao pipeline.

    Raises:
        SystemExit: Arquivo não encontrado ou colunas obrigatórias ausentes.
    """
    if not parquet_path.exists():
        _logger.error("Arquivo nao encontrado: %s", parquet_path)
        sys.exit(1)

    df_raw = pl.read_parquet(parquet_path)
    _logger.info(
        "✔ %d registros carregados | %d colunas  ←  %s",
        df_raw.shape[0], df_raw.shape[1], csv_path.name,
    )

    # —— etapa 2+3: validação + pipeline completo ModelSchema ————————————
    _log_block("VALIDACAO DO SCHEMA  (ModelSchema.build)")
    try:
        df_schema = ModelSchema(df_raw, _SCHEMA_FIELDS).build()
    except ValueError as exc:
        _logger.error("Schema invalido: %s", exc)
        sys.exit(1)

    # —— etapa 4: log de inspecão ———————————————————————————————
    ohe_prefixes = ("tipo_maquina_", "estacao_", "periodo_dia_")
    ohe_cols     = [c for c in df_schema.columns if c.startswith(ohe_prefixes)]

    _logger.info("Schema final: %d colunas  |  OHE geradas: %d",
                 df_schema.shape[1], len(ohe_cols))
    _logger.info("Coluna 'data' removida:         %s",
                 "data" not in df_schema.columns)
    _logger.info("Coluna 'machine_type' removida: %s",
                 "machine_type" not in df_schema.columns)

    # tipos de máquina após adjust_machine_type
    maquinas = sorted(df_raw["machine_type"].unique().to_list())
    _logger.info("%d tipo(s) de maquina detectado(s):", len(maquinas))
    for mt in maquinas:
        n = int((df_raw["machine_type"] == mt).sum())
        _logger.info("  %-40s  n=%d", mt, n)

    # estatísticas do target
    kwh = df_raw[_TARGET]
    _logger.info(
        "consumo_kwh: min=%.3f  p25=%.3f  median=%.3f  p75=%.3f  max=%.3f  mean=%.3f",
        float(kwh.min()), float(kwh.quantile(0.25)), float(kwh.median()),
        float(kwh.quantile(0.75)), float(kwh.max()), float(kwh.mean()),
    )

    # OHE geradas (amostra das 10 primeiras)
    _logger.info("Colunas OHE geradas (%d): %s%s",
                 len(ohe_cols),
                 ", ".join(ohe_cols[:10]),
                 " ..." if len(ohe_cols) > 10 else "")

    return df_raw


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUCAO DIRETA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    parquet_path = Path(r"use_case\files\final_dataframe.parquet")
    df_raw   = _load_raw_df(parquet_path)

    # -- 1. Configuracao ------------------------------------------------------
    cfg = DLPipelineConfig(
        embedding_dim=8,          # grupo_regional
        embedding_dim_hora=4,     # hora (0-23)        → Entity Embedding
        embedding_dim_mes=3,      # mes  (1-12)        → Entity Embedding
        embedding_dim_periodo=2,  # periodo_dia (0-3)  → Entity Embedding
        hidden_units=[256, 128, 64],
        dropout_rate=0.30,
        min_dropout_rate=0.10,
        epochs=200,
        batch_size=512,
        learning_rate=0.001,
        loss="huber",
        patience_stop=15,
        patience_lr=5,
        reduce_lr_factor=0.5,
        test_size=0.15,
        val_size=0.05,
        random_state=42,
        # Pré-etapa 1 — idêntica ao ML pipeline (__main__)
        noise_floor=0.59,
        iqr_factor=1.75,
        min_segment_size=35,
        noise_quantile=0.10,
        upper_quantile_cap=0.90,
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

    # -- 2-4. Pre-processamento (pré-etapa 1 integrada) -> Treino -> Avaliacao --
    pipe = DLPipeline(config=cfg)
    pipe.fit(df_raw)

    # -- 5. Persistencia (modelo global) --------------------------------------
    # Salva em subpasta dedicada para não misturar com artefatos segmentados.
    pipe.save(cfg.artifacts_dir / "dl_hvac" / "global")

    # -- 6. Pipeline segmentado (pré-etapa 2) ---------------------------------
    seg = SegmentedDLPipeline(config=cfg)
    seg.fit(df_raw)

    # -- 7. Persistencia (segmentado) -----------------------------------------
    seg.save(cfg.artifacts_dir / "dl_hvac")

    # -- 8. Demo: predição das 10 primeiras linhas (global vs segmentado) -----
    _log_block("DEMO — Predição DL das 10 primeiras linhas do dataset")

    # Aplica o mesmo filtro de outliers usado no treino para que o demo
    # contenha apenas registros representativos (sem ruído / extremos).
    df_clean = _filter_outliers(
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

    # Predição GLOBAL (modelo Wide & Deep treinado em todo o dataset)
    y_global = pipe.predict(df_input)

    # Predição SEGMENTADA (cada linha roteada ao modelo DL do seu machine_type)
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
    _demo_metrics_global = _compute_metrics(y_real, y_global)
    _demo_metrics_seg    = _compute_metrics(y_real, y_seg)

    _logger.info("")
    _logger.info(
        "  %-*s  %12s  %12.4f  %12.4f",
        4 + 2 + col_w_mt, "MAE (10 linhas):", "",
        _demo_metrics_global["MAE"],
        _demo_metrics_seg["MAE"],
    )
    _logger.info(
        "  %-*s  %12s  %12.4f  %12.4f",
        4 + 2 + col_w_mt, "RMSE (10 linhas):", "",
        _demo_metrics_global["RMSE"],
        _demo_metrics_seg["RMSE"],
    )
    _logger.info(
        "  %-*s  %12s  %12.4f  %12.4f",
        4 + 2 + col_w_mt, "R²  (10 linhas):", "",
        _demo_metrics_global["R2"],
        _demo_metrics_seg["R2"],
    )
    _logger.info("")
    _logger.info("Demo de predição DL finalizada ✔")

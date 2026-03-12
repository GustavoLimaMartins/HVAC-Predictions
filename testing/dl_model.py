"""
dl_models.py — Utilitário de predição DL para modelos HVAC
==========================================================

Disponibiliza a função ``predict_dl()`` que recebe dados no formato do
schema inicial (colunas brutas do ``final_dataframe.csv`` **sem** a coluna
``consumo_kwh``) e retorna as predições do modelo global e,
opcionalmente, de um modelo segmentado específico.

Toda a conversão de inputs é delegada ao módulo ``tools/normalizer.py``
(classe ``DLNormalizer``), que reproduz fielmente o fluxo de
``DLSchema.build()`` + sanitização + separação Embeddings/Dense — sem
duplicar lógica.

Schema inicial esperado (13 colunas, sem ``consumo_kwh``):
    hora, data, machine_type, estacao, grupo_regional,
    Temperatura_C, Temperatura_Percebida_C, Umidade_Relativa_%,
    Precipitacao_mm, Velocidade_Vento_kmh, Pressao_Superficial_hPa,
    is_dac, is_dut

Uso rápido:
    >>> from testing.dl_models import predict_dl, list_segments
    >>> list_segments()                                 # segmentos disponíveis
    >>> result = predict_dl(df_input)                   # só modelo global
    >>> result = predict_dl(df_input, segment="SPLIT HI-WALL")  # global + segmentado
    >>> result["global"]                                # np.ndarray de predições
    >>> result["segment"]                               # np.ndarray (ou None)

Execução direta (demo com 10 amostras):
    python testing/dl_models.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ── path de importação ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.dl_pipeline import (
    DLPipeline,
    SegmentedDLPipeline,
    _filter_outliers,
)
from model.pre_process.schema import ModelSchema
from tools.normalizer import DLNormalizer

# ── constantes ──────────────────────────────────────────────────────────────
_TARGET: str = "consumo_kwh"

_INPUT_FIELDS: list[str] = [
    "hora", "data", "machine_type", "estacao", "grupo_regional",
    "Temperatura_C", "Temperatura_Percebida_C",
    "Umidade_Relativa_%", "Precipitacao_mm",
    "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
    "is_dac", "is_dut",
]

_ROOT = Path(__file__).resolve().parent.parent
_GLOBAL_DIR = _ROOT / "model" / "artifacts" / "dl_hvac" / "global"
_SEGMENTED_DIR = _ROOT / "model" / "artifacts" / "dl_hvac"

_logger = logging.getLogger(__name__)

# ── cache de modelos (carregados uma única vez) ─────────────────────────────
_global_pipe: DLPipeline | None = None
_segmented_pipe: SegmentedDLPipeline | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS INTERNOS
# ═══════════════════════════════════════════════════════════════════════════

def _load_global() -> DLPipeline:
    """Carrega (ou retorna do cache) o modelo global."""
    global _global_pipe
    if _global_pipe is None:
        if not _GLOBAL_DIR.exists():
            raise FileNotFoundError(
                f"Artefato do modelo global não encontrado: {_GLOBAL_DIR}"
            )
        _global_pipe = DLPipeline.load(_GLOBAL_DIR)
    return _global_pipe


def _load_segmented() -> SegmentedDLPipeline:
    """Carrega (ou retorna do cache) o pipeline segmentado."""
    global _segmented_pipe
    if _segmented_pipe is None:
        if not _SEGMENTED_DIR.exists():
            raise FileNotFoundError(
                f"Artefatos segmentados não encontrados: {_SEGMENTED_DIR}"
            )
        _segmented_pipe = SegmentedDLPipeline.load(_SEGMENTED_DIR)
    return _segmented_pipe


def _validate_input(df: pl.DataFrame) -> None:
    """Valida que o DataFrame de entrada possui as colunas esperadas."""
    if _TARGET in df.columns:
        raise ValueError(
            f"O DataFrame de entrada não deve conter a coluna target "
            f"'{_TARGET}'. Remova-a antes de chamar predict_dl()."
        )
    missing = set(_INPUT_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes no DataFrame de entrada: "
            f"{sorted(missing)}\n"
            f"Colunas esperadas: {_INPUT_FIELDS}"
        )





# ═══════════════════════════════════════════════════════════════════════════
#  API PÚBLICA
# ═══════════════════════════════════════════════════════════════════════════

def list_segments() -> list[str]:
    """
    Retorna os nomes dos segmentos (``tipo_maquina`` normalizado)
    disponíveis no pipeline segmentado salvo em disco.

    Returns:
        Lista ordenada de nomes de segmento.

    Example:
        >>> list_segments()
        ['AR CONDICIONADO DE JANELA (ACJ)', 'SPLIT CASSETE', 'SPLIT DUTO', ...]
    """
    seg_pipe = _load_segmented()
    return sorted(seg_pipe.segments_.keys())


def predict_dl(
    df: pl.DataFrame,
    segment: str | None = None,
) -> dict[str, np.ndarray | None]:
    """
    Realiza predições de consumo HVAC (kWh) usando os modelos DL salvos.

    O DataFrame de entrada deve seguir o schema inicial (13 colunas brutas,
    **sem** ``consumo_kwh``). A conversão de inputs é delegada ao
    ``DLNormalizer`` (módulo ``tools/normalizer.py``), que reproduz
    fielmente o fluxo de ``DLSchema.build()`` + sanitização + separação
    Embeddings/Dense — idêntico ao usado no treino.

    Parameters
    ----------
    df : pl.DataFrame
        Dados de entrada no schema inicial:
            hora, data, machine_type, estacao, grupo_regional,
            Temperatura_C, Temperatura_Percebida_C, Umidade_Relativa_%,
            Precipitacao_mm, Velocidade_Vento_kmh, Pressao_Superficial_hPa,
            is_dac, is_dut
    segment : str | None
        Nome do segmento (``tipo_maquina`` normalizado) para predição
        segmentada. Se ``None``, retorna apenas a predição global.
        Apenas linhas cujo ``machine_type`` normalizado coincide com o
        segmento serão preditas; as demais recebem ``NaN``.
        Use ``list_segments()`` para ver os nomes válidos.

    Returns
    -------
    dict com:
        ``"global"``  : np.ndarray — predições do modelo global (sempre presente).
        ``"segment"`` : np.ndarray | None — predições do modelo segmentado
                        (somente para linhas do tipo correto; demais = NaN).
        ``"segment_name"`` : str | None — nome do segmento utilizado.

    Raises
    ------
    ValueError
        Se o DataFrame não tiver as colunas esperadas, contiver ``consumo_kwh``,
        ou se ``segment`` não estiver entre os disponíveis.
    FileNotFoundError
        Se os artefatos de modelo não existirem em disco.

    Example
    -------
    >>> import polars as pl
    >>> from testing.dl_models import predict_dl, list_segments
    >>>
    >>> df_input = pl.read_csv("use_case/files/final_dataframe.csv").head(10).drop("consumo_kwh")
    >>> result = predict_dl(df_input, segment="SPLIT HI-WALL")
    >>> result["global"]     # array de 10 predições (modelo global)
    >>> result["segment"]    # array de 10 predições (modelo segmentado)
    """
    # ── validação ────────────────────────────────────────────────────────
    _validate_input(df)

    if segment is not None:
        seg_pipe = _load_segmented()
        available = sorted(seg_pipe.segments_.keys())
        if segment not in seg_pipe.segments_:
            raise ValueError(
                f"Segmento '{segment}' não encontrado.\n"
                f"Segmentos disponíveis: {available}"
            )

    # ── predição global (DLNormalizer aplicado internamente) ─────────────
    global_pipe = _load_global()
    y_global = global_pipe.predict(df)
    _logger.info("Predição global concluída: %d amostras", len(y_global))

    # ── predição segmentada (opcional) ───────────────────────────────────
    # O modelo segmentado só prediz linhas cujo tipo de máquina normalizado
    # coincide com o segmento solicitado — as demais recebem NaN.
    y_segment: np.ndarray | None = None
    if segment is not None:
        seg_model = seg_pipe.segments_[segment]

        # 1. Filtra pelo tipo de máquina: só linhas do segmento correto
        norm_types = (
            ModelSchema(df.select("machine_type"), ["machine_type"])
            .adjust_machine_type()
            .df["tipo_maquina"]
        )
        type_mask = (norm_types == segment).to_numpy()

        # 2. Valida grupo_regional dentro dos limites do Embedding treinado
        #    (hora 0-23 < 24, mes 1-12 < 13, periodo 0-3 < 4 — sempre válidos)
        gr_arr    = df["grupo_regional"].to_numpy()
        max_group = int(seg_model.n_groups_)
        emb_mask  = (gr_arr >= 0) & (gr_arr < max_group)

        # 3. Combina: tipo correto + Embeddings válidos
        valid_mask = type_mask & emb_mask

        n_wrong_type  = int((~type_mask).sum())
        n_invalid_emb = int(type_mask.sum() - valid_mask.sum())

        if n_wrong_type > 0:
            _logger.info(
                "Segmento '%s': %d de %d linhas ignoradas (machine_type diferente).",
                segment, n_wrong_type, len(df),
            )
        if n_invalid_emb > 0:
            _logger.warning(
                "Segmento '%s': %d linhas do tipo correto com grupo_regional "
                "fora do range treinado (>=%d) — serão preditas como NaN.",
                segment, n_invalid_emb, max_group,
            )

        # 4. Prediz apenas linhas válidas (DLNormalizer aplicado internamente)
        y_segment = np.full(len(df), np.nan, dtype=np.float32)
        if valid_mask.any():
            df_valid = df.filter(pl.Series(valid_mask))
            y_segment[valid_mask] = seg_model.predict(df_valid)

        _logger.info(
            "Predição segmentada '%s' concluída: %d/%d amostras preditas "
            "(%d tipo correto, %d ignoradas por tipo diferente)",
            segment, int(valid_mask.sum()), len(df),
            int(type_mask.sum()), n_wrong_type,
        )

    return {
        "global": y_global,
        "segment": y_segment,
        "segment_name": segment,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO DIRETA — DEMO COM AMOSTRAS ALEATÓRIAS DO DATASET
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _SAMPLE_N = 100
    _SEED = 42

    # ── carrega dataset e amostra aleatória ─────────────────────────────
    csv_path = _ROOT / "use_case" / "files" / "final_dataframe.csv"
    if not csv_path.exists():
        _logger.error("CSV não encontrado: %s", csv_path)
        sys.exit(1)

    df_raw = pl.read_csv(csv_path)
    _logger.info("Dataset carregado: %d linhas", len(df_raw))

    # ── tratamento de outliers (idêntico ao usado no treinamento DL) ───
    _OUTLIER_PARAMS = {
        "noise_floor": 0.7,
        "iqr_factor": 1.5,
        "min_segment_size": 50,
        "noise_quantile": 0.08,
        "upper_quantile_cap": 0.99,
        "segment_params": {
            "AR CONDICIONADO DE JANELA (ACJ)": {
                "iqr_factor": 1.15,
                "noise_floor": 0.59,
                "upper_quantile_cap": 0.85,
                "noise_quantile": 0.15,
            },
            "SPLIT CASSETE": {
                "iqr_factor": 1.20,
                "noise_floor": 1.59,
                "upper_quantile_cap": 0.85,
                "noise_quantile": 0.15,
            },
            "SPLIT DUTO": {
                "iqr_factor": 1.25,
                "noise_floor": 1.69,
                "upper_quantile_cap": 0.85,
                "noise_quantile": 0.15,
            },
            "SPLIT HI-WALL": {
                "iqr_factor": 1.25,
                "noise_floor": 0.79,
                "upper_quantile_cap": 0.85,
                "noise_quantile": 0.15,
            },
            "SPLIT PISO-TETO": {
                "iqr_factor": 1.25,
                "noise_floor": 1.59,
                "upper_quantile_cap": 0.85,
                "noise_quantile": 0.15,
            },
            "SPLITÃO": {
                "iqr_factor": 1.5,
                "noise_floor": 5.49,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.10,
            },
            "SPLITÃO INVERTER": {
                "iqr_factor": 1.35,
                "noise_floor": 3.49,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.10,
            },
            "SPLITÃO ROOFTOP": {
                "iqr_factor": 1.5,
                "noise_floor": 5.99,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.10,
            },
            "SPLITÃO SELF CONTAINED": {
                "iqr_factor": 1.5,
                "noise_floor": 5.49,
                "upper_quantile_cap": 0.95,
                "noise_quantile": 0.10,
            },
        },
    }

    n_before = len(df_raw)
    df_full = _filter_outliers(df_raw, **_OUTLIER_PARAMS)
    n_after = len(df_full)
    _logger.info(
        "Outliers removidos: %d → %d linhas (-%d, -%.1f%%)",
        n_before, n_after, n_before - n_after,
        (n_before - n_after) / n_before * 100 if n_before else 0.0,
    )

    n = min(_SAMPLE_N, len(df_full))
    df_sample = df_full.sample(n=n, seed=_SEED)

    y_true = df_sample[_TARGET].to_numpy().astype(np.float32)
    df_input = df_sample.drop(_TARGET)

    _logger.info("Amostra aleatória (pós-outlier): %d linhas (seed=%d)", n, _SEED)
    _logger.info("\n%s", df_input.columns)

    # ══════════════════════════════════════════════════════════════════════
    #  TESTE INICIAL — 15 primeiras linhas do final_dataframe (real vs pred)
    # ══════════════════════════════════════════════════════════════════════
    _HEAD_N = 15
    df_head = df_raw.head(_HEAD_N)
    y_head_true = df_head[_TARGET].to_numpy().astype(np.float32)
    df_head_input = df_head.drop(_TARGET)

    # Normaliza machine_type para exibição e escolha de segmento
    _head_norm_types = (
        ModelSchema(df_head_input.select("machine_type"), ["machine_type"])
        .adjust_machine_type()
        .df["tipo_maquina"]
    )

    _logger.info("\n" + "═" * 100)
    _logger.info("  TESTE INICIAL — 15 primeiras linhas do final_dataframe.csv (Real vs Predito)")
    _logger.info("═" * 100)

    # Predição global
    _head_result = predict_dl(df_head_input)
    y_head_global = _head_result["global"]

    # Predição segmentada (cada linha com seu próprio segmento)
    y_head_seg = np.full(_HEAD_N, np.nan, dtype=np.float32)
    for _seg_name in sorted(set(_head_norm_types.to_list())):
        _seg_mask = (_head_norm_types == _seg_name).to_numpy()
        if not _seg_mask.any():
            continue
        try:
            _seg_result = predict_dl(df_head_input, segment=_seg_name)
            if _seg_result["segment"] is not None:
                _finite = np.isfinite(_seg_result["segment"])
                y_head_seg[_finite] = _seg_result["segment"][_finite]
        except Exception as _e:
            _logger.warning("  Segmento '%s' falhou: %s", _seg_name, _e)

    _logger.info(
        "  %3s  %-7s %-12s %-22s %10s %10s %10s %10s %8s",
        "#", "hora", "data", "machine_type", "Real", "Global", "Segmento", "Err_Glob", "Err%",
    )
    _logger.info("  " + "─" * 115)

    for _i in range(_HEAD_N):
        _real = y_head_true[_i]
        _glob = y_head_global[_i]
        _seg  = y_head_seg[_i]
        _err  = _glob - _real
        _pct  = abs(_err) / max(abs(_real), 1e-8) * 100
        _seg_str = f"{_seg:10.4f}" if np.isfinite(_seg) else "       N/A"
        _logger.info(
            "  %3d  %5d   %-12s %-22s %10.4f %10.4f %s %+10.4f %7.1f%%",
            _i,
            int(df_head["hora"][_i]),
            str(df_head["data"][_i]),
            str(df_head["machine_type"][_i]),
            _real, _glob, _seg_str, _err, _pct,
        )

    # Métricas resumo das 15 linhas
    _head_mae  = float(np.mean(np.abs(y_head_global - y_head_true)))
    _head_rmse = float(np.sqrt(np.mean((y_head_global - y_head_true) ** 2)))
    _head_denom = float(np.abs(y_head_true).sum())
    _head_wmape = float(np.abs(y_head_global - y_head_true).sum() / _head_denom * 100) if _head_denom > 0 else float("nan")

    _seg_finite = np.isfinite(y_head_seg)
    if _seg_finite.any():
        _seg_mae  = float(np.mean(np.abs(y_head_seg[_seg_finite] - y_head_true[_seg_finite])))
        _seg_rmse = float(np.sqrt(np.mean((y_head_seg[_seg_finite] - y_head_true[_seg_finite]) ** 2)))
        _seg_d = float(np.abs(y_head_true[_seg_finite]).sum())
        _seg_wmape = float(np.abs(y_head_seg[_seg_finite] - y_head_true[_seg_finite]).sum() / _seg_d * 100) if _seg_d > 0 else float("nan")
    else:
        _seg_mae = _seg_rmse = _seg_wmape = float("nan")

    _logger.info("  " + "─" * 115)
    _logger.info(
        "  Global    →  MAE=%.4f  RMSE=%.4f  WMAPE=%.2f%%  (%d linhas)",
        _head_mae, _head_rmse, _head_wmape, _HEAD_N,
    )
    _logger.info(
        "  Segmento  →  MAE=%.4f  RMSE=%.4f  WMAPE=%.2f%%  (%d linhas válidas)",
        _seg_mae, _seg_rmse, _seg_wmape, int(_seg_finite.sum()),
    )
    _logger.info("═" * 100 + "\n")

    # ══════════════════════════════════════════════════════════════════════
    #  VARREDURA VIA DLNormalizer.inspect()  —  validação estruturada
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "═" * 90)
    _logger.info("  VARREDURA DO DLNormalizer  —  Validação via inspect()")
    _logger.info("═" * 90)

    _dl_norm = DLNormalizer.from_artifact(_GLOBAL_DIR)
    _inspect = _dl_norm.inspect(df_input)

    _logger.info(
        "Shape pós-normalização: %d linhas × %d features densas + 4 embeddings",
        _inspect["n_rows"], _inspect["n_dense_features"],
    )

    # -- Entity Embeddings -----------------------------------------------
    _logger.info("\n" + "─" * 90)
    _logger.info("  ENTITY EMBEDDINGS — Detalhamento")
    _logger.info("─" * 90)
    _logger.info(
        "  %-18s %8s %8s %8s %10s %6s",
        "Embedding", "min", "max", "unique", "input_dim", "OOB",
    )
    _logger.info("  " + "─" * 62)
    for emb_name, emb_info in _inspect["embeddings"].items():
        oob_flag = " ⚠" if emb_info["oob"] > 0 else "  ✔"
        _logger.info(
            "  %-18s %8d %8d %8d %10d %5d%s",
            emb_name,
            emb_info["min"], emb_info["max"], emb_info["n_unique"],
            emb_info["input_dim"], emb_info["oob"], oob_flag,
        )

    # -- Features densas -------------------------------------------------
    _logger.info("\n" + "─" * 90)
    _logger.info("  FEATURES DENSAS — Estatísticas")
    _logger.info("─" * 90)
    _logger.info(
        "  %-3s  %-44s %10s %10s %6s",
        "#", "Feature", "Min", "Max", "NaN",
    )
    _logger.info("  " + "─" * 78)
    for idx, (feat_name, feat_info) in enumerate(_inspect["dense_stats"].items()):
        nan_flag = " ⚠" if feat_info["nan"] > 0 else ""
        _logger.info(
            "  %3d  %-44s %10.4f %10.4f %5d%s",
            idx, feat_name,
            feat_info["min"] if feat_info["min"] is not None else 0.0,
            feat_info["max"] if feat_info["max"] is not None else 0.0,
            feat_info["nan"], nan_flag,
        )

    # -- Checklist de sanidade -------------------------------------------
    _logger.info("\n" + "─" * 90)
    _logger.info("  CHECKLIST DE SANIDADE")
    _logger.info("─" * 90)

    _checks = []

    # 1. Nenhum warning do normalizer
    _checks.append((
        "Sem warnings do DLNormalizer",
        len(_inspect["warnings"]) == 0,
        f"{len(_inspect['warnings'])} warning(s)" if _inspect["warnings"] else "OK",
    ))
    for w in _inspect["warnings"]:
        _logger.warning("    ⚠ %s", w)

    # 2. Embeddings sem OOB
    for emb_name, emb_info in _inspect["embeddings"].items():
        _checks.append((
            f"Embedding '{emb_name}' sem OOB",
            emb_info["oob"] == 0,
            f"OOB={emb_info['oob']} (input_dim={emb_info['input_dim']})",
        ))

    # 3. Features densas sem NaN
    total_nan = sum(v["nan"] for v in _inspect["dense_stats"].values())
    _checks.append((
        "Features densas sem NaN",
        total_nan == 0,
        f"NaN={total_nan}",
    ))

    # 4. Dimensão esperada
    _checks.append((
        "n_dense_features correto",
        _inspect["n_dense_features"] == len(_dl_norm.feature_columns),
        f"{_inspect['n_dense_features']} (esperado {len(_dl_norm.feature_columns)})",
    ))

    for check_name, passed, detail in _checks:
        status = "✔" if passed else "✘"
        _logger.info("  %s  %-42s  %s", status, check_name, detail)

    n_passed = sum(1 for _, p, _ in _checks if p)
    n_total  = len(_checks)
    _logger.info(
        "\n  Resultado: %d/%d verificações OK%s",
        n_passed, n_total,
        "" if n_passed == n_total else "  ⚠ ATENÇÃO: há falhas!",
    )
    _logger.info("═" * 90 + "\n")

    # ── segmentos disponíveis ────────────────────────────────────────────
    segments = list_segments()
    _logger.info("Segmentos disponíveis: %s", segments)

    # ── normaliza machine_type para encontrar segmento predominante ──────
    norm_types = (
        ModelSchema(df_input.select("machine_type"), ["machine_type"])
        .adjust_machine_type()
        .df["tipo_maquina"]
    )
    df_input = df_input.with_columns(norm_types.alias("_norm_type"))

    # Escolhe o segmento com mais linhas na amostra
    type_counts = norm_types.value_counts().sort("count", descending=True)
    chosen_segment = str(type_counts["tipo_maquina"][0])
    _logger.info(
        "Segmento predominante na amostra: '%s' (%d linhas)",
        chosen_segment, int(type_counts["count"][0]),
    )

    # ── helpers de métricas ─────────────────────────────────────────────
    def _wmape_val(y_t: np.ndarray, y_p: np.ndarray) -> float:
        d = np.abs(y_t).sum()
        return float("nan") if d == 0 else float(np.abs(y_t - y_p).sum() / d * 100)

    def _mape_val(y_t: np.ndarray, y_p: np.ndarray) -> float:
        eps = 1e-8
        denom = np.maximum(np.abs(y_t), eps)
        return float(np.mean(np.abs(y_t - y_p) / denom) * 100)

    def _report(label: str, y_t: np.ndarray, y_p: np.ndarray) -> None:
        mask = np.isfinite(y_p)
        n_ok = int(mask.sum())
        if n_ok == 0:
            _logger.info("  %-20s  sem amostras válidas", label)
            return
        yt, yp = y_t[mask], y_p[mask]
        mae = mean_absolute_error(yt, yp)
        rmse = mean_squared_error(yt, yp) ** 0.5
        r2 = r2_score(yt, yp) if n_ok >= 2 else float("nan")
        mape = _mape_val(yt, yp)
        wmape = _wmape_val(yt, yp)
        _logger.info(
            "  %-20s  n=%3d  MAE=%.4f  RMSE=%.4f  R²=%.4f  MAPE=%.2f%%  WMAPE=%.2f%%",
            label, n_ok, mae, rmse, r2, mape, wmape,
        )

    # ── predição global sobre TODA a amostra (sem filtro) ──────────────
    result_global = predict_dl(df_input.drop("_norm_type"))
    y_global_all  = result_global["global"]

    # ── LOG 1: Predição Global — Real vs Predito (amostra completa) ─────
    _logger.info("\n══ MODELO GLOBAL — Real vs Predito (%d amostras) ══", n)
    _logger.info(
        "  %3s  %-22s %10s %10s %10s %9s",
        "#", "machine_type", "Real", "Predito", "Erro", "Erro%",
    )
    _logger.info("  %s", "─" * 80)

    for i in range(n):
        real = y_true[i]
        g = y_global_all[i]
        err = g - real
        pct = abs(err) / max(abs(real), 1e-8) * 100
        _logger.info(
            "  %3d  %-22s %10.4f %10.4f %+10.4f %8.1f%%",
            i, df_input["machine_type"][i], real, g, err, pct,
        )

    # ── Resumo do modelo global ──────────────────────────────────────────
    _logger.info("\n══ MÉTRICAS MODELO GLOBAL (%d amostras) ══", n)
    _report("Global (todas)", y_true, y_global_all)

    # métricas por tipo de máquina
    _logger.info("\n  ── por tipo de máquina ──")
    for mt in sorted(norm_types.unique().to_list()):
        mt_mask = (norm_types == mt).to_numpy()
        n_mt = int(mt_mask.sum())
        if n_mt == 0:
            continue
        _report(f"  {mt}", y_true[mt_mask], y_global_all[mt_mask])

    # ── filtra pelo tipo de máquina para comparação segmentada ───────────
    seg_mask = (df_input["_norm_type"] == chosen_segment)
    df_seg_input = df_input.filter(seg_mask).drop("_norm_type")
    y_seg_true   = y_true[seg_mask.to_numpy()]
    y_global_seg = y_global_all[seg_mask.to_numpy()]
    n_seg        = len(df_seg_input)

    _logger.info(
        "\nDataFrame filtrado para '%s': %d de %d linhas",
        chosen_segment, n_seg, n,
    )

    if n_seg == 0:
        _logger.warning("Nenhuma amostra do segmento '%s' — encerrando demo.", chosen_segment)
        sys.exit(0)

    # ── predição segmentada sobre o subset filtrado ──────────────────────
    result_seg = predict_dl(df_seg_input, segment=chosen_segment)
    y_segment  = result_seg["segment"]

    # ── LOG 2: Comparativo lado a lado — Global vs Segmentado ───────────
    _logger.info(
        "\n══ COMPARATIVO '%s' (%d amostras) — Global vs Segmentado ══",
        chosen_segment, n_seg,
    )
    _logger.info(
        "  %3s  %10s %10s %10s %10s %8s",
        "#", "Real", "Global", "Segmentado", "Melhor", "Ganho",
    )
    _logger.info("  %s", "─" * 60)

    n_global_wins = 0
    n_segment_wins = 0
    n_ties = 0
    for i in range(n_seg):
        real = y_seg_true[i]
        g = y_global_seg[i]
        s = y_segment[i]
        err_g = abs(g - real)

        if np.isfinite(s):
            err_s = abs(s - real)
            if err_s < err_g:
                melhor = "Seg"
                ganho = (err_g - err_s) / max(err_g, 1e-8) * 100
                n_segment_wins += 1
            elif err_g < err_s:
                melhor = "Global"
                ganho = (err_s - err_g) / max(err_s, 1e-8) * 100
                n_global_wins += 1
            else:
                melhor = "Empate"
                ganho = 0.0
                n_ties += 1
            _logger.info(
                "  %3d  %10.4f %10.4f %10.4f %10s %7.1f%%",
                i, real, g, s, melhor, ganho,
            )
        else:
            n_global_wins += 1
            _logger.info(
                "  %3d  %10.4f %10.4f %10s %10s %8s",
                i, real, g, "N/A", "Global", "—",
            )

    _logger.info("  %s", "─" * 60)
    _logger.info(
        "  Placar: Global=%d | Seg=%d | Empate=%d | total=%d amostras",
        n_global_wins, n_segment_wins, n_ties, n_seg,
    )

    _logger.info(
        "\n══ MÉTRICAS CONSOLIDADAS — '%s' (%d amostras) ══",
        chosen_segment, n_seg,
    )
    _report("Global", y_seg_true, y_global_seg)
    _report(f"Seg::{chosen_segment}", y_seg_true, y_segment)

    _logger.info("\nDemo concluída.")

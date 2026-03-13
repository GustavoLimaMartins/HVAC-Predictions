"""
ml_models.py — Utilitário de predição ML para modelos HVAC
==========================================================

Disponibiliza a função ``predict_ml()`` que recebe dados no formato do
schema inicial (colunas brutas do ``final_dataframe.csv`` **sem** a coluna
``consumo_kwh``) e retorna as predições do modelo global e,
opcionalmente, de um modelo segmentado específico.

Toda a conversão de inputs é delegada ao módulo ``tools/normalizer.py``
(classe ``MLNormalizer``), que reproduz fielmente o fluxo de
``MLPipeline._build_schema()`` + Target Encoding + alinhamento + sanitização
— sem duplicar lógica.

Schema inicial esperado (11 colunas, sem ``consumo_kwh``):
    hora, data, machine_type, estacao, grupo_regional,
    Temperatura_C, Temperatura_Percebida_C, Umidade_Relativa_%,
    Precipitacao_mm, Velocidade_Vento_kmh, Pressao_Superficial_hPa

Uso rápido:
    >>> from testing.ml_models import predict_ml, list_segments
    >>> list_segments()                                 # segmentos disponíveis
    >>> result = predict_ml(df_input)                   # só modelo global
    >>> result = predict_ml(df_input, segment="SPLIT HI-WALL")  # global + segmentado
    >>> result["global"]                                # np.ndarray de predições
    >>> result["segment"]                               # np.ndarray (ou None)

Execução direta (demo com 100 amostras):
    python testing/ml_models.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# ── path de importação ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.ml_pipeline import (
    MLPipeline,
    MLPipelineConfig,
    SegmentedMLPipeline,
    _filter_outliers,
)
from model.pre_process.schema import ModelSchema
from tools.normalizer import MLNormalizer

# ── constantes ──────────────────────────────────────────────────────────────
_TARGET: str = "consumo_kwh"

_INPUT_FIELDS: list[str] = [
    "hora", "data", "machine_type", "estacao", "grupo_regional",
    "Temperatura_C", "Temperatura_Percebida_C",
    "Umidade_Relativa_%", "Precipitacao_mm",
    "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
]

_ROOT = Path(__file__).resolve().parent.parent
_GLOBAL_DIR   = _ROOT / "model" / "artifacts" / "ml_hvac" / "global" / "best_pipeline.joblib"
_SEGMENTED_DIR = _ROOT / "model" / "artifacts" / "ml_hvac"

_logger = logging.getLogger(__name__)

# ── cache de modelos (carregados uma única vez) ─────────────────────────────
_global_pipe: MLPipeline | None = None
_segmented_pipe: SegmentedMLPipeline | None = None


# ═══════════════════════════════════════════════════════════════════════════
#  HELPERS INTERNOS
# ═══════════════════════════════════════════════════════════════════════════

def _load_global() -> MLPipeline:
    """Carrega (ou retorna do cache) o modelo global (.joblib)."""
    global _global_pipe
    if _global_pipe is None:
        if not _GLOBAL_DIR.exists():
            raise FileNotFoundError(
                f"Artefato do modelo global não encontrado: {_GLOBAL_DIR}"
            )
        _global_pipe = MLPipeline.load(_GLOBAL_DIR)
    return _global_pipe


def _load_segmented() -> SegmentedMLPipeline:
    """Carrega (ou retorna do cache) o pipeline segmentado."""
    global _segmented_pipe
    if _segmented_pipe is None:
        if not _SEGMENTED_DIR.exists():
            raise FileNotFoundError(
                f"Artefatos segmentados não encontrados: {_SEGMENTED_DIR}"
            )
        _segmented_pipe = SegmentedMLPipeline.load(_SEGMENTED_DIR)
    return _segmented_pipe


def _validate_input(df: pl.DataFrame) -> None:
    """Valida que o DataFrame de entrada possui as colunas esperadas."""
    if _TARGET in df.columns:
        raise ValueError(
            f"O DataFrame de entrada não deve conter a coluna target "
            f"'{_TARGET}'. Remova-a antes de chamar predict_ml()."
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


def predict_ml(
    df: pl.DataFrame,
    segment: str | None = None,
) -> dict[str, np.ndarray | None]:
    """
    Realiza predições de consumo HVAC (kWh) usando os modelos ML salvos.

    O DataFrame de entrada deve seguir o schema inicial (13 colunas brutas,
    **sem** ``consumo_kwh``). A conversão de inputs é delegada ao
    ``MLNormalizer`` (módulo ``tools/normalizer.py``), que reproduz
    fielmente o fluxo de ``MLPipeline._build_schema()`` + Target Encoding
    + alinhamento + sanitização — idêntico ao usado no treino.

    Parameters
    ----------
    df : pl.DataFrame
        Dados de entrada no schema inicial:
            hora, data, machine_type, estacao, grupo_regional,
            Temperatura_C, Temperatura_Percebida_C, Umidade_Relativa_%,
            Precipitacao_mm, Velocidade_Vento_kmh, Pressao_Superficial_hPa
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
    >>> from testing.ml_models import predict_ml, list_segments
    >>>
    >>> df_input = pl.read_csv("use_case/files/final_dataframe.csv").head(10).drop("consumo_kwh")
    >>> result = predict_ml(df_input, segment="SPLIT HI-WALL")
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

    # ── predição global ──────────────────────────────────────────────────
    # MLPipeline.predict() aplica todo o schema internamente
    global_pipe = _load_global()
    y_global = global_pipe.predict(df).astype(np.float32)
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

        n_wrong_type = int((~type_mask).sum())
        if n_wrong_type > 0:
            _logger.info(
                "Segmento '%s': %d de %d linhas ignoradas (machine_type diferente).",
                segment, n_wrong_type, len(df),
            )

        y_segment = np.full(len(df), np.nan, dtype=np.float32)
        if type_mask.any():
            df_valid = df.filter(pl.Series(type_mask))
            y_segment[type_mask] = seg_model.predict(df_valid).astype(np.float32)

        _logger.info(
            "Predição segmentada '%s' concluída: %d/%d amostras preditas "
            "(%d tipo correto, %d ignoradas por tipo diferente)",
            segment, int(type_mask.sum()), len(df),
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

    # ── tratamento de outliers (idêntico ao usado no treinamento ML) ───
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
    df_full, _ = _filter_outliers(df_raw, **_OUTLIER_PARAMS)
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
    #  VARREDURA VIA MLNormalizer.inspect()  —  validação estruturada
    # ══════════════════════════════════════════════════════════════════════
    _logger.info("\n" + "═" * 90)
    _logger.info("  VARREDURA DO MLNormalizer  —  Validação via inspect()")
    _logger.info("═" * 90)

    _ml_norm = MLNormalizer.from_artifact(_GLOBAL_DIR)
    _inspect = _ml_norm.inspect(df_input)

    _logger.info(
        "Shape pós-normalização: %d linhas × %d features",
        _inspect["n_rows"], _inspect["n_features"],
    )

    # -- Target Encoding -------------------------------------------------
    _logger.info("\n" + "─" * 90)
    _logger.info("  TARGET ENCODING — Colunas com TE")
    _logger.info("─" * 90)

    _te_specs = {
        "hora_target_enc": {"original": "hora (0-23)", "desc": "Média suavizada de consumo_kwh por hora"},
        "mes_target_enc":  {"original": "mes (1-12)",  "desc": "Média suavizada de consumo_kwh por mês"},
    }
    _logger.info(
        "  %-22s %10s %10s %6s  %s",
        "Coluna TE", "Min", "Max", "NaN", "Origem",
    )
    _logger.info("  " + "─" * 70)
    for te_col in _inspect["te_columns"]:
        st = _inspect["feature_stats"].get(te_col, {})
        spec = _te_specs.get(te_col, {"original": "?", "desc": ""})
        _logger.info(
            "  %-22s %10.4f %10.4f %5d   %s",
            te_col,
            st.get("min", 0.0) or 0.0,
            st.get("max", 0.0) or 0.0,
            st.get("nan", 0),
            spec["original"],
        )
        # Mostra mapa de TE do pipeline treinado (se disponível)
        _global = _load_global()
        raw_key = te_col.replace("_target_enc", "")
        if _global._te_map and raw_key in _global._te_map:
            te_entry = _global._te_map[raw_key]
            mapping_dict = te_entry.get("mapping", te_entry)
            global_mean  = te_entry.get("global_mean", None)
            sorted_items = sorted(mapping_dict.items(), key=lambda x: float(x[0]))
            _logger.info(
                "    Mapa TE : %s",
                ", ".join(f"{k}→{v:.4f}" for k, v in sorted_items[:12]),
            )
            if len(sorted_items) > 12:
                _logger.info("              ... + %d valores", len(sorted_items) - 12)
            if global_mean is not None:
                _logger.info("    Global mean (fallback): %.4f", global_mean)

    # -- Features (todas) ------------------------------------------------
    _logger.info("\n" + "─" * 90)
    _logger.info("  FEATURES — Estatísticas")
    _logger.info("─" * 90)
    _logger.info(
        "  %-3s  %-44s %10s %10s %6s",
        "#", "Feature", "Min", "Max", "NaN",
    )
    _logger.info("  " + "─" * 78)
    for idx, (feat_name, feat_info) in enumerate(_inspect["feature_stats"].items()):
        # Marca papel da coluna
        if feat_name.endswith("_target_enc"):
            role = " 📊TE"
        elif feat_name == "grupo_regional":
            role = " 🏷CAT"
        elif feat_name.startswith("tipo_maquina_"):
            role = " OHE"
        elif feat_name.startswith(("estacao_", "periodo_dia_")):
            role = " OHE"
        else:
            role = ""
        nan_flag = " ⚠" if feat_info["nan"] > 0 else ""
        _logger.info(
            "  %3d  %-44s %10.4f %10.4f %5d%s%s",
            idx, feat_name,
            feat_info["min"] if feat_info["min"] is not None else 0.0,
            feat_info["max"] if feat_info["max"] is not None else 0.0,
            feat_info["nan"], nan_flag, role,
        )

    # -- Checklist de sanidade -------------------------------------------
    _logger.info("\n" + "─" * 90)
    _logger.info("  CHECKLIST DE SANIDADE")
    _logger.info("─" * 90)

    _checks: list[tuple[str, bool, str]] = []

    # 1. Nenhum warning do normalizer
    _checks.append((
        "Sem warnings do MLNormalizer",
        len(_inspect["warnings"]) == 0,
        f"{len(_inspect['warnings'])} warning(s)" if _inspect["warnings"] else "OK",
    ))
    for w in _inspect["warnings"]:
        _logger.warning("    ⚠ %s", w)

    # 2. Target Encoding presentes
    for te in ["hora_target_enc", "mes_target_enc"]:
        present = te in _inspect["feature_stats"]
        _checks.append((
            f"TE '{te}' presente",
            present,
            f"min={_inspect['feature_stats'][te]['min']:.4f} max={_inspect['feature_stats'][te]['max']:.4f}"
            if present else "AUSENTE",
        ))

    # 3. TE valores razoáveis (> 0, < 100)
    for te in ["hora_target_enc", "mes_target_enc"]:
        if te in _inspect["feature_stats"]:
            st = _inspect["feature_stats"][te]
            _checks.append((
                f"{te} range razoável",
                (st["min"] or 0) > 0 and (st["max"] or 0) < 100,
                f"min={st['min']:.4f} max={st['max']:.4f}",
            ))

    # 4. Features sem NaN
    total_nan = sum(v["nan"] for v in _inspect["feature_stats"].values())
    _checks.append((
        "Features sem NaN",
        total_nan == 0,
        f"NaN={total_nan}",
    ))

    # 5. Dimensão esperada
    _checks.append((
        "n_features correto",
        _inspect["n_features"] == len(_ml_norm.feature_columns),
        f"{_inspect['n_features']} (esperado {len(_ml_norm.feature_columns)})",
    ))

    # 6. OHE tipo_maquina presente
    ohe_maquina = [c for c in _inspect["feature_stats"] if c.startswith("tipo_maquina_")]
    _checks.append((
        "OHE tipo_maquina presente",
        len(ohe_maquina) > 0,
        f"{len(ohe_maquina)} colunas",
    ))

    # 7. grupo_regional presente
    _checks.append((
        "grupo_regional presente",
        "grupo_regional" in _inspect["feature_stats"],
        "OK" if "grupo_regional" in _inspect["feature_stats"] else "AUSENTE",
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
    result_global = predict_ml(df_input.drop("_norm_type"))
    y_global_all  = result_global["global"]

    # ── LOG 1: Predição Global — Real vs Predito (amostra completa) ─────
    _logger.info("\n══ MODELO GLOBAL ML — Real vs Predito (%d amostras) ══", n)
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
    _logger.info("\n══ MÉTRICAS MODELO GLOBAL ML (%d amostras) ══", n)
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
    result_seg = predict_ml(df_seg_input, segment=chosen_segment)
    y_segment  = result_seg["segment"]

    # ── LOG 2: Comparativo lado a lado — Global vs Segmentado ───────────
    _logger.info(
        "\n══ COMPARATIVO ML '%s' (%d amostras) — Global vs Segmentado ══",
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
        "\n══ MÉTRICAS CONSOLIDADAS ML — '%s' (%d amostras) ══",
        chosen_segment, n_seg,
    )
    _report("Global", y_seg_true, y_global_seg)
    _report(f"Seg::{chosen_segment}", y_seg_true, y_segment)

    _logger.info("\nDemo concluída.")

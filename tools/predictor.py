"""
Predictor — Módulo unificado de predição HVAC (DL + ML)
=======================================================

Recebe dados no **formato inicial** (colunas brutas do ``final_dataframe.csv``,
opcionalmente sem ``consumo_kwh``) e devolve predições de consumo (kWh)
usando os modelos treinados (.keras para Deep Learning e .joblib para
Machine Learning).

A normalização é delegada inteiramente ao ``DLNormalizer`` / ``MLNormalizer``
do módulo ``tools.normalizer``, garantindo que os dados passem pelo mesmo
pipeline de pré-processamento usado no treinamento.

Modos de operação
-----------------
1. **Modelo global** — um único modelo treinado em todos os segmentos.
2. **Modelo segmentado** — um modelo por ``machine_type`` (SPLIT HI-WALL,
   SPLITÃO, etc.); cada linha é roteada ao modelo do seu tipo de máquina.

Modelos suportados
------------------
- **DL** (Deep Learning): Wide & Deep com Entity Embeddings (.keras)
- **ML** (Machine Learning): LightGBM / XGBoost (.joblib)

Uso rápido
----------
    >>> from tools.predictor import HVACPredictor
    >>>
    >>> pred = HVACPredictor()                       # carrega todos os artefatos
    >>> pred.list_segments()                         # segmentos disponíveis
    >>>
    >>> # Predição com ambos os engines (global)
    >>> result = pred.predict(df_input)
    >>> result["dl_global"]                          # np.ndarray
    >>> result["ml_global"]                          # np.ndarray
    >>>
    >>> # Predição segmentada (sem global)
    >>> result = pred.predict(df_input, mode="segment", segment="SPLIT HI-WALL")
    >>> result["dl_segment"]                         # np.ndarray (NaN para outros tipos)
    >>> result["ml_segment"]                         # np.ndarray (NaN para outros tipos)
    >>>
    >>> # Global + segmentada ao mesmo tempo
    >>> result = pred.predict(df_input, mode="both", segment="SPLIT HI-WALL")
    >>>
    >>> # Apenas DL ou ML (engine)
    >>> result = pred.predict(df_input, engine="dl")
    >>> result = pred.predict(df_input, engine="ml")
    >>>
    >>> # Combinando: apenas ML segmentado
    >>> result = pred.predict(df_input, engine="ml", mode="segment", segment="SPLITÃO")

Campos auto-derivados
---------------------
- ``estacao`` (estação do ano): derivada automaticamente da coluna ``data``
  usando o calendário astronômico brasileiro. Pode ser omitida.
- ``grupo_regional``: pode ser fornecido diretamente **ou** resolvido
  automaticamente via ``latitude``/``longitude`` (DBSCAN + Haversine KNN).
- ``mes`` e ``periodo_dia``: derivados internamente pelo Normalizer
  a partir de ``data`` e ``hora``, respectivamente. **Não** são input.

Entrada manual (dict → DataFrame)
----------------------------------
    >>> from tools.predictor import HVACPredictor, build_input
    >>>
    >>> # Com grupo_regional e estacao explícitos
    >>> row = build_input(
    ...     hora=14, data="2025-07-03", machine_type="splitao",
    ...     Temperatura_C=25.7, Temperatura_Percebida_C=24.9,
    ...     Umidade_Relativa_pct=57.0, Precipitacao_mm=0.0,
    ...     Velocidade_Vento_kmh=21.4, Pressao_Superficial_hPa=969.8,
    ...     is_dac=1, is_dut=0,
    ...     estacao="inverno", grupo_regional=103,
    ... )
    >>>
    >>> # Com lat/lon (grupo_regional e estacao auto-derivados)
    >>> row = build_input(
    ...     hora=14, data="2025-07-03", machine_type="splitao",
    ...     Temperatura_C=25.7, Temperatura_Percebida_C=24.9,
    ...     Umidade_Relativa_pct=57.0, Precipitacao_mm=0.0,
    ...     Velocidade_Vento_kmh=21.4, Pressao_Superficial_hPa=969.8,
    ...     is_dac=1, is_dut=0,
    ...     latitude=-8.07, longitude=-39.12,
    ... )
    >>> pred = HVACPredictor()
    >>> result = pred.predict(row)

Execução direta (demo):
    python tools/predictor.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

# ── Suprime mensagens do TensorFlow antes do import ─────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from sklearn.neighbors import BallTree

from tools.normalizer import DLNormalizer, MLNormalizer
from model.pre_process.schema import ModelSchema

_logger = logging.getLogger(__name__)

# ── Caminhos padrão dos artefatos ───────────────────────────────────────────
_DL_ARTIFACTS = _ROOT / "model" / "artifacts" / "dl_hvac"
_ML_ARTIFACTS = _ROOT / "model" / "artifacts" / "ml_hvac"
_GEO_REF_PATH = _ROOT / "model" / "artifacts" / "geo_reference.parquet"

_TARGET: str = "consumo_kwh"

# Colunas obrigatórias no schema final (após resolução de lat/lon e estacao)
_INPUT_FIELDS: list[str] = [
    "hora", "data", "machine_type", "estacao", "grupo_regional",
    "Temperatura_C", "Temperatura_Percebida_C",
    "Umidade_Relativa_%", "Precipitacao_mm",
    "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
    "is_dac", "is_dut",
]


# ══════════════════════════════════════════════════════════════════════════════
#  AUTO-RESOLUÇÃO — estação do ano e grupo regional
# ══════════════════════════════════════════════════════════════════════════════

def _derive_season(data_col: pl.Series) -> pl.Series:
    """
    Deriva a estação do ano brasileira a partir de uma coluna de datas.

    Reproduz a mesma lógica de
    ``dataframe.complementary_features.year_stations.YearStationsEnricher``
    sem prints e sem dependência de CSV.

    Regras astronômicas (Brasil):
        - Verão:     21/dez – 20/mar
        - Outono:    21/mar – 20/jun
        - Inverno:   21/jun – 22/set
        - Primavera: 23/set – 20/dez
    """
    df_tmp = (
        pl.DataFrame({"data": data_col})
        .with_columns(pl.col("data").cast(pl.Date))
        .with_columns([
            pl.col("data").dt.month().alias("_mes"),
            pl.col("data").dt.day().alias("_dia"),
        ])
        .with_columns(
            pl.when(
                ((pl.col("_mes") == 12) & (pl.col("_dia") >= 21))
                | pl.col("_mes").is_in([1, 2])
                | ((pl.col("_mes") == 3) & (pl.col("_dia") <= 20))
            ).then(pl.lit("verao"))
            .when(
                ((pl.col("_mes") == 3) & (pl.col("_dia") >= 21))
                | pl.col("_mes").is_in([4, 5])
                | ((pl.col("_mes") == 6) & (pl.col("_dia") <= 20))
            ).then(pl.lit("outono"))
            .when(
                ((pl.col("_mes") == 6) & (pl.col("_dia") >= 21))
                | pl.col("_mes").is_in([7, 8])
                | ((pl.col("_mes") == 9) & (pl.col("_dia") <= 22))
            ).then(pl.lit("inverno"))
            .otherwise(pl.lit("primavera"))
            .alias("estacao")
        )
    )
    return df_tmp["estacao"]


def _resolve_inputs(df: pl.DataFrame) -> pl.DataFrame:
    """
    Auto-resolve campos deriváveis antes da validação:

    1. **estacao** — se ausente, derivada automaticamente da coluna ``data``
       usando o calendário astronômico brasileiro.
    2. **grupo_regional** — se ausente mas ``latitude`` e ``longitude``
       estiverem presentes, resolve via ``RegionalGroupClassifier`` treinado
       no ``consumption_consolidated.csv`` (mesmos parâmetros do treino).

    Returns:
        DataFrame com ``estacao`` e ``grupo_regional`` garantidos.
    """
    # ── 1. Estação do ano ────────────────────────────────────────────────
    if "estacao" not in df.columns and "data" in df.columns:
        df = df.with_columns(_derive_season(df["data"]).alias("estacao"))
        _logger.info("Estação derivada automaticamente da coluna 'data'")

    # ── 2. Grupo regional ────────────────────────────────────────────────
    if "grupo_regional" not in df.columns:
        if "latitude" not in df.columns or "longitude" not in df.columns:
            raise ValueError(
                "Forneça 'grupo_regional' diretamente OU as colunas "
                "'latitude' e 'longitude' para resolução automática."
            )
        df = _assign_grupo_regional(df)
        # Remove lat/lon — não fazem parte do schema do modelo
        df = df.drop([c for c in ("latitude", "longitude") if c in df.columns])
        grupos = df["grupo_regional"].to_list()
        _logger.info("grupo_regional resolvido via lat/lon (KNN Haversine): %s", grupos)

    return df


# Cache do lookup geográfico (BallTree + labels carregados do artefato)
_geo_tree: BallTree | None = None
_geo_labels: np.ndarray | None = None


def _get_geo_lookup() -> tuple[BallTree, np.ndarray]:
    """
    Carrega o artefato ``geo_reference.parquet`` (gerado pelo DBSCAN
    no treino) e devolve um ``BallTree`` Haversine + vetor de labels.

    O artefato contém as 388 coordenadas únicas de treinamento já
    rotuladas com ``grupo_regional``.  Na inferência basta localizar
    o vizinho mais próximo — nenhum re-treinamento de DBSCAN ocorre.
    """
    global _geo_tree, _geo_labels
    if _geo_tree is not None:
        return _geo_tree, _geo_labels  # type: ignore[return-value]

    if not _GEO_REF_PATH.exists():
        raise FileNotFoundError(
            f"Artefato de referência geográfica não encontrado:\n"
            f"  {_GEO_REF_PATH}\n"
            f"Forneça 'grupo_regional' diretamente ou gere o artefato."
        )

    _logger.info("Carregando referência geográfica de %s ...", _GEO_REF_PATH.name)
    ref = pl.read_parquet(_GEO_REF_PATH)
    coords_rad = np.radians(ref.select(["latitude", "longitude"]).to_numpy())
    _geo_labels = ref["grupo_regional"].to_numpy()
    _geo_tree = BallTree(coords_rad, metric="haversine")
    return _geo_tree, _geo_labels


def _assign_grupo_regional(df: pl.DataFrame) -> pl.DataFrame:
    """
    Atribui ``grupo_regional`` a cada linha via KNN-1 Haversine sobre
    as coordenadas de referência do treino.

    • Linhas com lat/lon nulos recebem ``null``.
    • Coordenadas já conhecidas retornam o rótulo exato do treino.
    • Coordenadas novas recebem o grupo do vizinho mais próximo.
    """
    tree, labels = _get_geo_lookup()

    lats = df["latitude"].to_list()
    lons = df["longitude"].to_list()

    result: list[int | None] = []
    query_idxs: list[int] = []
    query_coords: list[tuple[float, float]] = []

    for i, (lat, lon) in enumerate(zip(lats, lons)):
        if lat is None or lon is None:
            result.append(None)
        else:
            result.append(None)  # placeholder
            query_idxs.append(i)
            query_coords.append((lat, lon))

    if query_idxs:
        coords_rad = np.radians(np.array(query_coords))
        _, indices = tree.query(coords_rad, k=1)
        for idx, ref_idx in zip(query_idxs, indices.ravel()):
            result[idx] = int(labels[ref_idx])

    return df.with_columns(
        pl.Series("grupo_regional", result, dtype=pl.Int32)
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER — Construção de input manual
# ══════════════════════════════════════════════════════════════════════════════

def build_input(
    hora: int,
    data: str,
    machine_type: str,
    Temperatura_C: float,
    Temperatura_Percebida_C: float,
    Umidade_Relativa_pct: float,
    Precipitacao_mm: float,
    Velocidade_Vento_kmh: float,
    Pressao_Superficial_hPa: float,
    is_dac: int,
    is_dut: int,
    *,
    estacao: str | None = None,
    grupo_regional: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> pl.DataFrame:
    """
    Constrói um ``pl.DataFrame`` de 1 linha a partir de parâmetros individuais.

    Campos auto-deriváveis:
        - **estacao** — se omitido, derivado automaticamente da ``data``
          (calendário astronômico brasileiro).
        - **grupo_regional** — se omitido, forneça ``latitude`` e ``longitude``
          para resolução automática via DBSCAN + Haversine.

    Parameters
    ----------
    hora : int
        Hora do dia (0–23).
    data : str
        Data no formato ``"YYYY-MM-DD"``.
    machine_type : str
        Tipo de máquina bruto (ex: ``"splitao"``, ``"split_hi-wall"``).
    Temperatura_C : float
        Temperatura em ºC.
    Temperatura_Percebida_C : float
        Temperatura percebida em ºC.
    Umidade_Relativa_pct : float
        Umidade relativa (%). O parâmetro usa ``_pct``; a coluna interna
        usa ``%``.
    Precipitacao_mm : float
        Precipitação em mm.
    Velocidade_Vento_kmh : float
        Velocidade do vento em km/h.
    Pressao_Superficial_hPa : float
        Pressão superficial em hPa.
    is_dac : int
        Flag de DAC (0 ou 1).
    is_dut : int
        Flag de DUT (0 ou 1).
    estacao : str | None
        Estação do ano. Se ``None``, derivada de ``data``.
    grupo_regional : int | None
        Código do grupo regional. Se ``None``, forneça ``latitude``/``longitude``.
    latitude : float | None
        Latitude (graus). Usada para resolver ``grupo_regional`` automaticamente.
    longitude : float | None
        Longitude (graus). Usada para resolver ``grupo_regional`` automaticamente.

    Returns
    -------
    pl.DataFrame
        DataFrame com 1 linha, pronto para ``HVACPredictor.predict()``.

    Examples
    --------
    Com grupo_regional explícito:

    >>> row = build_input(
    ...     hora=14, data="2025-07-03", machine_type="splitao",
    ...     Temperatura_C=25.7, Temperatura_Percebida_C=24.9,
    ...     Umidade_Relativa_pct=57.0, Precipitacao_mm=0.0,
    ...     Velocidade_Vento_kmh=21.4, Pressao_Superficial_hPa=969.8,
    ...     is_dac=1, is_dut=0,
    ...     grupo_regional=103,
    ... )

    Com lat/lon (grupo_regional e estacao auto-derivados):

    >>> row = build_input(
    ...     hora=14, data="2025-07-03", machine_type="splitao",
    ...     Temperatura_C=25.7, Temperatura_Percebida_C=24.9,
    ...     Umidade_Relativa_pct=57.0, Precipitacao_mm=0.0,
    ...     Velocidade_Vento_kmh=21.4, Pressao_Superficial_hPa=969.8,
    ...     is_dac=1, is_dut=0,
    ...     latitude=-8.0707, longitude=-39.1209,
    ... )
    """
    cols: dict = {
        "hora":                    [hora],
        "data":                    [data],
        "machine_type":            [machine_type],
        "Temperatura_C":           [Temperatura_C],
        "Temperatura_Percebida_C": [Temperatura_Percebida_C],
        "Umidade_Relativa_%":      [Umidade_Relativa_pct],
        "Precipitacao_mm":         [Precipitacao_mm],
        "Velocidade_Vento_kmh":    [Velocidade_Vento_kmh],
        "Pressao_Superficial_hPa": [Pressao_Superficial_hPa],
        "is_dac":                  [is_dac],
        "is_dut":                  [is_dut],
    }

    # Campos opcionais (resolvidos depois por _resolve_inputs se ausentes)
    if estacao is not None:
        cols["estacao"] = [estacao]
    if grupo_regional is not None:
        cols["grupo_regional"] = [grupo_regional]
    if latitude is not None:
        cols["latitude"] = [latitude]
    if longitude is not None:
        cols["longitude"] = [longitude]

    return pl.DataFrame(cols)


def build_input_batch(records: list[dict]) -> pl.DataFrame:
    """
    Constrói um ``pl.DataFrame`` de N linhas a partir de uma lista de dicts.

    Campos auto-deriváveis (podem ser omitidos):
        - ``estacao`` — derivada de ``data`` se ausente.
        - ``grupo_regional`` — resolvido via ``latitude``/``longitude`` se ausente.

    A chave ``Umidade_Relativa_%`` pode ser passada como
    ``"Umidade_Relativa_pct"`` (conveniência para evitar ``%`` em código).

    Parameters
    ----------
    records : list[dict]
        Lista de dicionários, cada um representando uma observação.

    Returns
    -------
    pl.DataFrame

    Example
    -------
    >>> rows = build_input_batch([
    ...     {"hora": 14, "data": "2025-07-03", "machine_type": "splitao",
    ...      "Temperatura_C": 25.7, "Temperatura_Percebida_C": 24.9,
    ...      "Umidade_Relativa_pct": 57.0, "Precipitacao_mm": 0.0,
    ...      "Velocidade_Vento_kmh": 21.4, "Pressao_Superficial_hPa": 969.8,
    ...      "is_dac": 1, "is_dut": 0,
    ...      "latitude": -8.07, "longitude": -39.12},
    ... ])
    """
    normalized = []
    for rec in records:
        r = dict(rec)
        if "Umidade_Relativa_pct" in r and "Umidade_Relativa_%" not in r:
            r["Umidade_Relativa_%"] = r.pop("Umidade_Relativa_pct")
        normalized.append(r)
    return pl.DataFrame(normalized)


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

def _validate_input(df: pl.DataFrame) -> pl.DataFrame:
    """
    Valida e prepara o DataFrame de entrada.

    Pipeline:
        1. Remove ``consumo_kwh`` se presente (modo inferência).
        2. Auto-resolve ``estacao`` (da ``data``) e ``grupo_regional``
           (de ``latitude``/``longitude``) se ausentes.
        3. Verifica a presença de todas as 13 colunas obrigatórias.

    Returns:
        DataFrame sem a coluna target, pronto para normalização.

    Raises:
        ValueError: Colunas obrigatórias ausentes após resolução.
    """
    if _TARGET in df.columns:
        df = df.drop(_TARGET)

    # Auto-resolve estacao e grupo_regional
    df = _resolve_inputs(df)

    missing = set(_INPUT_FIELDS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes: {sorted(missing)}\n"
            f"Colunas esperadas: {_INPUT_FIELDS}\n"
            f"Colunas recebidas: {df.columns}"
        )
    return df


def _normalize_machine_types(df: pl.DataFrame) -> pl.Series:
    """
    Retorna a coluna ``tipo_maquina`` normalizada (UPPER + mapeamento)
    usando o mesmo ``ModelSchema.adjust_machine_type()`` do treinamento.
    """
    schema = ModelSchema(df.select("machine_type"), ["machine_type"])
    schema.adjust_machine_type()
    return schema.df["tipo_maquina"]


# ══════════════════════════════════════════════════════════════════════════════
#  HVAC PREDICTOR — Classe principal
# ══════════════════════════════════════════════════════════════════════════════

class HVACPredictor:
    """
    Módulo unificado de predição HVAC usando os modelos treinados.

    Carrega os artefatos DL (.keras) e ML (.joblib) — globais e segmentados —
    e expõe uma interface única para predição, delegando a normalização
    ao ``DLNormalizer`` e ``MLNormalizer``.

    Parameters
    ----------
    dl_artifacts : str | Path
        Diretório raiz dos artefatos DL (padrão: ``model/artifacts/dl_hvac``).
    ml_artifacts : str | Path
        Diretório raiz dos artefatos ML (padrão: ``model/artifacts/ml_hvac``).
    load_dl : bool
        Se True, carrega os modelos Deep Learning.
    load_ml : bool
        Se True, carrega os modelos Machine Learning.

    Attributes
    ----------
    dl_global_model : keras.Model | None
        Modelo Keras global (carregado lazy na primeira predição).
    ml_global_pipe : MLPipeline | None
        Pipeline ML global (.joblib, carregado lazy).
    dl_segments : dict[str, (keras.Model, DLNormalizer)]
        Modelos DL segmentados {nome_segmento: (modelo, normalizer)}.
    ml_segments : dict[str, (MLPipeline, MLNormalizer)]
        Pipelines ML segmentados {nome_segmento: (pipeline, normalizer)}.
    """

    def __init__(
        self,
        dl_artifacts: str | Path = _DL_ARTIFACTS,
        ml_artifacts: str | Path = _ML_ARTIFACTS,
        load_dl: bool = True,
        load_ml: bool = True,
    ) -> None:
        self._dl_dir = Path(dl_artifacts)
        self._ml_dir = Path(ml_artifacts)

        # Modelos globais
        self.dl_global_model = None
        self.dl_global_norm: DLNormalizer | None = None
        self.ml_global_pipe = None
        self.ml_global_norm: MLNormalizer | None = None

        # Modelos segmentados: {segment_name: (model/pipe, normalizer)}
        self.dl_segments: dict[str, tuple] = {}
        self.ml_segments: dict[str, tuple] = {}

        # Manifests
        self._dl_manifest: dict | None = None
        self._ml_manifest: dict | None = None

        if load_dl:
            self._load_dl_artifacts()
        if load_ml:
            self._load_ml_artifacts()

    # ── Carregamento de Artefatos ────────────────────────────────────────

    def _load_dl_artifacts(self) -> None:
        """Carrega modelo DL global e segmentados."""
        import tensorflow as tf
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("absl").setLevel(logging.ERROR)

        # Global
        global_dir = self._dl_dir / "global"
        if (global_dir / "keras_model.keras").exists():
            self.dl_global_model = tf.keras.models.load_model(
                str(global_dir / "keras_model.keras")
            )
            self.dl_global_norm = DLNormalizer.from_artifact(global_dir)
            _logger.info("DL global carregado de: %s", global_dir)
        else:
            _logger.warning("Artefato DL global não encontrado: %s", global_dir)

        # Manifest (segmentos)
        manifest_path = self._dl_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as fh:
                self._dl_manifest = json.load(fh)

            for seg_name, seg_dir_name in self._dl_manifest.get("segment_dirs", {}).items():
                seg_path = self._dl_dir / seg_dir_name
                if (seg_path / "keras_model.keras").exists():
                    model = tf.keras.models.load_model(str(seg_path / "keras_model.keras"))
                    norm = DLNormalizer.from_artifact(seg_path)
                    self.dl_segments[seg_name] = (model, norm)
                    _logger.info("DL segmento '%s' carregado", seg_name)
        else:
            _logger.warning("Manifest DL não encontrado: %s", manifest_path)

    def _load_ml_artifacts(self) -> None:
        """Carrega pipeline ML global e segmentados."""
        # Global
        global_path = self._ml_dir / "global" / "best_pipeline.joblib"
        if global_path.exists():
            self.ml_global_norm = MLNormalizer.from_artifact(global_path)
            # O modelo já está dentro do MLPipeline deserializado
            import joblib
            import __main__
            from model.ml_pipeline import MLPipeline as _MLP, MLPipelineConfig as _MLPC
            if not hasattr(__main__, "MLPipeline"):
                __main__.MLPipeline = _MLP
            if not hasattr(__main__, "MLPipelineConfig"):
                __main__.MLPipelineConfig = _MLPC

            self.ml_global_pipe = joblib.load(global_path)
            _logger.info("ML global carregado de: %s", global_path)
        else:
            _logger.warning("Artefato ML global não encontrado: %s", global_path)

        # Manifest (segmentos)
        manifest_path = self._ml_dir / "manifest.json"
        if manifest_path.exists():
            with manifest_path.open(encoding="utf-8") as fh:
                self._ml_manifest = json.load(fh)

            import joblib
            for seg_name, seg_file in self._ml_manifest.get("segment_files", {}).items():
                seg_path = self._ml_dir / seg_file
                if seg_path.exists():
                    pipe = joblib.load(seg_path)
                    norm = MLNormalizer.from_artifact(seg_path)
                    self.ml_segments[seg_name] = (pipe, norm)
                    _logger.info("ML segmento '%s' carregado", seg_name)
        else:
            _logger.warning("Manifest ML não encontrado: %s", manifest_path)

    # ── Consulta ─────────────────────────────────────────────────────────

    def list_segments(self) -> dict[str, list[str]]:
        """
        Lista os segmentos (tipos de máquina) disponíveis em cada engine.

        Returns
        -------
        dict com:
            ``"dl"`` : list[str] — segmentos DL disponíveis
            ``"ml"`` : list[str] — segmentos ML disponíveis
        """
        return {
            "dl": sorted(self.dl_segments.keys()),
            "ml": sorted(self.ml_segments.keys()),
        }

    def model_info(self) -> dict:
        """
        Retorna informações sobre os modelos carregados.

        Returns
        -------
        dict com:
            ``"dl_global"``    : bool — modelo DL global disponível
            ``"ml_global"``    : bool — modelo ML global disponível
            ``"dl_segments"``  : list[str] — segmentos DL
            ``"ml_segments"``  : list[str] — segmentos ML
            ``"dl_metrics"``   : dict | None — métricas do manifest DL
            ``"ml_metrics"``   : dict | None — métricas do manifest ML
        """
        return {
            "dl_global":   self.dl_global_model is not None,
            "ml_global":   self.ml_global_pipe is not None,
            "dl_segments": sorted(self.dl_segments.keys()),
            "ml_segments": sorted(self.ml_segments.keys()),
            "dl_metrics":  self._dl_manifest.get("metrics") if self._dl_manifest else None,
            "ml_metrics":  self._ml_manifest.get("metrics") if self._ml_manifest else None,
        }

    # ── Predição ─────────────────────────────────────────────────────────

    def predict(
        self,
        df: pl.DataFrame,
        engine: Literal["dl", "ml", "both"] = "both",
        mode: Literal["global", "segment", "both", "auto"] = "auto",
        segment: str | None = None,
    ) -> dict[str, np.ndarray | None]:
        """
        Realiza predições de consumo HVAC (kWh).

        Parameters
        ----------
        df : pl.DataFrame
            Dados no formato do ``final_dataframe.csv`` (13 colunas brutas).
            A coluna ``consumo_kwh`` é removida automaticamente se presente.
        engine : ``"dl"`` | ``"ml"`` | ``"both"``
            Qual(is) engine(s) usar. Padrão: ambos.
        mode : ``"global"`` | ``"segment"`` | ``"both"`` | ``"auto"``
            Escopo da predição:

            - ``"global"``  — executa apenas os modelos globais.
            - ``"segment"`` — executa apenas os modelos segmentados
              (requer ``segment``).
            - ``"both"``    — executa modelos globais **e** segmentados
              (requer ``segment``).
            - ``"auto"``    — (padrão) global sempre; segmentado apenas
              se ``segment`` for fornecido (compatível com comportamento
              anterior).
        segment : str | None
            Nome do segmento (tipo de máquina) para predição segmentada.
            Obrigatório quando ``mode`` é ``"segment"`` ou ``"both"``.
            Linhas cujo ``machine_type`` difere do segmento recebem NaN.

        Returns
        -------
        dict com chaves (presentes conforme engine/mode escolhidos):
            ``"dl_global"``       : np.ndarray | None — predições DL global
            ``"ml_global"``       : np.ndarray | None — predições ML global
            ``"dl_segment"``      : np.ndarray | None — predições DL segmentado
            ``"ml_segment"``      : np.ndarray | None — predições ML segmentado
            ``"segment_name"``    : str | None — nome do segmento utilizado
            ``"n_rows"``          : int — quantidade de linhas preditas

        Raises
        ------
        ValueError
            Colunas ausentes, segmento não disponível, mode/engine inválido,
            ou ``segment`` ausente quando ``mode`` exige.
        RuntimeError
            Modelo solicitado não carregado.

        Examples
        --------
        >>> pred.predict(df, engine="dl", mode="global")   # só DL global
        >>> pred.predict(df, engine="ml", mode="segment",  # só ML segmentado
        ...              segment="SPLITÃO")
        >>> pred.predict(df, mode="both", segment="SPLIT HI-WALL")  # tudo
        """
        # ── Resolução do mode ────────────────────────────────────────────
        if mode == "auto":
            # Comportamento legado: global sempre, segment se fornecido
            run_global = True
            run_segment = segment is not None
        elif mode == "global":
            run_global = True
            run_segment = False
        elif mode == "segment":
            if segment is None:
                raise ValueError(
                    "mode='segment' requer o parâmetro 'segment' "
                    "(nome do tipo de máquina)."
                )
            run_global = False
            run_segment = True
        elif mode == "both":
            if segment is None:
                raise ValueError(
                    "mode='both' requer o parâmetro 'segment' "
                    "(nome do tipo de máquina)."
                )
            run_global = True
            run_segment = True
        else:
            raise ValueError(
                f"mode inválido: {mode!r}. "
                f"Use 'global', 'segment', 'both' ou 'auto'."
            )

        df = _validate_input(df)
        n = len(df)

        result: dict[str, np.ndarray | None] = {
            "dl_global":    None,
            "ml_global":    None,
            "dl_segment":   None,
            "ml_segment":   None,
            "segment_name": segment,
            "n_rows":       n,
        }

        # ── Predição DL Global ───────────────────────────────────────────
        if run_global and engine in ("dl", "both"):
            if self.dl_global_model is None:
                raise RuntimeError(
                    "Modelo DL global não carregado. Verifique os artefatos em "
                    f"{self._dl_dir / 'global'}"
                )
            inputs = self.dl_global_norm.transform(df)
            result["dl_global"] = (
                self.dl_global_model.predict(inputs, verbose=0)
                .flatten()
                .astype(np.float32)
            )
            _logger.info("DL global: %d predições", n)

        # ── Predição ML Global ───────────────────────────────────────────
        if run_global and engine in ("ml", "both"):
            if self.ml_global_pipe is None:
                raise RuntimeError(
                    "Pipeline ML global não carregado. Verifique os artefatos em "
                    f"{self._ml_dir / 'global'}"
                )
            X = self.ml_global_norm.transform(df)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
                result["ml_global"] = (
                    self.ml_global_pipe.model.predict(X)
                    .flatten()
                    .astype(np.float32)
                )
            _logger.info("ML global: %d predições", n)

        # ── Predição Segmentada ──────────────────────────────────────────
        if run_segment and segment is not None:
            norm_types = _normalize_machine_types(df)
            type_mask = (norm_types == segment).to_numpy()

            n_match = int(type_mask.sum())
            n_other = n - n_match
            if n_other > 0:
                _logger.info(
                    "Segmento '%s': %d de %d linhas ignoradas (machine_type diferente)",
                    segment, n_other, n,
                )

            # DL segmentado
            if engine in ("dl", "both"):
                if segment not in self.dl_segments:
                    _logger.warning(
                        "Segmento DL '%s' não disponível. Disponíveis: %s",
                        segment, sorted(self.dl_segments.keys()),
                    )
                else:
                    seg_model, seg_norm = self.dl_segments[segment]
                    y_seg = np.full(n, np.nan, dtype=np.float32)
                    if type_mask.any():
                        df_valid = df.filter(pl.Series(type_mask))
                        seg_inputs = seg_norm.transform(df_valid)

                        # Valida embedding bounds
                        emb_cols_limits = {
                            "grupo_regional": seg_norm.n_groups,
                            "hora":           seg_norm.n_horas,
                            "mes":            seg_norm.n_meses,
                            "periodo_dia":    seg_norm.n_periodos,
                        }
                        emb_valid = np.ones(len(df_valid), dtype=bool)
                        for col, max_dim in emb_cols_limits.items():
                            arr = seg_inputs[col].flatten()
                            emb_valid &= (arr >= 0) & (arr < max_dim)

                        if not emb_valid.all():
                            n_oob = int((~emb_valid).sum())
                            _logger.warning(
                                "DL segmento '%s': %d linhas com embedding OOB → NaN",
                                segment, n_oob,
                            )
                            # Filtra inputs válidos
                            for key in seg_inputs:
                                seg_inputs[key] = seg_inputs[key][emb_valid]

                        if seg_inputs["dense_features"].shape[0] > 0:
                            preds = (
                                seg_model.predict(seg_inputs, verbose=0)
                                .flatten()
                                .astype(np.float32)
                            )
                            # Mapeia de volta para posições corretas
                            valid_indices = np.where(type_mask)[0]
                            if not emb_valid.all():
                                valid_indices = valid_indices[emb_valid]
                            y_seg[valid_indices] = preds

                    result["dl_segment"] = y_seg
                    _logger.info(
                        "DL segmento '%s': %d/%d preditas", segment, n_match, n
                    )

            # ML segmentado
            if engine in ("ml", "both"):
                if segment not in self.ml_segments:
                    _logger.warning(
                        "Segmento ML '%s' não disponível. Disponíveis: %s",
                        segment, sorted(self.ml_segments.keys()),
                    )
                else:
                    seg_pipe, seg_norm = self.ml_segments[segment]
                    y_seg = np.full(n, np.nan, dtype=np.float32)
                    if type_mask.any():
                        df_valid = df.filter(pl.Series(type_mask))
                        X_seg = seg_norm.transform(df_valid)
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
                            preds = (
                                seg_pipe.model.predict(X_seg)
                                .flatten()
                                .astype(np.float32)
                            )
                        valid_indices = np.where(type_mask)[0]
                        y_seg[valid_indices] = preds

                    result["ml_segment"] = y_seg
                    _logger.info(
                        "ML segmento '%s': %d/%d preditas", segment, n_match, n
                    )

        return result

    # ── Predição com comparativo (DL vs ML) ──────────────────────────────

    def compare(
        self,
        df: pl.DataFrame,
        mode: Literal["global", "segment", "both", "auto"] = "auto",
        segment: str | None = None,
    ) -> pl.DataFrame:
        """
        Executa predição com ambos os engines e retorna um DataFrame
        comparativo lado a lado.

        Se o DataFrame de entrada contiver ``consumo_kwh``, ele é incluído
        como referência (``consumo_real``), permitindo avaliação direta.

        Parameters
        ----------
        df : pl.DataFrame
            Dados de entrada (pode conter ``consumo_kwh``).
        mode : ``"global"`` | ``"segment"`` | ``"both"`` | ``"auto"``
            Escopo da predição (ver ``predict()`` para detalhes).
        segment : str | None
            Segmento para predição segmentada.

        Returns
        -------
        pl.DataFrame com colunas (presentes conforme mode/engine):
            - ``idx``            : índice da linha
            - ``machine_type``   : tipo de máquina original
            - ``consumo_real``   : valor real (se disponível, senão None)
            - ``pred_dl_global`` : predição DL global
            - ``pred_ml_global`` : predição ML global
            - ``pred_dl_seg``    : predição DL segmentada (se solicitada)
            - ``pred_ml_seg``    : predição ML segmentada (se solicitada)
        """
        # Preserva consumo real se disponível
        y_real = None
        if _TARGET in df.columns:
            y_real = df[_TARGET].to_numpy().astype(np.float32)

        result = self.predict(df, engine="both", mode=mode, segment=segment)

        cols = {
            "idx": list(range(result["n_rows"])),
            "machine_type": df["machine_type"].to_list() if "machine_type" in df.columns else ["?"] * result["n_rows"],
        }

        if y_real is not None:
            cols["consumo_real"] = y_real.tolist()

        if result["dl_global"] is not None:
            cols["pred_dl_global"] = result["dl_global"].tolist()
        if result["ml_global"] is not None:
            cols["pred_ml_global"] = result["ml_global"].tolist()
        if result["dl_segment"] is not None:
            cols["pred_dl_seg"] = result["dl_segment"].tolist()
        if result["ml_segment"] is not None:
            cols["pred_ml_seg"] = result["ml_segment"].tolist()

        return pl.DataFrame(cols)

    # ── Métricas rápidas ─────────────────────────────────────────────────

    @staticmethod
    def evaluate(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """
        Calcula métricas de avaliação entre valores reais e preditos.

        Returns
        -------
        dict com: MAE, RMSE, R2, WMAPE, Acuracia
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        yt, yp = y_true[mask], y_pred[mask]

        if len(yt) == 0:
            return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                    "WMAPE": np.nan, "Acuracia": np.nan}

        mae  = mean_absolute_error(yt, yp)
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2   = r2_score(yt, yp)
        denom = float(np.abs(yt).sum())
        wmape = (float(np.abs(yt - yp).sum()) / denom * 100) if denom > 0 else np.nan
        acc   = 100.0 - wmape if np.isfinite(wmape) else np.nan

        return {
            "MAE":      round(mae, 4),
            "RMSE":     round(rmse, 4),
            "R2":       round(r2, 4),
            "WMAPE":    round(wmape, 2),
            "Acuracia": round(acc, 2),
        }


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO DIRETA — DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    SEP = "═" * 72
    SEP2 = "─" * 72

    # ── 1. Carrega predictor ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  HVAC PREDICTOR — Demo de Predição Unificada")
    print(SEP)

    pred = HVACPredictor(load_ml=True)
    info = pred.model_info()

    print(f"\n  Modelos carregados:")
    print(f"    DL global:    {'✔' if info['dl_global'] else '✘'}")
    print(f"    ML global:    {'✔' if info['ml_global'] else '✘'}")
    print(f"    DL segmentos: {len(info['dl_segments'])} — {info['dl_segments']}")
    print(f"    ML segmentos: {len(info['ml_segments'])} — {info['ml_segments']}")

    # ── 2. Input simplificado unitário ──────────────────────────────────
    print(f"\n{SEP2}")
    print("  TESTE — Input simplificado (lat/lon → grupo_regional, estacao auto)")
    print(SEP2)

    row = build_input(
        hora=10, data="2025-07-03", machine_type="splitao",
        Temperatura_C=23.8, Temperatura_Percebida_C=22.5,
        Umidade_Relativa_pct=62.0, Precipitacao_mm=0.1,
        Velocidade_Vento_kmh=23.2, Pressao_Superficial_hPa=973.0,
        is_dac=1, is_dut=0,
        latitude=-8.0707, longitude=-39.1209,
    )
    print(f"\n  Input: {row.to_dicts()[0]}")

    r1 = pred.predict(row, engine="both", mode="global")
    print(f"\n  Predição DL global: {r1['dl_global'][0]:.4f} kWh")
    print(f"  Predição ML global: {r1['ml_global'][0]:.4f} kWh")

    # ── 3. Demonstração dos diferentes modos ─────────────────────────────
    print(f"\n{SEP2}")
    print("  TESTE — Modos de predição (mode + engine)")
    print(SEP2)

    # Segmento disponível para teste (setup manual)
    # segs = pred.list_segments()
    seg_name = "SPLITÃO"
    if seg_name:
        # Remove acentos e converte para minúsculas → chave bruta
        # aceita pelo dicionário de adjust_machine_type
        import unicodedata
        raw_type = unicodedata.normalize("NFKD", seg_name).encode("ascii", "ignore").decode().lower()

        row_seg = build_input(
            hora=10, data="2025-07-03", machine_type=raw_type,
            Temperatura_C=23.8, Temperatura_Percebida_C=22.5,
            Umidade_Relativa_pct=62.0, Precipitacao_mm=0.1,
            Velocidade_Vento_kmh=23.2, Pressao_Superficial_hPa=973.0,
            is_dac=1, is_dut=0,
            latitude=-8.0707, longitude=-39.1209,
        )
        # engine = 'dl', 'ml' ou 'both' / mode = 'auto', 'global', 'segment' ou 'both'
        # Apenas segmentado (sem global) 
        r4 = pred.predict(row_seg, engine="dl", mode="both", segment=seg_name)
        print(f"\n  [engine=dl, mode=both, segment='{seg_name}']")
        print(f"    DL global:  {r4['dl_global']}")
        dl_val = r4['dl_segment'][0] if r4['dl_segment'] is not None else None
        print(f"    DL segment: {dl_val}")

        # Global + segmentado
        r5 = pred.predict(row_seg, engine="ml", mode="both", segment=seg_name)
        print(f"\n  [engine=ml, mode=both, segment='{seg_name}']")
        print(f"    ML global:  {r5['ml_global'][0]:.4f} kWh" if r5['ml_global'] is not None else "    ML global:  None")
        ml_val = r5['ml_segment'][0] if r5['ml_segment'] is not None else None
        print(f"    ML segment: {ml_val}")

    print(f"\n{SEP}")
    print("  Demo finalizada ✔")
    print(f"{SEP}\n")

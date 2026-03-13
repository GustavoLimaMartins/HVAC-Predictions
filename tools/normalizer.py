"""
Normalizer — Módulo único de normalização para inferência DL e ML
=================================================================

Converte dados no **schema inicial** (11 features brutas com latitude/longitude,
sem target) para o formato exato usado no treinamento, eliminando duplicação de
lógica entre ``DLPipeline.predict()``, ``tools/dl_pred.py`` e
``testing/dl_model.py``.

**MUDANÇA IMPORTANTE (Message 13):** grupo_regional é agora derivado AUTOMATICAMENTE
a partir de latitude/longitude usando o arquivo geo_reference.parquet (mapa DBSCAN
do treinamento com BallTree Haversine KNN-1 para lookup). Não é aceito como input.

Fluxo:

    dados brutos (11 cols: hora, data, lat, lon, machine_type, clima)
        │
        ├─ DLNormalizer.transform(df)
        │       0. FeatureDeriver.derive()
        │          ├─ Derivação de features de data (ano, mes, trimestre, etc)
        │          └─ Derivação de grupo_regional via BallTree KNN-1 (lat/lon → geo_reference.parquet)
        │       1. Insere dummy target (0.0)
        │       2. Extrai hora, mes, grupo_regional, periodo_dia brutos (Int32)
        │       3. ModelSchema.build()  → OHE, Clipping+MinMax, date features
        │       4. Remove artefatos ML incompatíveis (periodo_dia_*, mes cat, grupo cat)
        │       5. Adiciona Entity Embeddings Int32
        │       6. Alinha colunas com feature_columns_ do treino
        │       7. Sanitiza NaN/inf → 0.0
        │       8. Separa X_emb (dict int32) + X_dense (float32 array)
        │       └─ Retorna dict pronto para model.predict()
        │
        └─ MLNormalizer.transform(df)
                0. FeatureDeriver.derive()
                   ├─ Derivação de features de data (ano, mes, trimestre, etc)
                   └─ Derivação de grupo_regional via BallTree KNN-1 (lat/lon → geo_reference.parquet)
                1. Insere dummy target (0.0)
                2. ModelSchema.build()  → OHE, Clipping+MinMax, date features,
                   Target Encoding, Categorical encoding
                3. Alinha colunas com feature_columns_ do treino
                4. Sanitiza NaN/inf → 0.0
                └─ Retorna np.ndarray pronto para model.predict()

Uso:

    >>> from tools.normalizer import DLNormalizer, MLNormalizer
    >>>
    >>> # Deep Learning
    >>> norm = DLNormalizer.from_artifact("model/artifacts/dl_hvac/global")
    >>> inputs = norm.transform(df_raw)          # dict p/ model.predict()
    >>>                                           # df_raw REQUER: latitude, longitude
    >>> preds  = model.predict(inputs, verbose=0).flatten()
    >>>
    >>> # Machine Learning
    >>> norm = MLNormalizer.from_artifact("model/artifacts/ml_hvac/global")
    >>> X = norm.transform(df_raw)               # np.ndarray
    >>>                                          # df_raw REQUER: latitude, longitude
    >>> preds = pipeline.predict(X)
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.neighbors import BallTree

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from model.pre_process.schema import ModelSchema

_logger = logging.getLogger(__name__)

_TARGET: str = "consumo_kwh"

_SCHEMA_FIELDS: list[str] = [
    "hora", "data", "consumo_kwh", "machine_type",
    "estacao", "grupo_regional",
    "Temperatura_C", "Temperatura_Percebida_C",
    "Umidade_Relativa_%", "Precipitacao_mm",
    "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
]

# Colunas de Entity Embedding no DL (ordem fixa)
_EMB_COLS: list[str] = ["grupo_regional", "hora", "mes", "periodo_dia"]

# Caminho do artefato geográfico (mapa de coordenadas únicas → grupo_regional)
_GEO_REF_PATH = _ROOT / "use_case" / "files" / "geo_reference.parquet"

# Cache do lookup geográfico (BallTree + labels carregados do artefato)
_geo_tree: BallTree | None = None
_geo_labels: np.ndarray | None = None


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE DERIVER — Reaproveita ModelSchema + geo_reference.parquet
# ══════════════════════════════════════════════════════════════════════════════

def _get_geo_lookup() -> tuple[BallTree, np.ndarray]:
    """
    Carrega o artefato ``geo_reference.parquet`` (gerado pelo DBSCAN
    no treinamento) e devolve um ``BallTree`` Haversine + vetor de labels.

    O artefato contém as coordenadas únicas de treinamento já
    rotuladas com ``grupo_regional``. Na inferência basta localizar
    o vizinho mais próximo via KNN-1 Haversine — nenhum re-treinamento ocorre.
    
    Returns:
        tuple[BallTree, np.ndarray]: (tree, labels)
        
    Raises:
        FileNotFoundError: Se geo_reference.parquet não existe
    """
    global _geo_tree, _geo_labels
    if _geo_tree is not None:
        return _geo_tree, _geo_labels  # type: ignore[return-value]

    if not _GEO_REF_PATH.exists():
        raise FileNotFoundError(
            f"Artefato de referência geográfica não encontrado:\n"
            f"  {_GEO_REF_PATH}\n"
            f"Execute o pipeline de treinamento primeiro para gerar geo_reference.parquet"
        )

    _logger.info("Carregando referência geográfica de %s ...", _GEO_REF_PATH.name)
    ref = pl.read_parquet(_GEO_REF_PATH)
    coords_rad = np.radians(ref.select(["latitude", "longitude"]).to_numpy())
    _geo_labels = ref["grupo_regional"].to_numpy()
    _geo_tree = BallTree(coords_rad, metric="haversine")
    return _geo_tree, _geo_labels


def _assign_grupo_regional_knn(df: pl.DataFrame) -> pl.DataFrame:
    """
    Atribui ``grupo_regional`` a cada linha via KNN-1 Haversine sobre
    as coordenadas de referência do treinamento (geo_reference.parquet).

    • Linhas com lat/lon nulos recebem ``null``.
    • Coordenadas já conhecidas retornam o rótulo exato do treino.
    • Coordenadas novas recebem o grupo do vizinho mais próximo.
    
    Args:
        df: DataFrame com colunas 'latitude' e 'longitude'
        
    Returns:
        DataFrame com coluna 'grupo_regional' adicionada (Int32)
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

    # Lookup em batch: converte para radianos e consulta BallTree
    if query_idxs:
        coords_rad = np.radians(np.array(query_coords))
        _, indices = tree.query(coords_rad, k=1)
        for idx, ref_idx in zip(query_idxs, indices.ravel()):
            result[idx] = int(labels[ref_idx])

    return df.with_columns(
        pl.Series("grupo_regional", result, dtype=pl.Int32)
    )


class FeatureDeriver:
    """
    Reaproveita ModelSchema.add_date_features() + geo_reference.parquet para auto-derivar features.
    
    Delega a ModelSchema a responsabilidade de derivar features de data e usa
    o artefato geo_reference.parquet (gerado no treinamento via DBSCAN) para
    derivar grupo_regional de coordenadas via KNN-1 Haversine.
    
    **Features derivadas automaticamente:**
        1. ano, mes, dia          ← De 'data'
        2. trimestre              ← De mes (Q1-Q4)
        3. periodo_dia            ← De hora (Madrugada/Manhã/Tarde/Noite)
        4. is_feriado             ← De data (calendário BR)
        5. is_vespera_feriado     ← De data
        6. is_dia_util            ← De weekday + feriado
        7. estacao                ← De mes (Verão/Outono/Inverno/Primavera)
        8. grupo_regional         ← De latitude/longitude via BallTree Haversine KNN-1
    
    **Observação:** grupo_regional é derivado via lookup de coordenadas geográficas
    usando o mapa pré-computado no treinamento (geo_reference.parquet). 
    Requer coordenadas válidas no input.
    
    Uso:
        >>> df = pl.read_csv("data.csv")
        >>> df = FeatureDeriver.derive(df)  # Precisa de: data, hora, latitude, longitude, machine_type, clima
        >>> # Agora df contém todas as features derivadas
    """
    
    @staticmethod
    def derive(df: pl.DataFrame) -> pl.DataFrame:
        """
        Auto-deriva features usando ModelSchema.add_date_features() + geo lookup.
        
        Reusa a lógica centralizada de ModelSchema para derivar features de data,
        e o artefato geo_reference.parquet para derivar grupo_regional a partir 
        de coordenadas via KNN-1 Haversine.
        
        Args:
            df: DataFrame com as features de input (hora, data, latitude, longitude, 
                machine_type, clima, etc). Requer 'latitude' e 'longitude'.
        
        Returns:
            DataFrame com as features derivadas de data e geográficas adicionadas
            
        Raises:
            ValueError: Se latitude/longitude não estão presentes no DataFrame
            FileNotFoundError: Se geo_reference.parquet não existe
        """
        # Valida presença de coordenadas
        if "latitude" not in df.columns or "longitude" not in df.columns:
            raise ValueError(
                "FeatureDeriver.derive() requer colunas 'latitude' e 'longitude' "
                "para derivar grupo_regional via lookup geográfico"
            )
        
        # Normaliza nome de coluna: tipo_maquina → machine_type se necessário
        if "tipo_maquina" in df.columns and "machine_type" not in df.columns:
            df = df.rename({"tipo_maquina": "machine_type"})
        
        # Insere dummy target (obrigatório para ModelSchema)
        if "consumo_kwh" not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias("consumo_kwh"))
        
        # ── Deriva features de DATA ──────────────────────────────────────
        # Delega derivação de features de data ao ModelSchema
        schema = ModelSchema.__new__(ModelSchema)
        schema.df = df.clone()
        schema._schema_fields = []
        
        # Chama add_date_features() para derivar: mes, trimestre, 
        # is_feriado, is_vespera_feriado, is_dia_util, periodo_dia
        schema.add_date_features()
        df = schema.df
        
        # Adiciona estacao (derivada de mes, não é feita por ModelSchema)
        # Mapeamento: Verão (12,1,2), Outono (3,4,5), Inverno (6,7,8), Primavera (9,10,11)
        df = df.with_columns(
            pl.when(pl.col("mes").is_in([12, 1, 2]))
              .then(pl.lit("verao"))
              .when(pl.col("mes").is_in([3, 4, 5]))
              .then(pl.lit("outono"))
              .when(pl.col("mes").is_in([6, 7, 8]))
              .then(pl.lit("inverno"))
              .when(pl.col("mes").is_in([9, 10, 11]))
              .then(pl.lit("primavera"))
              .otherwise(pl.lit("desconhecida"))
              .alias("estacao")
        )
        
        # ── Deriva GRUPO REGIONAL via BallTree Haversine KNN-1 ──────────
        # Reaproveita geo_reference.parquet (gerado no treinamento)
        df = _assign_grupo_regional_knn(df)
        
        return df




# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sanitize(df: pl.DataFrame) -> pl.DataFrame:
    """Substitui NaN/±inf por 0.0 em colunas float."""
    for c in df.columns:
        if df[c].dtype in (pl.Float32, pl.Float64):
            if df[c].is_nan().any() or df[c].is_infinite().any():
                dt = np.float32 if df[c].dtype == pl.Float32 else np.float64
                arr = df[c].to_numpy().copy().astype(dt)
                arr = np.where(np.isfinite(arr), arr, dt(0.0))
                df = df.with_columns(pl.Series(c, arr))
    return df


def _align_columns(df: pl.DataFrame, expected: list[str]) -> pl.DataFrame:
    """Garante que df tenha exatamente as colunas esperadas, na ordem certa."""
    # Adiciona colunas ausentes (OHE de categorias não presentes na amostra)
    missing = set(expected) - set(df.columns)
    if missing:
        df = df.with_columns(
            [pl.lit(0.0, dtype=pl.Float32).alias(c) for c in missing]
        )
    # Remove colunas extras e reordena
    return df.select(expected)


def _ensure_target(df: pl.DataFrame) -> pl.DataFrame:
    """Insere coluna dummy de target se ausente (necessária para ModelSchema)."""
    if _TARGET not in df.columns:
        df = df.with_columns(pl.lit(0.0).alias(_TARGET))
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  DL NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DLNormalizer:
    """
    Normaliza dados brutos para o formato do modelo Deep Learning (.keras).

    Reproduz exatamente o fluxo de ``DLSchema.build()`` + alinhamento +
    sanitização + separação Embeddings/Dense, usando os metadados do
    artefato treinado para garantir consistência.

    Attributes:
        feature_columns : Nomes das features densas (Fluxo B), na ordem do treino.
        n_groups        : input_dim do Embedding de grupo_regional.
        n_horas         : input_dim do Embedding de hora.
        n_meses         : input_dim do Embedding de mes.
        n_periodos      : input_dim do Embedding de periodo_dia.
    """

    feature_columns: list[str]
    n_groups:        int
    n_horas:         int
    n_meses:         int
    n_periodos:      int

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_artifact(cls, path: str | Path) -> "DLNormalizer":
        """
        Carrega os metadados de normalização do artefato treinado.

        Args:
            path: Diretório contendo ``meta.json`` (gerado por DLPipeline.save()).

        Returns:
            DLNormalizer configurado com os parâmetros do treino.
        """
        meta_path = Path(path) / "meta.json"
        with meta_path.open(encoding="utf-8") as fh:
            meta = json.load(fh)

        return cls(
            feature_columns=meta["feature_columns"],
            n_groups=meta["n_groups"],
            n_horas=meta.get("n_horas", 24),
            n_meses=meta.get("n_meses", 13),
            n_periodos=meta.get("n_periodos", 4),
        )

    # ── Transformação ────────────────────────────────────────────────────

    def transform(self, df: pl.DataFrame) -> dict[str, np.ndarray]:
        """
        Converte DataFrame bruto no dict de inputs do modelo .keras.

        Pipeline:
            0. Derive features (ano, mes, dia, estacao, grupo_regional via lat/lng, etc) se ausentes
            1. Insere dummy target (0.0) se ausente
            2. Extrai hora, mes, grupo_regional, periodo_dia brutos (Int32)
            3. ModelSchema.build() → OHE, Clipping+MinMax, date features
            4. Remove artefatos ML (periodo_dia_*, mes cat, grupo_regional cat)
            5. Adiciona Entity Embeddings como Int32
            6. Sanitiza NaN/inf → 0.0
            7. Alinha colunas densas com feature_columns_ do treino
            8. Separa X_emb + X_dense

        Args:
            df: DataFrame com 11 features de input: hora, data, latitude, longitude, 
                machine_type, Temperatura_*, Umidade_*, Precipitacao_*, Velocidade_*, Pressao_*.
                
                **NOTA IMPORTANTE:** Requer 'latitude' e 'longitude' para derivar 
                'grupo_regional' via lookup geográfico (BallTree Haversine). 
                Não aceita 'grupo_regional' pré-computado.

        Returns:
            dict com chaves:
                - ``"grupo_regional"`` : int32 (n, 1)
                - ``"hora"``           : int32 (n, 1)
                - ``"mes"``            : int32 (n, 1)
                - ``"periodo_dia"``    : int32 (n, 1)
                - ``"dense_features"`` : float32 (n, d)
        """
        # ── 0. Auto-deriva features ausentes ─────────────────────────────
        df = FeatureDeriver.derive(df)
        
        # ── 0b. Renomeia tipo_maquina → machine_type para ModelSchema ──
        if "tipo_maquina" in df.columns and "machine_type" not in df.columns:
            df = df.rename({"tipo_maquina": "machine_type"})
        
        df = _ensure_target(df)

        # ── 1. Extrai valores brutos para Entity Embeddings ──────────────
        hora_arr  = df["hora"].to_numpy().astype(np.int32)
        
        # mes já foi derivado por FeatureDeriver, não precisa acessar data
        mes_arr   = df["mes"].to_numpy().astype(np.int32)
        grupo_arr = df["grupo_regional"].to_numpy().astype(np.int32)

        periodo_arr = np.where(
            hora_arr <= 6, 0,
            np.where(hora_arr <= 11, 1,
                     np.where(hora_arr <= 18, 2, 3)),
        ).astype(np.int32)

        # ── 2. ModelSchema: transformações pós-derivação ──────────────────
        # Apenas aplicamos: adjust_machine_type, categorical, OHE, clipping+minmax
        
        # Cria schema_fields sem 'data' (já foi removida por add_date_features)
        schema_fields_no_data = [f for f in _SCHEMA_FIELDS if f != "data"]
        
        # Cria schema temporário para usar métodos de transformação
        schema = ModelSchema.__new__(ModelSchema)
        schema.df = df.clone()
        schema._schema_fields = schema_fields_no_data
        
        # Aplica transformações (sem add_date_features() que precisa de 'data')
        schema.adjust_machine_type()
        schema.make_categorical_columns(["grupo_regional"])
        schema.make_one_hot_encode_columns(["tipo_maquina", "estacao", "periodo_dia"])
        schema.make_clipping_min_max_columns([
            "Temperatura_C", "Temperatura_Percebida_C",
            "Umidade_Relativa_%", "Precipitacao_mm",
            "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
        ])
        
        df_ml = schema.df

        # ── 3. Remove artefatos incompatíveis com DL ─────────────────────
        periodo_ohe = [c for c in df_ml.columns if c.startswith("periodo_dia_")]
        drop_cols = periodo_ohe + [
            c for c in ("mes", "grupo_regional") if c in df_ml.columns
        ]
        df_dl = df_ml.drop(drop_cols)

        # ── 4. Adiciona Entity Embeddings como Int32 ─────────────────────
        df_dl = df_dl.with_columns([
            pl.Series("hora",           hora_arr),
            pl.Series("mes",            mes_arr),
            pl.Series("grupo_regional", grupo_arr),
            pl.Series("periodo_dia",    periodo_arr),
        ])

        # ── 5. Sanitiza ─────────────────────────────────────────────────
        df_dl = _sanitize(df_dl)

        # ── 6. Valida limites dos Embeddings ─────────────────────────────
        self._validate_embeddings(hora_arr, mes_arr, grupo_arr, periodo_arr)

        # ── 7. Separa Embeddings e Dense ─────────────────────────────────
        X_emb = {
            col: df_dl[col].to_numpy().astype(np.int32).reshape(-1, 1)
            for col in _EMB_COLS
        }

        # Alinha features densas com a ordem do treino
        dense_df = _align_columns(
            df_dl.drop([_TARGET] + _EMB_COLS),
            self.feature_columns,
        )
        X_dense = dense_df.select(
            [pl.col(c).cast(pl.Float32) for c in self.feature_columns]
        ).to_numpy()

        return {**X_emb, "dense_features": X_dense}

    # ── Validação interna ────────────────────────────────────────────────

    def _validate_embeddings(
        self,
        hora: np.ndarray,
        mes: np.ndarray,
        grupo: np.ndarray,
        periodo: np.ndarray,
    ) -> None:
        """Loga warnings se algum valor excede o input_dim do Embedding."""
        checks = [
            ("hora",           hora,    self.n_horas),
            ("mes",            mes,     self.n_meses),
            ("grupo_regional", grupo,   self.n_groups),
            ("periodo_dia",    periodo, self.n_periodos),
        ]
        for name, arr, max_dim in checks:
            oob = int((arr >= max_dim).sum())
            if oob > 0:
                _logger.warning(
                    "Embedding '%s': %d valor(es) >= input_dim (%d) — "
                    "lookup out-of-bounds! max encontrado=%d",
                    name, oob, max_dim, int(arr.max()),
                )

    # ── Inspeção ─────────────────────────────────────────────────────────

    def inspect(self, df: pl.DataFrame) -> dict[str, object]:
        """
        Aplica transform e retorna relatório detalhado para depuração.

        Returns:
            dict com:
                - ``"n_rows"``           : int
                - ``"n_dense_features"`` : int
                - ``"embeddings"``       : {col: {min, max, n_unique, input_dim, oob}}
                - ``"dense_stats"``      : {col: {dtype, min, max, nulls, nan}}
                - ``"warnings"``         : list[str]
        """
        inputs = self.transform(df)
        warnings: list[str] = []

        emb_report: dict[str, dict] = {}
        emb_limits = {
            "grupo_regional": self.n_groups,
            "hora":           self.n_horas,
            "mes":            self.n_meses,
            "periodo_dia":    self.n_periodos,
        }
        for col in _EMB_COLS:
            arr = inputs[col].flatten()
            oob = int((arr >= emb_limits[col]).sum())
            emb_report[col] = {
                "min":       int(arr.min()),
                "max":       int(arr.max()),
                "n_unique":  int(len(np.unique(arr))),
                "input_dim": emb_limits[col],
                "oob":       oob,
            }
            if oob:
                warnings.append(
                    f"Embedding '{col}': {oob} valor(es) OOB (>= {emb_limits[col]})"
                )

        dense = inputs["dense_features"]
        dense_report: dict[str, dict] = {}
        for i, col in enumerate(self.feature_columns):
            col_data = dense[:, i]
            n_nan = int(np.isnan(col_data).sum())
            dense_report[col] = {
                "dtype":  str(col_data.dtype),
                "min":    float(np.nanmin(col_data)) if len(col_data) else None,
                "max":    float(np.nanmax(col_data)) if len(col_data) else None,
                "nulls":  0,
                "nan":    n_nan,
            }
            if n_nan:
                warnings.append(f"Dense '{col}': {n_nan} NaN(s) após sanitização")

        return {
            "n_rows":           dense.shape[0],
            "n_dense_features": dense.shape[1],
            "embeddings":       emb_report,
            "dense_stats":      dense_report,
            "warnings":         warnings,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  ML NORMALIZER
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MLNormalizer:
    """
    Normaliza dados brutos para o formato do modelo Machine Learning (.joblib).

    Reproduz exatamente o fluxo de ``MLPipeline._build_schema()`` +
    alinhamento + sanitização, usando os metadados do artefato treinado.

    Attributes:
        feature_columns : Nomes das features, na ordem do treino.
        te_map          : Mapa de Target Encoding {col: {mapping, global_mean}}.
    """

    feature_columns: list[str]
    te_map:          dict | None

    # ── Factory ──────────────────────────────────────────────────────────

    @classmethod
    def from_artifact(cls, path: str | Path) -> "MLNormalizer":
        """
        Carrega os metadados de um MLPipeline salvo (.joblib).

        Args:
            path: Caminho para o arquivo .joblib do MLPipeline.

        Returns:
            MLNormalizer configurado com os parâmetros do treino.
        """
        import joblib
        # O .joblib foi salvo com __main__.MLPipeline (executando ml_pipeline.py
        # diretamente) — para o pickle desserializar, as classes precisam estar
        # registradas no namespace __main__.
        import __main__
        from model.ml_pipeline import MLPipeline as _MLP, MLPipelineConfig as _MLPC
        if not hasattr(__main__, "MLPipeline"):
            __main__.MLPipeline = _MLP
        if not hasattr(__main__, "MLPipelineConfig"):
            __main__.MLPipelineConfig = _MLPC
        pipe = joblib.load(path)
        return cls(
            feature_columns=pipe.feature_columns_,
            te_map=getattr(pipe, "_te_map", None),
        )

    # ── Transformação ────────────────────────────────────────────────────

    def transform(self, df: pl.DataFrame) -> np.ndarray:
        """
        Converte DataFrame bruto no array numpy pronto para predict().

        Pipeline (replica ``MLPipeline._build_schema(inference=True)``):
            1. Auto-deriva features (ano, mes, dia, estacao, grupo_regional via lat/lng)
            2. Insere dummy target (0.0) se ausente
            3. add_date_features → adjust_machine_type
            4. Target Encoding de hora e mes (usando mapa do treino)
            5. make_categorical_columns, make_one_hot_encode_columns
            6. make_clipping_min_max_columns
            7. Sanitiza NaN/inf → 0.0
            8. Alinha colunas com feature_columns_ do treino

        Args:
            df: DataFrame com 11 features de input (hora, data, latitude, longitude, 
                machine_type, Temperatura_*, Umidade_*, Precipitacao_*, Velocidade_*, Pressao_*).
                
                **NOTA IMPORTANTE:** Requer 'latitude' e 'longitude' para derivar 
                'grupo_regional' via lookup geográfico (BallTree Haversine).

        Returns:
            np.ndarray float32 (n, d) pronto para model.predict().
        """
        # ── 0. Auto-deriva features ausentes ─────────────────────────────
        df = FeatureDeriver.derive(df)
        
        # ── 0b. Renomeia tipo_maquina → machine_type para ModelSchema ──
        if "tipo_maquina" in df.columns and "machine_type" not in df.columns:
            df = df.rename({"tipo_maquina": "machine_type"})
        
        df = _ensure_target(df)

        # ── 1. Resolve colunas de Target Encoding ────────────────────────
        te_cols = [
            c.replace("_target_enc", "")
            for c in self.feature_columns
            if c.endswith("_target_enc")
        ]

        # ── 2. ModelSchema: transformações pós-derivação ──────────────────
        # Em inferência, as features de data já foram derivadas por FeatureDeriver.
        # Apenas aplicamos: adjust_machine_type, TE, categorical, OHE, clipping+minmax
        
        # Cria schema_fields sem 'data' (já foi removida por add_date_features)
        schema_fields_no_data = [f for f in _SCHEMA_FIELDS if f != "data"]
        
        # Cria schema temporário para usar métodos de transformação
        schema = ModelSchema.__new__(ModelSchema)
        schema.df = df.clone()
        schema._schema_fields = schema_fields_no_data
        
        schema.adjust_machine_type()

        # ── 3. Target Encoding (antes de OHE, na mesma ordem do treino) ──
        if self.te_map and te_cols:
            schema.make_target_encoding_columns(te_cols, encoding_map=self.te_map)

        # ── 4. Categóricas + One-Hot Encoding ────────────────────────────
        schema.make_categorical_columns(["grupo_regional"])
        schema.make_one_hot_encode_columns(["tipo_maquina", "estacao", "periodo_dia"])

        # ── 5. Clipping + MinMax ─────────────────────────────────────────
        schema.make_clipping_min_max_columns([
            "Temperatura_C", "Temperatura_Percebida_C",
            "Umidade_Relativa_%", "Precipitacao_mm",
            "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
        ])

        df_ml = schema.df

        # ── 6. Categorias → códigos numéricos (UInt32) ──────────────────
        cat_cols = [c for c in df_ml.columns if df_ml[c].dtype == pl.Categorical]
        if cat_cols:
            df_ml = df_ml.with_columns(
                [pl.col(c).to_physical().alias(c) for c in cat_cols]
            )

        # ── 7. Sanitiza ─────────────────────────────────────────────────
        df_ml = _sanitize(df_ml)

        # ── 8. Alinha e converte ─────────────────────────────────────────
        df_aligned = _align_columns(df_ml, self.feature_columns)
        return df_aligned.select(
            [pl.col(c).cast(pl.Float32) for c in self.feature_columns]
        ).to_numpy()

    # ── Inspeção ─────────────────────────────────────────────────────────

    def inspect(self, df: pl.DataFrame) -> dict[str, object]:
        """
        Aplica transform e retorna relatório detalhado para depuração.

        Returns:
            dict com:
                - ``"n_rows"``       : int
                - ``"n_features"``   : int
                - ``"feature_stats"`` : {col: {dtype, min, max, nan}}
                - ``"te_columns"``   : list[str] — colunas com Target Encoding
                - ``"warnings"``     : list[str]
        """
        X = self.transform(df)
        warnings: list[str] = []

        feature_stats: dict[str, dict] = {}
        for i, col in enumerate(self.feature_columns):
            col_data = X[:, i]
            n_nan = int(np.isnan(col_data).sum())
            feature_stats[col] = {
                "dtype":  str(col_data.dtype),
                "min":    float(np.nanmin(col_data)) if len(col_data) else None,
                "max":    float(np.nanmax(col_data)) if len(col_data) else None,
                "nan":    n_nan,
            }
            if n_nan:
                warnings.append(f"Feature '{col}': {n_nan} NaN(s) após sanitização")

        te_cols = [c for c in self.feature_columns if c.endswith("_target_enc")]

        return {
            "n_rows":        X.shape[0],
            "n_features":    X.shape[1],
            "feature_stats": feature_stats,
            "te_columns":    te_cols,
            "warnings":      warnings,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO DIRETA — VALIDAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    SEP = "═" * 70

    # ── Carrega amostra do dataset ───────────────────────────────────────
    csv_path = Path(r"use_case\files\final_dataframe.csv")
    df_raw = pl.read_csv(csv_path)
    sample = df_raw.sample(10, seed=42)

    # Remove target para simular inferência
    if _TARGET in sample.columns:
        y_real = sample[_TARGET].to_numpy()
        sample = sample.drop(_TARGET)
    else:
        y_real = None

    # ── DL Normalizer ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DL NORMALIZER — Validação")
    print(SEP)

    dl_artifact = Path("model/artifacts/dl_hvac/global")
    if (dl_artifact / "meta.json").exists():
        dl_norm = DLNormalizer.from_artifact(dl_artifact)
        inputs = dl_norm.transform(sample)

        print(f"\n  Embeddings:")
        for col in _EMB_COLS:
            arr = inputs[col].flatten()
            print(f"    {col:20s}  shape={inputs[col].shape}  "
                  f"min={arr.min()}  max={arr.max()}  unique={len(np.unique(arr))}")

        dense = inputs["dense_features"]
        print(f"\n  Dense features:  shape={dense.shape}")
        print(f"  Feature columns: {len(dl_norm.feature_columns)}")
        for i, col in enumerate(dl_norm.feature_columns):
            v = dense[:, i]
            print(f"    {i:2d}. {col:45s}  min={v.min():.4f}  max={v.max():.4f}")

        # Inspeção detalhada
        report = dl_norm.inspect(sample)
        if report["warnings"]:
            print(f"\n  ⚠ Warnings:")
            for w in report["warnings"]:
                print(f"    - {w}")
        else:
            print(f"\n  ✔ Sem warnings — dados prontos para inferência DL")
    else:
        print(f"  ⚠ Artefato DL não encontrado em: {dl_artifact}")

    # ── ML Normalizer ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ML NORMALIZER — Validação")
    print(SEP)

    ml_artifact = Path("model/artifacts/ml_hvac/global/best_pipeline.joblib")
    if ml_artifact.exists():
        ml_norm = MLNormalizer.from_artifact(ml_artifact)
        X = ml_norm.transform(sample)

        print(f"\n  Array shape: {X.shape}")
        print(f"  Feature columns: {len(ml_norm.feature_columns)}")
        for i, col in enumerate(ml_norm.feature_columns):
            v = X[:, i]
            print(f"    {i:2d}. {col:45s}  min={v.min():.4f}  max={v.max():.4f}")

        # Inspeção detalhada
        report = ml_norm.inspect(sample)
        if report["warnings"]:
            print(f"\n  ⚠ Warnings:")
            for w in report["warnings"]:
                print(f"    - {w}")
        else:
            print(f"\n  ✔ Sem warnings — dados prontos para inferência ML")

        if report["te_columns"]:
            print(f"\n  Target Encoding aplicado em: {report['te_columns']}")
    else:
        print(f"  ⚠ Artefato ML não encontrado em: {ml_artifact}")

    # ── Comparação real vs normalizado — 10 primeiras linhas ────────────
    # Usa as 10 primeiras linhas (determinísticas) para rastreabilidade
    first10 = df_raw.head(10)
    if _TARGET in first10.columns:
        y_real_10 = first10[_TARGET].to_numpy()
        first10_input = first10.drop(_TARGET)
    else:
        y_real_10 = None
        first10_input = first10

    print(f"\n{SEP}")
    print("  VALIDAÇÃO DE INTEGRIDADE — 10 primeiras linhas")
    print("  (valores brutos do CSV  ↔  valores pós-normalização)")
    print(SEP)

    # ── Tabela 1 — Valores brutos originais ──────────────────────────────
    print(f"\n  ┌─ TABELA 1: Valores brutos (CSV original)")
    print(f"  │")
    header_nums = "  {:>3s}  {:>5s}  {:>6s}  {:>8s}  {:>8s}  {:>6s}  {:>6s}  {:>7s}  {:>8s}".format(
        "#", "hora", "g.reg", "Temp_C", "Tp_C", "UR_%", "Prec", "Vento", "Pressao",
    )
    print(header_nums)
    print("  " + "-" * (len(header_nums) - 2))
    for i in range(len(first10)):
        row = first10.row(i, named=True)
        print("  {:>3d}  {:>5d}  {:>6d}  {:>8.2f}  {:>8.2f}  {:>6.1f}  {:>6.1f}  {:>7.1f}  {:>8.1f}".format(
            i + 1,
            int(row["hora"]),
            int(row["grupo_regional"]),
            float(row["Temperatura_C"]),
            float(row["Temperatura_Percebida_C"]),
            float(row["Umidade_Relativa_%"]),
            float(row["Precipitacao_mm"]),
            float(row["Velocidade_Vento_kmh"]),
            float(row["Pressao_Superficial_hPa"]),
        ))

    print(f"\n  {'#':>3s}  {'machine_type':30s}  {'estacao':12s}  {'data':12s}  {'consumo_kwh':>12s}")
    print("  " + "-" * 75)
    for i in range(len(first10)):
        row = first10.row(i, named=True)
        consumo = f"{y_real_10[i]:.4f}" if y_real_10 is not None else "—"
        print("  {:>3d}  {:30s}  {:12s}  {:12s}  {:>12s}".format(
            i + 1,
            str(row["machine_type"]),
            str(row["estacao"]),
            str(row["data"]),
            consumo,
        ))

    # ── Tabela 2 — DL Normalizer: Embeddings reconvertidos ──────────────
    dl_artifact = Path("model/artifacts/dl_hvac/global")
    if (dl_artifact / "meta.json").exists():
        dl_norm = DLNormalizer.from_artifact(dl_artifact)
        dl_inputs = dl_norm.transform(first10_input)

        print(f"\n  ┌─ TABELA 2: DL Normalizer — Entity Embeddings (Int32)")
        print(f"  │  (devem coincidir com os valores brutos ou derivados)")
        print(f"\n  {'#':>3s}  {'hora_emb':>8s}  {'mes_emb':>7s}  {'grupo_emb':>9s}  {'periodo':>7s}  │  {'hora_raw':>8s}  {'mes_raw':>7s}  {'grupo_raw':>9s}  {'periodo_calc':>12s}")
        print("  " + "-" * 95)

        hora_emb    = dl_inputs["hora"].flatten()
        mes_emb     = dl_inputs["mes"].flatten()
        grupo_emb   = dl_inputs["grupo_regional"].flatten()
        periodo_emb = dl_inputs["periodo_dia"].flatten()

        _periodo_names = {0: "Madrugada", 1: "Manhã", 2: "Tarde", 3: "Noite"}

        for i in range(len(first10)):
            row = first10.row(i, named=True)
            raw_hora  = int(row["hora"])
            raw_data  = str(row["data"])
            raw_mes   = int(raw_data.split("-")[1]) if "-" in raw_data else "?"
            raw_grupo = int(row["grupo_regional"])

            # Calcula período esperado
            if raw_hora <= 6:
                exp_periodo = 0
            elif raw_hora <= 11:
                exp_periodo = 1
            elif raw_hora <= 18:
                exp_periodo = 2
            else:
                exp_periodo = 3

            # Marca discrepâncias com ✗
            ok_hora    = "✔" if hora_emb[i] == raw_hora else "✗"
            ok_mes     = "✔" if mes_emb[i] == raw_mes else "✗"
            ok_grupo   = "✔" if grupo_emb[i] == raw_grupo else "✗"
            ok_periodo = "✔" if periodo_emb[i] == exp_periodo else "✗"

            print("  {:>3d}  {:>8d}  {:>7d}  {:>9d}  {:>7d}  │  {:>6d} {}  {:>5} {}  {:>7d} {}  {:>9s} {}".format(
                i + 1,
                int(hora_emb[i]), int(mes_emb[i]),
                int(grupo_emb[i]), int(periodo_emb[i]),
                raw_hora, ok_hora,
                raw_mes, ok_mes,
                raw_grupo, ok_grupo,
                _periodo_names.get(exp_periodo, "?"), ok_periodo,
            ))

        # ── Tabela 3 — DL Dense features (clima normalizado) ────────────
        dense = dl_inputs["dense_features"]
        # Identificar colunas de clima no vetor denso
        _CLIMA_COLS = [
            "Temperatura_C", "Temperatura_Percebida_C",
            "Umidade_Relativa_%", "Precipitacao_mm",
            "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
        ]
        clima_indices = {
            col: dl_norm.feature_columns.index(col)
            for col in _CLIMA_COLS
            if col in dl_norm.feature_columns
        }

        if clima_indices:
            print(f"\n  ┌─ TABELA 3: DL Normalizer — Features contínuas (Clipping+MinMax → [0,1])")
            print(f"  │  (valores normalizados vs. brutos para validação de faixa)")
            print(f"\n  {'#':>3s}  {'Feature':30s}  {'Normalizado':>12s}  {'Bruto':>12s}")
            print("  " + "-" * 62)

            for i in range(len(first10)):
                row = first10.row(i, named=True)
                for col, idx in clima_indices.items():
                    norm_val = float(dense[i, idx])
                    raw_val  = float(row[col])
                    flag = "⚠" if norm_val < -0.01 or norm_val > 1.01 else " "
                    print("  {:>3d}  {:30s}  {:>12.6f}  {:>12.4f} {}".format(
                        i + 1, col, norm_val, raw_val, flag,
                    ))
                if i < len(first10) - 1:
                    print("  " + "·" * 62)
    else:
        print(f"\n  ⚠ DL: artefato não encontrado em {dl_artifact} — tabelas 2-3 omitidas")

    # ── Tabela 4 — ML Normalizer: features reconvertidas ────────────────
    ml_artifact = Path("model/artifacts/ml_hvac/global/best_pipeline.joblib")
    if ml_artifact.exists():
        ml_norm = MLNormalizer.from_artifact(ml_artifact)
        X_ml = ml_norm.transform(first10_input)

        # Identifica colunas de clima, TE e OHE
        clima_ml = {
            col: ml_norm.feature_columns.index(col)
            for col in _CLIMA_COLS
            if col in ml_norm.feature_columns
        }
        te_cols = {
            col: ml_norm.feature_columns.index(col)
            for col in ml_norm.feature_columns
            if col.endswith("_target_enc")
        }
        ohe_tipo = [
            (col, ml_norm.feature_columns.index(col))
            for col in ml_norm.feature_columns
            if col.startswith("tipo_maquina_")
        ]
        ohe_estacao = [
            (col, ml_norm.feature_columns.index(col))
            for col in ml_norm.feature_columns
            if col.startswith("estacao_")
        ]
        ohe_periodo = [
            (col, ml_norm.feature_columns.index(col))
            for col in ml_norm.feature_columns
            if col.startswith("periodo_dia_")
        ]

        print(f"\n  ┌─ TABELA 4: ML Normalizer — Features contínuas (Clipping+MinMax → [0,1])")
        print(f"\n  {'#':>3s}  {'Feature':30s}  {'Normalizado':>12s}  {'Bruto':>12s}")
        print("  " + "-" * 62)
        for i in range(len(first10)):
            row = first10.row(i, named=True)
            for col, idx in clima_ml.items():
                norm_val = float(X_ml[i, idx])
                raw_val  = float(row[col])
                flag = "⚠" if norm_val < -0.01 or norm_val > 1.01 else " "
                print("  {:>3d}  {:30s}  {:>12.6f}  {:>12.4f} {}".format(
                    i + 1, col, norm_val, raw_val, flag,
                ))
            if i < len(first10) - 1:
                print("  " + "·" * 62)

        # ── Tabela 5 — ML Target Encoding ────────────────────────────────
        if te_cols:
            print(f"\n  ┌─ TABELA 5: ML Normalizer — Target Encoding (hora/mes → média suavizada)")
            print(f"\n  {'#':>3s}  {'Feature':20s}  {'TE_value':>12s}  {'Valor bruto':>12s}")
            print("  " + "-" * 52)
            for i in range(len(first10)):
                row = first10.row(i, named=True)
                for col, idx in te_cols.items():
                    base_col = col.replace("_target_enc", "")
                    raw_val = row.get(base_col, "—")
                    if base_col == "mes":
                        raw_data = str(row.get("data", ""))
                        raw_val = int(raw_data.split("-")[1]) if "-" in raw_data else "?"
                    print("  {:>3d}  {:20s}  {:>12.6f}  {:>12s}".format(
                        i + 1, col, float(X_ml[i, idx]), str(raw_val),
                    ))

        # ── Tabela 6 — ML OHE: tipo_maquina ativo ───────────────────────
        if ohe_tipo:
            print(f"\n  ┌─ TABELA 6: ML Normalizer — OHE tipo_maquina (coluna ativa por linha)")
            print(f"\n  {'#':>3s}  {'machine_type (bruto)':30s}  {'OHE ativa':40s}")
            print("  " + "-" * 77)
            for i in range(len(first10)):
                row = first10.row(i, named=True)
                active = [
                    name.replace("tipo_maquina_", "")
                    for name, idx in ohe_tipo
                    if float(X_ml[i, idx]) >= 0.5
                ]
                active_str = ", ".join(active) if active else "(nenhuma)"
                print("  {:>3d}  {:30s}  {:40s}".format(
                    i + 1, str(row["machine_type"]), active_str,
                ))

        # ── Tabela 7 — ML OHE: estacao ativa ────────────────────────────
        if ohe_estacao:
            print(f"\n  ┌─ TABELA 7: ML Normalizer — OHE estacao (coluna ativa por linha)")
            print(f"\n  {'#':>3s}  {'estacao (bruta)':20s}  {'OHE ativa':30s}")
            print("  " + "-" * 57)
            for i in range(len(first10)):
                row = first10.row(i, named=True)
                active = [
                    name.replace("estacao_", "")
                    for name, idx in ohe_estacao
                    if float(X_ml[i, idx]) >= 0.5
                ]
                active_str = ", ".join(active) if active else "(nenhuma)"
                print("  {:>3d}  {:20s}  {:30s}".format(
                    i + 1, str(row["estacao"]), active_str,
                ))

        # ── Tabela 8 — ML OHE: periodo_dia ativo ────────────────────────
        if ohe_periodo:
            print(f"\n  ┌─ TABELA 8: ML Normalizer — OHE periodo_dia (coluna ativa por linha)")
            print(f"\n  {'#':>3s}  {'hora':>5s}  {'OHE ativa':30s}")
            print("  " + "-" * 42)
            for i in range(len(first10)):
                row = first10.row(i, named=True)
                active = [
                    name.replace("periodo_dia_", "")
                    for name, idx in ohe_periodo
                    if float(X_ml[i, idx]) >= 0.5
                ]
                active_str = ", ".join(active) if active else "(nenhuma)"
                print("  {:>3d}  {:>5d}  {:30s}".format(
                    i + 1, int(row["hora"]), active_str,
                ))
    else:
        print(f"\n  ⚠ ML: artefato não encontrado em {ml_artifact} — tabelas 4-8 omitidas")

    # ── Resumo final ─────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RESUMO DE VALIDAÇÃO")
    print(SEP)
    print("  Tabela 1 — Valores brutos originais (CSV)")
    print("  Tabela 2 — DL: Entity Embeddings (hora, mes, grupo, periodo) vs. brutos")
    print("  Tabela 3 — DL: Features contínuas normalizadas vs. brutas")
    print("  Tabela 4 — ML: Features contínuas normalizadas vs. brutas")
    print("  Tabela 5 — ML: Target Encoding (hora/mes → média suavizada)")
    print("  Tabela 6 — ML: OHE tipo_maquina → coluna ativa vs. valor bruto")
    print("  Tabela 7 — ML: OHE estacao → coluna ativa vs. valor bruto")
    print("  Tabela 8 — ML: OHE periodo_dia → coluna ativa vs. hora bruta")
    print(f"\n  ✔ Validação concluída — verifique ✗ e ⚠ nas tabelas acima\n")

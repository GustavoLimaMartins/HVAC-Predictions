"""
API de Inferência — Wrapper unificado para modelos HVAC Deep Learning
======================================================================

Fornece interface simples para predições com normalização automática
usando o DLNormalizer integrado em tools/normalizer.py.

**NOVO (Message 13):** predict_single() agora permite inferencia com valores
manuais, derivando automaticamente grupo_regional via BallTree Haversine
do arquivo geo_reference.parquet (sem re-treinamento DBSCAN).

**REFATORAÇÃO (Message 15):** inference_api.py substitui predictor.py.
HVACPredictor é agora um alias para HVACInferenceAPI para compatibilidade.

**ATUALIZAÇÃO:** Arquivo geo_reference.parquet salvo em use_case\files\
(gerado automaticamente a cada treinamento via RegionalGroupClassifier.export_mapping())

Fluxo:
  1. DLNormalizer.from_artifact() carrega metadata do treinamento
  2. Carrega modelo keras_model.keras
  3. DLNormalizer.transform() normaliza dados (com auto-feature derivation via lat/lon)
     - Lê geo_reference.parquet (use_case\files\) para atribuir grupo_regional
  4. model.predict(inputs) executa inferencia
  5. Retorna predicoes (kWh)

Uso como modulo:
  >>> from tools.inference_api import HVACInferenceAPI
  >>> api = HVACInferenceAPI("model/artifacts/dl_hvac/global")
  >>> 
  >>> # Predicao com valores manuais
  >>> pred = api.predict_single(
  ...     hora=14, data="2025-07-03", machine_type="splitao",
  ...     latitude=-27.24, longitude=-48.63,
  ...     Temperatura_C=25.7, Temperatura_Percebida_C=24.9,
  ...     Umidade_Relativa_pct=57.0, Precipitacao_mm=0.0,
  ...     Velocidade_Vento_kmh=21.4, Pressao_Superficial_hPa=969.8,
  ...     Irradiancia_Direta_Wm2=520.0, Irradiancia_Difusa_Wm2=180.0,
  ... )
  >>> print(f"Consumo previsto: {pred:.2f} kWh")
  >>>
  >>> # Predicao em batch
  >>> predictions = api.predict(df_batch)

Compatibilidade legada (predictor.py):
  >>> from tools.inference_api import HVACPredictor
  >>> pred = HVACPredictor("model/artifacts/dl_hvac/global")  # Alias para HVACInferenceAPI

Uso direto (para testes):
  >>> python -m tools.inference_api
"""

import logging
from pathlib import Path
from typing import Dict
import numpy as np
import polars as pl
import tensorflow as tf

# Import condicional: relativo se rodado como módulo, absoluto se rodado direto
try:
    from .normalizer import DLNormalizer, MLNormalizer
except ImportError:
    from normalizer import DLNormalizer, MLNormalizer

_logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  API DE INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

class HVACDLInferenceAPI:
    """
    API de Inferência para modelos HVAC Deep Learning.

    Carrega modelo e metadata de um artefato treinado via DLNormalizer,
    fornecendo interface simples para predições com normalização automática
    e derivação de features.

    Attributes:
        model_path      : Caminho da pasta com artefatos (contém keras_model.keras e meta.json).
        normalizer      : Instância de DLNormalizer carregada.
        model           : Modelo Keras carregado.
    """

    def __init__(self, model_path: str | Path):
        """
        Inicializa a API carregando modelo e normalizer.

        Args:
            model_path: Caminho para diretório contendo:
                        - keras_model.keras
                        - meta.json (gerado por DLPipeline.save())

        Raises:
            FileNotFoundError: Se arquivos não forem encontrados
            ValueError: Se meta.json inválido
        """
        self.model_path = Path(model_path)
        
        # Carrega normalizer (implicitamente carrega meta.json)
        self.normalizer = DLNormalizer.from_artifact(self.model_path)
        
        # Carrega modelo Keras
        model_file = self.model_path / "keras_model.keras"
        if not model_file.exists():
            raise FileNotFoundError(f"Modelo não encontrado em: {model_file}")
        
        self.model = tf.keras.models.load_model(model_file)
        _logger.info(f"Modelo carregado de {model_file}")

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Executa predição em um DataFrame.

        Pipeline:
            1. Deriva features automáticas (grupo_regional, trimestre, etc)
            2. Normaliza dados usando DLNormalizer.transform()
            3. Executa model.predict()

        Args:
            df: DataFrame com as features de input.
               Colunas esperadas (no mínimo):
               hora, data, machine_type, latitude, longitude,
               Temperatura_C, Temperatura_Percebida_C,
               Umidade_Relativa_%, Precipitacao_mm, Velocidade_Vento_kmh,
               Pressao_Superficial_hPa, Irradiancia_Direta_Wm2,
               Irradiancia_Difusa_Wm2
               (estacao e grupo_regional são derivados automaticamente)

        Returns:
            np.ndarray de predições (consumo em kWh) de shape (n,)
        """
        # Normaliza com DLNormalizer (que já inclui auto-feature derivation)
        inputs = self.normalizer.transform(df)
        
        # Executa predição
        predictions = self.model.predict(inputs, verbose=0).flatten()
        
        _logger.debug(f"Predições: {len(predictions)} linhas, "
                      f"min={predictions.min():.4f}, max={predictions.max():.4f}")
        
        return predictions

    def predict_batch(
        self,
        df: pl.DataFrame,
        batch_size: int = 1024,
    ) -> np.ndarray:
        """
        Executa predição em lotes para otimizar memória.

        Args:
            df: DataFrame com as features
            batch_size: Número de linhas por lote

        Returns:
            np.ndarray com todas as predições concatenadas
        """
        n_rows = len(df)
        predictions = []
        
        for i in range(0, n_rows, batch_size):
            batch = df.slice(i, min(batch_size, n_rows - i))
            batch_preds = self.predict(batch)
            predictions.append(batch_preds)
            _logger.debug(f"Lote {i//batch_size + 1}: {len(batch_preds)} predições")
        
        return np.concatenate(predictions) if predictions else np.array([])

    def get_normalizer_metadata(self) -> Dict[str, object]:
        """
        Retorna metadata do normalizer para inspeção.

        Returns:
            dict com:
                - feature_columns: lista de features densas
                - embedding_config: {grupo_regional, hora, mes, periodo_dia} input_dims
                - geo_reference_path: caminho do arquivo geo_reference.parquet usado
        """
        return {
            "feature_columns": self.normalizer.feature_columns,
            "embedding_config": {
                "grupo_regional": self.normalizer.n_groups,
                "hora": self.normalizer.n_horas,
                "mes": self.normalizer.n_meses,
                "periodo_dia": self.normalizer.n_periodos,
            },
            "geo_reference_path": "use_case/files/geo_reference.parquet",
        }

    def predict_single(
        self,
        hora: int,
        data: str,
        machine_type: str,
        latitude: float,
        longitude: float,
        Temperatura_C: float,
        Temperatura_Percebida_C: float,
        Umidade_Relativa_pct: float,
        Precipitacao_mm: float,
        Velocidade_Vento_kmh: float,
        Pressao_Superficial_hPa: float,
        Irradiancia_Direta_Wm2: float,
        Irradiancia_Difusa_Wm2: float,
    ) -> float:
        """
        Executa predição para um único registro com valores manuais.

        Args:
            hora: Hora do dia (0–23)
            data: Data no formato "YYYY-MM-DD"
            machine_type: Tipo de máquina (ex: "splitao", "split_hi-wall")
            latitude: Latitude em graus
            longitude: Longitude em graus
            Temperatura_C: Temperatura em °C
            Temperatura_Percebida_C: Temperatura percebida em °C
            Umidade_Relativa_pct: Umidade relativa (%)
            Precipitacao_mm: Precipitação em mm
            Velocidade_Vento_kmh: Velocidade do vento em km/h
            Pressao_Superficial_hPa: Pressão em hPa
            Irradiancia_Direta_Wm2: Irradiância direta normal em W/m²
            Irradiancia_Difusa_Wm2: Irradiância difusa horizontal em W/m²

        Returns:
            float: Predição de consumo em kWh

        Example:
            >>> api = HVACInferenceAPI("model/artifacts/dl_hvac/global")
            >>> pred = api.predict_single(
            ...     hora=14,
            ...     data="2025-07-03",
            ...     machine_type="splitao",
            ...     latitude=-27.24,
            ...     longitude=-48.63,
            ...     Temperatura_C=25.7,
            ...     Temperatura_Percebida_C=24.9,
            ...     Umidade_Relativa_pct=57.0,
            ...     Precipitacao_mm=0.0,
            ...     Velocidade_Vento_kmh=21.4,
            ...     Pressao_Superficial_hPa=969.8,
            ...     Irradiancia_Direta_Wm2=520.0,
            ...     Irradiancia_Difusa_Wm2=180.0,
            ... )
            >>> print(f"Consumo previsto: {pred:.2f} kWh")
        """
        df = pl.DataFrame({
            "hora": [hora],
            "data": [data],
            "machine_type": [machine_type],
            "latitude": [latitude],
            "longitude": [longitude],
            "Temperatura_C": [Temperatura_C],
            "Temperatura_Percebida_C": [Temperatura_Percebida_C],
            "Umidade_Relativa_%": [Umidade_Relativa_pct],
            "Precipitacao_mm": [Precipitacao_mm],
            "Velocidade_Vento_kmh": [Velocidade_Vento_kmh],
            "Pressao_Superficial_hPa": [Pressao_Superficial_hPa],
            "Irradiancia_Direta_Wm2": [Irradiancia_Direta_Wm2],
            "Irradiancia_Difusa_Wm2": [Irradiancia_Difusa_Wm2],
        }).with_columns(pl.col("data").cast(pl.Date))
        
        predictions = self.predict(df)
        return float(predictions[0])


# ══════════════════════════════════════════════════════════════════════════════
#  API DE INFERÊNCIA ML
# ══════════════════════════════════════════════════════════════════════════════

class HVACMLInferenceAPI:
    """
    API de Inferência para modelos HVAC Machine Learning (.joblib).

    Espelha a interface de HVACInferenceAPI para pipelines scikit-learn/XGBoost/LGBM
    treinados via MLPipeline (modelos segmentados por tipo de máquina).

    Attributes:
        model_path : Caminho da pasta contendo best_pipeline.joblib.
        normalizer : Instância de MLNormalizer carregada.
        pipeline   : Pipeline sklearn/XGBoost/LGBM carregado.
    """

    def __init__(self, model_path: str | Path):
        """
        Inicializa a API carregando pipeline e normalizer.

        Args:
            model_path: Caminho para diretório contendo best_pipeline.joblib.

        Raises:
            FileNotFoundError: Se best_pipeline.joblib não for encontrado.
        """
        import joblib

        self.model_path = Path(model_path)
        joblib_file = self.model_path / "best_pipeline.joblib"
        if not joblib_file.exists():
            raise FileNotFoundError(f"Modelo ML não encontrado em: {joblib_file}")

        # from_artifact registra MLPipeline/MLPipelineConfig em __main__ antes
        # do segundo joblib.load — sem isso o pickle falha ao desserializar.
        self.normalizer = MLNormalizer.from_artifact(joblib_file)
        self.pipeline   = joblib.load(joblib_file)
        _logger.info(f"Modelo ML carregado de {joblib_file}")

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        """
        Executa predição em um DataFrame.

        MLPipeline.predict() espera um pl.DataFrame sem a coluna target e
        aplica internamente normalização + predição. Não usar self.normalizer
        aqui — ele existe apenas para registrar MLPipeline em __main__ no __init__.

        Args:
            df: DataFrame com as features de input (mesmo schema de HVACDLInferenceAPI.predict).
               A coluna 'consumo_kwh', se presente, é removida automaticamente.

        Returns:
            np.ndarray de predições (consumo em kWh) de shape (n,)
        """
        import warnings

        # MLPipeline.predict() proíbe consumo_kwh no input
        df_input = df.drop("consumo_kwh") if "consumo_kwh" in df.columns else df

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            predictions = self.pipeline.predict(df_input).flatten()

        _logger.debug(f"Predições ML: {len(predictions)} linhas, "
                      f"min={predictions.min():.4f}, max={predictions.max():.4f}")
        return predictions

    def predict_single(
        self,
        hora: int,
        data: str,
        machine_type: str,
        latitude: float,
        longitude: float,
        Temperatura_C: float,
        Temperatura_Percebida_C: float,
        Umidade_Relativa_pct: float,
        Precipitacao_mm: float,
        Velocidade_Vento_kmh: float,
        Pressao_Superficial_hPa: float,
        Irradiancia_Direta_Wm2: float,
        Irradiancia_Difusa_Wm2: float,
    ) -> float:
        """
        Executa predição para um único registro com valores manuais.

        Mesma assinatura de HVACInferenceAPI.predict_single para facilitar
        comparação direta entre DL e ML.

        Returns:
            float: Predição de consumo em kWh
        """
        df = pl.DataFrame({
            "hora":                   [hora],
            "data":                   [data],
            "machine_type":           [machine_type],
            "latitude":               [latitude],
            "longitude":              [longitude],
            "Temperatura_C":          [Temperatura_C],
            "Temperatura_Percebida_C":[Temperatura_Percebida_C],
            "Umidade_Relativa_%":     [Umidade_Relativa_pct],
            "Precipitacao_mm":        [Precipitacao_mm],
            "Velocidade_Vento_kmh":   [Velocidade_Vento_kmh],
            "Pressao_Superficial_hPa":[Pressao_Superficial_hPa],
            "Irradiancia_Direta_Wm2": [Irradiancia_Direta_Wm2],
            "Irradiancia_Difusa_Wm2": [Irradiancia_Difusa_Wm2],
        }).with_columns(pl.col("data").cast(pl.Date))
        return float(self.predict(df)[0])


# ══════════════════════════════════════════════════════════════════════════════
#  TESTE / EXECUÇÃO DIRETA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    root_dir     = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root_dir))

    dl_path      = root_dir / "model" / "artifacts" / "dl_hvac" / "global"
    ml_path      = root_dir / "model" / "artifacts" / "ml_hvac" / "global"
    parquet_path = root_dir / "use_case" / "files" / "final_dataframe.parquet"

    for p in (dl_path / "keras_model.keras", ml_path / "best_pipeline.joblib", parquet_path):
        if not p.exists():
            print(f"\n  ERRO: Arquivo não encontrado: {p}")
            sys.exit(1)

    W = 72
    print("\n" + "=" * W)
    print("  INFERÊNCIA — Modelo Global  DL vs ML")
    print("=" * W)
    print(f"  DL  : {dl_path.relative_to(root_dir)}")
    print(f"  ML  : {ml_path.relative_to(root_dir)}")
    print(f"  Data: {parquet_path.relative_to(root_dir)}")

    # ── Carrega modelos globais ───────────────────────────────────────────────
    try:
        api_dl = HVACDLInferenceAPI(dl_path)
        api_ml = HVACMLInferenceAPI(ml_path)
    except Exception as exc:
        print(f"\n  ERRO ao carregar modelos: {exc}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print("  Modelos carregados com sucesso.")

    # ══════════════════════════════════════════════════════════════════════════
    #  TESTE 1 — Inferência Manual (valores configurados)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "─" * W)
    print("  [TESTE 1] Inferência Manual — valores configurados")
    print("─" * W)

    # Valores de entrada configurados manualmente
    _INPUT = dict(
        hora                 = 14,
        data                 = "2025-07-03",
        machine_type         = "splitao",
        latitude             = -23.883839,
        longitude            = -46.4200317682745,
        Temperatura_C        = 25.7,
        Temperatura_Percebida_C = 24.9,
        Umidade_Relativa_pct = 57.0,
        Precipitacao_mm      = 0.0,
        Velocidade_Vento_kmh = 21.4,
        Pressao_Superficial_hPa = 969.8,
        Irradiancia_Direta_Wm2  = 520.0,
        Irradiancia_Difusa_Wm2  = 180.0,
    )

    print(f"\n  hora={_INPUT['hora']}  data={_INPUT['data']}  machine_type={_INPUT['machine_type']!r}")
    print(f"  lat={_INPUT['latitude']}  lon={_INPUT['longitude']}")
    print(f"  Temp={_INPUT['Temperatura_C']}°C  Umid={_INPUT['Umidade_Relativa_pct']}%"
          f"  Vento={_INPUT['Velocidade_Vento_kmh']}km/h  Pressão={_INPUT['Pressao_Superficial_hPa']}hPa")
    print(f"  IrrDir={_INPUT['Irradiancia_Direta_Wm2']}W/m²  IrrDif={_INPUT['Irradiancia_Difusa_Wm2']}W/m²")

    try:
        pred_dl_1 = api_dl.predict_single(**_INPUT)
    except Exception as exc:
        print(f"  [ERRO DL] {exc}"); pred_dl_1 = float("nan")

    try:
        pred_ml_1 = api_ml.predict_single(**_INPUT)
    except Exception as exc:
        print(f"  [ERRO ML] {exc}"); pred_ml_1 = float("nan")

    delta_1 = pred_dl_1 - pred_ml_1
    pct_1   = (delta_1 / pred_ml_1 * 100) if pred_ml_1 != 0 else float("nan")

    print(f"\n  {'Modelo':<20} {'Predito (kWh)':>14}")
    print(f"  {'─' * 20}  {'─' * 14}")
    print(f"  {'Deep Learning':<20}  {pred_dl_1:>12.4f}")
    print(f"  {'Machine Learning':<20}  {pred_ml_1:>12.4f}")
    print(f"  {'Δ (DL − ML)':<20}  {delta_1:>+12.4f}  ({pct_1:+.1f}%)")

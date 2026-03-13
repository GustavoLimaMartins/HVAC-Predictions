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
  ...     Velocidade_Vento_kmh=21.4, Pressao_Superficial_hPa=969.8
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
    from .normalizer import DLNormalizer
except ImportError:
    from normalizer import DLNormalizer

_logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  API DE INFERÊNCIA
# ══════════════════════════════════════════════════════════════════════════════

class HVACInferenceAPI:
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
            df: DataFrame com as 9 features de input.
               Colunas esperadas (no mínimo):
               hora, tipo_maquina, Temperatura_C, Temperatura_Percebida_C,
               Umidade_Relativa_%, Precipitacao_mm, Velocidade_Vento_kmh,
               Pressao_Superficial_hPa, data (ou mes)

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
        }).with_columns(pl.col("data").cast(pl.Date))
        
        predictions = self.predict(df)
        return float(predictions[0])


# ══════════════════════════════════════════════════════════════════════════════
#  TESTE / EXECUÇÃO DIRETA
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Obtém caminho da raiz do projeto (2 níveis acima deste arquivo)
    root_dir = Path(__file__).resolve().parent.parent
    
    # Tenta encontrar modelo em múltiplos caminhos possíveis
    possible_paths = [
        root_dir / "model" / "artifacts" / "dl_hvac" / "global",
        root_dir / "model" / "artifacts" / "dl_hvac",
    ]
    
    artifact_path = None
    for path in possible_paths:
        if (path / "keras_model.keras").exists():
            artifact_path = path
            break
    
    if artifact_path:
        try:
            api = HVACInferenceAPI(artifact_path)
            
            print("\n" + "="*70)
            print("  INFERENCIA - HVACInferenceAPI")
            print("="*70)
            print(f"  Artefato: {artifact_path}")
            print(f"  Geo Reference: use_case/files/geo_reference.parquet")
            print("="*70)
            
            # -- TESTE 1: Inferencia com Valores Manuais --
            print("\n  [TESTE 1] Predicao com Valores Manuais")
            print("  " + "-"*70)
            
            # Cria DataFrame para validar features derivativas
            test_df = pl.DataFrame({
                "hora": [14],
                "data": ["2025-07-03"],
                "machine_type": ["splitao"],
                "latitude": [-27.24],
                "longitude": [-48.63],
                "Temperatura_C": [25.7],
                "Temperatura_Percebida_C": [24.9],
                "Umidade_Relativa_%": [57.0],
                "Precipitacao_mm": [0.0],
                "Velocidade_Vento_kmh": [21.4],
                "Pressao_Superficial_hPa": [969.8],
            }).with_columns(pl.col("data").cast(pl.Date))
            
            try:
                from .normalizer import FeatureDeriver
            except ImportError:
                from normalizer import FeatureDeriver
            
            test_df_derived = FeatureDeriver.derive(test_df)
        
            
            # Extrai e mostra features derivadas (Polars DataFrame)
            derived_features = {
                "grupo_regional": test_df_derived["grupo_regional"][0] if "grupo_regional" in test_df_derived.columns else "N/A",
                "estacao": test_df_derived["estacao"][0] if "estacao" in test_df_derived.columns else "N/A",
                "mes": test_df_derived["mes"][0] if "mes" in test_df_derived.columns else "N/A",
                "periodo_dia": test_df_derived["periodo_dia"][0] if "periodo_dia" in test_df_derived.columns else "N/A",
                "trimestre": test_df_derived["trimestre"][0] if "trimestre" in test_df_derived.columns else "N/A",
                "dia_semana": test_df_derived["dia_semana"][0] if "dia_semana" in test_df_derived.columns else "N/A",
                "is_feriado": test_df_derived["is_feriado"][0] if "is_feriado" in test_df_derived.columns else "N/A",
                "is_vespera_feriado": test_df_derived["is_vespera_feriado"][0] if "is_vespera_feriado" in test_df_derived.columns else "N/A",
                "is_dia_util": test_df_derived["is_dia_util"][0] if "is_dia_util" in test_df_derived.columns else "N/A",
            }
            
            print(f"\n    Features Derivadas:")
            print(f"      Grupo Regional:        {derived_features['grupo_regional']}")
            print(f"      Estação:               {derived_features['estacao']}")
            print(f"      Mês:                   {derived_features['mes']}")
            print(f"      Período do Dia:        {derived_features['periodo_dia']}")
            print(f"      Trimestre:             {derived_features['trimestre']}")
            print(f"      Dia Semana:            {derived_features['dia_semana']}")
            print(f"      É Feriado:             {derived_features['is_feriado']}")
            print(f"      É Véspera Feriado:     {derived_features['is_vespera_feriado']}")
            print(f"      É Dia Útil:            {derived_features['is_dia_util']}")
            
            # ── DEBUG: Verificar geo_reference.parquet ──
            geo_ref_path = root_dir / "use_case" / "files" / "geo_reference.parquet"
            print(f"\n    === DEBUG: GEO REFERENCE PARQUET ===")
            if geo_ref_path.exists():
                try:
                    geo_ref = pl.read_parquet(geo_ref_path)
                    print(f"    Arquivo encontrado em: {geo_ref_path}")
                    print(f"    Linhas: {len(geo_ref)}")
                    print(f"    Colunas: {geo_ref.columns}")
                    print(f"    Grupos únicos: {sorted(geo_ref['grupo_regional'].unique().to_list())}")
                    
                    # Acha registros próximos à coordenada de teste
                    from math import radians, cos, sin, asin, sqrt
                    def haversine(lon1, lat1, lon2, lat2):
                        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
                        dlon = lon2 - lon1
                        dlat = lat2 - lat1
                        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                        c = 2 * asin(sqrt(a))
                        km = 6371 * c
                        return km
                    
                    test_lat, test_lon = -23.883839, -46.4200317682745
                    
                    distances = [haversine(test_lon, test_lat, row['longitude'], row['latitude']) 
                                for row in geo_ref.select(["latitude", "longitude"]).to_dicts()]
                    geo_ref = geo_ref.with_columns(
                        pl.Series("distance_km", distances)
                    )
                    nearest = geo_ref.sort("distance_km").head(5)
                    print(f"\n    Coordenadas mais próximas a (-23.883839, -46.4200317682745):")
                    for row in nearest.to_dicts():
                        print(f"      lat={row['latitude']:.6f}, lon={row['longitude']:.6f}, "
                              f"grupo={row['grupo_regional']}, dist={row['distance_km']:.2f} km")
                except Exception as e:
                    print(f"    Erro ao ler geo_reference.parquet: {e}")
            else:
                print(f"    Arquivo não encontrado em: {geo_ref_path}")
                print(f"    Execute python main.py para gerar o arquivo.")
            
            print(f"\n    === PREDIÇÃO ===")
            
            pred_manual = api.predict_single(
                hora=14,
                data="2025-07-03",
                machine_type="splitao",
                latitude=-23.883839,
                longitude=-46.4200317682745,
                Temperatura_C=25.7,
                Temperatura_Percebida_C=24.9,
                Umidade_Relativa_pct=57.0,
                Precipitacao_mm=0.0,
                Velocidade_Vento_kmh=21.4,
                Pressao_Superficial_hPa=969.8,
            )
            
            print(f"    Horario:        14:00")
            print(f"    Data:           2025-07-03 (Inverno)")
            print(f"    Maquina:        SPLITAO")
            print(f"    Localizacao:    lat=-23.883839 graus, lon=-46.4200317682745 graus")
            print(f"    Temperatura:    25.7 C")
            print(f"    Umidade:        57%")
            print(f"    Vento:          21.4 km/h")
            print(f"\n    > Consumo Previsto: {pred_manual:.4f} kWh")
            
            # -- TESTE 2: Inferencia em Batch (CSV com lat/lon) --
            print("\n  [TESTE 2] Predicao em Batch (valores manuais repetidos)")
            print("  " + "-"*70)
            
            # Cria batch com valores manuais repetidos para demonstracao
            batch_data = pl.DataFrame({
                "data": ["2025-07-03"] * 10,
                "hora": list(range(8, 18)),
                "machine_type": ["splitao"] * 10,
                "latitude": [-23.883839] * 10,
                "longitude": [-46.4200317682745] * 10,
                "Temperatura_C": [25.7 + i*0.5 for i in range(10)],
                "Temperatura_Percebida_C": [24.9 + i*0.5 for i in range(10)],
                "Umidade_Relativa_%": [57.0 - i*2 for i in range(10)],
                "Precipitacao_mm": [0.0] * 10,
                "Velocidade_Vento_kmh": [21.4 - i for i in range(10)],
                "Pressao_Superficial_hPa": [969.8] * 10,
            }).with_columns(pl.col("data").cast(pl.Date))
            
            # Predicao
            predictions = api.predict(batch_data)
            
            # Deriva features para validação
            batch_data_derived = FeatureDeriver.derive(batch_data)
            
            print(f"\n    === FEATURES DERIVATIVAS (Amostra) ===")
            print(f"    Horas:              {batch_data['hora'].to_list()}")
            if "grupo_regional" in batch_data_derived.columns:
                print(f"    Grupos Regionais:   {batch_data_derived['grupo_regional'].to_list()}")
            if "mes" in batch_data_derived.columns:
                print(f"    Meses:              {batch_data_derived['mes'].to_list()}")
            if "periodo_dia" in batch_data_derived.columns:
                print(f"    Períodos Dia:       {batch_data_derived['periodo_dia'].to_list()}")
            if "is_feriado" in batch_data_derived.columns:
                print(f"    É Feriado:          {batch_data_derived['is_feriado'].to_list()}")
            if "is_vespera_feriado" in batch_data_derived.columns:
                print(f"    É Véspera Feriado:  {batch_data_derived['is_vespera_feriado'].to_list()}")
            if "is_dia_util" in batch_data_derived.columns:
                print(f"    É Dia Útil:         {batch_data_derived['is_dia_util'].to_list()}")
            
            # Relatorio
            print(f"\n    === PREDIÇÕES ===")
            print(f"    Amostra: {len(batch_data)} linhas (simulado)")
            print(f"    Predicoes: shape={predictions.shape}")
            print(f"    Min: {predictions.min():.4f} kWh")
            print(f"    Max: {predictions.max():.4f} kWh")
            print(f"    Mean: {predictions.mean():.4f} kWh")
            print(f"    Std: {predictions.std():.4f} kWh")
            
            # Amostra de predicoes
            print(f"\n    Predicoes por hora (08:00 - 17:00):")
            for i in range(len(predictions)):
                hora = 8 + i
                print(f"      {hora:02d}:00 - {predictions[i]:8.4f} kWh")
            
            print("\n" + "="*70 + "\n")
            
        except Exception as e:
            print(f"\n  ERRO: {e}\n")
            import traceback
            traceback.print_exc()
    else:
        print("\n" + "="*70)
        print("  AVISO: Artefato de modelo não encontrado")
        print("="*70)
        print(f"\n  Caminhos procurados:")
        for path in possible_paths:
            print(f"    - {path}")
        print(f"\n  Para executar testes:")
        print(f"    1. Execute o pipeline de treinamento:")
        print(f"       >>> python main.py")
        print(f"\n    2. Isso gerará:")
        print(f"       - model/artifacts/dl_hvac/global/keras_model.keras")
        print(f"       - use_case/files/geo_reference.parquet")
        print(f"\n    3. Então teste novamente:")
        print(f"       >>> python -m tools.inference_api")
        print("\n" + "="*70 + "\n")

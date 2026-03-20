import requests
import polars as pl
import time
from typing import Union
from pathlib import Path
from datetime import datetime

class WeatherEnricher:
    """Classe para enriquecimento de dados com informações meteorológicas do OpenMeteo.
    
    Adiciona colunas meteorológicas ao DataFrame baseado em data, hora, latitude e longitude.
    Utiliza cache inteligente para evitar requisições duplicadas.
    
    Attributes:
        df (pl.DataFrame): DataFrame com os dados
        cache (Dict): Cache de requisições já realizadas
    """
    
    def __init__(self, df: pl.DataFrame, cache_path: Union[str, Path] = 'use_case/files/weather_requests_openmeteo.parquet'):
        """Inicializa o enriquecedor de dados meteorológicos.
        
        Args:
            df (pl.DataFrame): DataFrame Polars com colunas 'data', 'hora', 'latitude', 'longitude'
            cache_path (Union[str, Path]): Caminho para arquivo Parquet de cache
        
        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes
        """
        required_columns = ['data', 'hora', 'latitude', 'longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Colunas ausentes no DataFrame: {missing_columns}")
        
        self.df = df.clone()
        self.cache = {}
        self.cache_path = Path(cache_path)
        self.request_delay = 0.105  # 60s / 600 req = 0.1s, usando 0.105s para margem de segurança
        self._load_cache_from_parquet()
        
        print(f"✓ WeatherEnricher inicializado")
        print(f"  Registros: {self.df.shape[0]}")
        print(f"  Rate limit: ~571 requisições/minuto (máximo 600)")
        print(f"  Cache Parquet: {self.cache_path}")
    
    def _ensure_schema_compliance(self, df: pl.DataFrame) -> pl.DataFrame:
        """Garante que o DataFrame tenha o esquema correto e consistente.
        
        Args:
            df (pl.DataFrame): DataFrame com dados climáticos
            
        Returns:
            pl.DataFrame: DataFrame com esquema padronizado
        """
        return df.select([
            pl.col('hora').cast(pl.Int64),
            pl.col('Temperatura_C').cast(pl.Float64),
            pl.col('Temperatura_Percebida_C').cast(pl.Float64),
            pl.col('Umidade_Relativa_%').cast(pl.Float64),
            pl.col('Precipitacao_mm').cast(pl.Float64),
            pl.col('Velocidade_Vento_kmh').cast(pl.Float64),
            pl.col('Pressao_Superficial_hPa').cast(pl.Float64),
            pl.col('Irradiancia_Direta_Wm2').cast(pl.Float64),
            pl.col('Irradiancia_Difusa_Wm2').cast(pl.Float64),
            pl.col('data_str').cast(pl.Utf8) if 'data_str' in df.columns else pl.lit(None).cast(pl.Utf8).alias('data_str'),
            pl.col('latitude').cast(pl.Float64) if 'latitude' in df.columns else pl.lit(None).cast(pl.Float64).alias('latitude'),
            pl.col('longitude').cast(pl.Float64) if 'longitude' in df.columns else pl.lit(None).cast(pl.Float64).alias('longitude')
        ])
    
    def _fetch_weather_data(self, lat: float, lon: float, data: str) -> pl.DataFrame:
        """Busca dados meteorológicos para uma localização e data específicas.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            data (str): Data no formato YYYY-MM-DD
        
        Returns:
            pl.DataFrame: DataFrame com dados meteorológicos horários
        """
        cache_key = (lat, lon, data)
        
        # Verifica cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        url = "https://archive-api.open-meteo.com/v1/archive"
        variaveis = "temperature_2m," \
                    "apparent_temperature," \
                    "relative_humidity_2m," \
                    "precipitation," \
                    "wind_speed_10m," \
                    "surface_pressure," \
                    "direct_normal_irradiance," \
                    "diffuse_radiation"
        
        parametros = {
            "latitude": lat,
            "longitude": lon,
            "start_date": data,
            "end_date": data,
            "hourly": variaveis,
            "timezone": "America/Sao_Paulo"
        }
        
        try:
            resposta = requests.get(url, params=parametros, timeout=10)
            resposta.raise_for_status()
            dados = resposta.json()
            
            # Rate limiting: aguarda antes da próxima requisição
            time.sleep(self.request_delay)
            
            df_weather = (
                pl.DataFrame(dados['hourly'])
                .rename({
                    "time": "Data_Hora",
                    "temperature_2m": "Temperatura_C",
                    "apparent_temperature": "Temperatura_Percebida_C",
                    "relative_humidity_2m": "Umidade_Relativa_%",
                    "precipitation": "Precipitacao_mm",
                    "surface_pressure": "Pressao_Superficial_hPa",
                    "wind_speed_10m": "Velocidade_Vento_kmh",
                    "direct_normal_irradiance": "Irradiancia_Direta_Wm2",
                    "diffuse_radiation": "Irradiancia_Difusa_Wm2",
                })
                .with_columns([
                    pl.col("Data_Hora").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M")
                ])
                .with_columns([
                    pl.col("Data_Hora").dt.hour().cast(pl.Int64).alias("hora")
                ])
                .drop("Data_Hora")
                .with_columns([
                    pl.col("Temperatura_C").cast(pl.Float64),
                    pl.col("Temperatura_Percebida_C").cast(pl.Float64),
                    pl.col("Umidade_Relativa_%").cast(pl.Float64),
                    pl.col("Precipitacao_mm").cast(pl.Float64),
                    pl.col("Velocidade_Vento_kmh").cast(pl.Float64),
                    pl.col("Pressao_Superficial_hPa").cast(pl.Float64),
                    pl.col("Irradiancia_Direta_Wm2").cast(pl.Float64),
                    pl.col("Irradiancia_Difusa_Wm2").cast(pl.Float64),
                ])
            )
            
            # Armazena no cache
            self.cache[cache_key] = df_weather
            return df_weather
            
        except requests.exceptions.RequestException as e:
            print(f"⚠ Erro ao buscar dados para {data} (lat={lat}, lon={lon}): {e}")
            return None
    
    def _load_cache_from_parquet(self) -> None:
        """Carrega cache de requisições anteriores do arquivo Parquet."""
        if not self.cache_path.exists():
            print(f"  ℹ Cache não encontrado. Será criado: {self.cache_path}")
            return
        
        # Verifica data de modificação do arquivo (informativo apenas)
        mod_time = datetime.fromtimestamp(self.cache_path.stat().st_mtime)
        today = datetime.now()
        
        if mod_time.date() < today.date():
            print(f"  ℹ Cache de {mod_time.strftime('%Y-%m-%d')} será reutilizado (evita requisições duplicadas)")
        else:
            print(f"  ✓ Cache atualizado (modificado hoje às {mod_time.strftime('%H:%M:%S')})")
        
        try:
            df_cache = pl.read_parquet(self.cache_path)
            print(f"  ✓ Cache carregado: {df_cache.shape[0]} requisições em cache")
            
            # Agrupa por chave de cache de forma vetorizada - MUITO mais performático que iter_rows()
            for (lat, lon, data_str), group_df in df_cache.group_by(['latitude', 'longitude', 'data_str']):
                cache_key = (lat, lon, data_str)
                # Seleciona colunas e converte tipos em uma operação
                weather_df = group_df.select([
                    pl.col('hora').cast(pl.Int64),
                    pl.col('Temperatura_C').cast(pl.Float64),
                    pl.col('Temperatura_Percebida_C').cast(pl.Float64),
                    pl.col('Umidade_Relativa_%').cast(pl.Float64),
                    pl.col('Precipitacao_mm').cast(pl.Float64),
                    pl.col('Velocidade_Vento_kmh').cast(pl.Float64),
                    pl.col('Pressao_Superficial_hPa').cast(pl.Float64),
                    pl.col('Irradiancia_Direta_Wm2').cast(pl.Float64),
                    pl.col('Irradiancia_Difusa_Wm2').cast(pl.Float64),
                ])
                self.cache[cache_key] = weather_df

        except Exception as e:
            print(f"  ⚠ Erro ao carregar cache: {e}")
    
    def _get_cached_unique_combinations(self) -> pl.DataFrame:
        """Extrai combinações únicas de (data, latitude, longitude) do cache Parquet.
        
        Returns:
            pl.DataFrame: DataFrame com colunas ['data_str', 'latitude', 'longitude'] das combinações em cache
        """
        if not self.cache_path.exists():
            return pl.DataFrame({
                'data_str': pl.Series([], dtype=pl.Utf8),
                'latitude': pl.Series([], dtype=pl.Float64),
                'longitude': pl.Series([], dtype=pl.Float64)
            })
        
        try:
            df_cache = pl.read_parquet(self.cache_path)
            # Extrai combinações únicas e garante que data_str seja Utf8
            df_unique = (
                df_cache
                .select(['data_str', 'latitude', 'longitude'])
                .with_columns([
                    pl.col('data_str').cast(pl.Utf8),
                    pl.col('latitude').cast(pl.Float64),
                    pl.col('longitude').cast(pl.Float64)
                ])
                .unique()
            )
            return df_unique
        except Exception as e:
            print(f"  ⚠ Erro ao extrair combinações do cache: {e}")
            return pl.DataFrame({
                'data_str': pl.Series([], dtype=pl.Utf8),
                'latitude': pl.Series([], dtype=pl.Float64),
                'longitude': pl.Series([], dtype=pl.Float64)
            })
    
    def _save_cache_to_parquet(self) -> None:
        """Salva cache de requisições no arquivo Parquet."""
        if not self.cache:
            print("  ℹ Nenhum dado para salvar no cache")
            return
        
        try:
            # Cria diretório se não existir
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Converte cache para DataFrame
            cache_rows = []
            for (lat, lon, data_str), df_weather in self.cache.items():
                for row in df_weather.iter_rows(named=True):
                    cache_rows.append({
                        'latitude': lat,
                        'longitude': lon,
                        'data_str': data_str,
                        'hora': row['hora'],
                        'Temperatura_C': row['Temperatura_C'],
                        'Temperatura_Percebida_C': row['Temperatura_Percebida_C'],
                        'Umidade_Relativa_%': row['Umidade_Relativa_%'],
                        'Precipitacao_mm': row['Precipitacao_mm'],
                        'Velocidade_Vento_kmh': row['Velocidade_Vento_kmh'],
                        'Pressao_Superficial_hPa': row['Pressao_Superficial_hPa'],
                        'Irradiancia_Direta_Wm2': row['Irradiancia_Direta_Wm2'],
                        'Irradiancia_Difusa_Wm2': row['Irradiancia_Difusa_Wm2'],
                    })
            
            df_cache = pl.DataFrame(cache_rows)
            df_cache.write_parquet(self.cache_path)
            print(f"\n✓ Cache salvo: {df_cache.shape[0]} registros em {self.cache_path}")
            
        except Exception as e:
            print(f"\n⚠ Erro ao salvar cache: {e}")
    
    def add_weather_columns(self) -> pl.DataFrame:
        """Adiciona colunas meteorológicas ao DataFrame.
        
        Processo otimizado:
        1. Identifica combinações únicas de (data, latitude, longitude) no DataFrame
        2. Compara com combinações já presentes no cache Parquet
        3. Faz requisições apenas para combinações novas (não presentes no cache)
        4. Mapeia resultados de volta para todos os registros
        
        Returns:
            pl.DataFrame: DataFrame com novas colunas meteorológicas
        """
        print("\n✓ Iniciando enriquecimento com dados meteorológicos...")
        
        # Prepara coluna de data como string
        df_work = self.df.with_columns([
            pl.col('data').cast(pl.Date).cast(pl.Utf8).alias('data_str')
        ])
        
        # Identifica combinações únicas no DataFrame
        unique_combinations = (
            df_work
            .select(['data_str', 'latitude', 'longitude'])
            .with_columns([
                pl.col('data_str').cast(pl.Utf8),
                pl.col('latitude').cast(pl.Float64),
                pl.col('longitude').cast(pl.Float64)
            ])
            .unique()
        )
        
        print(f"  Combinações únicas no DataFrame: {unique_combinations.shape[0]}")
        
        # Remove combinações com latitude ou longitude nulas
        combinations_before = unique_combinations.shape[0]
        unique_combinations = unique_combinations.filter(
            (pl.col('latitude').is_not_null()) & (pl.col('longitude').is_not_null())
        )
        combinations_after = unique_combinations.shape[0]
        
        if combinations_before > combinations_after:
            print(f"  ⚠ Removidas {combinations_before - combinations_after} combinações com lat/lon nulos")
        
        # Obtém combinações já presentes no cache Parquet
        cached_combinations = self._get_cached_unique_combinations()
        print(f"  Combinações únicas no cache Parquet: {cached_combinations.shape[0]}")
        
        # Identifica combinações novas (não presentes no cache)
        if cached_combinations.shape[0] > 0:
            # Anti-join: mantém apenas combinações do DataFrame que NÃO estão no cache
            new_combinations = unique_combinations.join(
                cached_combinations,
                on=['data_str', 'latitude', 'longitude'],
                how='anti'
            )
            print(f"  Combinações novas (não presentes no cache): {new_combinations.shape[0]}")
        else:
            new_combinations = unique_combinations
            print(f"  Cache vazio. Todas as {new_combinations.shape[0]} combinações são novas")
        
        print(f"  Requisições necessárias: {new_combinations.shape[0]}")
        
        # Coleta dados meteorológicos APENAS para combinações novas
        weather_data_list = []
        new_requests = 0
        
        # Processa apenas as combinações novas
        if new_combinations.shape[0] > 0:
            print(f"\n  [Requisitando] Dados para combinações novas...")
            # to_dicts() é mais rápido que iter_rows() para iteração
            new_combinations_list = new_combinations.to_dicts()
            for idx, row in enumerate(new_combinations_list, 1):
                data_str = row['data_str']
                lat = row['latitude']
                lon = row['longitude']
                
                # Handle None values for lat/lon
                lat_str = f"{lat:.4f}" if lat is not None else "None"
                lon_str = f"{lon:.4f}" if lon is not None else "None"
                
                print(f"  [{idx}/{new_combinations.shape[0]}] 🌐 {data_str} (lat={lat_str}, lon={lon_str})...", end='\r')
                
                new_requests += 1
                df_weather = self._fetch_weather_data(lat, lon, data_str)
                
                if df_weather is not None:
                    # Adiciona informações de localização
                    df_weather = df_weather.with_columns([
                        pl.lit(data_str).cast(pl.Utf8).alias('data_str'),
                        pl.lit(lat).cast(pl.Float64).alias('latitude'),
                        pl.lit(lon).cast(pl.Float64).alias('longitude')
                    ])
                    # Garante conformidade do esquema
                    df_weather = self._ensure_schema_compliance(df_weather)
                    weather_data_list.append(df_weather)

                # Checkpoint: salva cache a cada 500 requisições concluídas
                if new_requests % 500 == 0:
                    print(f"\n  💾 Checkpoint: {new_requests} requisições concluídas. Salvando cache...")
                    self._save_cache_to_parquet()

            print(f"\n  ✓ Requisições concluídas: {new_requests}")
        
        # Consolidação: reúne dados de cache e novas requisições
        print(f"\n  [Consolidação] Reunindo dados...")
        all_weather_data = weather_data_list.copy()
        cache_hits = 0
        
        # Usa set para verificação O(1)
        new_keys_set = set()
        if new_combinations.shape[0] > 0:
            for row in new_combinations.to_dicts():
                new_keys_set.add((row['latitude'], row['longitude'], row['data_str']))
        
        # Processa combinações do cache que NÃO foram requisitadas novamente
        for row in unique_combinations.to_dicts():
            cache_key = (row['latitude'], row['longitude'], row['data_str'])
            # Se está em cache E não é uma nova combinação requisitada
            if cache_key in self.cache and cache_key not in new_keys_set:
                df_weather = self.cache[cache_key].with_columns([
                    pl.lit(row['data_str']).cast(pl.Utf8).alias('data_str'),
                    pl.lit(row['latitude']).cast(pl.Float64).alias('latitude'),
                    pl.lit(row['longitude']).cast(pl.Float64).alias('longitude')
                ])
                df_weather = self._ensure_schema_compliance(df_weather)
                all_weather_data.append(df_weather)
                cache_hits += 1
        
        weather_data_list = all_weather_data
        
        print(f"\n✓ Processamento concluído:")
        print(f"  📊 Combinações do cache reutilizadas: {cache_hits}")
        print(f"  📊 Novas requisições: {new_requests}")
        print(f"  📊 Total de combinações processadas: {len(weather_data_list)}")
        
        if not weather_data_list:
            raise ValueError("Nenhum dado meteorológico foi obtido com sucesso")
        
        # Consolida todos os dados meteorológicos
        df_weather_all = pl.concat(weather_data_list)
        
        # Faz join com o DataFrame original
        df_enriched = (
            df_work
            .join(
                df_weather_all,
                on=['data_str', 'hora', 'latitude', 'longitude'],
                how='left'
            )
            .drop('data_str')
        )
        
        print(f"✓ Enriquecimento concluído: {df_enriched.shape[0]} registros")
        
        # Salva cache atualizado
        self._save_cache_to_parquet()
        
        return df_enriched
    
    def get_weather_columns_only(self) -> pl.DataFrame:
        """Retorna apenas as colunas meteorológicas para integração posterior.
        
        Returns:
            pl.DataFrame: DataFrame com apenas as colunas meteorológicas
        """
        df_enriched = self.add_weather_columns()
        
        weather_columns = [
            'Temperatura_C',
            'Temperatura_Percebida_C',
            'Umidade_Relativa_%',
            'Precipitacao_mm',
            'Velocidade_Vento_kmh',
            'Pressao_Superficial_hPa',
            'Irradiancia_Direta_Wm2',
            'Irradiancia_Difusa_Wm2',
        ]
        
        return df_enriched.select(weather_columns)


def get_weather_columns_from_parquet(parquet_path: Union[str, Path]) -> pl.DataFrame:
    """Carrega Parquet e retorna apenas colunas meteorológicas do OpenMeteo.
    Processo integrado: carrega, busca dados meteorológicos e retorna apenas novas colunas.
    
    Args:
        parquet_path (Union[str, Path]): Caminho do arquivo Parquet
    
    Returns:
        pl.DataFrame: DataFrame com apenas as colunas meteorológicas (mesmo número de linhas do Parquet original)
    
    Uso típico:
        >>> # Gera e integra em 2 linhas
        >>> df_weather = get_weather_columns_from_parquet('consumption_consolidated.parquet')
        >>> df_full = pl.read_parquet('consumption_consolidated.parquet')
        >>> df_final = df_full.hstack(df_weather)
    
    Exemplo:
        >>> import polars as pl
        >>> from dataframe.complementary_features.temperature_openmeteo import get_weather_columns_from_parquet
        >>> 
        >>> # Método otimizado (uma linha)
        >>> df_final = pl.read_parquet('data.parquet').hstack(get_weather_columns_from_parquet('data.parquet'))
    """
    # Carrega apenas as colunas necessárias
    required_columns = ['data', 'hora', 'latitude', 'longitude']
    df = pl.read_parquet(parquet_path, columns=required_columns)
    
    # Enriquece com dados meteorológicos
    enricher = WeatherEnricher(df)
    return enricher.get_weather_columns_only()


def get_weather_columns_from_parquet(parquet_path: Union[str, Path]) -> pl.DataFrame:
    """[DEPRECATED] Carrega Parquet e retorna apenas colunas meteorológicas do OpenMeteo.
    
    AVISO: Esta função está DEPRECATED. Use get_weather_columns_from_parquet() em vez disso.
    Processo integrado: carrega, busca dados meteorológicos e retorna apenas novas colunas.
    
    Args:
        parquet_path (Union[str, Path]): Caminho do arquivo Parquet
    
    Returns:
        pl.DataFrame: DataFrame com apenas as colunas meteorológicas (mesmo número de linhas do Parquet original)
    
    Uso típico (DEPRECATED):
        >>> # Gera e integra em 2 linhas
        >>> df_weather = get_weather_columns_from_parquet('consumption_consolidated.parquet')
        >>> df_full = pl.read_parquet('consumption_consolidated.parquet')
        >>> df_final = df_full.hstack(df_weather)
    
    Novo uso recomendado:
        >>> import polars as pl
        >>> from dataframe.complementary_features.temperature_openmeteo import get_weather_columns_from_parquet
        >>> 
        >>> # Método otimizado (uma linha)
        >>> df_final = pl.read_parquet('data.parquet').hstack(get_weather_columns_from_parquet('data.parquet'))
    """
    # Carrega apenas as colunas necessárias
    required_columns = ['data', 'hora', 'latitude', 'longitude']
    df = pl.read_parquet(parquet_path, columns=required_columns)
    
    # Enriquece com dados meteorológicos
    enricher = WeatherEnricher(df)
    return enricher.get_weather_columns_only()


if __name__ == "__main__":
    """Exemplo de uso otimizado do WeatherEnricher."""
    
    print("=" * 80)
    print("ENRIQUECIMENTO COM DADOS METEOROLÓGICOS - OpenMeteo")
    print("=" * 80)
    
    # Exemplo 1: Teste com dados mockados
    print("\n[Exemplo 1] Teste com dados de exemplo")
    print("-" * 80)
    
    # Cria DataFrame de exemplo
    df_example = pl.DataFrame({
        'data': ['2026-03-02', '2026-03-02', '2026-03-02'],
        'hora': [10, 11, 12],
        'latitude': [-23.596351105380737, -23.596351105380737, -23.596351105380737],
        'longitude': [-46.68850466086873, -46.68850466086873, -46.68850466086873],
        'consumo_kwh': [100.5, 95.3, 88.7]
    })
    
    print("\nDataFrame original:")
    print(df_example)
    
    try:
        enricher = WeatherEnricher(df_example)
        df_enriched = enricher.add_weather_columns()
        
        print("\nDataFrame enriquecido:")
        print(df_enriched)
        
        print("\nApenas colunas meteorológicas:")
        df_weather_only = enricher.get_weather_columns_only()
        print(df_weather_only)
        
    except Exception as e:
        print(f"\n✗ Erro: {e}")
    
    # Exemplo 2: Teste com dados reais do projeto
    print("\n\n[Exemplo 2] Enriquecimento com dados reais do projeto")
    print("-" * 80)
    
    consumption_path = Path(r'use_case\files\consumption_consolidated.parquet')
    if consumption_path.exists():
        try:
            print(f"\nCarregando dados de consumo de: {consumption_path}")
            df_consumption = pl.read_parquet(consumption_path)
            print(f"✓ {df_consumption.shape[0]} registros carregados")
            print(f"  Colunas: {df_consumption.columns}")
            
            # Verifica se possui as colunas necessárias
            required_cols = ['data', 'hora', 'latitude', 'longitude']
            if all(col in df_consumption.columns for col in required_cols):
                print(f"\n✓ Colunas necessárias presentes. Iniciando enriquecimento...")
                
                enricher = WeatherEnricher(df_consumption)
                df_enriched = enricher.add_weather_columns()
                
                print(f"\n✓ Enriquecimento concluído:")
                print(f"  Registros: {df_enriched.shape[0]}")
                print(f"  Colunas: {df_enriched.columns}")
                print(f"  Primeiras linhas:")
                print(df_enriched.head())
                
            else:
                missing = [col for col in required_cols if col not in df_consumption.columns]
                print(f"\n✗ Colunas ausentes: {missing}")
                print(f"  Colunas disponíveis: {df_consumption.columns}")
        
        except Exception as e:
            print(f"\n✗ Erro ao processar dados reais: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠ Arquivo não encontrado: {consumption_path}")
        print(f"  Execute main.py primeiro para gerar o arquivo de consumo consolidado")

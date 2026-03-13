"""
Orquestrador de features complementares para enriquecimento de dados HVAC.

Este módulo integra três pipelines de enriquecimento:
1. Estações do ano (year_stations)
2. Dados climáticos (temperature_openmeteo)
3. Grupos regionais K-means (regional_group/model)

Retorna apenas as novas colunas para integração em DataFrames existentes.
"""

import polars as pl

from .complementary_features.year_stations import enrich_with_seasons
from .complementary_features.temperature_openmeteo import WeatherEnricher
from .complementary_features.regional_group.model import RegionalGroupClassifier


def enrich_dataframe_with_all_features(df: pl.DataFrame, export_geo_reference: bool = True) -> pl.DataFrame:
    """
    Pipeline completo de enriquecimento: estações, clima e grupos regionais.
    
    Args:
        df: DataFrame Polars com dados base (deve conter: data, hora, latitude, longitude)
        export_geo_reference: Se True, exporta o mapeamento de grupos para geo_reference.parquet
    
    Returns:
        DataFrame Polars original enriquecido com novas colunas:
        - estacao (str)
        - grupo_regional (int)
        - Temperatura_C (float)
        - Temperatura_Percebida_C (float)
        - Umidade_Relativa_% (float)
        - Precipitacao_mm (float)
        - Velocidade_Vento_kmh (float)
        - Pressao_Superficial_hPa (float)
    
    Example:
        >>> df_enriched = enrich_dataframe_with_all_features(df_original)
    """
    print("=" * 80)
    print("PIPELINE DE ENRIQUECIMENTO DE FEATURES")
    print("=" * 80)
    
    # 1. Estações do ano (apenas coluna 'data')
    print("\n[1/3] Classificando estações do ano...")
    df_with_season = enrich_with_seasons(df.select('data'))
    df_season = df_with_season.select('estacao')
    print(f"✓ {len(df_season)} registros processados")
    
    # 2. Grupos regionais DBSCAN + Haversine (latitude, longitude)
    print("\n[2/3] Classificando grupos regionais...")
    regional_clf = RegionalGroupClassifier(radius_km=40, min_samples=2)
    regional_clf.fit_predict(df.select(['latitude', 'longitude']))
    df_groups = regional_clf.predict(df.select(['latitude', 'longitude'])).select('grupo_regional')
    
    # Exporta mapeamento de grupos para arquivo Parquet se solicitado
    if export_geo_reference:
        print("\n[2.1/3] Exportando mapeamento de grupos regionais...")
        regional_clf.export_mapping()
    
    print(f"✓ {len(df_groups)} registros processados")
    
    # 3. Dados climáticos OpenMeteo (data, hora, latitude, longitude)
    print("\n[3/3] Enriquecendo com dados climáticos...")
    weather_enricher = WeatherEnricher(df.select(['data', 'hora', 'latitude', 'longitude']))
    df_weather = weather_enricher.get_weather_columns_only()
    print(f"✓ {len(df_weather)} registros processados")
    
    # Concatena DataFrame original com novas features
    print("\n[✓] Consolidando features ao DataFrame original...")
    df_enriched = pl.concat([df, df_season, df_groups, df_weather], how='horizontal')
    
    print(f"\n✓ Enriquecimento concluído!")
    print(f"  Total de registros: {len(df_enriched)}")
    print(f"  Total de colunas: {len(df_enriched.columns)} (original: {len(df.columns)} + novas: {len(df_season.columns) + len(df_groups.columns) + len(df_weather.columns)})")
    print(f"  Novas colunas adicionadas: {df_season.columns + df_groups.columns + df_weather.columns}")
    
    return df_enriched


def get_season_only(df: pl.DataFrame) -> pl.DataFrame:
    """Retorna apenas coluna de estações do ano."""
    df_with_season = enrich_with_seasons(df.select('data'))
    return df_with_season.select('estacao')


def get_regional_groups_only(df: pl.DataFrame) -> pl.DataFrame:
    """Retorna apenas coluna de grupos regionais."""
    return RegionalGroupClassifier(radius_km=5, min_samples=2).fit_predict(df.select(['latitude', 'longitude'])).select('grupo_regional')


def get_weather_only(df: pl.DataFrame) -> pl.DataFrame:
    """Retorna apenas colunas de dados climáticos."""
    weather_enricher = WeatherEnricher(df.select(['data', 'hora', 'latitude', 'longitude']))
    return weather_enricher.get_weather_columns_only()


def main():
    """Exemplo de uso do orquestrador"""
    print("\n" + "=" * 80)
    print("EXEMPLO: Enriquecimento de consumption_consolidated.csv")
    print("=" * 80)
    
    csv_path = r'use_case\files\consumption_consolidated.csv'
    
    # Carrega DataFrame
    print("\nCarregando CSV...")
    df = pl.read_csv(csv_path)
    print(f"✓ {len(df)} registros carregados")
    print(f"✓ Colunas originais: {len(df.columns)}")
    
    # Pipeline completo
    df_enriched = enrich_dataframe_with_all_features(df)
    
    print("\n" + "=" * 80)
    print("PREVIEW DO DATAFRAME ENRIQUECIDO (5 primeiras linhas)")
    print("=" * 80)
    print(df_enriched.head(5))
    
    print("\n" + "=" * 80)
    print("✓ Pipeline concluído com sucesso!")
    print("=" * 80)


if __name__ == "__main__":
    main()

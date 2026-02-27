import polars as pl
from pathlib import Path


def aggregate_consumption_by_unit(input_csv_path: str, output_csv_path: str = None) -> pl.DataFrame:
    """Agrega consumo de dispositivos por unidade, data, hora e método.
    
    Esta função lê o arquivo de consumo consolidado e agrega os dados removendo a 
    granularidade de dispositivo individual (device_id, device_version), somando o 
    consumo total por unidade em cada hora.
    
    Args:
        input_csv_path (str): Caminho para o arquivo CSV de entrada (consumption_consolidated.csv)
        output_csv_path (str, optional): Caminho para salvar o arquivo agregado. 
                                         Se None, não salva arquivo.
    
    Returns:
        pl.DataFrame: DataFrame agregado com colunas:
                     - unit_id: ID da unidade
                     - data: Data do registro
                     - hora: Hora do registro (0-23)
                     - metodo: Método de coleta ('direto' ou 'indireto')
                     - consumo_kwh_total: Soma do consumo de todos dispositivos da unidade
                     - qtd_dispositivos: Quantidade de dispositivos que contribuíram
    
    Exemplo:
        >>> df_aggregated = aggregate_consumption_by_unit(
        ...     input_csv_path='use_case/consumption_consolidated.csv',
        ...     output_csv_path='use_case/consumption_aggregated_by_unit.csv'
        ... )
        >>> print(df_aggregated.head())
    """
    
    # Lê o arquivo CSV consolidado
    df = pl.read_csv(input_csv_path)
    
    print(f"✓ Arquivo carregado: {df.shape[0]} registros")
    print(f"  Colunas originais: {df.columns}")
    
    # Valida presença das colunas necessárias
    required_columns = ['unit_id', 'data', 'hora', 'metodo', 'consumo_kwh']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas ausentes no CSV: {missing_columns}")
    
    # Garante tipos corretos antes da agregação
    df = df.with_columns([
        pl.col('unit_id').cast(pl.Int64),
        pl.col('data').cast(pl.Date),
        pl.col('hora').cast(pl.Int64),
        pl.col('metodo').cast(pl.Utf8),
        pl.col('consumo_kwh').cast(pl.Float64)
    ])
    
    # Agrega por unit_id, data, hora e método
    df_aggregated = (
        df
        .group_by(['unit_id', 'data', 'hora', 'metodo'])
        .agg([
            pl.col('consumo_kwh').sum().round(4).alias('consumo_kwh_total'),
            pl.col('device_id').n_unique().alias('qtd_dispositivos')
        ])
        .sort(['unit_id', 'data', 'hora', 'metodo'])
    )
    
    print(f"\n✓ Agregação concluída: {df_aggregated.shape[0]} registros")
    print(f"  Colunas finais: {df_aggregated.columns}")
    print(f"\n=== ESTATÍSTICAS DA AGREGAÇÃO ===")
    print(f"  Unidades únicas: {df_aggregated['unit_id'].n_unique()}")
    print(f"  Período: {df_aggregated['data'].min()} a {df_aggregated['data'].max()}")
    print(f"  Consumo total agregado: {df_aggregated['consumo_kwh_total'].sum():.2f} kWh")
    
    # Salva em arquivo se caminho foi fornecido
    if output_csv_path:
        df_aggregated.write_csv(output_csv_path)
        print(f"\n✓ Arquivo salvo: {output_csv_path}")
    
    return df_aggregated


def get_aggregation_summary(df_aggregated: pl.DataFrame) -> pl.DataFrame:
    """Gera resumo estatístico da agregação por unidade.
    
    Args:
        df_aggregated (pl.DataFrame): DataFrame agregado retornado por aggregate_consumption_by_unit
    
    Returns:
        pl.DataFrame: Resumo com estatísticas por unidade:
                     - unit_id
                     - consumo_total_kwh
                     - dias_com_dados
                     - registros_direto
                     - registros_indireto
                     - dispositivos_medio
    """
    
    summary = (
        df_aggregated
        .group_by('unit_id')
        .agg([
            pl.col('consumo_kwh_total').sum().round(4).alias('consumo_total_kwh'),
            pl.col('data').n_unique().alias('dias_com_dados'),
            pl.col('metodo').filter(pl.col('metodo') == 'direto').count().alias('registros_direto'),
            pl.col('metodo').filter(pl.col('metodo') == 'indireto').count().alias('registros_indireto'),
            pl.col('qtd_dispositivos').mean().round(2).alias('dispositivos_medio')
        ])
        .sort('unit_id')
    )
    
    return summary


if __name__ == "__main__":
    """Exemplo de uso do módulo."""
    
    print("=" * 80)
    print("AGREGAÇÃO DE CONSUMO POR UNIDADE")
    print("=" * 80)
    
    # Define caminhos
    input_path = r'use_case\output_files\consumption_consolidated.csv'
    output_path = r'use_case\output_files\consumption_aggregated_by_unit.csv'
    
    # Verifica se arquivo de entrada existe
    if not Path(input_path).exists():
        print(f"✗ ERRO: Arquivo não encontrado: {input_path}")
        print("  Execute main.py primeiro para gerar o arquivo consolidado.")
    else:
        # Executa agregação
        df_agg = aggregate_consumption_by_unit(input_path, output_path)
        
        # Gera e exibe resumo
        print("\n" + "=" * 80)
        print("RESUMO POR UNIDADE")
        print("=" * 80)
        df_summary = get_aggregation_summary(df_agg)
        print(df_summary)
        
        # Exibe amostra dos dados agregados
        print("\n" + "=" * 80)
        print("AMOSTRA DOS DADOS AGREGADOS (primeiras 10 linhas)")
        print("=" * 80)
        print(df_agg.head(10))

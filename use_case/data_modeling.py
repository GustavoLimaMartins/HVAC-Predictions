import sys
from pathlib import Path

# Garante que a raiz do projeto está no sys.path independente de onde o script é executado
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from typing import Union
import csv
from db_setup.my_sql import SyntaxMySQL
from db_setup.postgre_sql import SyntaxPostgreeSQL, Client as PostgreSQLClient


def detect_delimiter(file_path: Union[str, Path]) -> str:
    """
    Detecta automaticamente o delimitador de um arquivo CSV.
    Tenta: ';', ',', '\t', '|'
    
    Args:
        file_path: Caminho para o arquivo CSV
    
    Returns:
        str: O delimitador detectado (padrão: ',')
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sample = f.read(2048)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=';,\t|')
                return dialect.delimiter
            except csv.Error:
                # Se Sniffer falhar, tenta heurística simples
                first_line = sample.split('\n')[0]
                for delim in [';', ',', '\t', '|']:
                    if delim in first_line:
                        return delim
                return ','
    except Exception:
        return ','


def read_csv_to_polars(file_path: Union[str, Path], sep: str = None) -> pl.DataFrame:
    """
    Lê arquivo CSV e converte para Polars DataFrame.
    
    Args:
        file_path: Caminho para o arquivo CSV
        sep: Delimitador do CSV. Se None, detecta automaticamente.
             Valores comuns: ',', ';', '\t', '|'
    
    Returns:
        pl.DataFrame: DataFrame com os dados do CSV
        
    Exemplo:
        >>> df = read_csv_to_polars('dados.csv')  # auto-detecta
        >>> df = read_csv_to_polars('dados.csv', sep=';')  # força ponto-e-vírgula
    """
    if sep is None:
        sep = detect_delimiter(file_path)
    return pl.read_csv(file_path, separator=sep)


def process_client_units(file_path: Union[str, Path], date_auto_name: str, sep: str = None) -> pl.DataFrame:
    """
    Processa arquivo CSV de unidades Bradesco, adicionando a coluna data_instalacao.
    
    A data_instalacao é calculada como: data_inicio_automacao - dias_antes_automacao
    
    Args:
        file_path: Caminho para o arquivo CSV. Ex: 'BradescoUnidadesSemAuto.csv'
        date_auto_name: Nome da coluna com data de início da automação
        sep: Delimitador do CSV. Se None, detecta automaticamente.
    
    Returns:
        pl.DataFrame: DataFrame processado com a coluna data_instalacao
    
    Exemplo:
        >>> df = process_client_units('BradescoUnidadesSemAuto.csv', 'data_inicio')
        >>> print(df.columns)
        ['id_bradesco', 'unit_name', 'data_inicio_automacao', 'dias_antes_automacao', 'data_instalacao']
    """
    # Lê o CSV
    if sep is None:
        sep = detect_delimiter(file_path)
    df = pl.read_csv(file_path, separator=sep)
    
    # Converte data_inicio_automacao para o formato de data (está em MM/DD/YY)
    df = df.with_columns(
        pl.col(date_auto_name).str.to_date('%m/%d/%y').alias('data_inicio_automacao')
    )
    
    # Calcula data_instalacao = data_inicio_automacao - dias_antes_automacao
    df = df.with_columns(
        (pl.col('data_inicio_automacao') - pl.duration(days=pl.col('dias_antes_automacao')+1))
        .alias('data_instalacao')
    )
    
    return df


def enrich_csv_with_reference_id(
    file_path: Union[str, Path],
    mysql_client: SyntaxMySQL,
    output_path: Union[str, Path] = None,
    sep: str = None,
) -> pl.DataFrame:
    """
    Adiciona a coluna 'reference_id' ao CSV de unidades a partir dos UNIT_IDs
    retornados pela query get_devices_by_units_by_unit_names.

    O mapeamento é feito via join entre a coluna 'unit_name' do CSV e
    a coluna UNIT_NAME retornada pela query, preservando todas as linhas
    originais (left join).

    Args:
        file_path (str | Path): Caminho para o CSV de unidades.
            Ex: 'use_case/files/BradescoUnidadesSemAuto.csv'
        mysql_client (SyntaxMySQL): Instância conectada para executar a query federada.
        output_path (str | Path | None): Destino do CSV enriquecido.
            Se None, sobrescreve file_path.
        sep (str | None): Delimitador do CSV. Se None, detecta automaticamente.

    Returns:
        pl.DataFrame: DataFrame original acrescido da coluna 'reference_id' (Int64).
    """
    if sep is None:
        sep = detect_delimiter(file_path)
    df = pl.read_csv(file_path, separator=sep)

    if "reference_id" in df.columns:
        print("  ↳ 'reference_id' já presente no CSV. Etapa ignorada.")
        return df

    unit_names = df["unit_name"].to_list()

    query = SyntaxMySQL.fetch_devices_by_unit_names(unit_names)
    result = mysql_client.execute_query(query, verbose=False)

    # Mapeamento único unit_name → unit_id (uma linha por unidade)
    unit_id_map = (
        result
        .select(["UNIT_NAME", "UNIT_ID"])
        .unique()
        .rename({"UNIT_NAME": "unit_name", "UNIT_ID": "reference_id"})
    )

    df = df.join(unit_id_map, on="unit_name", how="left")

    dest = Path(output_path) if output_path else Path(file_path)
    df.write_csv(dest, separator=sep)

    return df


def enrich_csv_with_unit_id(
    file_path: Union[str, Path],
    pg_client: PostgreSQLClient,
    output_path: Union[str, Path] = None,
    sep: str = None,
) -> pl.DataFrame:
    """
    Adiciona a coluna 'unit_id' ao CSV de unidades a partir dos IDs internos
    do PostgreSQL, usando o mapeamento reference_id → id da tabela units.

    Pré-requisito: o CSV já deve conter a coluna 'reference_id' gerada por
    enrich_csv_with_reference_id().

    O join é feito pela coluna 'reference_id', preservando todas as linhas
    originais (left join). Linhas sem correspondência ficam com unit_id nulo.

    Args:
        file_path (str | Path): Caminho para o CSV de unidades.
            Ex: 'use_case/files/BradescoUnidadesSemAuto.csv'
        pg_client (PostgreSQLClient): Instância conectada ao PostgreSQL.
        output_path (str | Path | None): Destino do CSV enriquecido.
            Se None, sobrescreve file_path.
        sep (str | None): Delimitador do CSV. Se None, detecta automaticamente.

    Returns:
        pl.DataFrame: DataFrame original acrescido da coluna 'unit_id' (Int64).

    Raises:
        ValueError: Se a coluna 'reference_id' não existir no CSV.
    """
    if sep is None:
        sep = detect_delimiter(file_path)
    df = pl.read_csv(file_path, separator=sep)

    if "unit_id" in df.columns:
        print("  ↳ 'unit_id' já presente no CSV. Etapa ignorada.")
        return df

    if "reference_id" not in df.columns:
        raise ValueError(
            "Coluna 'reference_id' não encontrada no CSV. "
            "Execute enrich_csv_with_reference_id() antes desta etapa."
        )

    reference_ids = df["reference_id"].drop_nulls().cast(pl.Int64).unique().to_list()

    query = SyntaxPostgreeSQL.get_unit_ids_by_reference_ids(reference_ids)
    result = pg_client.query(query)

    unit_id_map = (
        pl.from_arrow(result.to_arrow())
        .select(["reference_id", "id"])
        .rename({"id": "unit_id"})
        .with_columns(pl.col("reference_id").cast(pl.Int64))
        
    )

    df = df.with_columns(pl.col("reference_id").cast(pl.Int64))
    df = df.join(unit_id_map, on="reference_id", how="left")

    dest = Path(output_path) if output_path else Path(file_path)
    df.write_csv(dest, separator=sep)

    return df


if __name__ == "__main__":
    csv_path = r'use_case\files\BradescoUnidadesSemAuto_updated.CSV'

    # 1. Processa datas e data_instalacao
    df_processed = process_client_units(csv_path, date_auto_name="data_inicio")
    print("Primeiras 5 linhas do DataFrame processado:")
    print(df_processed.head())

    # 2. Enriquece com reference_id via BigQuery / MySQL
    mysql_client = SyntaxMySQL()
    df_enriched = enrich_csv_with_reference_id(csv_path, mysql_client)
    print("\nDataFrame com reference_id:")
    print(df_enriched.head())

    # 3. Enriquece com unit_id via PostgreSQL
    pg_client = PostgreSQLClient()
    df_final = enrich_csv_with_unit_id(csv_path, pg_client)
    print("\nDataFrame com unit_id:")
    print(df_final.head())
    pg_client.close()

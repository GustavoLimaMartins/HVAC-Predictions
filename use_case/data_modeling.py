import sys
from pathlib import Path

# Garante que a raiz do projeto está no sys.path independente de onde o script é executado
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl
from typing import Union
from db_setup.my_sql import SyntaxMySQL
from db_setup.postgre_sql import SyntaxPostgreeSQL, Client as PostgreSQLClient

def read_csv_to_polars(file_path: Union[str, Path]) -> pl.DataFrame:
    """
    Lê arquivo CSV e converte para Polars DataFrame.
    
    Args:
        file_path: Caminho para o arquivo CSV
    
    Returns:
        pl.DataFrame: DataFrame com os dados do CSV
    """
    return pl.read_csv(file_path)


def process_client_units(file_path: Union[str, Path], date_auto_name: str) -> pl.DataFrame:
    """
    Processa arquivo CSV de unidades Bradesco, adicionando a coluna data_instalacao.
    
    A data_instalacao é calculada como: data_inicio_automacao - dias_antes_automacao
    
    Args:
        file_path: Caminho para o arquivo CSV. Ex: 'BradescoUnidadesSemAuto.csv'
    
    Returns:
        pl.DataFrame: DataFrame processado com a coluna data_instalacao
    
    Exemplo:
        >>> df = process_bradesco_units('BradescoUnidadesSemAuto.csv')
        >>> print(df.columns)
        ['id_bradesco', 'unit_name', 'data_inicio_automacao', 'dias_antes_automacao', 'data_instalacao']
    """
    # Lê o CSV
    df = pl.read_csv(file_path)
    
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

    Returns:
        pl.DataFrame: DataFrame original acrescido da coluna 'reference_id' (Int64).
    """
    df = pl.read_csv(file_path)

    if "reference_id" in df.columns:
        print("  ↳ 'reference_id' já presente no CSV. Etapa ignorada.")
        return df

    unit_names = df["unit_name"].to_list()

    query = SyntaxMySQL.get_devices_by_units_by_unit_names(unit_names)
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
    df.write_csv(dest)

    return df


def enrich_csv_with_unit_id(
    file_path: Union[str, Path],
    pg_client: PostgreSQLClient,
    output_path: Union[str, Path] = None,
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

    Returns:
        pl.DataFrame: DataFrame original acrescido da coluna 'unit_id' (Int64).

    Raises:
        ValueError: Se a coluna 'reference_id' não existir no CSV.
    """
    df = pl.read_csv(file_path)

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
    df.write_csv(dest)

    return df


if __name__ == "__main__":
    csv_path = r'use_case\files\BradescoUnidadesSemAuto.csv'

    # 1. Processa datas e data_instalacao
    df_processed = process_client_units(csv_path)
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

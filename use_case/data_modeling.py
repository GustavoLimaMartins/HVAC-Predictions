import polars as pl
from pathlib import Path
from typing import Union

def read_csv_to_polars(file_path: Union[str, Path]) -> pl.DataFrame:
    """
    Lê arquivo CSV e converte para Polars DataFrame.
    
    Args:
        file_path: Caminho para o arquivo CSV
    
    Returns:
        pl.DataFrame: DataFrame com os dados do CSV
    """
    return pl.read_csv(file_path)


def process_client_units(file_path: Union[str, Path]) -> pl.DataFrame:
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
        pl.col('data_inicio_automacao').str.to_date('%m/%d/%y').alias('data_inicio_automacao')
    )
    
    # Calcula data_instalacao = data_inicio_automacao - dias_antes_automacao
    df = df.with_columns(
        (pl.col('data_inicio_automacao') - pl.duration(days=pl.col('dias_antes_automacao')+1))
        .alias('data_instalacao')
    )
    
    return df


if __name__ == "__main__":
    # Exemplo de uso
    csv_path = Path(__file__).parent / 'BradescoUnidadesSemAuto.csv'
    
    # Processa o arquivo
    df = process_client_units(csv_path)
    
    # Exibe as primeiras linhas
    print("Primeiras 5 linhas do DataFrame processado:")
    print(df.head())

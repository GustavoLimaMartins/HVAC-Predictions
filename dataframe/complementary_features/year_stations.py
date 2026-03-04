import polars as pl
from typing import Union
from pathlib import Path

class YearStationsEnricher:
    """Classe para enriquecimento de dados com estações do ano brasileiras.
    
    Adiciona uma coluna 'estacao' ao DataFrame baseada na coluna 'data',
    classificando cada registro em uma das quatro estações do ano:
    - verão
    - outono
    - inverno
    - primavera
    
    As estações seguem o calendário astronômico brasileiro:
    - Verão: 21/22 dezembro até 20/21 março
    - Outono: 20/21 março até 20/21 junho
    - Inverno: 20/21 junho até 22/23 setembro
    - Primavera: 22/23 setembro até 21/22 dezembro
    
    Attributes:
        df (pl.DataFrame): DataFrame com os dados enriquecidos
    """
    
    def __init__(self, df: pl.DataFrame, date_column: str = 'data'):
        """Inicializa o enriquecedor de estações do ano.
        
        Args:
            df (pl.DataFrame): DataFrame Polars com coluna de data
            date_column (str): Nome da coluna de data (padrão: 'data')
        
        Raises:
            ValueError: Se a coluna de data não existir no DataFrame
        """
        if date_column not in df.columns:
            raise ValueError(f"Coluna '{date_column}' não encontrada no DataFrame. Colunas disponíveis: {df.columns}")
        
        self.date_column = date_column
        self.df = df.clone()
        
        # Garante que a coluna de data está no tipo correto
        self.df = self.df.with_columns([
            pl.col(self.date_column).cast(pl.Date)
        ])
        
        print(f"✓ YearStationsEnricher inicializado")
        print(f"  Registros: {self.df.shape[0]}")
        print(f"  Coluna de data: '{self.date_column}'")
    
    def add_season_column(self) -> pl.DataFrame:
        """Adiciona coluna 'estacao' ao DataFrame.
        
        A classificação é feita com base em intervalos aproximados das estações
        astronômicas brasileiras. Para simplificar, usa-se:
        - Verão: 21 dezembro até 20 março
        - Outono: 21 março até 20 junho
        - Inverno: 21 junho até 22 setembro
        - Primavera: 23 setembro até 20 dezembro
        
        Returns:
            pl.DataFrame: DataFrame com nova coluna 'estacao'
        """
        self.df = self.df.with_columns([
            pl.col(self.date_column).dt.month().alias('_mes'),
            pl.col(self.date_column).dt.day().alias('_dia')
        ]).with_columns([
            pl.when(
                # Verão: 21/dez até 20/mar
                ((pl.col('_mes') == 12) & (pl.col('_dia') >= 21)) |
                (pl.col('_mes').is_in([1, 2])) |
                ((pl.col('_mes') == 3) & (pl.col('_dia') <= 20))
            ).then(pl.lit('verao'))
            .when(
                # Outono: 21/mar até 20/jun
                ((pl.col('_mes') == 3) & (pl.col('_dia') >= 21)) |
                (pl.col('_mes').is_in([4, 5])) |
                ((pl.col('_mes') == 6) & (pl.col('_dia') <= 20))
            ).then(pl.lit('outono'))
            .when(
                # Inverno: 21/jun até 22/set
                ((pl.col('_mes') == 6) & (pl.col('_dia') >= 21)) |
                (pl.col('_mes').is_in([7, 8])) |
                ((pl.col('_mes') == 9) & (pl.col('_dia') <= 22))
            ).then(pl.lit('inverno'))
            .otherwise(pl.lit('primavera'))  # Primavera: 23/set até 20/dez
            .alias('estacao')
        ]).drop(['_mes', '_dia'])
        
        print(f"✓ Coluna 'estacao' adicionada com sucesso")
        
        # Mostra distribuição das estações
        season_counts = (
            self.df
            .group_by('estacao')
            .agg(pl.len().alias('quantidade'))
            .sort('estacao')
        )
        print(f"  Distribuição por estação:")
        for row in season_counts.iter_rows(named=True):
            print(f"    {row['estacao']}: {row['quantidade']} registros")
        
        return self.df
    
    def get_enriched_dataframe(self) -> pl.DataFrame:
        """Retorna o DataFrame enriquecido.
        
        Returns:
            pl.DataFrame: DataFrame com a coluna 'estacao'
        """
        if 'estacao' not in self.df.columns:
            print("⚠ Coluna 'estacao' ainda não foi adicionada. Execute add_season_column() primeiro.")
        
        return self.df
    
    def save_to_csv(self, output_path: Union[str, Path]) -> None:
        """Salva o DataFrame enriquecido em arquivo CSV.
        
        Args:
            output_path (Union[str, Path]): Caminho do arquivo de saída
        """
        self.df.write_csv(output_path)
        print(f"✓ Arquivo salvo: {output_path}")


def enrich_with_seasons(df: pl.DataFrame, date_column: str = 'data') -> pl.DataFrame:
    """Função auxiliar para enriquecer DataFrame com estações do ano.
    
    Args:
        df (pl.DataFrame): DataFrame com coluna de data
        date_column (str): Nome da coluna de data (padrão: 'data')
    
    Returns:
        pl.DataFrame: DataFrame enriquecido com coluna 'estacao'
    
    Exemplo:
        >>> import polars as pl
        >>> df = pl.DataFrame({'data': ['2024-01-15', '2024-07-20'], 'valor': [100, 200]})
        >>> df_enriched = enrich_with_seasons(df)
        >>> print(df_enriched)
    """
    enricher = YearStationsEnricher(df, date_column)
    return enricher.add_season_column()


def get_season_column_from_csv(csv_path: Union[str, Path], date_column: str = 'data') -> pl.DataFrame:
    """Carrega apenas a coluna de data de um CSV e retorna apenas a coluna 'estacao'.
    
    OTIMIZADO para processar grandes arquivos com mínimo uso de memória.
    Processo integrado: carrega, processa e retorna em uma única operação.
    
    Args:
        csv_path (Union[str, Path]): Caminho do arquivo CSV
        date_column (str): Nome da coluna de data (padrão: 'data')
    
    Returns:
        pl.DataFrame: DataFrame com apenas a coluna 'estacao' (mesmo número de linhas do CSV original)
    
    Uso típico:
        >>> # Gera e integra em 2 linhas
        >>> df_full = pl.read_csv('consumption_consolidated.csv')
        >>> df_final = df_full.hstack(get_season_column_from_csv('consumption_consolidated.csv'))
    
    Exemplo:
        >>> import polars as pl
        >>> from dataframe.complementary_features.year_stations import get_season_column_from_csv
        >>> 
        >>> # Método otimizado (uma linha)
        >>> df_final = pl.read_csv('data.csv').hstack(get_season_column_from_csv('data.csv'))
    """
    # Carrega apenas a coluna de data e processa em pipeline integrado
    df_season = (
        pl.read_csv(csv_path, columns=[date_column])
        .with_columns([
            pl.col(date_column).cast(pl.Date)
        ])
        .with_columns([
            pl.col(date_column).dt.month().alias('_mes'),
            pl.col(date_column).dt.day().alias('_dia')
        ])
        .with_columns([
            pl.when(
                # Verão: 21/dez até 20/mar
                ((pl.col('_mes') == 12) & (pl.col('_dia') >= 21)) |
                (pl.col('_mes').is_in([1, 2])) |
                ((pl.col('_mes') == 3) & (pl.col('_dia') <= 20))
            ).then(pl.lit('verao'))
            .when(
                # Outono: 21/mar até 20/jun
                ((pl.col('_mes') == 3) & (pl.col('_dia') >= 21)) |
                (pl.col('_mes').is_in([4, 5])) |
                ((pl.col('_mes') == 6) & (pl.col('_dia') <= 20))
            ).then(pl.lit('outono'))
            .when(
                # Inverno: 21/jun até 22/set
                ((pl.col('_mes') == 6) & (pl.col('_dia') >= 21)) |
                (pl.col('_mes').is_in([7, 8])) |
                ((pl.col('_mes') == 9) & (pl.col('_dia') <= 22))
            ).then(pl.lit('inverno'))
            .otherwise(pl.lit('primavera'))  # Primavera: 23/set até 20/dez
            .alias('estacao')
        ])
        .select(['estacao'])
    )
    
    return df_season


if __name__ == "__main__":
    """Exemplo de uso otimizado da função get_season_column_from_csv."""
    
    print("=" * 80)
    print("ENRIQUECIMENTO COM ESTAÇÕES DO ANO")
    print("=" * 80)
    
    try:
        csv_path = r'use_case\output_files\consumption_consolidated.csv'
        if not Path(csv_path).exists():
            csv_path = r'use_case\files\consumption_consolidated.csv'
        
        if Path(csv_path).exists():
            
            df_season = get_season_column_from_csv(csv_path)
            df_full = pl.read_csv(csv_path)
            df_final = df_full.hstack(df_season)
            
            print(f"\n✓ Coluna 'estacao' gerada: {df_season.shape}")
            print(f"✓ DataFrame final: {df_final.shape}")
            print(f"✓ Colunas: {df_final.columns}")
            
            print("\n" + "-" * 80)
            print("Distribuição por estação:")
            season_dist = df_final.group_by('estacao').agg(pl.len().alias('quantidade')).sort('estacao')
            for row in season_dist.iter_rows(named=True):
                print(f"  {row['estacao']}: {row['quantidade']} registros")
            
            print("\nAmostra do resultado final (5 linhas):")
            print(df_final.select(['unit_id', 'data', 'estacao', 'consumo_kwh']).head())
            
        else:
            print(f"\n⚠ Arquivo não encontrado em:")
            print("  - use_case\\output_files\\consumption_consolidated.csv")
            print("  - use_case\\files\\consumption_consolidated.csv")
            
    except Exception as e:
        print(f"\n✗ Erro: {e}")

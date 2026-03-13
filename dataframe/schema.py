import polars as pl
from dataframe.complementary import enrich_dataframe_with_all_features


class DataFrameFormatter:
    """
    Formata DataFrame para modelo de machine learning.
    
    Pipeline completo:
    1. Enriquece com features complementares (estações, clima, grupos regionais)
    2. Remove colunas desnecessárias para modelo
    
    Example:
        >>> formatter = DataFrameFormatter()
        >>> df_formatted = formatter.prepare_from_csv('consumption_consolidated.csv')
    """
    
    def __init__(self):
        """Inicializa o formatador de DataFrame."""
        self.columns_to_drop = [
            'unit_id',
            'device_id',
            'device_version',
            'data_instalacao',
            'data_inicio_automacao',
            'metodo',
            'latitude',
            'longitude'
        ]
    
    def drop_unnecessary_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove colunas desnecessárias para modelo.
        
        Args:
            df: DataFrame com todas as colunas
        
        Returns:
            DataFrame sem colunas desnecessárias
        """
        existing_columns = [col for col in self.columns_to_drop if col in df.columns]
        return df.drop(existing_columns) if existing_columns else df
    
    def drop_null_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Remove linhas com valores nulos em qualquer coluna.
        
        Args:
            df: DataFrame com possíveis valores nulos
        
        Returns:
            DataFrame sem linhas contendo valores nulos
        """
        return df.drop_nulls()
    
    def format_date_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Converte colunas de data para o tipo pl.Date.

        Colunas do tipo String com formato 'YYYY-MM-DD' são convertidas
        automaticamente. Colunas já tipadas como Date são ignoradas.

        Args:
            df: DataFrame com possíveis colunas de data como String

        Returns:
            DataFrame com colunas de data no tipo pl.Date
        """
        date_columns = [
            col for col in df.columns
            if df[col].dtype == pl.Utf8
            and df[col].drop_nulls().head(1).to_list()
            and len(str(df[col].drop_nulls().head(1).to_list()[0])) == 10
            and str(df[col].drop_nulls().head(1).to_list()[0]).count('-') == 2
        ]
        if not date_columns:
            return df
        return df.with_columns([
            pl.col(col).str.to_date("%Y-%m-%d").alias(col)
            for col in date_columns
        ])

    def format_for_model(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Pipeline completo de formatação para modelo.
        
        Remove automaticamente linhas com valores nulos em qualquer coluna.
        
        Args:
            df: DataFrame bruto com dados de consumo
        
        Returns:
            DataFrame formatado pronto para modelo (sem nulos)
        
        Example:
            >>> formatter = DataFrameFormatter()
            >>> df_ready = formatter.format_for_model(df_raw)
        """
        # 1. Formata colunas de data
        df_dated = self.format_date_columns(df)
        
        # 2. Remove colunas desnecessárias
        df_formatted = self.drop_unnecessary_columns(df_dated)
        
        # 3. Remove linhas com nulos
        df_clean = self.drop_null_rows(df_formatted)
        
        return df_clean
    
    def prepare_from_csv(self, csv_path: str) -> pl.DataFrame:
        """
        Pipeline completo: carrega CSV, enriquece e formata para modelo.
        
        Args:
            csv_path: Caminho do arquivo CSV com dados de consumo
        
        Returns:
            DataFrame formatado e enriquecido pronto para modelo
        
        Example:
            >>> formatter = DataFrameFormatter()
            >>> df_ready = formatter.prepare_from_csv('consumption_consolidated.csv')
        """
        # 1. Carrega CSV
        df = pl.read_csv(csv_path)
        
        # 2. Remove linhas com valores nulos
        df_clean = self.drop_null_rows(df)
        
        # 3. Enriquece com features complementares
        df_enriched = enrich_dataframe_with_all_features(df_clean)
        
        # 4. Formata para modelo
        df_formatted = self.format_for_model(df_enriched)
        
        return df_formatted


def prepare_dataframe_for_model(csv_path: str) -> pl.DataFrame:
    """
    Função helper para preparação completa de DataFrame a partir de CSV.
    
    Args:
        csv_path: Caminho do arquivo CSV
    
    Returns:
        DataFrame formatado e enriquecido pronto para modelo
    
    Example:
        >>> df_ready = prepare_dataframe_for_model('consumption_consolidated.csv')
    """
    formatter = DataFrameFormatter()
    return formatter.prepare_from_csv(csv_path)

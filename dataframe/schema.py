import polars as pl
from dataframe.complementary import enrich_dataframe_with_all_features


class DataFrameFormatter:
    """
    Formata DataFrame para modelo de machine learning.
    
    Pipeline completo:
    1. Enriquece com features complementares (estações, clima, grupos regionais)
    2. Cria features binárias is_dac e is_dut baseadas em device_id
    3. Remove colunas desnecessárias para modelo
    
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
    
    def create_device_type_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Cria features binárias baseadas no tipo de dispositivo.
        
        Args:
            df: DataFrame com coluna 'device_id'
        
        Returns:
            DataFrame com novas colunas 'is_dac' e 'is_dut' (0 ou 1)
        """
        return df.with_columns([
            pl.col('device_id').str.starts_with('DAC').cast(pl.Int8).alias('is_dac'),
            pl.col('device_id').str.starts_with('DUT').cast(pl.Int8).alias('is_dut')
        ])
    
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
        print(f"\n[Formatação] Registros iniciais: {df.shape[0]}")
        print(f"[Formatação] Colunas iniciais: {df.columns}")
        
        # 1. Cria features binárias
        df_with_features = self.create_device_type_features(df)
        print(f"[Formatação] ✓ Features binárias criadas: is_dac, is_dut")
        
        # 2. Remove colunas desnecessárias
        existing_to_drop = [col for col in self.columns_to_drop if col in df_with_features.columns]
        df_formatted = self.drop_unnecessary_columns(df_with_features)
        if existing_to_drop:
            print(f"[Formatação] ✓ Colunas removidas: {existing_to_drop}")
        
        # 3. Remove linhas com nulos
        null_count_before = df_formatted.shape[0]
        df_clean = self.drop_null_rows(df_formatted)
        removed_nulls = null_count_before - df_clean.shape[0]
        
        if removed_nulls > 0:
            print(f"[Formatação] ✓ Removidos {removed_nulls} registros com valores nulos")
        else:
            print(f"[Formatação] ✓ Nenhum registro com nulos encontrado")
        
        print(f"[Formatação] Registros finais: {df_clean.shape[0]}")
        print(f"[Formatação] Colunas finais ({len(df_clean.columns)}): {df_clean.columns}")
        
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

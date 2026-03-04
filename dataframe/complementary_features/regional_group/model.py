from typing import Optional
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import sys

# Adiciona path para importar módulos do projeto
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from db_setup.my_sql import SyntaxMySQL


class RegionalGroupClassifier:
    """
    Classificador de grupos regionais usando K-means clustering.
    
    Utiliza aprendizado não-supervisionado para identificar grupos regionais
    baseados em coordenadas geográficas (latitude/longitude). O número de clusters
    é automaticamente definido pela quantidade de cidades únicas no banco de dados.
    
    Attributes:
        n_clusters (int): Número de clusters (igual ao número de cidades)
        kmeans (KMeans): Modelo K-means treinado
        scaler (StandardScaler): Normalizador de coordenadas
        is_fitted (bool): Indica se o modelo foi treinado
        
    Example:
        >>> classifier = RegionalGroupClassifier()
        >>> classifier.fit(df_locations)
        >>> df_enriched = classifier.predict(df_data)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Inicializa o classificador de grupos regionais.
        
        Args:
            random_state (int): Seed para reprodutibilidade (padrão: 42)
        """
        self.n_clusters: Optional[int] = None
        self.kmeans: Optional[KMeans] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.random_state = random_state
        self.mysql_client = SyntaxMySQL()
        
        print("✓ RegionalGroupClassifier inicializado")
    
    def _get_number_of_cities(self) -> int:
        """
        Obtém o número de cidades únicas do banco de dados.
        
        Returns:
            int: Número de cidades únicas
        """
        query = SyntaxMySQL.get_unique_cities()
        df_cities = self.mysql_client.execute_query(query, verbose=False)
        n_cities = len(df_cities)
        
        print(f"✓ Número de cidades únicas encontradas: {n_cities}")
        return n_cities
    
    def fit(self, df: pl.DataFrame) -> 'RegionalGroupClassifier':
        """
        Treina o modelo K-means com as coordenadas únicas do DataFrame.
        
        O número de clusters é automaticamente definido pela quantidade
        de cidades únicas no banco de dados MySQL.
        
        IMPORTANTE: Remove automaticamente linhas com valores nulos em latitude/longitude
        antes do treinamento, pois K-means não aceita valores NaN.
        
        Args:
            df (pl.DataFrame): DataFrame com colunas 'latitude' e 'longitude'
        
        Returns:
            RegionalGroupClassifier: Self para method chaining
        
        Raises:
            ValueError: Se colunas latitude/longitude não existirem
        """
        # Validações
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("DataFrame deve conter colunas 'latitude' e 'longitude'")
        
        # Remove nulos (K-means não aceita NaN)
        original_count = df.shape[0]
        df_clean = df.drop_nulls(subset=['latitude', 'longitude'])
        removed_count = original_count - df_clean.shape[0]
        
        if removed_count > 0:
            print(f"⚠️  Removidos {removed_count} registros com valores nulos em latitude/longitude")
            print(f"   Registros válidos: {df_clean.shape[0]}")
        
        # Obtém número de cidades para definir clusters
        self.n_clusters = self._get_number_of_cities()
        
        # Extrai coordenadas únicas
        df_coords = df_clean.select(['latitude', 'longitude']).unique()
        
        if len(df_coords) < self.n_clusters:
            print(f"⚠️  Aviso: Apenas {len(df_coords)} coordenadas únicas encontradas, mas {self.n_clusters} clusters solicitados")
            print(f"   Ajustando para {len(df_coords)} clusters")
            self.n_clusters = len(df_coords)
        
        # Converte para numpy array
        coords_array = df_coords.to_numpy()
        
        # Normaliza as coordenadas
        coords_scaled = self.scaler.fit_transform(coords_array)
        
        # Treina K-means
        print(f"⏳ Treinando K-means com {self.n_clusters} clusters...")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        self.kmeans.fit(coords_scaled)
        
        self.is_fitted = True
        print(f"✓ Modelo treinado com sucesso!")
        print(f"  Inércia: {self.kmeans.inertia_:.2f}")
        
        return self
    
    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Adiciona coluna 'grupo_regional' ao DataFrame com os clusters preditos.
        
        Registros com latitude/longitude nulos recebem valor null na coluna 'grupo_regional'.
        Isso mantém o mesmo número de linhas do DataFrame original para compatibilidade.
        
        Args:
            df (pl.DataFrame): DataFrame com colunas 'latitude' e 'longitude'
        
        Returns:
            pl.DataFrame: DataFrame original com nova coluna 'grupo_regional'
        
        Raises:
            ValueError: Se modelo não foi treinado ou colunas necessárias não existem
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute fit() primeiro.")
        
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            raise ValueError("DataFrame deve conter colunas 'latitude' e 'longitude'")
        
        # Identifica registros com coordenadas válidas (não-nulos)
        mask_valid = (~df['latitude'].is_null()) & (~df['longitude'].is_null())
        
        # Filtra apenas coordenadas válidas para predição
        df_valid = df.filter(mask_valid)
        
        if len(df_valid) == 0:
            # Todos os registros têm coordenadas nulas
            df_result = df.with_columns(
                pl.lit(None).cast(pl.Utf8).alias('grupo_regional')
            )
            print("⚠️  Todos os registros possuem coordenadas nulas")
            return df_result
        
        # Prepara coordenadas válidas
        coords = df_valid.select(['latitude', 'longitude']).to_numpy()
        
        # Normaliza e prediz
        coords_scaled = self.scaler.transform(coords)
        predictions = self.kmeans.predict(coords_scaled)
        
        # Cria coluna temporária com índices
        df_with_idx = df.with_row_index("__idx__")
        df_valid_with_idx = df_with_idx.filter(mask_valid)
        
        # Cria DataFrame com predições e índices
        df_predictions = pl.DataFrame({
            '__idx__': df_valid_with_idx['__idx__'].to_list(),
            'grupo_regional': predictions.astype(str)
        })
        
        # Join para adicionar predições (nulos onde não havia coordenadas válidas)
        df_result = (
            df_with_idx
            .join(df_predictions, on='__idx__', how='left')
            .drop('__idx__')
        )
        
        valid_count = len(df_valid)
        null_count = len(df) - valid_count
        
        print(f"✓ Grupos regionais adicionados: {valid_count} registros")
        if null_count > 0:
            print(f"⚠️  {null_count} registros com coordenadas nulas (grupo_regional = null)")
        
        return df_result
    
    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Treina o modelo e retorna DataFrame com grupos regionais em uma única operação.
        
        Args:
            df (pl.DataFrame): DataFrame com colunas 'latitude' e 'longitude'
        
        Returns:
            pl.DataFrame: DataFrame original com nova coluna 'grupo_regional'
        """
        self.fit(df)
        return self.predict(df)
    
    def get_cluster_centers(self) -> Optional[pl.DataFrame]:
        """
        Retorna as coordenadas dos centróides dos clusters.
        
        Returns:
            pl.DataFrame: DataFrame com latitude/longitude dos centróides, ou None se não treinado
        """
        if not self.is_fitted:
            print("⚠️  Modelo não foi treinado ainda")
            return None
        
        # Desnormaliza os centróides
        centers_scaled = self.kmeans.cluster_centers_
        centers = self.scaler.inverse_transform(centers_scaled)
        
        df_centers = pl.DataFrame({
            'grupo_regional': [str(i) for i in range(self.n_clusters)],
            'centroide_latitude': centers[:, 0],
            'centroide_longitude': centers[:, 1]
        })
        
        return df_centers
    
    def save_model_info(self, output_path: str = "regional_groups_info.csv") -> None:
        """
        Salva informações dos centróides em arquivo CSV.
        
        Args:
            output_path (str): Caminho para salvar o arquivo
        """
        if not self.is_fitted:
            print("⚠️  Modelo não foi treinado ainda")
            return
        
        df_centers = self.get_cluster_centers()
        df_centers.write_csv(output_path)
        print(f"✓ Informações dos centróides salvas em: {output_path}")


def enrich_with_regional_groups(df: pl.DataFrame) -> pl.DataFrame:
    """
    Função helper para enriquecer DataFrame com grupos regionais.
    
    Args:
        df (pl.DataFrame): DataFrame com colunas 'latitude' e 'longitude'
    
    Returns:
        pl.DataFrame: DataFrame enriquecido com coluna 'grupo_regional'
    
    Example:
        >>> df_enriched = enrich_with_regional_groups(df_original)
    """
    classifier = RegionalGroupClassifier()
    return classifier.fit_predict(df)


def get_regional_groups_from_csv(
    csv_path: str,
    output_path: Optional[str] = None
) -> pl.DataFrame:
    """
    Carrega CSV, aplica classificação regional e retorna apenas coluna de grupos.
    
    Esta função otimizada carrega apenas as colunas necessárias (latitude, longitude)
    e retorna apenas a coluna 'grupo_regional' para integração posterior.
    
    Args:
        csv_path (str): Caminho do arquivo CSV de entrada
        output_path (str, optional): Se fornecido, salva resultado em CSV
    
    Returns:
        pl.DataFrame: DataFrame com apenas a coluna 'grupo_regional'
    
    Example:
        >>> df_grupos = get_regional_groups_from_csv('consumption_consolidated.csv')
        >>> # Integrar no DataFrame principal
        >>> df_final = df_original.with_columns(df_grupos)
    """
    print(f"⏳ Carregando coordenadas de {csv_path}...")
    
    # Carrega apenas colunas necessárias (otimização de memória)
    df_coords = pl.read_csv(csv_path, columns=['latitude', 'longitude'])
    
    print(f"✓ {len(df_coords)} registros carregados")
    
    # Aplica classificação
    classifier = RegionalGroupClassifier()
    df_classified = classifier.fit_predict(df_coords)
    
    # Retorna apenas coluna de grupos
    df_result = df_classified.select('grupo_regional')
    
    # Salva se solicitado
    if output_path:
        df_result.write_csv(output_path)
        print(f"✓ Grupos regionais salvos em: {output_path}")
    
    return df_result


def main():
    """Função de exemplo/teste do módulo"""
    print("=" * 80)
    print("CLASSIFICAÇÃO DE GRUPOS REGIONAIS - K-MEANS")
    print("=" * 80)
    
    # Exemplo com dados simulados
    print("\n📊 Exemplo 1: Dados simulados")
    print("-" * 80)
    
    # Cria dados de exemplo
    np.random.seed(42)
    df_example = pl.DataFrame({
        'unit_id': [f'UNIT_{i}' for i in range(100)],
        'latitude': np.random.uniform(-30, -10, 100),  # Sul/Sudeste do Brasil
        'longitude': np.random.uniform(-55, -35, 100),
        'consumo_kwh': np.random.uniform(10, 100, 100)
    })
    
    print(f"Dataset exemplo: {len(df_example)} registros")
    print(df_example.head(5))
    
    # Aplica classificação
    print("\n⏳ Aplicando classificação regional...")
    df_classified = enrich_with_regional_groups(df_example)
    
    print("\n✓ Resultado (primeiras 5 linhas):")
    print(df_classified.head(5))
    
    print("\n📈 Distribuição por grupo regional:")
    df_dist = df_classified.group_by('grupo_regional').agg([
        pl.len().alias('quantidade'),
        pl.mean('latitude').alias('lat_media'),
        pl.mean('longitude').alias('lon_media'),
        pl.mean('consumo_kwh').alias('consumo_medio')
    ]).sort('grupo_regional')
    print(df_dist)
    
    # Mostra centróides
    classifier = RegionalGroupClassifier()
    classifier.fit(df_example)
    df_centers = classifier.get_cluster_centers()
    print("\n🎯 Centróides dos clusters:")
    print(df_centers)
    
    print("\n" + "=" * 80)
    print("✓ Exemplo concluído com sucesso!")
    print("=" * 80)


if __name__ == "__main__":
    main()

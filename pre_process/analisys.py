import polars as pl
import matplotlib.pyplot as plt

class DataAnalisys:
    def __init__(self):
        """
        Inicializa a instância de DataAnalysis.

        Atributos:
            data (pl.DataFrame | None): DataFrame carregado para análise.
                                        Inicialmente None até que um dado seja carregado.
        """
        self.data = None

    def load_csv_to_polars(self, csv_path: str):
        """
        Carrega um arquivo CSV em um DataFrame Polars.

        Args:
            csv_path (str): Caminho absoluto ou relativo para o arquivo CSV.

        Returns:
            pl.DataFrame: DataFrame com os dados do arquivo CSV.
        """
        self.data = pl.read_csv(csv_path)
        return self.data
    
    def load_data_from_polars(self, data: pl.DataFrame):
        """
        Carrega um DataFrame Polars já existente para análise.

        Args:
            data (pl.DataFrame): DataFrame Polars a ser utilizado nas análises.

        Returns:
            pl.DataFrame: O mesmo DataFrame recebido, agora armazenado em self.data.
        """
        self.data = data
        return self.data
    
    def filter_numerical_columns_list(self) -> list[str]:
        """
        Retorna a lista de colunas com tipos numéricos do DataFrame carregado.

        Considera os tipos: Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64,
        Float32 e Float64.

        Returns:
            list[str]: Nomes das colunas de tipo numérico.
        """
        numeric_types = (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.Float32, pl.Float64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
        return [
            col for col in self.data.columns
            if isinstance(self.data[col].dtype, numeric_types)
        ]

    def filter_categorical_columns_list(self) -> list[str]:
        """
        Retorna a lista de colunas com tipo textual (Utf8) do DataFrame carregado.

        Returns:
            list[str]: Nomes das colunas de tipo categórico (string).
        """
        return [
            col for col in self.data.columns
            if self.data[col].dtype == pl.Utf8
        ]

    def categorical_cardinality_analyze(self, categorical_columns: list = None):
        """
        Analisa e plota a cardinalidade das colunas categóricas.

        Exibe um gráfico de barras com o número de valores únicos por coluna e
        imprime no console os valores únicos de cada coluna analisada.

        Args:
            categorical_columns (list | None): Lista de colunas categóricas a analisar.
                Se None, todas as colunas do tipo Utf8 são consideradas.

        Returns:
            None
        """
        data = self.data.select(
            pl.col(categorical_columns) if categorical_columns else pl.col(pl.Utf8)
        )
        cardinality = {col: data[col].n_unique() for col in data.columns}
        plt.bar(cardinality.keys(), cardinality.values())
        plt.xlabel("Categorical Columns")
        plt.ylabel("Cardinality")
        plt.title("Cardinality of Each Column")
        plt.show()

        for col, count in cardinality.items():
            print(f"Column: {col}, Cardinality: {count}")
            print(f"Unique values in {col}: {data[col].unique().to_list()}\n")
    
    def plot_categorical_distribution(self, columns: list[str]):
        """
        Plota a distribuição de frequência de cada coluna categórica fornecida.

        Para cada coluna, exibe um gráfico de barras com os valores ordenados
        pela frequência absoluta em ordem decrescente.

        Args:
            columns (list[str]): Lista de nomes de colunas categóricas a plotar.

        Returns:
            None
        """
        for column in columns:
            if column not in self.data.columns:
                print(f"Column '{column}' not found in the data.")
                return
            
            distribution = self.data[column].value_counts().sort("count", descending=True)
            plt.bar(distribution[column], distribution["count"])
            plt.xlabel(column)
            plt.ylabel("Count")
            plt.title(f"Distribution of {column}")
            plt.xticks(rotation=45)
            plt.show()

    def plot_numerical_distribution(self, columns: list[str]):
        """
        Plota a distribuição de frequência de cada coluna numérica fornecida.

        Para cada coluna, exibe um histograma com 30 bins para visualização
        da distribuição dos valores.

        Args:
            columns (list[str]): Lista de nomes de colunas numéricas a plotar.

        Returns:
            None
        """
        for column in columns:
            if column not in self.data.columns:
                print(f"Column '{column}' not found in the data.")
                return
            
            plt.hist(self.data[column].to_numpy(), bins=30, edgecolor='k')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {column}")
            plt.show()

    def aggregate_proportional_distribution(
        self,
        categorical_columns: list[str],
        numerical_columns: list[str],
        num_bins: int = 10
    ) -> dict[str, pl.DataFrame]:
        """
        Agrega a quantidade de instâncias por cardinalidade em termos proporcionais.

        Para colunas categóricas, calcula a proporção de cada valor único.
        Para colunas numéricas, divide os valores em bins e calcula a proporção
        de cada intervalo.

        Args:
            categorical_columns: Lista de colunas categóricas.
            numerical_columns: Lista de colunas numéricas.
            num_bins: Número de bins para colunas numéricas (padrão: 10).

        Returns:
            Dicionário {coluna: DataFrame com colunas 'valor'/'intervalo' e 'proporcao'}.
        """
        total = self.data.shape[0]
        result: dict[str, pl.DataFrame] = {}

        for col in categorical_columns:
            counts = (
                self.data[col]
                .value_counts()
                .rename({"count": "instancias"})
                .with_columns((pl.col("instancias") / total).alias("proporcao"))
                .sort("proporcao", descending=True)
            )
            result[col] = counts

        for col in numerical_columns:
            series = self.data[col].drop_nulls().to_numpy()
            import numpy as np
            counts, edges = np.histogram(series, bins=num_bins)
            labels = [f"({edges[i]:.0f}, {edges[i+1]:.0f})" for i in range(len(edges) - 1)]
            result[col] = pl.DataFrame({
                "intervalo": labels,
                "instancias": counts.tolist(),
                "proporcao": (counts / total).tolist()
            }).sort("proporcao", descending=True)

        return result

    def plot_correlation_matrix(self, numerical_columns: list[str]):
        """
        Plota a matriz de correlação das colunas numéricas fornecidas.

        Utiliza o coeficiente de Pearson via numpy e exibe o resultado como
        um heatmap com escala de cores 'coolwarm' (de -1 a 1).

        Args:
            numerical_columns (list[str]): Lista de nomes de colunas numéricas.
                Pressupõe-se que todas as colunas sejam de tipo numérico.

        Returns:
            None
        """
        import numpy as np
        data_np = self.data.select(numerical_columns).to_numpy().astype(float).T
        correlation_matrix = np.corrcoef(data_np)
        plt.figure(figsize=(10, 8))
        plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(numerical_columns)), numerical_columns, rotation=45)
        plt.yticks(range(len(numerical_columns)), numerical_columns)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    analysis = DataAnalisys()
    data = analysis.load_csv_to_polars(r"use_case\files\final_dataframe.csv")
    categorical_columns = analysis.filter_categorical_columns_list()
    numerical_columns = analysis.filter_numerical_columns_list()
    analysis.categorical_cardinality_analyze(categorical_columns)
    analysis.plot_categorical_distribution(categorical_columns)
    analysis.plot_numerical_distribution(numerical_columns)
    analysis.plot_correlation_matrix(numerical_columns)
    proportional_distribution = analysis.aggregate_proportional_distribution(categorical_columns, numerical_columns)
    for col, df in proportional_distribution.items():
        print(f"Proportional distribution for column '{col}':")
        print(df)
        print("\n")


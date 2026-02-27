import os
from dotenv import load_dotenv
from google.cloud import bigquery
import polars as pl


class SyntaxMySQL:
    """
    Classe para executar consultas MySQL via BigQuery EXTERNAL_QUERY (CloudSQL Federation).
    Utiliza o SDK do Google Cloud já configurado na máquina local.
    """
    
    def __init__(self, project_id: str = None, connection_region: str = None, connection_name: str = None):
        """
        Inicializa conexão MySQL via BigQuery Federation.
        
        Args:
            project_id (str): ID do projeto BigQuery. Se None, usa variável BIGQUERY_PROJECT_ID ou padrão.
            connection_region (str): Região da conexão. Se None, usa variável BQ_CONNECTION_REGION ou padrão.
            connection_name (str): Nome da conexão. Se None, usa variável BQ_CONNECTION_NAME ou padrão.
        """
        # Carrega variáveis de ambiente do arquivo .env
        load_dotenv()
        
        # Configurações BigQuery
        self.project_id = project_id or os.getenv('BIGQUERY_PROJECT_ID', 'prod-default-1')
        self.connection_region = connection_region or os.getenv('BQ_CONNECTION_REGION', 'us-central1')
        self.connection_name = connection_name or os.getenv('BQ_CONNECTION_NAME', 'Celsius-Prod')
        
        # ID completo da conexão
        self.connection_id = f"{self.project_id}.{self.connection_region}.{self.connection_name}"
        
        # Cliente BigQuery
        self.client = bigquery.Client(project=self.project_id)
        
        print(f"✓ SyntaxMySQL inicializado")
        print(f"  Project: {self.project_id}")
        print(f"  Connection: {self.connection_id}")
    
    def execute_query(self, query_mysql: str, verbose: bool = True) -> pl.DataFrame:
        """
        Executa uma query MySQL via BigQuery EXTERNAL_QUERY e retorna Polars DataFrame.
        
        Args:
            query_mysql (str): Query MySQL a ser executada
            verbose (bool): Se True, exibe mensagens de progresso
        
        Returns:
            pl.DataFrame: Resultado da query como Polars DataFrame
        """
        # Query BigQuery com EXTERNAL_QUERY para federated MySQL
        query_bigquery = f"""
        SELECT * FROM EXTERNAL_QUERY(
            '{self.connection_id}',
            '''{query_mysql}'''
        );
        """
        
        try:
            if verbose:
                print(f"\n⏳ Executando federated query MySQL via BigQuery...")
            
            # Executa a query
            query_job = self.client.query(query_bigquery)
            
            # Converte para Polars DataFrame via Apache Arrow
            arrow_table = query_job.to_arrow()
            df = pl.from_arrow(arrow_table)
            
            if verbose:
                print(f"✓ Query executada com sucesso: {len(df)} registros retornados")
            
            return df
        
        except Exception as e:
            print(f"✗ Erro BigQuery: {e}")
            raise
    
    @staticmethod
    def get_environment_monitoring_data() -> str:
        """
        Retorna query SQL para obter dados de monitoramento de ambientes com DUT3.
        
        Returns:
            str: Query MySQL para dados de ambientes, máquinas, assets e DUT3
        """
        return """
        SELECT
            e.ID                AS ENVIRONMENT_ID,
            e.ENVIRONMENT_NAME,
            m.NAME              AS MACHINE_NAME,
            m.TYPE              AS MACHINE_TYPE,
            a.NAME              AS ASSET_NAME,
            ahh.H_DESC          AS HEALTH_DESCRIPTION,
            d.DEVICE_CODE       AS DUT3_DEVICE_CODE,
            rt.USEPERIOD        AS ROOM_TYPE_USEPERIOD
        FROM ENVIRONMENTS e
        LEFT JOIN DUTS_MONITORING dm        ON dm.ENVIRONMENT_ID = e.ID
        LEFT JOIN DUTS_REFERENCE dr         ON dr.DUT_MONITORING_ID = dm.ID
        LEFT JOIN MACHINES m                ON m.ID = dr.MACHINE_ID
        LEFT JOIN EVAPORATORS ev            ON ev.MACHINE_ID = m.ID
        LEFT JOIN ASSETS a                  ON a.ID = ev.ASSET_ID
        LEFT JOIN ASSETS_HEALTH ah          ON ah.ASSET_ID = a.ID
        LEFT JOIN ASSETS_HEALTH_HIST ahh    ON ahh.ID = ah.HEALTH_HIST_ID
        LEFT JOIN DUTS_DEVICES dd           ON dd.ID = dm.DUT_DEVICE_ID
        LEFT JOIN DEVICES d                 ON d.ID = dd.DEVICE_ID
                                            AND d.DEVICE_CODE LIKE 'DUT3%'
        LEFT JOIN ENVIRONMENTS_ROOM_TYPES ert ON ert.ENVIRONMENT_ID = e.ID
        LEFT JOIN ROOMTYPES rt              ON rt.RTYPE_ID = ert.RTYPE_ID
        WHERE e.ID IS NOT NULL
        """


def consultar_mysql_via_bigquery():
    """
    Função helper para compatibilidade com código legado.
    Executa consulta de monitoramento de ambientes.
    """
    mysql_client = SyntaxMySQL()
    query = SyntaxMySQL.get_environment_monitoring_data()
    return mysql_client.execute_query(query)


def main():
    """Função principal de teste"""
    print("=" * 80)
    print("TESTE DE CONSULTA MYSQL (via BigQuery Federated Query)")
    print("=" * 80)
    
    try:
        # Inicializa cliente MySQL
        mysql_client = SyntaxMySQL()
        
        # Obtém query de exemplo
        query = SyntaxMySQL.get_environment_monitoring_data()
        
        # Executa consulta
        df_result = mysql_client.execute_query(query)
        
        if df_result is not None and len(df_result) > 0:
            print("\n" + "=" * 80)
            print("RESULTADOS (primeiras 10 linhas)")
            print("=" * 80)
            print(df_result.head(10))
            
            print("\n" + "=" * 80)
            print("ESTATÍSTICAS")
            print("=" * 80)
            print(f"Total de registros: {len(df_result)}")
            print(f"Colunas: {', '.join(df_result.columns)}")
            print(f"\nTipos de dados:")
            print(df_result.dtypes)
        else:
            print("\n⚠️  Nenhum resultado retornado")
    
    except Exception as e:
        print(f"\n✗ Erro na execução: {e}")


if __name__ == "__main__":
    main()

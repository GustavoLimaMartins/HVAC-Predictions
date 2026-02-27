from google.cloud import bigquery
import db_setup.postgre_sql as postgre_sql
import polars as pl

class DatabaseConnectionClient:
    def __init__(self, db_type: int):
        """
        Inicializa o cliente de conexão com o banco de dados, podendo ser Google BigQuery ou PostgreeSQL.

        Args:
            db_type (int): Tipo do banco de dados. 1 para Google BigQuery, 2 para PostgreeSQL.
        """
        if db_type == 1:
            self.client = bigquery.Client()
        elif db_type == 2:
            self.client = postgre_sql.Client()
    
    def data_convert_to_polars(self, query: str) -> pl.DataFrame:
        """
        Retorna um DataFrame do Polars a partir de uma query SQL executada no GoogleBigQuery ou PostgreeSQL.
        
        Args:
            query (str): Query SQL a ser executada no GoogleBigQuery ou PostgreeSQL (SyntaxBigQuery/SyntaxPostgreeSQL).
        
        Returns:
            pl.DataFrame: DataFrame com os dados convertidos para Polars
        """
        query_job = self.client.query(query)
        # 1. Faz o download direto em formato colunar (Arrow), muito mais rápido na rede
        arrow_table = query_job.to_arrow() 
        # 2. Converte para Polars instantaneamente (frequentemente "Zero-Copy", sem duplicar memória)
        df_polars = pl.from_arrow(arrow_table) 
        
        return df_polars

if __name__ == "__main__":
    from google_big_query import SyntaxBigQuery as bq_query
    device_version = 'DAC40324'
    date_init = '2025-06-09'
    date_final = '2025-06-09'

    bq_client = DatabaseConnectionClient(db_type=1)
    QUERY_BQ = bq_query.get_consumption_by_hour(device_version, date_init, date_final)
    df_bq = bq_client.data_convert_to_polars(QUERY_BQ)
    print(df_bq)

    from postgre_sql import SyntaxPostgreeSQL as pg_query
    pg_client = DatabaseConnectionClient(db_type=2)
    QUERY_PG = pg_query.get_devices_by_units(units=[302, 895, 1322])
    df_pg = pg_client.data_convert_to_polars(QUERY_PG)
    print(df_pg)

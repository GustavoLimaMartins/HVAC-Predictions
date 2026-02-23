from google.cloud import bigquery
from consumption_query import get_consumption_query
import polars as pl

class BigQueryClient:
    def __init__(self):
        self.client = bigquery.Client()
    
    def fetch_consumptions_to_polars(self, dev_version, date_init, date_final):
        QUERY = get_consumption_query(dev_version, date_init, date_final)
        query_job = self.client.query(QUERY)
        # 1. Faz o download direto em formato colunar (Arrow), muito mais rápido na rede
        arrow_table = query_job.to_arrow() 
        # 2. Converte para Polars instantaneamente (frequentemente "Zero-Copy", sem duplicar memória)
        df_polars = pl.from_arrow(arrow_table) 
        
        return df_polars

def main():
    device_version = 'DAC40324'
    date_init = '2025-06-09'
    date_final = '2025-06-09'
    
    bq_client = BigQueryClient()
    df = bq_client.fetch_consumptions_to_polars(device_version, date_init, date_final)
    print(df)

if __name__ == "__main__":
    main()

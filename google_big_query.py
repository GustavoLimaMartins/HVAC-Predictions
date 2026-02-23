from google.cloud import bigquery
from consumption_query import get_consumption_query
import polars as pl

class BigQueryClient:
    def __init__(self):
        self.client = bigquery.Client()
    
    def fetch_devices_consumptions_by_version_and_period(self, dev_version, date_init, date_final):
        QUERY = (get_consumption_query(dev_version, date_init, date_final))
        query_job = self.client.query(QUERY)
        return query_job.result()

    def create_polars_dataframe(self, instances):
        data = []
        for row in instances:
            data.append({
                'device_id': row.device_id,
                'data': row.data,
                'hora': row.hora,
                'consumo_kwh': row.consumo_kwh
            })
        return pl.DataFrame(data)

def main():
    # Configurações para a consulta
    device_version = 'DAC40324'
    date_init = '2025-06-09'
    date_final = '2025-06-09'
    # Criar cliente BigQuery e buscar os dados
    bq_client = BigQueryClient()
    instances = bq_client.fetch_devices_consumptions_by_version_and_period(device_version, date_init, date_final)
    # Criar DataFrame Polars a partir dos resultados
    df = bq_client.create_polars_dataframe(instances)
    print(df)

if __name__ == "__main__":
    main()

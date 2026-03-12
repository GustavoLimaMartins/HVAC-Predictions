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
        Retorna query SQL para obter dados de dispositivos com localização e tipo de máquina.
        
        Returns:
            str: Query MySQL para device_code, machine_type, latitude e longitude
        """
        return """
        -- Query para associar device_code com machine_type e localização
        -- Database: dashprod
        -- Utilizando CTEs para melhor organização e legibilidade
        WITH dut_associations AS (
            -- Associação de DUTs com dados de máquina e localização
            SELECT DISTINCT
                d.DEVICE_CODE,
                m.TYPE AS machine_type,
                u.LAT AS latitude,
                u.LON AS longitude
            FROM 
                DEVICES d
                INNER JOIN DUTS_DEVICES dut_dev ON d.ID = dut_dev.DEVICE_ID
                INNER JOIN DUTS_AUTOMATION dut_auto ON dut_dev.ID = dut_auto.DUT_DEVICE_ID
                INNER JOIN MACHINES m ON dut_auto.MACHINE_ID = m.ID
                LEFT JOIN CLUNITS u ON m.UNIT_ID = u.UNIT_ID
            WHERE u.LAT IS NOT NULL AND u.LON IS NOT NULL
        ),
        dac_condenser_associations AS (
            -- Associação de DACs (condensers) com dados de máquina e localização
            SELECT DISTINCT
                d.DEVICE_CODE,
                m.TYPE AS machine_type,
                u.LAT AS latitude,
                u.LON AS longitude
            FROM 
                DEVICES d
                INNER JOIN DACS_DEVICES dac_dev ON d.ID = dac_dev.DEVICE_ID
                INNER JOIN DACS_CONDENSERS dac_cond ON dac_dev.ID = dac_cond.DAC_DEVICE_ID
                INNER JOIN CONDENSERS c ON dac_cond.CONDENSER_ID = c.ID
                INNER JOIN MACHINES m ON c.MACHINE_ID = m.ID
                LEFT JOIN CLUNITS u ON m.UNIT_ID = u.UNIT_ID
            WHERE u.LAT IS NOT NULL AND u.LON IS NOT NULL
        )
        -- Combinação de todas as associações, removendo duplicatas
        SELECT DISTINCT
            DEVICE_CODE,
            machine_type,
            latitude,
            longitude
        FROM dut_associations
        UNION
        SELECT DISTINCT
            DEVICE_CODE,
            machine_type,
            latitude,
            longitude
        FROM dac_condenser_associations
        ORDER BY DEVICE_CODE;
        """
    
    @staticmethod
    def get_unique_cities() -> str:
        """
        Retorna query SQL para obter cidades únicas.
        
        Returns:
            str: Query MySQL para obter cidades únicas
        """
        return """
        SELECT DISTINCT u.CITY_ID as cidade
        FROM 
            CLUNITS u
        WHERE u.CITY_ID IS NOT NULL
        ORDER BY 
            u.CITY_ID ASC;
        """

    @staticmethod
    def get_devices_by_units_by_unit_names(unit_names: list[str]) -> str:
        """
        Retorna query SQL para obter dispositivos (DUT/DAC) associados a unidades pelo nome.

        Cobre todos os caminhos de associação possíveis:
            DUT: via automação de máquina | via monitoramento de ambiente
            DAC: via condensadora | via evaporadora | via trocador de calor | via automação de máquina

        Args:
            unit_names (list[str]): Lista de nomes de unidades (UNIT_NAME) para filtrar.

        Returns:
            str: Query MySQL retornando pares únicos (DEVICE_CODE, UNIT_ID).
        """
        return f"""
        -- Query geral para associar device_code de DUT e DAC com unit_id
        -- Cobre todos os caminhos de associação possíveis:
        --   DUT : via automação de máquina  |  via monitoramento de ambiente
        --   DAC : via condensadora  |  via evaporadora  |  via trocador de calor  |  via automação de máquina
        -- Database: dashprod

        WITH dut_via_automation AS (
            -- DUTs vinculados a unidades através da automação de uma máquina HVAC
            SELECT
                'DUT'        AS device_type,
                'automation' AS association_path,
                d.DEVICE_CODE,
                m.UNIT_ID
            FROM DEVICES d
            INNER JOIN DUTS_DEVICES     dut_dev  ON d.ID            = dut_dev.DEVICE_ID
            INNER JOIN DUTS_AUTOMATION  dut_auto ON dut_dev.ID      = dut_auto.DUT_DEVICE_ID
            INNER JOIN MACHINES         m        ON dut_auto.MACHINE_ID = m.ID
            WHERE m.UNIT_ID IS NOT NULL
        ),
        dut_via_monitoring AS (
            -- DUTs vinculados a unidades através do monitoramento de um ambiente (ENVIRONMENTS)
            SELECT
                'DUT'        AS device_type,
                'monitoring' AS association_path,
                d.DEVICE_CODE,
                e.UNIT_ID
            FROM DEVICES d
            INNER JOIN DUTS_DEVICES    dut_dev ON d.ID              = dut_dev.DEVICE_ID
            INNER JOIN DUTS_MONITORING dut_mon ON dut_dev.ID        = dut_mon.DUT_DEVICE_ID
            INNER JOIN ENVIRONMENTS    e       ON dut_mon.ENVIRONMENT_ID = e.ID
            WHERE e.UNIT_ID IS NOT NULL
        ),
        dac_via_condenser AS (
            -- DACs vinculados a unidades através de uma condensadora
            SELECT
                'DAC'        AS device_type,
                'condenser'  AS association_path,
                d.DEVICE_CODE,
                m.UNIT_ID
            FROM DEVICES d
            INNER JOIN DACS_DEVICES    dac_dev  ON d.ID               = dac_dev.DEVICE_ID
            INNER JOIN DACS_CONDENSERS dac_cond ON dac_dev.ID         = dac_cond.DAC_DEVICE_ID
            INNER JOIN CONDENSERS      c        ON dac_cond.CONDENSER_ID = c.ID
            INNER JOIN MACHINES        m        ON c.MACHINE_ID       = m.ID
            WHERE m.UNIT_ID IS NOT NULL
        ),
        dac_via_evaporator AS (
            -- DACs vinculados a unidades através de uma evaporadora
            SELECT
                'DAC'        AS device_type,
                'evaporator' AS association_path,
                d.DEVICE_CODE,
                m.UNIT_ID
            FROM DEVICES d
            INNER JOIN DACS_DEVICES     dac_dev  ON d.ID                  = dac_dev.DEVICE_ID
            INNER JOIN DACS_EVAPORATORS dac_evap ON dac_dev.ID            = dac_evap.DAC_DEVICE_ID
            INNER JOIN EVAPORATORS      evap     ON dac_evap.EVAPORATOR_ID = evap.ID
            INNER JOIN MACHINES         m        ON evap.MACHINE_ID       = m.ID
            WHERE m.UNIT_ID IS NOT NULL
        ),
        dac_via_automation AS (
            -- DACs vinculados a unidades através da automação direta de uma máquina HVAC
            SELECT
                'DAC'        AS device_type,
                'automation' AS association_path,
                d.DEVICE_CODE,
                m.UNIT_ID
            FROM DEVICES d
            INNER JOIN DACS_DEVICES     dac_dev  ON d.ID                = dac_dev.DEVICE_ID
            INNER JOIN DACS_AUTOMATIONS dac_auto ON dac_dev.ID          = dac_auto.DAC_DEVICE_ID
            INNER JOIN MACHINES         m        ON dac_auto.MACHINE_ID = m.ID
            WHERE m.UNIT_ID IS NOT NULL
        ),
        dac_via_heat_exchanger AS (
            -- DACs vinculados a unidades através de um trocador de calor (asset heat exchanger)
            SELECT
                'DAC'            AS device_type,
                'heat_exchanger' AS association_path,
                d.DEVICE_CODE,
                m.UNIT_ID
            FROM DEVICES d
            INNER JOIN DACS_DEVICES                dac_dev ON d.ID                          = dac_dev.DEVICE_ID
            INNER JOIN DACS_ASSET_HEAT_EXCHANGERS  dac_ahe ON dac_dev.ID                   = dac_ahe.DAC_DEVICE_ID
            INNER JOIN ASSET_HEAT_EXCHANGERS       ahe     ON dac_ahe.ASSET_HEAT_EXCHANGER_ID = ahe.ID
            INNER JOIN MACHINES                    m       ON ahe.MACHINE_ID                = m.ID
            WHERE m.UNIT_ID IS NOT NULL
        ),
        all_associations AS (
            SELECT * FROM dut_via_automation
            UNION ALL
            SELECT * FROM dut_via_monitoring
            UNION ALL
            SELECT * FROM dac_via_condenser
            UNION ALL
            SELECT * FROM dac_via_evaporator
            UNION ALL
            SELECT * FROM dac_via_automation
            UNION ALL
            SELECT * FROM dac_via_heat_exchanger
        )
        -- Resultado final: par único (device_code, unit_id, unit_name) por tipo de dispositivo
        SELECT DISTINCT
            a.DEVICE_CODE,
            a.UNIT_ID,
            u.UNIT_NAME
        FROM all_associations a
        LEFT JOIN CLUNITS u ON a.UNIT_ID = u.UNIT_ID
        WHERE u.UNIT_NAME IN ({','.join([f"'{name}'" for name in unit_names])})
        ORDER BY
            a.DEVICE_CODE,
            a.UNIT_ID;
    """


def query_mysql_by_bigquery():
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
    print("CONSULTA MYSQL (via BigQuery Federated Query)")
    print("=" * 80)
    
    try:
        # Inicializa cliente MySQL
        mysql_client = SyntaxMySQL()
        
        # Obtém query de exemplo
        query = SyntaxMySQL.get_unique_cities()
        
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

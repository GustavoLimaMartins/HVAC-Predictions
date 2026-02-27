class SyntaxBigQuery:
    """
    Classe com métodos estáticos para gerar queries SQL para Google BigQuery.
    """
    def __init__(self):
        pass
    
    @staticmethod
    def test_table_exists(device_version: str) -> str:
        """
        Retorna query para testar se a tabela existe e quantos registros tem.
        
        Args:
            device_version (str): Versão do dispositivo/tabela. Ex: 'DAC12345'
        
        Returns:
            str: Dados de total de registros, data mínima e data máxima para a tabela especificada.
        """
        return f'''
        SELECT 
            COUNT(*) as total_rows,
            MIN(day) as min_date,
            MAX(day) as max_date
        FROM `prod-default-1.cache_dev_gen.{device_version}`
        LIMIT 1
        '''
    
    @staticmethod
    def get_consumption_by_hour(device_version: str, date_init: str, date_final: str) -> str:
        """
        Retorna query SQL para consumo por hora de dispositivos na tabela cache_dev_gen.
        
        Args:
            device_version (str): Versão do dispositivo/tabela. Ex: 'DAC12345'
            date_init (str): Data inicial no formato 'YYYY-MM-DD'. Ex: '2026-01-01'
            date_final (str): Data final no formato 'YYYY-MM-DD'. Ex: '2026-01-31'
        
        Returns:
            str: Dados de dispositivo, data, hora e consumo para dispositivos da versão especificada entre as datas fornecidas, filtrando apenas consumos maiores que zero.
        """

        return f'''
        WITH base AS (
          SELECT
            dev_id,
            day,
            JSON_VALUE(charts_detailed, '$.Curr') AS payload
          FROM `prod-default-1.cache_dev_gen.{device_version}`
          WHERE day BETWEEN "{date_init}" AND "{date_final}"
            AND STARTS_WITH(dev_id, 'DAC')  -- Filtra apenas dispositivos DAC
        ),

        exploded AS (
          -- explode mantendo a ordem (offset)
          SELECT
            dev_id,
            day,
            TRIM(part) AS part,
            offset AS idx
          FROM base,
          UNNEST(SPLIT(payload, ',')) AS part WITH OFFSET AS offset
          WHERE payload IS NOT NULL
            AND TRIM(part) <> ''
            AND NOT STARTS_WITH(TRIM(part), '*')  -- ignora entradas que começam com "*"
        ),

        parsed AS (
          -- separar corrente e tempo (tempo = 1 se ausente)
          SELECT
            dev_id,
            day,
            idx,
            SAFE_CAST(SPLIT(part, '*')[SAFE_OFFSET(0)] AS FLOAT64) AS corrente,
            COALESCE(SAFE_CAST(SPLIT(part, '*')[SAFE_OFFSET(1)] AS FLOAT64), 1) AS tempo_seg
          FROM exploded
          WHERE SAFE_CAST(SPLIT(part, '*')[SAFE_OFFSET(0)] AS FLOAT64) IS NOT NULL
            AND COALESCE(SAFE_CAST(SPLIT(part, '*')[SAFE_OFFSET(1)] AS FLOAT64), 1) > 0
        ),

        cumul AS (
          -- calcular o segundo acumulado fim (cumulative_end) para cada medição
          SELECT
            *,
            SUM(tempo_seg) OVER (PARTITION BY dev_id, day ORDER BY idx
                                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_end
          FROM parsed
        ),

        spans AS (
          -- cada medição vira um intervalo [start_sec, end_sec)
          SELECT
            dev_id,
            day,
            idx,
            corrente,
            tempo_seg,
            cumulative_end - tempo_seg AS start_sec,
            cumulative_end AS end_sec
          FROM cumul
        ),

        per_hour AS (
          -- para cada medição, gerar as horas que ela toca e calcular segundos de overlap por hora
          SELECT
            s.dev_id,
            s.day,
            hour AS hour_index,
            GREATEST(LEAST(s.end_sec, (hour + 1) * 3600) - GREATEST(s.start_sec, hour * 3600), 0) AS overlap_seconds,
            s.corrente
          FROM spans s,
          UNNEST(GENERATE_ARRAY(
            SAFE_CAST(FLOOR(s.start_sec / 3600) AS INT64),
            SAFE_CAST(FLOOR((s.end_sec - 1) / 3600) AS INT64)
          )) AS hour
          WHERE s.tempo_seg > 0
        )

        SELECT
          dev_id as device_id,
          day as data,
          hour_index AS hora,
          ROUND(SUM((310.94 * corrente * (overlap_seconds / 3600)) / 1000), 6) AS consumo_kwh
          --ROUND(SUM(corrente * (overlap_seconds / 3600)), 6) AS consumo_ah
        FROM per_hour
        WHERE hour_index BETWEEN 0 AND 23
        GROUP BY dev_id, day, hour_index
        ORDER BY dev_id, day, hour_index;
      '''
    

def get_consumption_query(device_version, date_init, date_final):
    return f'''
    WITH base AS (
      SELECT
        dev_id,
        day,
        JSON_VALUE(charts_detailed, '$.Curr') AS payload
      FROM `prod-default-1.cache_dev_gen.{device_version}`
      WHERE day BETWEEN "{date_init}" AND "{date_final}"
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
      ROUND(SUM((310.86 * corrente * overlap_seconds / 3600) / 1000), 6) AS consumo_kwh
    FROM per_hour
    WHERE hour_index BETWEEN 0 AND 23
    GROUP BY dev_id, day, hour_index
    ORDER BY dev_id, day, hour_index;
  '''
from use_case import data_modeling
from db_setup import client
from db_setup.postgre_sql import SyntaxPostgreeSQL as pg_query
from db_setup.google_big_query import SyntaxBigQuery as bq_query
import polars as pl
from tqdm import tqdm


def save_consolidated_consumption(df_direct: pl.DataFrame, df_indirect: pl.DataFrame = None) -> None:
    """Consolida e salva dados de consumo direto e indireto em CSV.
    
    Args:
        df_direct: DataFrame com dados de consumo direto
        df_indirect: DataFrame opcional com dados de consumo indireto padronizado
    """
    # Consolida com consumo indireto se disponível
    if df_indirect is not None and df_indirect.shape[0] > 0:
        # Valida esquema antes de concatenar
        print("\n=== VALIDAÇÃO DE ESQUEMA ===")
        print(f"Schema direto: {df_direct.schema}")
        print(f"Schema indireto: {df_indirect.schema}")
        
        df_consolidated = pl.concat([df_direct, df_indirect], how='vertical_relaxed')
        print(f"\n=== CONSOLIDAÇÃO FINAL ===")
        print(f"Total de registros (direto + indireto): {df_consolidated.shape[0]}")
        print(f"Registros direto: {df_direct.shape[0]}")
        print(f"Registros indireto: {df_indirect.shape[0]}")
    else:
        df_consolidated = df_direct
        print(f"\n=== CONSOLIDAÇÃO FINAL ===")
        print(f"Total de registros (apenas direto): {df_consolidated.shape[0]}")
    
    print(f"Colunas: {df_consolidated.columns}")
    
    # Salva DataFrame consolidado em CSV
    output_path = r'use_case\output_files\consumption_consolidated.csv'
    df_consolidated.write_csv(output_path)
    print(f"Arquivo salvo: {output_path}")


if __name__ == "__main__":
    print("=" * 80)
    print("INICIANDO PROCESSAMENTO DE CONSUMO HVAC")
    print("=" * 80)
    
    # Processa o arquivo CSV de unidades Bradesco
    print("\n[1/7] Carregando dados das unidades...")
    csv_path = r'use_case\BradescoUnidadesSemAuto.csv'
    df_units = data_modeling.process_client_units(csv_path)
    print(f"✓ {df_units.shape[0]} unidades carregadas")

    # Conexão com PostgreSQL e execução de query
    print("\n[2/7] Consultando dispositivos no PostgreSQL...")
    pg_client = client.DatabaseConnectionClient(db_type=2)
    query_dev_by_units = pg_query.get_devices_by_units(units=df_units['id_bradesco'].to_list())
    df_devices_by_units = pg_client.data_convert_to_polars(query_dev_by_units)
    print(f"✓ {df_devices_by_units.shape[0]} dispositivos encontrados")
    
    # Consulta registros de disponibilidade para filtrar datas válidas
    print("\n[3/7] Consultando disponibilidade dos dispositivos...")
    date_min_global = df_units['data_instalacao'].min()
    date_max_global = df_units['data_inicio_automacao'].max()
    query_disponibility = pg_query.get_record_dates_above_disponibility_threshold(
        units=df_units['id_bradesco'].to_list(),
        threshold=75,
        date_init=str(date_min_global),
        date_final=str(date_max_global)
    )
    df_disponibility = pg_client.data_convert_to_polars(query_disponibility).with_columns([
        pl.col('device_code').cast(pl.Utf8),
        pl.col('record_date').cast(pl.Date)
    ]).select(['device_code', 'record_date'])  # Seleciona apenas colunas necessárias
    print(f"✓ {df_disponibility.shape[0]} registros de disponibilidade encontrados")
    
    # Extrai os 8 primeiros caracteres (versões) e cria nova coluna
    df_devices_by_units = df_devices_by_units.with_columns(
        df_devices_by_units['device_code'].str.slice(0, 8).alias('device_version')
    ).select(['unit_id', 'device_code', 'device_version'])

    df_versions_by_units = df_devices_by_units.select(['unit_id', 'device_version']).unique()
    versions = df_versions_by_units['device_version'].unique().to_list()
    
    # Consulta famílias de dispositivos com dados válidos de consumo no PostgreSQL
    print("\n[4/7] Identificando versões com dados de corrente...")
    query_valid_families = pg_query.get_devices_families_with_current_parameter()
    df_valid_families = pg_client.data_convert_to_polars(query_valid_families)
    valid_device_prefixes = df_valid_families['device_prefix'].to_list()
    
    # Filtra apenas versões que têm dados válidos de consumo
    versions_filtered = [v for v in versions if v in valid_device_prefixes]
    
    print(f"✓ {len(versions)} versões encontradas, {len(versions_filtered)} com dados válidos")
    print(f"  Versões selecionadas: {versions_filtered}")
    
    # Cria cliente BigQuery uma única vez
    print("\n[5/7] Conectando ao BigQuery...")
    bq_client = client.DatabaseConnectionClient(db_type=1)
    print("✓ Conexão estabelecida")
    
    # Lista para acumular dados de consumo de todas as versões
    all_consumption_data = []
    
    # Pré-calcula dataframe com unit_id e datas para reutilização
    units_with_dates = df_units.select(['id_bradesco', 'data_instalacao', 'data_inicio_automacao'])

    print("\n[6/7] Processando consumo DIRETO (BigQuery)...")
    for version in tqdm(versions_filtered, desc="Progresso BigQuery", unit="versão"):
        df_target_units = df_versions_by_units.filter(pl.col('device_version') == version).select('unit_id')
        df_date_units = units_with_dates.join(df_target_units, left_on='id_bradesco', right_on='unit_id', how='inner')
        date_min = df_date_units['data_instalacao'].min()
        date_max = df_date_units['data_inicio_automacao'].max()
        # Pré-computa lista de unit_ids para reutilização
        target_unit_ids = df_target_units['unit_id'].to_list()
        
        # Execução de query no BigQuery
        query_consumption = bq_query.get_consumption_by_hour(
            device_version=version, 
            date_init=str(date_min), 
            date_final=str(date_max)
        )
        
        # Filtra dispositivos da versão atual e adiciona datas em uma única operação
        df_devices_with_dates = df_devices_by_units.filter(
            pl.col('device_version') == version
        ).join(
            df_date_units,
            left_on='unit_id',
            right_on='id_bradesco',
            how='inner'
        ).select(['device_code', 'unit_id', 'data_instalacao', 'data_inicio_automacao'])
        
        df_direct_consumption = bq_client.data_convert_to_polars(query_consumption)
        
        # Verifica se há dados de consumo
        if df_direct_consumption.shape[0] == 0:
            continue

        # Pipeline otimizado: tipos, join, filtros em sequência encadeada
        df_direct_consumption = (
            df_direct_consumption
            .with_columns([
                pl.col('consumo_kwh').cast(pl.Float64),
                pl.col('hora').cast(pl.Int64),
                pl.col('data').cast(pl.Date),
                pl.col('device_id').cast(pl.Utf8)
            ])
            .join(
                df_devices_with_dates,
                left_on='device_id',
                right_on='device_code',
                how='inner'
            )
            .with_columns([
                pl.col('unit_id').cast(pl.Int64),
                pl.col('data_instalacao').cast(pl.Date),
                pl.col('data_inicio_automacao').cast(pl.Date)
            ])
            .filter(
                (pl.col('data') >= pl.col('data_instalacao')) &
                (pl.col('data') <= pl.col('data_inicio_automacao')) &
                (pl.col('consumo_kwh') > 0)
            )
            .join(
                df_disponibility,
                left_on=['device_id', 'data'],
                right_on=['device_code', 'record_date'],
                how='inner'
            )
            .with_columns([
                pl.lit(version).cast(pl.Utf8).alias('device_version'),
                pl.lit('direto').cast(pl.Utf8).alias('metodo')
            ])
            .select([
                'unit_id', 
                'device_id',
                'device_version',
                'hora',
                'data',
                'consumo_kwh',
                'data_instalacao',
                'data_inicio_automacao',
                'metodo'
            ])
        )
        
        if df_direct_consumption.shape[0] > 0:
            all_consumption_data.append(df_direct_consumption)
    
    # Consolida todos os dados em um DataFrame Polars
    if all_consumption_data:
        df_final = pl.concat(all_consumption_data, how='vertical_relaxed')
        
        # Garante integridade do schema final com tipos consistentes
        df_final = df_final.with_columns([
            pl.col('unit_id').cast(pl.Int64),
            pl.col('device_id').cast(pl.Utf8),
            pl.col('device_version').cast(pl.Utf8),
            pl.col('hora').cast(pl.Int64),
            pl.col('data').cast(pl.Date),
            pl.col('consumo_kwh').cast(pl.Float64),
            pl.col('data_instalacao').cast(pl.Date),
            pl.col('data_inicio_automacao').cast(pl.Date),
            pl.col('metodo').cast(pl.Utf8)
        ])
        
        print(f"\n✓ Consumo direto processado: {df_final.shape[0]} registros")
        print(f"Schema final direto: {df_final.schema}")
        
        # Obtém lista de dispositivos únicos com consumo já calculado
        devices_with_consumption = df_final['device_id'].unique().to_list()
        
        # Identifica dispositivos sem consumo direto (todos os dispositivos das unidades menos os que já têm)
        all_devices = df_devices_by_units['device_code'].unique().to_list()
        devices_without_consumption = [d for d in all_devices if d not in devices_with_consumption]
        
        print(f"\n[7/7] Processando consumo INDIRETO (PostgreSQL)...")
        print(f"  Dispositivos com consumo direto: {len(devices_with_consumption)}")
        print(f"  Dispositivos para método indireto: {len(devices_without_consumption)}")
        
        if devices_without_consumption:
            # Cria DataFrame com dispositivos sem consumo e suas respectivas datas por unidade
            df_devices_for_indirect = df_devices_by_units.filter(
                pl.col('device_code').is_in(devices_without_consumption)
            ).join(
                units_with_dates,
                left_on='unit_id',
                right_on='id_bradesco',
                how='inner'
            ).select(['device_code', 'unit_id', 'data_instalacao', 'data_inicio_automacao'])
            
            all_indirect_data = []
            
            for row in tqdm(df_devices_for_indirect.iter_rows(named=True), 
                          desc="Progresso PostgreSQL", 
                          unit="device",
                          total=df_devices_for_indirect.shape[0]):
                device_code = row['device_code']
                date_init = str(row['data_instalacao'])
                date_final = str(row['data_inicio_automacao'])
                
                # Consulta consumo por método indireto para este dispositivo com suas datas específicas
                query_indirect = pg_query.get_device_consumptions_by_indirect_method(
                    device_code=device_code,
                    date_init=date_init, 
                    date_final=date_final
                )
                df_device_indirect = pg_client.data_convert_to_polars(query_indirect)
                
                if df_device_indirect.shape[0] > 0:
                    all_indirect_data.append(df_device_indirect)
            
            # Consolida todos os dados indiretos
            if all_indirect_data:
                df_indirect_consumption = pl.concat(all_indirect_data).with_columns([
                    pl.col('consumption').cast(pl.Float64),
                    pl.col('device_code').cast(pl.Utf8),
                    pl.col('record_date').cast(pl.Datetime)  # Garante datetime para extração de hora
                ])
                print(f"\n✓ Dados brutos coletados: {df_indirect_consumption.shape[0]} registros")
                
                if df_indirect_consumption.shape[0] > 0:
                    
                    # Pipeline otimizado: join, filtro, conversão e padronização em sequência
                    df_indirect_standardized = (
                        df_indirect_consumption
                        .join(df_devices_for_indirect, on='device_code', how='inner')
                        .filter(pl.col('consumption') > 0)
                        .with_columns([
                            pl.col('unit_id').cast(pl.Int64),
                            pl.col('device_code').cast(pl.Utf8).alias('device_id'),
                            pl.col('device_code').str.slice(0, 8).cast(pl.Utf8).alias('device_version'),
                            pl.col('record_date').dt.hour().cast(pl.Int64).alias('hora'),
                            pl.col('record_date').cast(pl.Date).alias('data'),
                            pl.col('consumption').cast(pl.Float64).alias('consumo_kwh'),
                            pl.col('data_instalacao').cast(pl.Date),
                            pl.col('data_inicio_automacao').cast(pl.Date),
                            pl.lit('indireto').cast(pl.Utf8).alias('metodo')
                        ])
                        .join(
                            df_disponibility,
                            left_on=['device_id', 'data'],
                            right_on=['device_code', 'record_date'],
                            how='inner'
                        )
                        .select([
                            'unit_id',
                            'device_id',
                            'device_version',
                            'hora',
                            'data',
                            'consumo_kwh',
                            'data_instalacao',
                            'data_inicio_automacao',
                            'metodo'
                        ])
                    )
                    
                    if df_indirect_standardized.shape[0] > 0:
                        print(f"✓ Consumo indireto processado: {df_indirect_standardized.shape[0]} registros")
                        print(f"\n=== VALIDAÇÃO SCHEMA INDIRETO ===")
                        print(f"Schema: {df_indirect_standardized.schema}")
                        
                        save_consolidated_consumption(df_final, df_indirect_standardized)
                    else:
                        print("\n⚠ Nenhum registro indireto após filtros")
                        save_consolidated_consumption(df_final)
                else:
                    print("\n⚠ Nenhum registro indireto encontrado")
                    save_consolidated_consumption(df_final)
            else:
                print("\n⚠ Nenhum dado indireto coletado")
                save_consolidated_consumption(df_final)
        else:
            print("\n✓ Todos os dispositivos já têm consumo direto")
            save_consolidated_consumption(df_final)
    else:
        df_final = None
        print("\n✗ ERRO: Nenhum dado de consumo encontrado.")

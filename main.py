from use_case import data_modeling
from use_case.data_modeling import enrich_csv_with_reference_id, enrich_csv_with_unit_id
from db_setup import client
from db_setup.postgre_sql import SyntaxPostgreeSQL as pg_query
from db_setup.google_big_query import SyntaxBigQuery as bq_query
from db_setup.my_sql import SyntaxMySQL as mysql_query
from dataframe.schema import DataFrameFormatter
from dataframe.complementary import enrich_dataframe_with_all_features
import polars as pl
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, date


def save_consolidated_consumption(df_direct: pl.DataFrame, df_indirect: pl.DataFrame = None) -> None:
    """Consolida e salva dados de consumo direto e indireto em CSV. Dados brutos para validação e formatação final para ML.
    
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
    
    # Salva DataFrame consolidado bruto em CSV
    output_path_raw = r'use_case\files\consumption_consolidated.csv'
    df_consolidated.write_csv(output_path_raw)
    print(f"Arquivo bruto salvo: {output_path_raw}")
    
    # Enriquece com features complementares (estações, clima, grupos regionais)
    print("\n=== ENRIQUECIMENTO COM FEATURES COMPLEMENTARES ===")
    df_enriched = enrich_dataframe_with_all_features(df_consolidated)
    print(f"Registros após enriquecimento: {df_enriched.shape[0]}")
    print(f"Colunas após enriquecimento: {df_enriched.columns}")
    
    # Aplica formatação para schema final de ML
    print("\n=== FORMATAÇÃO FINAL DO SCHEMA ===")
    formatter = DataFrameFormatter()
    df_formatted = formatter.format_for_model(df_enriched)
    
    print(f"Registros após formatação: {df_formatted.shape[0]}")
    print(f"Colunas finais: {df_formatted.columns}")
    print(f"Schema final: {df_formatted.schema}")
    
    # Salva DataFrame formatado em CSV
    output_path_final = r'use_case\files\final_dataframe.csv'
    df_formatted.write_csv(output_path_final)
    print(f"Arquivo formatado salvo: {output_path_final}")


if __name__ == "__main__":
    print("=" * 80)
    print("INICIANDO PROCESSAMENTO DE CONSUMO HVAC")
    print("=" * 80)
    
    csv_path = r'use_case\files\BradescoUnidadesSemAuto.csv'
    pg_client = client.DatabaseConnectionClient(db_type=2)
    mysql_client = client.DatabaseConnectionClient(db_type=3)

    # Pré-etapa: enriquecimento do CSV ocorre sempre, independente de cache
    print("\n=== PRÉ-ETAPA: ENRIQUECIMENTO DO CSV ===")
    print("[1/10] Enriquecendo CSV com reference_id (MySQL → UNIT_ID)...")
    enrich_csv_with_reference_id(csv_path, mysql_client.client)
    print("✓ reference_id verificado/adicionado ao CSV")

    print("[2/10] Enriquecendo CSV com unit_id (PostgreSQL → id interno)...")
    enrich_csv_with_unit_id(csv_path, pg_client.client)
    print("✓ unit_id verificado/adicionado ao CSV")

    # Verifica se o arquivo consolidado já existe e foi atualizado hoje
    output_path_raw = Path(r'use_case\files\consumption_consolidated.csv')
    output_path_final = Path(r'use_case\files\final_dataframe.csv')
    
    if output_path_raw.exists():
        # Obtém a data de modificação do arquivo
        modification_time = datetime.fromtimestamp(output_path_raw.stat().st_mtime)
        modification_date = modification_time.date()
        today = date.today()
        
        print(f"\n=== VERIFICAÇÃO DE CACHE ===")
        print(f"Arquivo encontrado: {output_path_raw}")
        print(f"Data de modificação: {modification_date}")
        print(f"Data de hoje: {today}")
        
        if modification_date == today:
            print("✓ Arquivo foi atualizado hoje. Reaproveitando dados existentes...")
            
            # Carrega o arquivo consolidado existente
            df_consolidated = pl.read_csv(str(output_path_raw))
            print(f"✓ {df_consolidated.shape[0]} registros carregados do cache")
            print(f"  Colunas: {df_consolidated.columns}")
            
            # Enriquece com features complementares (estações, clima, grupos regionais)
            print("\n=== ENRIQUECIMENTO COM FEATURES COMPLEMENTARES ===")
            df_enriched = enrich_dataframe_with_all_features(df_consolidated)
            print(f"Registros após enriquecimento: {df_enriched.shape[0]}")
            print(f"Colunas após enriquecimento: {df_enriched.columns}")
            
            # Aplica formatação para schema final de ML
            print("\n=== FORMATAÇÃO FINAL DO SCHEMA ===")
            formatter = DataFrameFormatter()
            df_formatted = formatter.format_for_model(df_enriched)
            
            print(f"Registros após formatação: {df_formatted.shape[0]}")
            print(f"Colunas finais: {df_formatted.columns}")
            print(f"Schema final: {df_formatted.schema}")
            
            # Salva DataFrame formatado em CSV
            df_formatted.write_csv(str(output_path_final))
            print(f"Arquivo formatado salvo: {output_path_final}")
            
            print("\n" + "=" * 80)
            print("PROCESSAMENTO CONCLUÍDO (CACHE REUTILIZADO)")
            print("=" * 80)
            exit(0)
        else:
            print("⚠ Arquivo desatualizado. Iniciando reprocessamento completo...")
    else:
        print("\n⚠ Arquivo consolidado não encontrado. Iniciando processamento completo...")
    
    # Processa o arquivo CSV de unidades Bradesco
    print("\n[3/10] Carregando dados das unidades...")
    df_units = data_modeling.process_client_units(csv_path, date_auto_name='data_inicio')
    print(f"✓ {df_units.shape[0]} unidades carregadas")

    # Consulta dispositivos no PostgreSQL
    print("\n[4/10] Consultando dispositivos no PostgreSQL...")
    query_dev_by_units = pg_query.get_devices_by_units(units=df_units['unit_id'].drop_nulls().cast(pl.Int64).to_list())
    df_devices_by_units = pg_client.data_convert_to_polars(query_dev_by_units)
    print(f"✓ {df_devices_by_units.shape[0]} dispositivos encontrados")
    
    # Consulta dados de localização e tipo de máquina via MySQL (BigQuery Federation)
    print("\n[5/10] Consultando dados de localização e machine_type via MySQL...")
    query_location = mysql_query.get_environment_monitoring_data()
    df_location_data = mysql_client.data_convert_to_polars(query_location)
    
    # Renomeia colunas para match com o schema existente e converte tipos
    df_location_data = df_location_data.rename({
        'DEVICE_CODE': 'device_code'
    }).with_columns([
        pl.col('device_code').cast(pl.Utf8),
        pl.col('machine_type').cast(pl.Utf8),
        pl.col('latitude').cast(pl.Float64),
        pl.col('longitude').cast(pl.Float64)
    ])
    
    print(f"✓ {df_location_data.shape[0]} registros de localização obtidos")
    print(f"  Colunas: {df_location_data.columns}")
    
    # Consulta registros de disponibilidade para filtrar datas válidas
    print("\n[6/10] Consultando disponibilidade dos dispositivos...")
    date_min_global = df_units['data_instalacao'].min()
    date_max_global = df_units['data_inicio_automacao'].max()
    query_disponibility = pg_query.get_record_dates_above_disponibility_threshold(
        units=df_units['unit_id'].drop_nulls().cast(pl.Int64).to_list(),
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
    print("\n[7/10] Identificando versões com dados de corrente...")
    query_valid_families = pg_query.get_devices_families_with_current_parameter()
    df_valid_families = pg_client.data_convert_to_polars(query_valid_families)
    valid_device_prefixes = df_valid_families['device_prefix'].to_list()
    
    # Filtra apenas versões que têm dados válidos de consumo
    versions_filtered = [v for v in versions if v in valid_device_prefixes]
    
    print(f"✓ {len(versions)} versões encontradas, {len(versions_filtered)} com dados válidos")
    print(f"  Versões selecionadas: {versions_filtered}")
    
    # Cria cliente BigQuery uma única vez
    print("\n[8/10] Conectando ao BigQuery...")
    bq_client = client.DatabaseConnectionClient(db_type=1)
    print("✓ Conexão estabelecida")
    
    # Lista para acumular dados de consumo de todas as versões
    all_consumption_data = []
    
    # Pré-calcula dataframe com unit_id e datas para reutilização
    units_with_dates = df_units.select(['unit_id', 'data_instalacao', 'data_inicio_automacao'])

    print("\n[9/10] Processando consumo DIRETO (BigQuery)...")
    for version in tqdm(versions_filtered, desc="Progresso BigQuery", unit="versão"):
        df_target_units = df_versions_by_units.filter(pl.col('device_version') == version).select('unit_id')
        df_date_units = units_with_dates.join(df_target_units, on='unit_id', how='inner')
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
            on='unit_id',
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
            .join(
                df_location_data,
                left_on='device_id',
                right_on='device_code',
                how='left'
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
                'metodo',
                'machine_type',
                'latitude',
                'longitude'
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
            pl.col('metodo').cast(pl.Utf8),
            pl.col('machine_type').cast(pl.Utf8),
            pl.col('latitude').cast(pl.Float64),
            pl.col('longitude').cast(pl.Float64)
        ])
        
        print(f"\n✓ Consumo direto processado: {df_final.shape[0]} registros")
        print(f"Schema final direto: {df_final.schema}")
        
        # Obtém lista de dispositivos únicos com consumo já calculado
        devices_with_consumption = df_final['device_id'].unique().to_list()
        
        # Identifica dispositivos sem consumo direto (todos os dispositivos das unidades menos os que já têm)
        all_devices = df_devices_by_units['device_code'].unique().to_list()
        devices_without_consumption = [d for d in all_devices if d not in devices_with_consumption]
        
        print(f"\n[10/10] Processando consumo INDIRETO (PostgreSQL)...")
        print(f"  Dispositivos com consumo direto: {len(devices_with_consumption)}")
        print(f"  Dispositivos para método indireto: {len(devices_without_consumption)}")
        
        if devices_without_consumption:
            # Cria DataFrame com dispositivos sem consumo e suas respectivas datas por unidade
            df_devices_for_indirect = df_devices_by_units.filter(
                pl.col('device_code').is_in(devices_without_consumption)
            ).join(
                units_with_dates,
                on='unit_id',
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
                        .join(
                            df_location_data,
                            left_on='device_id',
                            right_on='device_code',
                            how='left'
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
                            'metodo',
                            'machine_type',
                            'latitude',
                            'longitude'
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

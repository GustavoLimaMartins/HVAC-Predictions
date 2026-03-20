"""
Engenharia de Features Temporais — Lag e Rolling Mean (Médias por Máquina)
===========================================================================

Módulo para computação de features de lag (consumo_lag_1h, consumo_lag_24h)
e rolling mean (consumo_rolling_mean_3h) baseadas em VALORES MÉDIOS por máquina,
abstraindo a granularidade de device_code.

Todos os devices de uma mesma máquina/unidade (unit_id, machine_type) 
compartilham os mesmos lags e rolling means, calculados a partir das médias 
horárias de consumo da máquina.

Features criadas
-----------------
consumo_lag_1h      : float — Consumo médio (machine_type) exatamente 1 hora atrás
                              (mesma unit_id, machine_type)

consumo_lag_24h     : float — Consumo médio (machine_type) na mesma hora do dia anterior
                              (mesma unit_id, machine_type)

consumo_rolling_mean_3h: float — Média do consumo médio nas últimas 3 horas
                                 (mesma unit_id, machine_type)

Estratégias de Performance
--------------------------
- < 1M registros: Window functions + shift nativas
- 1M-10M registros: Parallelização por machine_type (recomendado)
- > 10M registros: Streaming com chunks

Uso
---
>>> from dataframe.complementary_features.lag_features import add_lag_features
>>> import polars as pl
>>> 
>>> df = pl.read_parquet('consumption_consolidated.parquet')
>>> df_with_lags = add_lag_features(df, strategy='auto')
>>> df_with_lags.write_parquet('consumption_consolidated.parquet')
"""

from __future__ import annotations

import polars as pl
import multiprocessing
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal


def _add_lag_shift_native(df: pl.DataFrame) -> pl.DataFrame:
    """
    Estratégia 1: Window functions nativas com shift.
    
    Ideal para < 1M registros. Simples, sem joins.
    
    Agrupa por (unit_id, machine_type) e calcula médias horárias,
    abstraindo a granularidade de device_code. Todos os devices do mesmo
    tipo em uma unidade compartilham os mesmos lags e rolling mean.
    
    lag_1h: Usa shift(1) com validação de timestamp
    lag_24h: Usa join com (unit_id, machine_type, data-1, hora)
    """
    grouping_for_lags = ['unit_id', 'machine_type']
    
    # Remove colunas de lags antigas (se existirem de runs anteriores)
    lag_cols = ['consumo_lag_1h', 'consumo_lag_24h', 'consumo_rolling_mean_3h',
                'consumo_lag_1h_right', 'consumo_lag_24h_right', 'consumo_rolling_mean_3h_right',
                'consumo_medio']
    cols_to_drop = [col for col in lag_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
    
    # Passo 1: Agregação por (unit_id, machine_type, data, hora)
    df_agg = (
        df
        .lazy()
        .group_by(['unit_id', 'machine_type', 'data', 'hora'])
        .agg([
            pl.col('consumo_kwh').mean().alias('consumo_medio'),
        ])
        .sort(['unit_id', 'machine_type', 'data', 'hora'])
        .collect()
    )
    
    # Passo 2: Calcula lag_1h
    df_with_lag1h = (
        df_agg
        .with_columns([
            (pl.col('data').cast(pl.Datetime) + pl.duration(hours=pl.col('hora')))
            .alias('_ts_'),
        ])
        .with_columns([
            pl.col('_ts_').shift(1).over(grouping_for_lags).alias('_ts_prev_'),
            pl.col('consumo_medio').shift(1).over(grouping_for_lags).alias('_lag1h_raw_'),
        ])
        .with_columns([
            pl.when(
                pl.col('_ts_') - pl.col('_ts_prev_') == pl.duration(hours=1)
            )
            .then(pl.col('_lag1h_raw_'))
            .otherwise(pl.lit(None))
            .alias('consumo_lag_1h'),
        ])
        .drop(['_ts_', '_ts_prev_', '_lag1h_raw_'])
    )
    
    # Passo 3: Calcula lag_24h via join com dia anterior
    # Cria lookup table: (unit_id, machine_type, data+1, hora) → consumo_medio
    df_lookup = df_agg.select([
        'unit_id', 'machine_type',
        (pl.col('data') + pl.duration(days=1)).alias('_data_next_'),
        pl.col('hora').alias('_hora_'),
        pl.col('consumo_medio').alias('consumo_lag_24h'),
    ])
    
    df_with_lags = df_with_lag1h.join(
        df_lookup,
        left_on=['unit_id', 'machine_type', 'data', 'hora'],
        right_on=['unit_id', 'machine_type', '_data_next_', '_hora_'],
        how='left',
    )
    
    # Remove colunas de join se existirem
    cols_to_remove = [c for c in ['_data_next_', '_hora_'] if c in df_with_lags.columns]
    if cols_to_remove:
        df_with_lags = df_with_lags.drop(cols_to_remove)
    
    # Passo 4: Calcula rolling_mean
    df_with_lags = df_with_lags.with_columns([
        pl.col('consumo_medio')
        .rolling_mean(window_size=3, min_samples=1)
        .over(grouping_for_lags)
        .alias('consumo_rolling_mean_3h'),
    ]).drop(['consumo_medio'])
    
    # Passo 5: Arredonda para 4 casas decimais
    df_with_lags = df_with_lags.with_columns([
        pl.col('consumo_lag_1h').round(4),
        pl.col('consumo_lag_24h').round(4),
        pl.col('consumo_rolling_mean_3h').round(4),
    ])
    
    # Passo 6: Join de volta com dados originais
    result = df.join(
        df_with_lags,
        on=['unit_id', 'machine_type', 'data', 'hora'],
        how='left',
    )
    
    return result


def _add_lag_parallel_by_device(df: pl.DataFrame, n_workers: int = 8) -> pl.DataFrame:
    """
    Estratégia 2: Particionamento por machine_type + processamento paralelo.
    
    Ideal para 1M-50M registros. Escalável, usa múltiplos cores.
    
    lag_1h: Usa shift(1) com validação de timestamp
    lag_24h: Usa join com (unit_id, machine_type, data-1, hora)
    """
    grouping_for_partition = ['unit_id', 'machine_type']
    grouping_for_lags = ['unit_id', 'machine_type']
    
    # Remove colunas de lags antigas
    lag_cols = ['consumo_lag_1h', 'consumo_lag_24h', 'consumo_rolling_mean_3h',
                'consumo_lag_1h_right', 'consumo_lag_24h_right', 'consumo_rolling_mean_3h_right',
                'consumo_medio']
    cols_to_drop = [col for col in lag_cols if col in df.columns]
    if cols_to_drop:
        df = df.drop(cols_to_drop)
    
    # Particiona por (unit_id, machine_type)
    device_groups = df.partition_by(grouping_for_partition, as_dict=True)
    
    def process_machine(machine_df: pl.DataFrame) -> pl.DataFrame:
        """Processa uma máquina/unidade (grupo agregado em memória)."""
        # Passo 1: Agrega por (unit_id, machine_type, data, hora)
        df_agg = (
            machine_df
            .lazy()
            .group_by(['unit_id', 'machine_type', 'data', 'hora'])
            .agg([
                pl.col('consumo_kwh').mean().alias('consumo_medio'),
            ])
            .sort(['data', 'hora'])
            .collect()
        )
        
        # Passo 2: Calcula lag_1h
        df_with_lag1h = (
            df_agg
            .with_columns([
                (pl.col('data').cast(pl.Datetime) + pl.duration(hours=pl.col('hora')))
                .alias('_ts_'),
            ])
            .with_columns([
                pl.col('_ts_').shift(1).over(grouping_for_lags).alias('_ts_prev_'),
                pl.col('consumo_medio').shift(1).over(grouping_for_lags).alias('_lag1h_raw_'),
            ])
            .with_columns([
                pl.when(
                    pl.col('_ts_') - pl.col('_ts_prev_') == pl.duration(hours=1)
                )
                .then(pl.col('_lag1h_raw_'))
                .otherwise(pl.lit(None))
                .alias('consumo_lag_1h'),
            ])
            .drop(['_ts_', '_ts_prev_', '_lag1h_raw_'])
        )
        
        # Passo 3: Calcula lag_24h
        df_lookup = df_agg.select([
            'unit_id', 'machine_type',
            (pl.col('data') + pl.duration(days=1)).alias('_data_next_'),
            pl.col('hora').alias('_hora_'),
            pl.col('consumo_medio').alias('consumo_lag_24h'),
        ])
        
        df_with_lags = df_with_lag1h.join(
            df_lookup,
            left_on=['unit_id', 'machine_type', 'data', 'hora'],
            right_on=['unit_id', 'machine_type', '_data_next_', '_hora_'],
            how='left',
        )
        
        # Remove colunas de join se existirem
        cols_to_remove = [c for c in ['_data_next_', '_hora_'] if c in df_with_lags.columns]
        if cols_to_remove:
            df_with_lags = df_with_lags.drop(cols_to_remove)
        
        # Passo 4: Calcula rolling_mean
        df_with_lags = df_with_lags.with_columns([
            pl.col('consumo_medio')
            .rolling_mean(window_size=3, min_samples=1)
            .over(grouping_for_lags)
            .alias('consumo_rolling_mean_3h'),
        ]).drop(['consumo_medio'])
        
        # Passo 5: Arredonda para 4 casas decimais
        df_with_lags = df_with_lags.with_columns([
            pl.col('consumo_lag_1h').round(4),
            pl.col('consumo_lag_24h').round(4),
            pl.col('consumo_rolling_mean_3h').round(4),
        ])
        
        # Passo 6: Join de volta
        result = machine_df.join(
            df_with_lags,
            on=['unit_id', 'machine_type', 'data', 'hora'],
            how='left',
        )
        
        return result
    
    # Paraleliza
    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_machine, machine_df): key
            for key, machine_df in device_groups.items()
        }
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return pl.concat(results)


def add_lag_features(
    df: pl.DataFrame,
    strategy: Literal['auto', 'shift', 'parallel', 'streaming'] = 'auto',
    n_workers: int | None = None,
) -> pl.DataFrame:
    """
    Adiciona features de lag e rolling mean baseadas em MÉDIAS HORÁRIAS por máquina.
    
    Agrupa por (unit_id, machine_type) e calcula lags a partir de valores médios,
    abstraindo a granularidade de device_code. Todos os devices do mesmo tipo em 
    uma unidade compartilham os mesmos lags.

    Features adicionadas
    --------------------
    consumo_lag_1h      : float — Consumo médio (máquina_tipo) uma hora atrás
    consumo_lag_24h     : float — Consumo médio (máquina_tipo) 24h atrás (mesmo horário)
    consumo_rolling_mean_3h: float — Média das últimas 3h de consumo médio (máquina_tipo)

    Args:
        df: DataFrame com colunas: unit_id, machine_type, consumo_kwh, hora, data
        strategy: 'auto' (detecta pelo tamanho), 'shift' (< 1M), 
                 'parallel' (1M-50M), 'streaming' (> 50M)
        n_workers: Número de workers paralelos (default: cpu_count)

    Returns:
        DataFrame original com features temporais adicionadas

    Raises:
        ValueError: Se colunas obrigatórias estiverem ausentes
    """
    # Validação de colunas
    required_cols = {'consumo_kwh', 'hora', 'data', 'unit_id', 'machine_type'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")
    
    # Define workers
    if n_workers is None:
        n_workers = min(8, multiprocessing.cpu_count())
    
    # Garante tipos
    cast_exprs = [
        pl.col("data").cast(pl.Date),
        pl.col("hora").cast(pl.Int64),
        pl.col("consumo_kwh").cast(pl.Float64),
        pl.col("unit_id").cast(pl.Int64),
        pl.col("machine_type").cast(pl.Utf8),
    ]
    
    df = df.with_columns(cast_exprs)
    
    # Escolhe estratégia
    if strategy == 'auto':
        n_rows = len(df)
        if n_rows < 1_000_000:
            strategy = 'shift'
        elif n_rows < 50_000_000:
            strategy = 'parallel'
        else:
            strategy = 'streaming'
    
    # Executa estratégia escolhida
    if strategy == 'shift':
        return _add_lag_shift_native(df)
    elif strategy == 'parallel':
        return _add_lag_parallel_by_device(df, n_workers)
    elif strategy == 'streaming':
        # Por enquanto, usar parallel (streaming é mais complexo)
        # TODO: implementar streaming para > 50M
        return _add_lag_parallel_by_device(df, n_workers)
    else:
        raise ValueError(f"Estratégia desconhecida: {strategy}")



if __name__ == "__main__":
    """
    Execução direta: carrega consumption_consolidated.parquet, 
    adiciona features de lag e salva atualizado.
    """
    import sys
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    input_path = Path(r'use_case\files\consumption_consolidated.parquet')
    
    if not input_path.exists():
        print(f"✗ ERRO: Arquivo não encontrado: {input_path}")
        sys.exit(1)
    
    try:
        print("=" * 80)
        print("ADIÇÃO DE FEATURES TEMPORAIS — consumption_consolidated.parquet")
        print("=" * 80)
        
        print(f"\n[1/3] Carregando dados de consumo consolidado...")
        start_load = time.time()
        df = pl.read_parquet(str(input_path))
        elapsed_load = time.time() - start_load
        # Estima tamanho usando método do Polars
        df_size_mb = sum(df[col].estimated_size('mb') for col in df.columns)
        print(f"✓ {df.shape[0]:,} registros carregados em {elapsed_load:.2f}s")
        print(f"  Colunas iniciais: {df.columns}")
        print(f"  Tamanho estimado: {df_size_mb:.1f} MB")
        
        print(f"\n[2/3] Adicionando features temporais (estratégia: auto)...")
        start_lag = time.time()
        df_with_lags = add_lag_features(df, strategy='auto')
        elapsed_lag = time.time() - start_lag
        print(f"✓ Features temporais criadas em {elapsed_lag:.2f}s")
        print(f"  Velocidade: {df_with_lags.shape[0] / elapsed_lag / 1_000_000:.2f}M registros/seg")
        
        # Relatório de NaNs
        temporal_cols = ['consumo_lag_1h', 'consumo_lag_24h', 'consumo_rolling_mean_3h']
        print(f"\n  Status das features temporais:")
        for col in temporal_cols:
            if col in df_with_lags.columns:
                nan_count = df_with_lags[col].null_count()
                pct_nan = 100 * nan_count / len(df_with_lags)
                print(f"    - {col}: {nan_count:,} NaNs ({pct_nan:.1f}%)")
        
        # Salva atualizado
        print(f"\n[3/3] Salvando arquivo atualizado...")
        start_save = time.time()
        df_with_lags.write_parquet(str(input_path))
        elapsed_save = time.time() - start_save
        print(f"✓ Arquivo atualizado com sucesso em {elapsed_save:.2f}s")
        print(f"  Shape final: {df_with_lags.shape}")
        # Estima tamanho final
        df_final_size_mb = sum(df_with_lags[col].estimated_size('mb') for col in df_with_lags.columns)
        print(f"  Tamanho final: {df_final_size_mb:.1f} MB")
        
        # Resumo de performance
        total_time = elapsed_load + elapsed_lag + elapsed_save
        print(f"\n" + "=" * 80)
        print(f"PERFORMANCE")
        print(f"=" * 80)
        print(f"  Tempo total: {total_time:.2f}s")
        print(f"  - Carregamento: {elapsed_load:.2f}s ({100*elapsed_load/total_time:.0f}%)")
        print(f"  - Processamento: {elapsed_lag:.2f}s ({100*elapsed_lag/total_time:.0f}%)")
        print(f"  - Salvamento: {elapsed_save:.2f}s ({100*elapsed_save/total_time:.0f}%)")
        print(f"  Throughput: {df_with_lags.shape[0] / total_time / 1_000_000:.2f}M registros/seg")
        
        print(f"\n" + "=" * 80)
        print("PROCESSAMENTO CONCLUÍDO")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ ERRO durante processamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

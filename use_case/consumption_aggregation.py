import polars as pl
from pathlib import Path
from typing import Optional


class ConsumptionAnalyzer:
    """Classe para análise e agregação de dados de consumo consolidado.
    
    Attributes:
        df (pl.DataFrame): DataFrame com dados de consumo consolidado
        input_path (str): Caminho do arquivo CSV de entrada
    """
    
    def __init__(self, input_csv_path: str):
        """Inicializa o analisador carregando o arquivo CSV consolidado.
        
        Args:
            input_csv_path (str): Caminho para o arquivo consumption_consolidated.csv
        
        Raises:
            FileNotFoundError: Se o arquivo não existir
            ValueError: Se colunas obrigatórias estiverem ausentes
        """
        if not Path(input_csv_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_csv_path}")
        
        self.input_path = input_csv_path
        self.df = pl.read_csv(input_csv_path)
        
        print(f"✓ Arquivo carregado: {self.df.shape[0]} registros")
        print(f"  Colunas: {self.df.columns}")
        
        # Valida colunas obrigatórias
        required_columns = ['unit_id', 'device_id', 'device_version', 'data', 'hora', 'metodo', 'consumo_kwh']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Colunas ausentes no CSV: {missing_columns}")
        
        # Garante tipos corretos
        self.df = self.df.with_columns([
            pl.col('unit_id').cast(pl.Int64),
            pl.col('device_id').cast(pl.Utf8),
            pl.col('device_version').cast(pl.Utf8),
            pl.col('data').cast(pl.Date),
            pl.col('hora').cast(pl.Int64),
            pl.col('metodo').cast(pl.Utf8),
            pl.col('consumo_kwh').cast(pl.Float64)
        ])
    
    def aggregate_device_availability_by_hour(self, output_csv_path: Optional[str] = None) -> pl.DataFrame:
        """Agrega disponibilidade de dispositivos por unidade e hora.
        
        Identifica quantos dispositivos (DAC e DUT) apresentaram registros em cada hora,
        permitindo detectar períodos com dispositivos abaixo do critério de disponibilidade.
        Calcula o peso médio composto (frequência + consumo) por tipo de dispositivo.
        
        Args:
            output_csv_path (str, optional): Caminho para salvar resultado
        
        Returns:
            pl.DataFrame: Dados agregados com colunas:
                - unit_id: ID da unidade
                - data: Data do registro
                - hora: Hora do registro (0-23)
                - qtd_devices_total: Total de devices com registro
                - qtd_dac: Quantidade de dispositivos DAC
                - qtd_dut: Quantidade de dispositivos DUT
                - peso_medio_dac: Peso médio por DAC (50% frequência + 50% consumo, 0-1)
                - peso_medio_dut: Peso médio por DUT (50% frequência + 50% consumo, 0-1)
                - consumo_kwh_total: Soma do consumo
                - metodos: Métodos utilizados
        """
        # Primeiro calcula o peso individual de cada device_code por hora
        df_device_weights = (
            self.df
            .with_columns([
                # Classifica tipo de dispositivo
                pl.when(pl.col('device_id').str.slice(0, 3) == 'DAC')
                .then(pl.lit('DAC'))
                .when(pl.col('device_id').str.slice(0, 3) == 'DUT')
                .then(pl.lit('DUT'))
                .otherwise(pl.lit('OUTRO'))
                .alias('device_type')
            ])
            .group_by(['unit_id', 'data', 'hora', 'device_id', 'device_type'])
            .agg([
                pl.col('consumo_kwh').sum().alias('consumo_device')
            ])
        )
        
        # Calcula totais por unidade/data/hora para normalização
        df_totals = (
            self.df
            .group_by(['unit_id', 'data', 'hora'])
            .agg([
                pl.col('device_id').n_unique().alias('qtd_devices_total'),
                pl.col('consumo_kwh').sum().alias('consumo_total')
            ])
        )
        
        # Junta e calcula pesos individuais
        df_with_weights = (
            df_device_weights
            .join(df_totals, on=['unit_id', 'data', 'hora'], how='inner')
            .with_columns([
                # Peso de frequência normalizado: 50% do peso total (0.5 * 1/total_devices)
                (0.5 / pl.col('qtd_devices_total')).alias('peso_frequencia'),
                # Peso de consumo normalizado: 50% do peso total (0.5 * contribuição percentual)
                (0.5 * (pl.col('consumo_device') / pl.col('consumo_total'))).alias('peso_consumo'),
            ])
            .with_columns([
                # Peso composto: soma dos dois componentes (entre 0 e 1)
                (pl.col('peso_frequencia') + pl.col('peso_consumo')).alias('peso_total_device')
            ])
        )
        
        # Agrega por tipo de dispositivo
        df_availability = (
            df_with_weights
            .group_by(['unit_id', 'data', 'hora'])
            .agg([
                # Contagem total e por tipo
                pl.col('device_id').n_unique().alias('qtd_devices_total'),
                pl.col('device_id').filter(pl.col('device_type') == 'DAC').n_unique().alias('qtd_dac'),
                pl.col('device_id').filter(pl.col('device_type') == 'DUT').n_unique().alias('qtd_dut'),
                
                # Peso médio por tipo (média dos pesos individuais)
                pl.col('peso_total_device').filter(pl.col('device_type') == 'DAC').mean().alias('peso_medio_dac'),
                pl.col('peso_total_device').filter(pl.col('device_type') == 'DUT').mean().alias('peso_medio_dut'),
                
                # Consumo total
                pl.col('consumo_total').first().alias('consumo_kwh_total'),
            ])
            .join(
                # Adiciona informação de métodos do dataframe original
                self.df.group_by(['unit_id', 'data', 'hora']).agg([
                    pl.col('metodo').unique().sort().str.join(', ').alias('metodos')
                ]),
                on=['unit_id', 'data', 'hora'],
                how='inner'
            )
            .with_columns([
                # Arredonda valores e substitui null por 0
                pl.col('peso_medio_dac').fill_null(0.0).round(4),
                pl.col('peso_medio_dut').fill_null(0.0).round(4),
                pl.col('consumo_kwh_total').round(4)
            ])
            .select([
                'unit_id',
                'data',
                'hora',
                'qtd_devices_total',
                'qtd_dac',
                'qtd_dut',
                'peso_medio_dac',
                'peso_medio_dut',
                'consumo_kwh_total',
                'metodos'
            ])
            .sort(['unit_id', 'data', 'hora'])
        )
        
        print(f"\n✓ Agregação de disponibilidade concluída: {df_availability.shape[0]} registros")
        print(f"  Período: {df_availability['data'].min()} a {df_availability['data'].max()}")
        
        if output_csv_path:
            df_availability.write_csv(output_csv_path)
            print(f"✓ Arquivo salvo: {output_csv_path}")
        
        return df_availability
    
    def detect_low_availability_hours(self, df_availability: pl.DataFrame, 
                                     min_dac_expected: int = None,
                                     min_dut_expected: int = None) -> pl.DataFrame:
        """Detecta horas com possível baixa disponibilidade de dispositivos.
        
        Args:
            df_availability (pl.DataFrame): DataFrame de aggregate_device_availability_by_hour
            min_dac_expected (int, optional): Quantidade mínima esperada de DACs
            min_dut_expected (int, optional): Quantidade mínima esperada de DUTs
        
        Returns:
            pl.DataFrame: Registros com possível baixa disponibilidade
        """
        filters = []
        
        if min_dac_expected is not None:
            filters.append(pl.col('qtd_dac') < min_dac_expected)
        
        if min_dut_expected is not None:
            filters.append(pl.col('qtd_dut') < min_dut_expected)
        
        if not filters:
            # Se nenhum limite foi definido, retorna todos
            return df_availability
        
        # Combina filtros com OR
        combined_filter = filters[0]
        for f in filters[1:]:
            combined_filter = combined_filter | f
        
        df_low_availability = df_availability.filter(combined_filter).sort(['unit_id', 'data', 'hora'])
        
        print(f"\n⚠ Detectados {df_low_availability.shape[0]} registros com possível baixa disponibilidade")
        
        return df_low_availability
    
    def get_statistics(self) -> dict:
        """Retorna estatísticas gerais do dataset.
        
        Returns:
            dict: Dicionário com estatísticas
        """
        stats = {
            'total_registros': self.df.shape[0],
            'unidades_unicas': self.df['unit_id'].n_unique(),
            'dispositivos_unicos': self.df['device_id'].n_unique(),
            'periodo_inicio': str(self.df['data'].min()),
            'periodo_fim': str(self.df['data'].max()),
            'consumo_total_kwh': round(self.df['consumo_kwh'].sum(), 4),
            'registros_direto': self.df.filter(pl.col('metodo') == 'direto').shape[0],
            'registros_indireto': self.df.filter(pl.col('metodo') == 'indireto').shape[0]
        }
        
        return stats


if __name__ == "__main__":
    """Exemplo de uso da classe ConsumptionAnalyzer."""
    
    print("=" * 80)
    print("ANÁLISE DE CONSUMO HVAC")
    print("=" * 80)
    
    # Define caminhos
    input_path = r'use_case\output_files\consumption_consolidated.csv'
    output_avail_path = r'use_case\output_files\device_availability_by_hour.csv'
    
    try:
        # Inicializa analisador
        analyzer = ConsumptionAnalyzer(input_path)
        
        # Exibe estatísticas gerais
        print("\n" + "=" * 80)
        print("ESTATÍSTICAS GERAIS")
        print("=" * 80)
        stats = analyzer.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Análise de disponibilidade de dispositivos por hora
        print("\n" + "=" * 80)
        print("DISPONIBILIDADE DE DISPOSITIVOS POR HORA")
        print("=" * 80)
        df_avail = analyzer.aggregate_device_availability_by_hour(output_avail_path)
        print(f"\nAmostra (primeiras 5 linhas):")
        print(df_avail.head(5))
        
        # Detecta horas com baixa disponibilidade (exemplo: esperado pelo menos 2 DACs)
        print("\n" + "=" * 80)
        print("DETECÇÃO DE BAIXA DISPONIBILIDADE")
        print("=" * 80)
        df_low = analyzer.detect_low_availability_hours(df_avail, min_dac_expected=2)
        if df_low.shape[0] > 0:
            print(f"\nAmostra de horas com < 2 DACs (primeiras 10 linhas):")
            print(df_low.head(10))
        
    except FileNotFoundError as e:
        print(f"\n✗ ERRO: {e}")
        print("  Execute main.py primeiro para gerar o arquivo consolidado.")
    except Exception as e:
        print(f"\n✗ ERRO: {e}")

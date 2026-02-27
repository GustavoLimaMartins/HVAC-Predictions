import psycopg2
import pyarrow as pa
import os
from typing import Optional
from dotenv import load_dotenv

class QueryResult:
    """Wrapper para resultado de query PostgreSQL."""
    
    def __init__(self, cursor, connection):
        self.cursor = cursor
        self.connection = connection
        self._rows = None
        self._column_names = None
    
    @property
    def rows(self):
        """Retorna todas as linhas do resultado."""
        if self._rows is None:
            self._rows = self.cursor.fetchall()
        return self._rows
    
    @property
    def column_names(self):
        """Retorna os nomes das colunas."""
        if self._column_names is None:
            self._column_names = [desc[0] for desc in self.cursor.description]
        return self._column_names
    
    def to_arrow(self) -> pa.Table:
        """
        Converte o resultado da query para PyArrow Table de forma eficiente.
        
        Returns:
            pa.Table: Tabela Arrow com os dados da query
        """
        # Obtém nomes das colunas
        column_names = self.column_names
        
        # Usa fetchmany para processar em batches (evita carregar tudo na memória de uma vez)
        batch_size = 10000
        all_batches = []
        
        while True:
            rows = self.cursor.fetchmany(batch_size)
            if not rows:
                break
            
            # Converte batch para dicionário de listas
            if not all_batches:
                # Primeira batch: cria estrutura
                batch_data = {col: [] for col in column_names}
            else:
                batch_data = {col: [] for col in column_names}
            
            for row in rows:
                for col, value in zip(column_names, row):
                    batch_data[col].append(value)
            
            all_batches.append(pa.Table.from_pydict(batch_data))
        
        # Combina todos os batches em uma única tabela
        if not all_batches:
            # Se não há dados, retorna tabela vazia com schema correto
            return pa.Table.from_pydict({col: [] for col in column_names})
        
        return pa.concat_tables(all_batches)


class Client:
    """
    Cliente PostgreSQL compatível com a interface DatabaseConnectionClient.
    
    Variáveis de ambiente necessárias:
        POSTGRES_HOST: Host do servidor PostgreSQL (padrão: 'localhost')
        POSTGRES_PORT: Porta do servidor PostgreSQL (padrão: '5432')
        POSTGRES_DB: Nome do banco de dados
        POSTGRES_USER: Usuário do banco de dados
        POSTGRES_PASSWORD: Senha do banco de dados
    """
    
    def __init__(self, 
                 host: Optional[str] = None,
                 port: Optional[str] = None,
                 database: Optional[str] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None):
        """
        Inicializa conexão com PostgreSQL.
        
        Args:
            host: Host do servidor (padrão: env POSTGRES_HOST ou 'localhost')
            port: Porta do servidor (padrão: env POSTGRES_PORT ou '5432')
            database: Nome do banco (padrão: env POSTGRES_DB)
            user: Usuário (padrão: env POSTGRES_USER)
            password: Senha (padrão: env POSTGRES_PASSWORD)
        """
        # Carrega variáveis de ambiente do arquivo .env
        load_dotenv()

        self.host = host or os.getenv('POSTGRES_HOST', 'localhost')
        self.port = port or os.getenv('POSTGRES_PORT', '5432')
        self.database = database or os.getenv('POSTGRES_DB')
        self.user = user or os.getenv('POSTGRES_USER')
        self.password = password or os.getenv('POSTGRES_PASSWORD')
        
        self.connection = None
        self._connect()
    
    def _connect(self):
        """Estabelece conexão com o banco de dados."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
        except psycopg2.Error as e:
            raise Exception(f"Erro ao conectar ao PostgreSQL: {e}")
    
    def query(self, sql: str) -> QueryResult:
        """
        Executa uma query SQL e retorna resultado compatível com PyArrow.
        
        Args:
            sql: Query SQL a ser executada
        
        Returns:
            QueryResult: Objeto com método to_arrow() para conversão
        """
        if not self.connection or self.connection.closed:
            self._connect()
        
        cursor = self.connection.cursor()
        cursor.execute(sql)
        
        return QueryResult(cursor, self.connection)
    
    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.connection and not self.connection.closed:
            self.connection.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self):
        self.close()


class SyntaxPostgreeSQL:
    """
    Classe com métodos estáticos para gerar queries SQL para PostgreSQL.
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def get_devices_by_units(units: list[int]) -> str:
        """
        Retorna query SQL para obter dispositivos únicos por unidade usando a tabela device_disponiblity_hist.
        Filtra apenas dispositivos que começam com 'DAC'.
        
        Args:
            units (list[int]): Lista de IDs de unidades. Ex: [895, 1322, 302]
        
        Returns:
            str: Dados de código do dispositivo e ID da unidade para dispositivos únicos por unidade e com disponibilidade maior ou igual a 75%.
        """
        # Converte para lista de inteiros para garantir tipo correto
        units_int = [int(u) for u in units]
        
        # Para listas pequenas, IN é mais simples e igualmente eficiente
        if len(units_int) < 1000:
            units_str = ', '.join(map(str, units_int))
            where_clause = f"unit_id IN ({units_str})"
        else:
            # Para listas grandes, ANY(VALUES) é mais eficiente
            values_str = ', '.join(f"({u})" for u in units_int)
            where_clause = f"unit_id = ANY(VALUES {values_str})"
        
        return f'''
        WITH dispositivos_unicos AS (
            SELECT DISTINCT ON (device_code) 
                device_code, 
                unit_id 
            FROM device_disponibility_hist
            WHERE {where_clause}
                AND disponibility >= 75
              --AND device_code LIKE 'DAC%'
        )
        SELECT * FROM dispositivos_unicos
        ORDER BY unit_id;
        '''

    @staticmethod
    def get_devices_families_with_current_parameter() -> str:
        """
        Retorna query SQL para obter famílias de dispositivos com parâmetro de corrente elétrica (Curr) na tabela device_current_consumption.
            
        Returns:
            str: Dados de prefixo de dispositivo (primeiros 8 caracteres do device_code) para dispositivos que possuem consumo em Ah maior que zero.
        """
        return '''
        SELECT DISTINCT LEFT(device_code, 8) AS device_prefix
        FROM device_current_consumption
        WHERE consumption_ah > 0;
        '''
    
    @staticmethod
    def get_device_consumptions_by_indirect_method(device_code: str, date_init: str, date_final: str) -> str:
        """
        Retorna query SQL para obter consumo de um dispositivo usando método indireto na tabela energy_efficiency_hour_hist.
        
        Args:
            device_code (str): Código do dispositivo. Ex: 'DAC12345'
            date_init (str): Data inicial no formato 'YYYY-MM-DD'. Ex: '2026-01-01'
            date_final (str): Data final no formato 'YYYY-MM-DD'. Ex: '2026-01-31'
        
        Returns:
            str: Dados de dispositivo, data e consumo para o dispositivo especificado entre as datas fornecidas, filtrando apenas consumos maiores que zero.
        """
        
        return f'''
        SELECT 
            device_code,
            record_date,
            consumption
        FROM energy_efficiency_hour_hist
        WHERE device_code = '{device_code}'
          AND record_date BETWEEN '{date_init}' AND '{date_final}'
          AND consumption > 0;
        '''
    
    @staticmethod
    def get_record_dates_above_disponibility_threshold(units: list[int], threshold: int = 75, date_init: str = None, date_final: str = None) -> str:
        """
        Retorna query SQL para obter dispositivos e datas onde a disponibilidade ficou acima de um certo limiar.
        
        Args:
            units (list[int]): Lista de IDs de unidades. Ex: [895, 1322, 302]
            threshold (int): Limiar de disponibilidade (padrão: 75)
        
        Returns:
            str: Dados de dispositivos e datas onde a disponibilidade ficou acima do limiar para as unidades especificadas
        """
        units_int = [int(u) for u in units]
        
        if len(units_int) < 1000:
            units_str = ', '.join(map(str, units_int))
            where_clause = f"unit_id IN ({units_str})"
        else:
            values_str = ', '.join(f"({u})" for u in units_int)
            where_clause = f"unit_id = ANY(VALUES {values_str})"
        
        return f'''
        SELECT 
            device_code,
            record_date
        FROM device_disponibility_hist
        WHERE {where_clause}
          AND disponibility >= {threshold}
          AND record_date BETWEEN '{date_init}' AND '{date_final}';
        '''

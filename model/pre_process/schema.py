"""
ModelSchema — Pré-processamento e Engenharia de Features para Treinamento
=========================================================================

Transforma o `final_dataframe.csv` (saída do pipeline de coleta) no esquema
final esperado pelo modelo de ML, adicionando features derivadas que não
existem no arquivo consolidado:

    final_dataframe                     ModelSchema (saída)
    ─────────────────────────────────   ─────────────────────────────────────
    hora                                hora
    data                                (removida após extração)
    consumo_kwh                         consumo_kwh
    machine_type                        tipo_maquina
    estacao                             estacao
    grupo_regional                      grupo_regional
    Temperatura_C                       Temperatura_C
    Temperatura_Percebida_C             Temperatura_Percebida_C
    Umidade_Relativa_%                  Umidade_Relativa_%
    Precipitacao_mm                     Precipitacao_mm
    Velocidade_Vento_kmh                Velocidade_Vento_kmh
    Pressao_Superficial_hPa             Pressao_Superficial_hPa
    Irradiancia_Direta_Wm2              Irradiancia_Direta_Wm2
    Irradiancia_Difusa_Wm2              Irradiancia_Difusa_Wm2
                                        ── features de data ──
                                        ano              (ex: 2025)
                                        mes              (1 a 12)
                                        dia              (1 a 31)
                                        trimestre        (1 a 4)
                                        dia_semana       (nome)
                                        is_dia_util      (bool)
                                        is_feriado       (bool)
                                        is_vespera_feriado (bool)
"""

from __future__ import annotations

import polars as pl
import holidays

class ModelSchema:
    """
    Classe de pré-processamento e engenharia de features para o modelo de ML.

    Recebe o DataFrame consolidado (final_dataframe.csv) e expõe métodos
    para enriquecê-lo com features derivadas necessárias ao treinamento.

    O método principal é `build()`, que executa o pipeline completo. Cada
    método de transformação pode ser chamado individualmente para inspeção
    ou composição parcial.

    Attributes:
        df (pl.DataFrame): DataFrame que será transformado ao longo do pipeline.

    Example:
        >>> schema = ModelSchema(pl.read_csv("final_dataframe.csv"))
        >>> df_model = schema.build()
    """

    def __init__(self, df: pl.DataFrame, schema_fields: list[str]) -> None:
        """
        Inicializa o schema com o DataFrame consolidado.

        Args:
            df (pl.DataFrame): DataFrame de entrada com as colunas do
                final_dataframe.csv.

        Raises:
            ValueError: Se colunas obrigatórias estiverem ausentes.
        """
        required = set(schema_fields)  # Converte a lista em conjunto para verificação
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing)}")

        self._schema_fields = schema_fields
        # ✅ Remove linhas com valores nulos em qualquer coluna obrigatória
        self.df = df.clone().drop_nulls(subset=schema_fields)
        self.clipping_limits_ = {}  # Armazena limites de clipping para persistência

    # ── Pipeline completo ────────────────────────────────────────────────────

    def build(self) -> pl.DataFrame:
        """
        Executa o pipeline completo de engenharia de features.

        Etapas:
            1. add_date_features          — features derivadas da coluna 'data'
            2. add_temporal_features      — lag e rolling mean (requer device_id)
            3. adjust_machine_type        — padroniza nomes de equipamento
            4. make_categorical_columns   — declara grupo_regional e mes como categóricos
            5. make_one_hot_encode_columns — OHE em tipo_maquina, estacao, etc.

        NOTA: make_clipping_min_max_columns NÃO é chamado aqui pois deve ser
        aplicado APENAS ao conjunto de treino (não em validação/teste/inferência).
        Use `.make_clipping_min_max_columns()` manualmente após `build()` apenas
        para dados de treino.

        Returns:
            pl.DataFrame pronto para treinamento ou inferência.
        """
        return (
            ModelSchema(self.df, self._schema_fields)
            .add_date_features()
            .adjust_machine_type()
            .make_categorical_columns(["grupo_regional"])
            .make_one_hot_encode_columns([
                "tipo_maquina", "estacao", "periodo_dia"
            ])
            .df
        )

    # ── Features de data ─────────────────────────────────────────────────────

    def add_date_features(self) -> "ModelSchema":
        """
        Adiciona features temporais derivadas da coluna 'data' e a remove.

        Todas as expressões são avaliadas em paralelo em C/Rust pelo Polars
        dentro de um único `with_columns`, minimizando overhead de cópia.
        Ao final, a coluna 'data' é descartada — todas as informações
        relevantes já foram decompostas em features numéricas/booleanas.

        Features adicionadas
        --------------------
        ano               : int  — Ano (ex: 2025).
        mes               : int  — Número do mês (1 a 12).
        dia               : int  — Dia do mês (1 a 31).
        trimestre         : int  — Trimestre do ano (1 a 4).
        dia_semana_num    : int  — Dia ISO da semana (1=Segunda … 7=Domingo).
        dia_semana        : str  — Nome do dia em português.
        is_feriado        : bool — True se o dia é feriado nacional brasileiro.
        is_vespera_feriado: bool — True se o dia seguinte é feriado nacional.
        is_dia_util       : bool — True se é dia de semana e não é feriado.

        A lista de feriados é calculada uma única vez para os anos presentes
        no DataFrame (otimização: evita chamadas repetidas à lib `holidays`).

        Returns:
            Self (para method chaining).
        """
        # Garante que 'data' é pl.Date antes de operar
        df = self.df.with_columns(
            pl.col("data").cast(pl.Date)
        )

        # Extrai anos únicos e gera feriados brasileiros apenas para eles
        anos = df["data"].dt.year().unique().to_list()
        feriados_br = pl.Series(
            "feriado",
            list(holidays.BR(years=anos).keys()),
            dtype=pl.Date,
        )
        vesperas_br = pl.Series(
            "vespera",
            [d + __import__("datetime").timedelta(days=-1)
             for d in holidays.BR(years=anos).keys()],
            dtype=pl.Date,
        )

        self.df = (
            df.with_columns([
                # Componentes da data
                pl.col("data").dt.year().alias("ano"),
                pl.col("data").dt.month().alias("mes"),
                pl.col("data").dt.day().alias("dia"),

                # Trimestre
                ((pl.col("data").dt.month() - 1) // 3 + 1).alias("trimestre"),

                # Dia da semana numérico (ISO 8601: 1=Seg … 7=Dom)
                #pl.col("data").dt.weekday().alias("dia_semana_num"),

                # Nome do dia em português
                #pl.col("data").dt.weekday()
                #  .replace_strict(_WEEKDAY_MAP, return_dtype=pl.Utf8)
                #  .alias("dia_semana"),

                # Feriado nacional
                pl.col("data").is_in(feriados_br).cast(pl.Int8).alias("is_feriado"),

                # Véspera de feriado
                pl.col("data").is_in(vesperas_br).cast(pl.Int8).alias("is_vespera_feriado"),

                # Dia útil: seg–sex e não feriado
                (
                    (pl.col("data").dt.weekday() < 6) &
                    ~pl.col("data").is_in(feriados_br)
                ).cast(pl.Int8).alias("is_dia_util"),

                # Período do dia — agrupa horas pelo comportamento do AC
                pl.when(pl.col("hora").is_between(0, 6))
                  .then(pl.lit("Madrugada"))
                  .when(pl.col("hora").is_between(7, 11))
                  .then(pl.lit("Manhã"))
                  .when(pl.col("hora").is_between(12, 18))
                  .then(pl.lit("Tarde"))
                  .otherwise(pl.lit("Noite"))
                  .alias("periodo_dia"),
            ])
            .drop(["data", "dia", "ano"])  # data completamente decomposta — coluna removida
        )

        return self
    
    def adjust_machine_type(self) -> "ModelSchema":
        """
        Ajusta a coluna 'machine_type' para garantir consistência.

        Substitui valores inconsistentes ou ausentes por 'Desconhecido' e
        padroniza o formato (ex: capitalização).

        Returns:
            Self (para method chaining).
        """
        # Dicionário de mapeamento (De -> Para)
        dicionario_maquinas = {
            'split-wall': 'SPLIT HI-WALL',
            'split wall': 'SPLIT HI-WALL',
            'splitao-inverter': 'SPLITÃO INVERTER',
            'splitao': 'SPLITÃO',
            'rooftop': 'SPLITÃO ROOFTOP',
            'ar condicionado de janela': 'AR CONDICIONADO DE JANELA (ACJ)',
            'split-duto': 'SPLIT DUTO',
            'self': 'SPLITÃO SELF CONTAINED',
            'split-piso-teto': 'SPLIT PISO-TETO',
            'split piso teto': 'SPLIT PISO-TETO',
            'split-cassete': 'SPLIT CASSETE',
            'split cassete': 'SPLIT CASSETE',
            'Acj': 'AR CONDICIONADO DE JANELA (ACJ)',
            'ACJ (ar condicionado de janela)': 'AR CONDICIONADO DE JANELA (ACJ)',
            'ACJ (Ar condicionado Janela)': 'AR CONDICIONADO DE JANELA (ACJ)',
            'Câmara Fria': 'CÂMARA FRIA',
            'Cassete': 'SPLIT CASSETE',
            'Chiller-Água': 'CHILLER ÁGUA',
            'Chiller-Ar': 'CHILLER AR',
            'Cold Head': 'COLD HEAD',
            'Cortina de ar': 'CORTINA DE AR',
            'Fan Coil': 'FANCOIL',
            'Fancoil': 'FANCOIL',
            'Fancolete': 'FANCOIL',
            'Hi-Wall': 'SPLIT HI-WALL',
            'Multisplit': 'MULTISPLIT',
            'piso-teto': 'SPLIT PISO-TETO',
            'Piso-Teto Embutido': 'SPLIT PISO-TETO',
            'Rooftop': 'SPLITÃO ROOFTOP',
            'Self': 'SPLITÃO SELF CONTAINED',
            'Self Condensação A Água': 'SPLITÃO SELF CONTAINED ÁGUA',
            'Self Containde': 'SPLITÃO SELF CONTAINED',
            'Self Contaneid': 'SPLITÃO SELF CONTAINED',
            'Self-Contained': 'SPLITÃO SELF CONTAINED',
            'Spitão-Inverter': 'SPLITÃO INVERTER',
            'Spli Hi-Wall': 'SPLIT HI-WALL',
            'Spli K7': 'SPLIT CASSETE',
            'Spli Tipo K7': 'SPLIT CASSETE',
            'Split': 'SPLIT HI-WALL',
            'Split Hi-Wall': 'SPLIT HI-WALL',
            'Split Hi0wall': 'SPLIT HI-WALL',
            'Split Hiwall': 'SPLIT HI-WALL',
            'Split K7': 'SPLIT CASSETE',
            'Split Kassete': 'SPLIT CASSETE',
            'Split Tipo K7': 'SPLIT CASSETE',
            'Split-Cassete': 'SPLIT CASSETE',
            'Split-Duto': 'SPLIT DUTO',
            'Split-Piso Teto': 'SPLIT PISO-TETO',
            'Split-Wall': 'SPLIT HI-WALL',
            'Splitão-Inverter': 'SPLITÃO INVERTER',
            'SplitDuto': 'SPLIT DUTO',
            'SplitWall': 'SPLIT HI-WALL',
            'Splt Hi-Wall': 'SPLIT HI-WALL',
            'tipo-split duto': 'SPLIT DUTO',
            'tipo-split-cassete': 'SPLIT CASSETE',
            'tipo-split-hi-wall': 'SPLIT HI-WALL',
            'tipo-split-piso-teto': 'SPLIT PISO-TETO',
            'tipo-split-teto': 'SPLIT PISO-TETO',
            'TROCADOR': 'TROCADOR DE CALOR',
            'Trocador de Calor': 'TROCADOR DE CALOR',
            'VAV': 'VAV'
        }

        self.df = self.df.with_columns(
            pl.col("machine_type")
            .str.to_lowercase()
            .replace(dicionario_maquinas)
            .fill_null("Desconhecido")
            .alias("tipo_maquina")
        ).drop("machine_type")

        return self

    def make_categorical_columns(self, columns: list[str]) -> "ModelSchema":
        """
        Declara colunas ordinais/nominais como pl.Categorical.

        Colunas inteiras (ex: mes 1–12, grupo_regional 0–135) são convertidas
        para Utf8 e em seguida para pl.Categorical. A representação física
        interna do Polars (UInt32 de códigos) é preservada e usada pela
        camada de conversão numpy (_to_numpy) via Expr.to_physical().

        No LightGBM, as colunas presentes em _NATIVE_CAT_COLS são declaradas
        via categorical_feature, ativando splits categóricos nativos no lugar
        de splits numéricos contínuos — mais eficiente para variáveis de ID
        ordinais como grupo regional e mês do ano.

        Args:
            columns (list[str]): Colunas a serem declaradas como categóricas.

        Returns:
            Self (para method chaining).
        """
        self.df = self.df.with_columns([
            pl.col(c).cast(pl.Utf8).cast(pl.Categorical)
            for c in columns
        ])
        return self

    def make_one_hot_encode_columns(self, columns: list[str]) -> "ModelSchema":
        """
        Aplica one-hot encoding às colunas categóricas especificadas.

        Args:
            columns (list[str]): Lista de nomes de colunas a serem codificadas.
        Returns:
            Self (para method chaining).
        """
        for col in columns:
            dummies = self.df.select(pl.col(col)).to_dummies()
            self.df = self.df.with_columns(dummies)
            self.df = self.df.drop(col)
        return self
    
    def make_clipping_min_max_columns(self, columns: list[str], use_persisted_limits: bool = False) -> "ModelSchema":
        """
        Aplica Clipping + Min/Max às colunas numéricas especificadas.

        Pipeline por coluna:
            1. Calcula Q1, Q3 e IQR (Q3 - Q1).
            2. Define os limites de Tukey:
                   lower = Q1 - 1.5 * IQR
                   upper = Q3 + 1.5 * IQR
            3. Realiza o clipping — valores fora dos limites são
               truncados para `lower` / `upper`, eliminando outliers extremos.
            4. Aplica Min/Max com os próprios limites como âncoras:
                   (x_clipped - lower) / (upper - lower)
               Resultado final no intervalo [0, 1].

        Args:
            columns (list[str]): Lista de nomes de colunas a serem normalizadas.
            use_persisted_limits (bool): ✅ Novo - Se True, usar limites de self.clipping_limits_ 
                                         (para inferência com dados de treino).

        Returns:
            Self (para method chaining).
        """
        for col in columns:
            # ✅ Skip colunas que não existem no DataFrame
            if col not in self.df.columns:
                continue
            
            # ✅ Novo: se use_persisted_limits=True, usar limites pré-calculados
            if use_persisted_limits and col in self.clipping_limits_:
                limits = self.clipping_limits_[col]
                lower = limits["lower"]
                upper = limits["upper"]
            else:
                # Calcular limites novamente (comportamento antigo para treino)
                q1 = self.df[col].quantile(0.15)
                q3 = self.df[col].quantile(0.85)
                
                # ✅ Skip se quantile retorna None (coluna vazia ou todos NaN)
                if q1 is None or q3 is None:
                    continue
                
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr

                # ✅ Armazena limites para persistência em metadata_norm.json
                self.clipping_limits_[col] = {
                    "q1": float(q1),
                    "q3": float(q3),
                    "iqr": float(iqr),
                    "lower": float(lower),
                    "upper": float(upper),
                }

            self.df = self.df.with_columns(
                ((pl.col(col).clip(lower, upper) - lower) / (upper - lower)).alias(col)
            )
        return self
    
    def make_target_encoding_columns(
        self,
        columns: list[str],
        target: str = "consumo_kwh",
        smoothing: float = 10.0,
        encoding_map: dict[str, dict] | None = None,
    ) -> "ModelSchema":
        """
        Aplica Target Encoding (média suavizada do target) a features numéricas.

        Substitui cada valor da coluna pela média suavizada do target
        (``consumo_kwh``) observada para aquele valor. Captura relações
        não-lineares (ex: consumo médio por hora do dia) sem explodir a
        dimensionalidade como OHE.

        Fórmula de suavização bayesiana (evita overfitting em categorias raras):

        .. math::

            \\text{encoding}_i = \\frac{n_i \\cdot \\bar{y}_i + m \\cdot \\bar{y}}{n_i + m}

        onde:
            - :math:`n_i`        = nº de observações com valor *i*
            - :math:`\\bar{y}_i` = média do target para o valor *i*
            - :math:`m`          = fator de suavização (*smoothing*)
            - :math:`\\bar{y}`   = média global do target

        Modos de operação
        -----------------
        **Treino** (``encoding_map=None``):
            Calcula as estatísticas a partir dos dados atuais e armazena o
            mapa em ``self.target_encoding_map_`` para reutilização.

        **Inferência** (``encoding_map`` fornecido):
            Usa o mapa pré-computado do treino. Valores não vistos recebem
            a média global armazenada como fallback.

        .. warning::
            Na inferência o pipeline insere ``consumo_kwh = 0.0`` como
            dummy — portanto é **obrigatório** passar ``encoding_map``
            para não corromper as médias.

        Args:
            columns:      Colunas a serem codificadas (ex: ``["hora", "mes"]``).
            target:       Nome da coluna target (default: ``"consumo_kwh"``).
            smoothing:    Fator de suavização *m* (default: 10.0).
                          Valores maiores puxam mais para a média global.
            encoding_map: Mapa pré-computado
                          ``{col: {"mapping": {val: enc}, "global_mean": float}}``.
                          Se ``None``, calcula a partir dos dados (modo treino).

        Returns:
            Self (para method chaining).

        Attributes:
            target_encoding_map_ : dict[str, dict]
                Mapa gerado (ou recebido) pronto para ser persistido junto
                ao pipeline e reutilizado em inferência.

        Example:
            >>> # Treino
            >>> schema = ModelSchema(df_train, fields)
            >>> schema.make_target_encoding_columns(["hora", "mes"])
            >>> te_map = schema.target_encoding_map_  # salvar junto ao modelo
            >>>
            >>> # Inferência
            >>> schema_inf = ModelSchema(df_new, fields)
            >>> schema_inf.make_target_encoding_columns(["hora", "mes"], encoding_map=te_map)
        """
        fit_mode = encoding_map is None
        if fit_mode:
            encoding_map = {}
            if target not in self.df.columns:
                raise ValueError(
                    f"Coluna target '{target}' ausente no DataFrame. "
                    f"Em modo treino ela é necessária para calcular as médias."
                )
            global_mean = float(self.df[target].mean())

        for col in columns:
            if col not in self.df.columns:
                continue

            if fit_mode:
                # ── estatísticas por valor ────────────────────────────────
                stats = (
                    self.df
                    .group_by(col)
                    .agg([
                        pl.col(target).mean().alias("_te_mean"),
                        pl.len().alias("_te_count"),
                    ])
                )

                # ── suavização bayesiana ──────────────────────────────────
                stats = stats.with_columns(
                    (
                        (pl.col("_te_count") * pl.col("_te_mean")
                         + smoothing * global_mean)
                        / (pl.col("_te_count") + smoothing)
                    ).alias("_te_value")
                )

                col_encoding: dict[int | str, float] = {
                    row[col]: float(row["_te_value"])
                    for row in stats.iter_rows(named=True)
                }
                encoding_map[col] = {
                    "mapping": col_encoding,
                    "global_mean": global_mean,
                }

            # ── aplicar mapa ao DataFrame ─────────────────────────────────
            map_info = encoding_map[col]
            mapping = map_info["mapping"]
            fallback = float(map_info["global_mean"])

            original_dtype = self.df[col].dtype
            te_col_name = f"{col}_target_enc"

            mapping_df = pl.DataFrame({
                col: pl.Series(list(mapping.keys())).cast(original_dtype),
                te_col_name: pl.Series(
                    list(mapping.values()), dtype=pl.Float64,
                ),
            })

            self.df = (
                self.df
                .join(mapping_df, on=col, how="left")
                .with_columns(pl.col(te_col_name).fill_null(fallback))
                .drop(col)
            )

        self.target_encoding_map_ = encoding_map
        return self

    def make_cyclical_encoding(self, column: str, period: int) -> "ModelSchema":
        """
        Aplica codificação cíclica a uma coluna temporal (ex: hora, dia).

        Transforma a coluna em duas novas colunas usando funções trigonométricas:
            sin_col = sin(2π * col / period)
            cos_col = cos(2π * col / period)

        Args:
            column (str): Nome da coluna a ser codificada.
            period (int): Período do ciclo (ex: 24 para horas, 7 para dias da semana).

        Returns:
            Self (para method chaining).
        """
        self.df = self.df.with_columns([
            (pl.col(column).cast(pl.Float64) * (2 * pl.pi / period)).alias("angle"),
            pl.col("angle").sin().alias(f"{column}_sin"),
            pl.col("angle").cos().alias(f"{column}_cos")
        ]).drop([column, "angle"])
        return self


# ── Execução direta ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    csv_path = Path(r"use_case\files\final_dataframe.parquet")
    if not csv_path.exists():
        print(f"✗ Arquivo não encontrado: {csv_path}")
        sys.exit(1)

    df_raw = pl.read_parquet(csv_path)
    print(f"✓ {df_raw.shape[0]} registros carregados | colunas: {df_raw.columns}")

    schema_fields = [
        "hora", "data", "consumo_kwh", "machine_type",
        "estacao", "grupo_regional",
        "Temperatura_C", "Temperatura_Percebida_C",
        "Umidade_Relativa_%", "Precipitacao_mm",
        "Velocidade_Vento_kmh", "Pressao_Superficial_hPa",
        "Irradiancia_Direta_Wm2", "Irradiancia_Difusa_Wm2",
        "consumo_lag_1h", "consumo_lag_24h", "temp_rolling_mean_3h"
    ]
    schema = ModelSchema(df_raw, schema_fields)
    df_model = schema.build()

    print(f"\nSchema final ({df_model.shape[1]} colunas):")
    print(df_model.schema)

    # Colunas one-hot geradas (machine_type, estacao, dia_semana, hora, dia, mes, trimestre)
    ohe_prefixes = ("machine_type_", "estacao_", "periodo_dia_")  # prefixos definidos no método make_one_hot_encode_columns
    ohe_cols = [c for c in df_model.columns if c.startswith(ohe_prefixes)]
    print(f"\nColunas one-hot geradas ({len(ohe_cols)}): {ohe_cols}")

    print(f"\nColuna 'data' removida:        {'data' not in df_model.columns}")
    print(f"Coluna 'machine_type' removida: {'machine_type' not in df_model.columns}")

    print(f"\nAmostra — features numéricas normalizadas (Clipping + Min/Max):")
    print(df_model.select(schema_fields).head(10))

    print(f"\nAmostra — features booleanas e temporais:")
    print(df_model.select([
        "is_feriado", "is_vespera_feriado", "is_dia_util",
    ]).head(10))

    print(f"\nAmostra — variável alvo e tipo de máquina padronizado:")
    print(df_model.head(10))

    output_path = Path(r"use_case\files\final_schema_sample.csv")
    df_model.head(10).write_csv(output_path)
    print(f"\n✓ Amostra salva em: {output_path}")

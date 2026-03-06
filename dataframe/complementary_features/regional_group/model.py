"""
Regional Group Classifier — DBSCAN + Haversine
===============================================

Descobre automaticamente "zonas quentes" (clusters geográficos) a partir de
combinações únicas de latitude/longitude, sem necessidade de definir o número
de grupos previamente.

Por que DBSCAN + Haversine?
──────────────────────────
• DBSCAN não exige que o número de clusters seja informado.
• Identifica pontos fora de qualquer zona ("ruído").
• A métrica Haversine mede distâncias reais na superfície da Terra, levando em
  conta sua forma esférica — impossível com geometria plana (Pitágoras).
• O parâmetro `radius_km` é intuitivo: "agrupe pontos a até X km de distância".

Fórmula de Haversine (usada internamente pelo sklearn)
──────────────────────────────────────────────────────
Dados dois pontos (φ₁,λ₁) e (φ₂,λ₂) em radianos:

    a = sin²(Δφ/2) + cos(φ₁)·cos(φ₂)·sin²(Δλ/2)
    d = 2R · arcsin(√a)          onde R ≈ 6 371 km
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

# ── Constante física ────────────────────────────────────────────────────────────
EARTH_RADIUS_KM: float = 6_371.0


# ── Classe base: contrato comum a modelos geográficos não-supervisionados ───────

class UnsupervisedGeoModel:
    """
    Classe base para modelos de clustering geográfico não-supervisionados.

    Define a interface mínima compartilhada por qualquer algoritmo de agrupamento
    espacial: fit / predict / fit_predict e validação de colunas de coordenadas.

    Subclasses devem obrigatoriamente implementar fit() e predict().
    """

    def __init__(self) -> None:
        self.is_fitted: bool = False

    def fit_predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Treina o modelo e retorna o DataFrame com grupos atribuídos.

        Equivalente a chamar fit() seguido de predict() sobre o mesmo DataFrame.

        Args:
            df: DataFrame contendo ao menos as colunas 'latitude' e 'longitude'.

        Returns:
            DataFrame original com a coluna 'grupo_regional' adicionada.
        """
        self.fit(df)
        return self.predict(df)

    # ── Utilitários protegidos ───────────────────────────────────────────────

    def _validate_geo_columns(self, df: pl.DataFrame) -> None:
        """
        Valida a presença das colunas de coordenadas no DataFrame.

        Raises:
            ValueError: Se 'latitude' ou 'longitude' estiverem ausentes.
        """
        missing = [c for c in ("latitude", "longitude") if c not in df.columns]
        if missing:
            raise ValueError(f"Colunas ausentes no DataFrame: {missing}")

    @staticmethod
    def _to_radians(df_coords: pl.DataFrame) -> np.ndarray:
        """
        Converte colunas latitude/longitude (em graus) para radianos.

        O sklearn com metric='haversine' exige entrada em radianos.

        Args:
            df_coords: DataFrame com colunas 'latitude' e 'longitude'.

        Returns:
            Array numpy de shape (n, 2) com valores em radianos.
        """
        return np.radians(df_coords.select(["latitude", "longitude"]).to_numpy())


# ── Classifier ──────────────────────────────────────────────────────────────────

class RegionalGroupClassifier(UnsupervisedGeoModel):
    """
    Classificador de grupos regionais usando DBSCAN + distância Haversine.

    Descobre "zonas quentes" a partir da concentração de coordenadas únicas,
    sem que o número de grupos precise ser informado previamente.

    Parâmetros
    ----------
    radius_km   : Raio de vizinhança em km (parâmetro eps do DBSCAN).
                  Semântica: "dois pontos fazem parte do mesmo grupo se
                  estiverem a menos de radius_km km um do outro".
    min_samples : Mínimo de pontos (inclusive o próprio) para que uma
                  região seja considerada um cluster e não ruído.
                  Valor padrão 4 → "pelo menos 3 vizinhos no raio".
    noise_label : Rótulo atribuído a pontos fora de qualquer zona quente.

    Atributos pós-treino
    --------------------
    is_fitted   : bool — True após fit().
    n_clusters_ : int  — Número de zonas quentes encontradas (ruído excluído).
    n_noise_    : int  — Pontos classificados como ruído (label DBSCAN = -1).

    Como funciona internamente
    --------------------------
    1. Extrai combinações únicas de (latitude, longitude) sem nulos.
    2. Converte para radianos (requisito do sklearn com metric='haversine').
    3. DBSCAN agrupa pontos dentro de `radius_km` com ≥ `min_samples` vizinhos.
    4. Mapeia cada coordenada única ao seu rótulo de cluster.
    5. Para predição de pontos novos (não vistos no treino), um KNN 1-vizinho
       com a mesma métrica Haversine é usado como fallback.

    Example
    -------
    >>> clf = RegionalGroupClassifier(radius_km=10, min_samples=4)
    >>> df_enriched = clf.fit_predict(df)
    >>> print(clf.n_clusters_)
    """

    def __init__(
        self,
        radius_km: float = 10.0,
        min_samples: int = 4,
        noise_label: str = "ruido",
    ) -> None:
        super().__init__()

        self.radius_km = radius_km
        self.min_samples = min_samples
        self.noise_label = noise_label

        # eps convertido para radianos: distância / raio médio da Terra
        self._eps_rad: float = radius_km / EARTH_RADIUS_KM

        self._dbscan: Optional[DBSCAN] = None
        self._knn: Optional[KNeighborsClassifier] = None
        self._unique_coords: Optional[pl.DataFrame] = None
        self._coord_labels: Optional[np.ndarray] = None

        self.n_clusters_: int = 0
        self.n_noise_: int = 0

    # ── Treino ──────────────────────────────────────────────────────────────

    def fit(self, df: pl.DataFrame) -> "RegionalGroupClassifier":
        """
        Treina o DBSCAN sobre as coordenadas únicas do DataFrame.

        Linhas com latitude ou longitude nulos são ignoradas no treino.

        Args:
            df: DataFrame com colunas 'latitude' e 'longitude'.

        Returns:
            Self (para method chaining).

        Raises:
            ValueError: Se as colunas de coordenadas não existirem.
        """
        self._validate_geo_columns(df)

        # Extrai coordenadas únicas válidas
        self._unique_coords = (
            df.select(["latitude", "longitude"])
            .drop_nulls()
            .unique()
        )

        coords_rad = self._to_radians(self._unique_coords)

        # DBSCAN com métrica Haversine (sklearn exige entrada em radianos)
        self._dbscan = DBSCAN(
            eps=self._eps_rad,
            min_samples=self.min_samples,
            algorithm="ball_tree",
            metric="haversine",
        )
        self._dbscan.fit(coords_rad)
        self._coord_labels = self._dbscan.labels_

        self.n_clusters_ = len(set(self._coord_labels)) - (
            1 if -1 in self._coord_labels else 0
        )
        self.n_noise_ = int((self._coord_labels == -1).sum())

        # KNN 1-vizinho como fallback para coordenadas não vistas no treino.
        # Treinado apenas com pontos que pertencem a algum cluster (não ruído).
        valid_mask = self._coord_labels != -1
        if valid_mask.sum() > 0:
            self._knn = KNeighborsClassifier(
                n_neighbors=1,
                algorithm="ball_tree",
                metric="haversine",
            )
            self._knn.fit(
                coords_rad[valid_mask],
                self._coord_labels[valid_mask].tolist(),
            )

        self.is_fitted = True
        print(
            f"✓ DBSCAN concluído: {self.n_clusters_} zonas quentes | "
            f"{self.n_noise_} pontos isolados → absorvidos via KNN "
            f"(raio={self.radius_km} km, min_samples={self.min_samples})"
        )
        return self

    # ── Predição ─────────────────────────────────────────────────────────────

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Atribui grupo regional a cada linha do DataFrame.

        Estratégia por tipo de ponto:
        • Coordenada de cluster (DBSCAN ≥ 0)  → rótulo direto do DBSCAN.
        • Ponto de ruído (DBSCAN label -1)    → KNN 1-vizinho ao cluster mais próximo.
        • Coordenada nova (não vista)         → KNN 1-vizinho ao cluster mais próximo.
        • Lat/lon nulo                        → recebe null.

        Nenhum ponto recebe rótulo de ruído — todos são absorvidos pela zona
        geograficamente mais próxima via KNN.

        Args:
            df: DataFrame com colunas 'latitude' e 'longitude'.

        Returns:
            DataFrame original com nova coluna 'grupo_regional' (Utf8).

        Raises:
            ValueError: Se o modelo não foi treinado.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")

        self._validate_geo_columns(df)

        # Monta mapa (lat, lon) → rótulo apenas para pontos de cluster (label ≥ 0).
        # Pontos de ruído (label == -1) são excluídos intencionalmente para que
        # sejam enfileirados no KNN junto com coordenadas novas.
        train_lats = self._unique_coords["latitude"].to_list()
        train_lons = self._unique_coords["longitude"].to_list()
        coord_map: dict[tuple, int] = {
            (lat, lon): label
            for (lat, lon), label in zip(
                zip(train_lats, train_lons), self._coord_labels
            )
            if label != -1
        }

        labels: list[Optional[int]] = []
        knn_idxs: list[int] = []
        knn_coords: list[tuple[float, float]] = []

        for i, (lat, lon) in enumerate(
            zip(df["latitude"].to_list(), df["longitude"].to_list())
        ):
            if lat is None or lon is None:
                labels.append(None)
            elif (lat, lon) in coord_map:
                labels.append(coord_map[(lat, lon)])
            else:
                # Ponto de ruído ou coordenada nova → resolvido via KNN
                labels.append(None)  # placeholder
                knn_idxs.append(i)
                knn_coords.append((lat, lon))

        # Resolve ruído + pontos novos via KNN Haversine
        if knn_idxs:
            if self._knn is not None:
                coords_rad = np.radians(np.array(knn_coords))
                knn_labels = self._knn.predict(coords_rad)
                for idx, label in zip(knn_idxs, knn_labels):
                    labels[idx] = int(label)
            else:
                # Caso extremo: todos os pontos do treino eram ruído
                for idx in knn_idxs:
                    labels[idx] = None

        return df.with_columns(
            pl.Series("grupo_regional", labels, dtype=pl.Int32)
        )

    # ── Utilitários ─────────────────────────────────────────────────────────

    def get_cluster_summary(self) -> Optional[pl.DataFrame]:
        """
        Retorna um resumo estatístico por zona quente.

        Para cada cluster encontrado, calcula o centróide geográfico
        (média de lat/lon) e o número de pontos únicos pertencentes.

        Returns:
            pl.DataFrame com colunas:
                - grupo_regional  : rótulo do cluster
                - centroide_lat   : latitude média do cluster
                - centroide_lon   : longitude média do cluster
                - n_pontos_unicos : quantidade de coordenadas únicas no cluster
            None se o modelo não foi treinado.
        """
        if not self.is_fitted:
            return None

        lats = self._unique_coords["latitude"].to_list()
        lons = self._unique_coords["longitude"].to_list()

        groups: dict[str, list[tuple[float, float]]] = {}
        for (lat, lon), label in zip(zip(lats, lons), self._coord_labels):
            key = int(label) if label != -1 else None
            if key is not None:
                groups.setdefault(key, []).append((lat, lon))

        rows = [
            {
                "grupo_regional": k,
                "centroide_lat": float(np.mean([p[0] for p in pts])),
                "centroide_lon": float(np.mean([p[1] for p in pts])),
                "n_pontos_unicos": len(pts),
            }
            for k, pts in sorted(groups.items())
        ]
        return pl.DataFrame(rows).with_columns(pl.col("grupo_regional").cast(pl.Int32))


# ── Funções helper ──────────────────────────────────────────────────────────────

def enrich_with_regional_groups(
    df: pl.DataFrame,
    radius_km: float = 5.0,
    min_samples: int = 2,
) -> pl.DataFrame:
    """
    Enriquece o DataFrame com a coluna 'grupo_regional' via DBSCAN + Haversine.

    Função de conveniência que instancia RegionalGroupClassifier e executa
    fit_predict em uma única chamada.

    Args:
        df          : DataFrame com colunas 'latitude' e 'longitude'.
        radius_km   : Raio de vizinhança em km (padrão: 5 km).
        min_samples : Mínimo de pontos para formar um cluster (padrão: 2).

    Returns:
        DataFrame original com coluna 'grupo_regional' adicionada.

    Example:
        >>> df_enriched = enrich_with_regional_groups(df, radius_km=5, min_samples=2)
    """
    clf = RegionalGroupClassifier(radius_km=radius_km, min_samples=min_samples)
    return clf.fit_predict(df)

if __name__ == "__main__":
    # Exemplo de uso
    import polars as pl

    # Carrega dados de exemplo (substitua pelo caminho real do CSV)
    df = pl.read_csv(r"use_case\files\consumption_consolidated.csv")

    # Enriquece com grupos regionais
    df_enriched = enrich_with_regional_groups(df, radius_km=5, min_samples=2)
    print(df_enriched.select(["grupo_regional"]).unique())
    print(df_enriched.select(["latitude", "longitude", "grupo_regional"]).head())

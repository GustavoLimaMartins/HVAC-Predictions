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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

    Atributos pós-treino
    --------------------
    is_fitted      : bool — True após fit().
    n_clusters_    : int  — Número de zonas multi-ponto encontradas.
    n_noise_       : int  — Pontos isolados (sem vizinhos suficientes no raio).
    n_total_groups_: int  — Total de grupos = n_clusters_ + n_noise_.

    Como funciona internamente
    --------------------------
    1. Extrai combinações únicas de (latitude, longitude) sem nulos.
    2. Converte para radianos (requisito do sklearn com metric='haversine').
    3. DBSCAN agrupa pontos dentro de `radius_km` com ≥ `min_samples` vizinhos.
    4. Cada ponto isolado (label DBSCAN = -1) recebe um grupo único próprio,
       preservando a fidelidade regional climática de cada localização.
    5. Para coordenadas novas (não vistas no treino), um KNN 1-vizinho é usado
       como fallback, buscando o ponto treinado geograficamente mais próximo
       (seja cluster ou grupo isolado).

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
    ) -> None:
        super().__init__()

        self.radius_km = radius_km
        self.min_samples = min_samples

        # eps convertido para radianos: distância / raio médio da Terra
        self._eps_rad: float = radius_km / EARTH_RADIUS_KM

        self._dbscan: Optional[DBSCAN] = None
        self._knn: Optional[KNeighborsClassifier] = None
        self._unique_coords: Optional[pl.DataFrame] = None
        self._coord_labels: Optional[np.ndarray] = None

        self.n_clusters_: int = 0
        self.n_noise_: int = 0
        self.n_total_groups_: int = 0

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

        # Extrai coordenadas únicas válidas (ordenadas para determinismo)
        self._unique_coords = (
            df.select(["latitude", "longitude"])
            .drop_nulls()
            .unique()
            .sort(["latitude", "longitude"])
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
        self._coord_labels = self._dbscan.labels_.copy()

        self.n_clusters_ = len(set(self._coord_labels)) - (
            1 if -1 in self._coord_labels else 0
        )
        self.n_noise_ = int((self._coord_labels == -1).sum())

        # Cada ponto isolado recebe um grupo único próprio, preservando a
        # fidelidade regional climática de cada localização geográfica.
        noise_indices = np.where(self._coord_labels == -1)[0]
        for i, idx in enumerate(noise_indices):
            self._coord_labels[idx] = self.n_clusters_ + i

        self.n_total_groups_ = self.n_clusters_ + self.n_noise_

        # KNN 1-vizinho treinado sobre TODOS os pontos (clusters + isolados)
        # como fallback exclusivo para coordenadas não vistas no treino.
        self._knn = KNeighborsClassifier(
            n_neighbors=1,
            algorithm="ball_tree",
            metric="haversine",
        )
        self._knn.fit(coords_rad, self._coord_labels.tolist())

        self.is_fitted = True
        print(
            f"✓ DBSCAN concluído: {self.n_clusters_} zonas multi-ponto | "
            f"{self.n_noise_} grupos isolados | "
            f"{self.n_total_groups_} grupos no total "
            f"(raio={self.radius_km} km, min_samples={self.min_samples})"
        )
        return self

    # ── Predição ─────────────────────────────────────────────────────────────

    def predict(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Atribui grupo regional a cada linha do DataFrame.

        Estratégia por tipo de ponto:
        • Coordenada de cluster (grupo < n_clusters_) → rótulo direto do treino.
        • Coordenada isolada (grupo ≥ n_clusters_)    → rótulo único atribuído no fit().
        • Coordenada nova (não vista no treino)       → KNN 1-vizinho mais próximo.
        • Lat/lon nulo                                → recebe null.

        Pontos isolados mantêm seu grupo individual — nenhuma localização
        é forçada a um cluster geograficamente distante.

        Args:
            df: DataFrame com colunas 'latitude' e 'longitude'.

        Returns:
            DataFrame original com nova coluna 'grupo_regional' (Int32).

        Raises:
            ValueError: Se o modelo não foi treinado.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")

        self._validate_geo_columns(df)

        # Mapa completo: todos os pontos do treino → rótulo (sem label -1).
        train_lats = self._unique_coords["latitude"].to_list()
        train_lons = self._unique_coords["longitude"].to_list()
        coord_map: dict[tuple, int] = {
            (lat, lon): int(label)
            for (lat, lon), label in zip(
                zip(train_lats, train_lons), self._coord_labels
            )
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
                # Coordenada nova (não vista no treino) → resolvida via KNN
                labels.append(None)  # placeholder
                knn_idxs.append(i)
                knn_coords.append((lat, lon))

        # Resolve coordenadas novas via KNN Haversine
        if knn_idxs:
            coords_rad = np.radians(np.array(knn_coords))
            knn_labels = self._knn.predict(coords_rad)
            for idx, label in zip(knn_idxs, knn_labels):
                labels[idx] = int(label)

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

        groups: dict[int, list[tuple[float, float]]] = {}
        for (lat, lon), label in zip(zip(lats, lons), self._coord_labels):
            groups.setdefault(int(label), []).append((lat, lon))

        rows = [
            {
                "grupo_regional": k,
                "centroide_lat": float(np.mean([p[0] for p in pts])),
                "centroide_lon": float(np.mean([p[1] for p in pts])),
                "n_pontos_unicos": len(pts),
                "is_isolado": k >= self.n_clusters_,
            }
            for k, pts in sorted(groups.items())
        ]
        return pl.DataFrame(rows).with_columns(pl.col("grupo_regional").cast(pl.Int32))

    def plot_clusters(
        self,
        title: str = "Grupos Regionais — DBSCAN + Haversine",
        figsize: tuple = (12, 8),
        save_path: str = None,
    ) -> None:
        """
        Plota os clusters geográficos e pontos isolados (ruído pré-KNN).

        Cada cluster recebe uma cor distinta. Pontos classificados como ruído
        pelo DBSCAN (antes do fallback KNN) são exibidos em cinza com marcador
        'x' para facilitar a análise do parâmetro radius_km.
        Centróides de cada cluster são marcados com uma estrela.

        Args:
            title     : Título do gráfico.
            figsize   : Tamanho da figura (largura, altura) em polegadas.
            save_path : Se fornecido, salva a figura no caminho especificado
                        em vez de exibi-la.

        Raises:
            ValueError: Se o modelo não foi treinado.
        """
        if not self.is_fitted:
            raise ValueError("Modelo não treinado. Execute fit() primeiro.")

        lats = self._unique_coords["latitude"].to_numpy()
        lons = self._unique_coords["longitude"].to_numpy()
        labels = self._coord_labels

        cluster_ids = sorted(k for k in set(labels) if k < self.n_clusters_)
        isolated_ids = sorted(k for k in set(labels) if k >= self.n_clusters_)

        fig, ax = plt.subplots(figsize=figsize)

        # Paleta de cores para clusters multi-ponto
        if len(cluster_ids) <= 20:
            colors = cm.tab20.colors[:max(len(cluster_ids), 1)]
        else:
            colors = cm.hsv(np.linspace(0, 1, len(cluster_ids), endpoint=False))

        # Clusters multi-ponto: círculos + estrela no centróide
        for cluster_id, color in zip(cluster_ids, colors):
            mask = labels == cluster_id
            ax.scatter(
                lons[mask], lats[mask],
                c=[color], s=65, alpha=0.80, zorder=2,
                label=f"Grupo {cluster_id} ({mask.sum()} pts)",
            )
            ax.scatter(
                lons[mask].mean(), lats[mask].mean(),
                c=[color], s=220, marker="*", edgecolors="black",
                linewidths=0.8, zorder=4,
            )

        # Grupos isolados: losangos (◆) — cor única para não poluir a legenda
        if isolated_ids:
            iso_mask = np.isin(labels, isolated_ids)
            ax.scatter(
                lons[iso_mask], lats[iso_mask],
                c="#E07B39", s=55, marker="D", edgecolors="#7B3000",
                linewidths=0.7, alpha=0.75, zorder=3,
                label=f"Isolados — grupo único por local ({len(isolated_ids)} pts)",
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(
            f"{title}\n"
            f"raio={self.radius_km} km | min_samples={self.min_samples} | "
            f"{len(cluster_ids)} clusters | {len(isolated_ids)} grupos isolados"
        )
        ax.legend(loc="best", fontsize=8, framealpha=0.8)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"\u2713 Gráfico salvo em: {save_path}")
        else:
            plt.show()

        plt.close(fig)


if __name__ == "__main__":
    import polars as pl

    df = pl.read_csv(r"use_case\files\consumption_consolidated.csv")

    clf = RegionalGroupClassifier(radius_km=40, min_samples=2)
    df_enriched = clf.fit_predict(df)
    print(df_enriched.select(["grupo_regional"]).unique())
    print(df_enriched.select(["latitude", "longitude", "grupo_regional"]).head())
    print(clf.get_cluster_summary())
    clf.plot_clusters()

"""
REST API Server para HVAC Predictions
======================================

FastAPI server que expõe endpoints para predições com normalização automática.

**Deploy (Railway):**
    A porta é lida automaticamente da variável de ambiente PORT injetada pelo Railway.
    Localmente, define PORT=8000 ou deixa o padrão.

**Uso local:**
    >>> python -m tools.api_server
    >>> uvicorn tools.api_server:app --port 8000   # desenvolvimento

**Cliente (curl) — Railway:**
    >>> BASE="https://SEU-DOMINIO.up.railway.app"
    >>> curl -X POST $BASE/predict \
    ...   -H "Content-Type: application/json" \
    ...   -d '{
    ...     "hora": 14,
    ...     "data": "2025-07-03",
    ...     "machine_type": "splitao",
    ...     "latitude": -23.88,
    ...     "longitude": -46.42,
    ...     "temperatura_c": 25.7,
    ...     "temperatura_percebida_c": 24.9,
    ...     "umidade_relativa_pct": 57.0,
    ...     "precipitacao_mm": 0.0,
    ...     "velocidade_vento_kmh": 21.4,
    ...     "pressao_superficial_hpa": 969.8,
    ...     "irradiancia_direta_wm2": 520.0,
    ...     "irradiancia_difusa_wm2": 180.0
    ...   }'

**Cliente Python — Railway:**
    >>> import requests
    >>> BASE = "https://SEU-DOMINIO.up.railway.app"
    >>> resp = requests.post(f"{BASE}/predict", json={
    ...     "hora": 14, "data": "2025-07-03", "machine_type": "splitao",
    ...     "latitude": -23.88, "longitude": -46.42,
    ...     "temperatura_c": 25.7, "temperatura_percebida_c": 24.9,
    ...     "umidade_relativa_pct": 57.0, "precipitacao_mm": 0.0,
    ...     "velocidade_vento_kmh": 21.4, "pressao_superficial_hpa": 969.8,
    ...     "irradiancia_direta_wm2": 520.0, "irradiancia_difusa_wm2": 180.0,
    ... })
    >>> print(resp.json())   # {"consumo_kwh": 6.32, "timestamp": "..."}

**Documentação Interativa:**
    - Swagger UI: https://SEU-DOMINIO.up.railway.app/docs
    - ReDoc:      https://SEU-DOMINIO.up.railway.app/redoc
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import condicional: relativo se rodado como módulo, absoluto se rodado direto
try:
    from .inference_runner import HVACDLInferenceAPI
except ImportError:
    from tools.inference_runner import HVACDLInferenceAPI

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

_logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)

# Porta injetada pelo Railway em produção; fallback local para 8000
_PORT = int(os.environ.get("PORT", 8000))

app = FastAPI(
    title="HVAC Predictions API",
    description="API REST para predições de consumo energético de sistemas HVAC",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS — permite requisições do domínio Railway e de clientes externos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Inicializa a API de inferência (carregada uma única vez)
_ROOT = Path(__file__).resolve().parent.parent
_ARTIFACT_PATH = _ROOT / "model" / "artifacts" / "dl_hvac" / "global"

_inference_api: Optional[HVACDLInferenceAPI] = None


# ══════════════════════════════════════════════════════════════════════════════
#  MODELOS PYDANTIC
# ══════════════════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    """Requisição de predição para um único registro."""
    
    hora: int = Field(..., ge=0, le=23, description="Hora do dia (0-23)")
    data: str = Field(..., description="Data no formato YYYY-MM-DD")
    machine_type: str = Field(..., description="Tipo de máquina (ex: splitao, split_hi-wall)")
    latitude: float = Field(..., ge=-90, le=90, description="Latitude em graus")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude em graus")
    temperatura_c: float = Field(..., description="Temperatura em °C")
    temperatura_percebida_c: float = Field(..., description="Temperatura percebida em °C")
    umidade_relativa_pct: float = Field(..., ge=0, le=100, description="Umidade relativa (%)")
    precipitacao_mm: float = Field(..., ge=0, description="Precipitação em mm")
    velocidade_vento_kmh: float = Field(..., ge=0, description="Velocidade do vento em km/h")
    pressao_superficial_hpa: float = Field(..., description="Pressão em hPa")
    irradiancia_direta_wm2: float = Field(..., ge=0, description="Irradiância direta normal em W/m²")
    irradiancia_difusa_wm2: float = Field(..., ge=0, description="Irradiância difusa horizontal em W/m²")

    class Config:
        json_schema_extra = {
            "example": {
                "hora": 14,
                "data": "2025-07-03",
                "machine_type": "splitao",
                "latitude": -23.88,
                "longitude": -46.42,
                "temperatura_c": 25.7,
                "temperatura_percebida_c": 24.9,
                "umidade_relativa_pct": 57.0,
                "precipitacao_mm": 0.0,
                "velocidade_vento_kmh": 21.4,
                "pressao_superficial_hpa": 969.8,
                "irradiancia_direta_wm2": 520.0,
                "irradiancia_difusa_wm2": 180.0,
            }
        }


class BatchPredictionRequest(BaseModel):
    """Requisição de predição em lote."""
    
    records: list[PredictionRequest] = Field(..., description="Lista de registros para predição")


class PredictionResponse(BaseModel):
    """Resposta de predição."""
    
    consumo_kwh: float = Field(..., description="Consumo previsto em kWh")
    timestamp: str = Field(..., description="Timestamp da predição (ISO 8601)")


class BatchPredictionResponse(BaseModel):
    """Resposta de predição em lote."""
    
    predictions: list[float] = Field(..., description="Lista de predições em kWh")
    n_records: int = Field(..., description="Número de registros processados")
    timestamp: str = Field(..., description="Timestamp da predição (ISO 8601)")


class HealthResponse(BaseModel):
    """Resposta de health check."""
    
    status: str = Field(..., description="Status do serviço")
    artifact_path: str = Field(..., description="Caminho do artefato carregado")
    model_loaded: bool = Field(..., description="Se o modelo está carregado")


class ErrorResponse(BaseModel):
    """Resposta de erro."""
    
    error: str = Field(..., description="Mensagem de erro")
    detail: Optional[str] = Field(None, description="Detalhes adicionais")


# ══════════════════════════════════════════════════════════════════════════════
#  INICIALIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Carrega o modelo ao iniciar o servidor."""
    global _inference_api
    
    try:
        _logger.info(f"Carregando artefato de {_ARTIFACT_PATH} ...")
        _inference_api = HVACDLInferenceAPI(_ARTIFACT_PATH)
        _logger.info("✔ Modelo carregado com sucesso")
    except Exception as e:
        _logger.error(f"✗ Erro ao carregar modelo: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup ao desligar o servidor."""
    _logger.info("Desligando servidor...")


# ══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check — verifica se o serviço está operacional.
    
    Returns:
        HealthResponse com status do serviço
    """
    return HealthResponse(
        status="healthy" if _inference_api is not None else "unhealthy",
        artifact_path=str(_ARTIFACT_PATH),
        model_loaded=_inference_api is not None,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Dados de entrada inválidos"},
        500: {"model": ErrorResponse, "description": "Erro interno do servidor"},
    },
    tags=["Prediction"],
)
async def predict_single(request: PredictionRequest):
    """
    Realiza predição para um único registro.
    
    Args:
        request: Dados de entrada com as 13 features obrigatórias
        
    Returns:
        PredictionResponse com consumo previsto em kWh
        
    Raises:
        HTTPException: Se houver erro na predição
    """
    if _inference_api is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado",
        )
    
    try:
        # Cria DataFrame a partir da requisição
        df = pl.DataFrame({
            "hora": [request.hora],
            "data": [request.data],
            "machine_type": [request.machine_type],
            "latitude": [request.latitude],
            "longitude": [request.longitude],
            "Temperatura_C": [request.temperatura_c],
            "Temperatura_Percebida_C": [request.temperatura_percebida_c],
            "Umidade_Relativa_%": [request.umidade_relativa_pct],
            "Precipitacao_mm": [request.precipitacao_mm],
            "Velocidade_Vento_kmh": [request.velocidade_vento_kmh],
            "Pressao_Superficial_hPa": [request.pressao_superficial_hpa],
            "Irradiancia_Direta_Wm2": [request.irradiancia_direta_wm2],
            "Irradiancia_Difusa_Wm2": [request.irradiancia_difusa_wm2],
        }).with_columns(pl.col("data").cast(pl.Date))
        
        # Prediz
        pred = _inference_api.predict(df)[0]
        
        return PredictionResponse(
            consumo_kwh=float(pred),
            timestamp=datetime.now().isoformat(),
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dados inválidos: {str(e)}",
        )
    except Exception as e:
        _logger.error(f"Erro na predição: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predição: {str(e)}",
        )


@app.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Dados de entrada inválidos"},
        500: {"model": ErrorResponse, "description": "Erro interno do servidor"},
    },
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest):
    """
    Realiza predições em lote.
    
    Args:
        request: Lista de registros para predição
        
    Returns:
        BatchPredictionResponse com todas as predições
        
    Raises:
        HTTPException: Se houver erro na predição
    """
    if _inference_api is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modelo não carregado",
        )
    
    if not request.records:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Lista de registros vazia",
        )
    
    try:
        # Cria DataFrame a partir da lista de requisições
        data_dict = {
            "hora": [r.hora for r in request.records],
            "data": [r.data for r in request.records],
            "machine_type": [r.machine_type for r in request.records],
            "latitude": [r.latitude for r in request.records],
            "longitude": [r.longitude for r in request.records],
            "Temperatura_C": [r.temperatura_c for r in request.records],
            "Temperatura_Percebida_C": [r.temperatura_percebida_c for r in request.records],
            "Umidade_Relativa_%": [r.umidade_relativa_pct for r in request.records],
            "Precipitacao_mm": [r.precipitacao_mm for r in request.records],
            "Velocidade_Vento_kmh": [r.velocidade_vento_kmh for r in request.records],
            "Pressao_Superficial_hPa": [r.pressao_superficial_hpa for r in request.records],
            "Irradiancia_Direta_Wm2": [r.irradiancia_direta_wm2 for r in request.records],
            "Irradiancia_Difusa_Wm2": [r.irradiancia_difusa_wm2 for r in request.records],
        }
        
        df = pl.DataFrame(data_dict).with_columns(pl.col("data").cast(pl.Date))
        
        # Prediz
        preds = _inference_api.predict(df)
        
        return BatchPredictionResponse(
            predictions=[float(p) for p in preds],
            n_records=len(request.records),
            timestamp=datetime.now().isoformat(),
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Dados inválidos: {str(e)}",
        )
    except Exception as e:
        _logger.error(f"Erro na predição em lote: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro ao processar predições em lote: {str(e)}",
        )


@app.get("/", tags=["Info"])
async def root():
    """Informações gerais da API."""
    return {
        "name": "HVAC Predictions API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "predict": "POST /predict",
            "predict_batch": "POST /predict_batch",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXECUÇÃO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import uvicorn

    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    # ── Modo: --test  →  cliente Railway | padrão → inicia servidor ──────────
    # Uso:  python -m tools.api_server --test [URL]
    #   URL opcional — sobrescreve RAILWAY_URL do ambiente.
    #   Exemplo: python -m tools.api_server --test https://meu-app.up.railway.app

    if "--test" in sys.argv:
        import json
        import requests as _req
        from dotenv import load_dotenv

        # Carrega .env da raiz do projeto (dois níveis acima de tools/)
        load_dotenv(root_dir / ".env")

        idx = sys.argv.index("--test")
        _base = (
            sys.argv[idx + 1]
            if idx + 1 < len(sys.argv) and not sys.argv[idx + 1].startswith("-")
            else os.environ.get("RAILWAY_URL", "http://localhost:8000")
        ).rstrip("/")

        W = 72
        print("\n" + "=" * W)
        print("  TESTE DE ENDPOINTS — Railway")
        print("=" * W)
        print(f"  Base URL: {_base}\n")

        _PAYLOAD = {
            "hora": 14,
            "data": "2025-07-03",
            "machine_type": "splitao",
            "latitude": -23.883839,
            "longitude": -46.4200317682745,
            "temperatura_c": 25.7,
            "temperatura_percebida_c": 24.9,
            "umidade_relativa_pct": 57.0,
            "precipitacao_mm": 0.0,
            "velocidade_vento_kmh": 21.4,
            "pressao_superficial_hpa": 969.8,
            "irradiancia_direta_wm2": 520.0,
            "irradiancia_difusa_wm2": 180.0,
        }

        def _req_test(method: str, path: str, **kwargs):
            url = f"{_base}{path}"
            try:
                resp = _req.request(method, url, timeout=30, **kwargs)
                ok   = resp.status_code < 300
                tag  = "OK " if ok else "ERR"
                print(f"  [{tag}] {method} {path}  →  HTTP {resp.status_code}")
                return resp if ok else None
            except _req.exceptions.RequestException as exc:
                print(f"  [ERR] {method} {path}  →  {exc}")
                return None

        # ── GET / ─────────────────────────────────────────────────────────────
        print("─" * W)
        print("  [1] Root info")
        print("─" * W)
        r = _req_test("GET", "/")
        if r:
            print(f"  {json.dumps(r.json(), ensure_ascii=False, indent=4)}")

        # ── GET /health ───────────────────────────────────────────────────────
        print("\n" + "─" * W)
        print("  [2] Health check")
        print("─" * W)
        r = _req_test("GET", "/health")
        if r:
            h = r.json()
            print(f"  status       : {h.get('status')}")
            print(f"  model_loaded : {h.get('model_loaded')}")
            print(f"  artifact_path: {h.get('artifact_path')}")

        # ── POST /predict ─────────────────────────────────────────────────────
        print("\n" + "─" * W)
        print("  [3] Predição simples  (/predict)")
        print("─" * W)
        print(f"  Payload: hora={_PAYLOAD['hora']}  data={_PAYLOAD['data']}"
              f"  machine_type={_PAYLOAD['machine_type']!r}")
        r = _req_test("POST", "/predict", json=_PAYLOAD)
        if r:
            body = r.json()
            print(f"  consumo_kwh : {body.get('consumo_kwh'):.4f} kWh")
            print(f"  timestamp   : {body.get('timestamp')}")

        # ── POST /predict_batch ───────────────────────────────────────────────
        print("\n" + "─" * W)
        print("  [4] Predição em lote  (/predict_batch)  — 3 registros")
        print("─" * W)
        _batch_variants = [
            {**_PAYLOAD, "hora": h, "temperatura_c": t}
            for h, t in [(8, 18.0), (14, 25.7), (20, 22.3)]
        ]
        r = _req_test("POST", "/predict_batch", json={"records": _batch_variants})
        if r:
            body = r.json()
            preds = body.get("predictions", [])
            print(f"  n_records : {body.get('n_records')}")
            for i, (v, p) in enumerate(zip(_batch_variants, preds)):
                print(f"  [{i}] hora={v['hora']:>2}  Temp={v['temperatura_c']}°C"
                      f"  →  {p:.4f} kWh")

        print("\n" + "=" * W + "\n")

    else:
        uvicorn.run(
            "tools.api_server:app",
            host="0.0.0.0",
            port=_PORT,
            reload=False,
            log_level="info",
        )

# HVAC Predictions - Deploy Guide

## Local Development

```bash
# Ativar venv
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Rodar API
python -m uvicorn tools.api_server:app --reload --port 8000

# Acessar Swagger UI
http://localhost:8000/docs
```

---

## Railway Deployment

### 1. Criar conta no Railway
[https://railway.app](https://railway.app)

### 2. Conectar repositório Git
```bash
# Fazer push do código para GitHub
git add .
git commit -m "Deploy Railway"
git push origin main
```

### 3. Deploy via CLI (alternativa)
```bash
# Instalar Railway CLI
npm i -g @railway/cli

# Login
railway login

# Deploy
railway up
```

### 4. Variáveis de Ambiente (se necessário)
No painel do Railway, adicione:
```
PORT=8000
ARTIFACT_PATH=model/artifacts/dl_hvac/global
```

---

## Endpoints Disponíveis

**Base URL (Local):** `http://localhost:8000`  
**Base URL (Railway):** `https://seu-app.railway.app`

### Health Check
```bash
GET /health
```

### Predição Single
```bash
POST /predict
Content-Type: application/json

{
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
  "pressao_superficial_hpa": 969.8
}
```

### Predição Batch
```bash
POST /predict_batch
Content-Type: application/json

{
  "records": [
    { ...record1... },
    { ...record2... }
  ]
}
```

---

## Documentação Interativa

- **Swagger UI:** `/docs`
- **ReDoc:** `/redoc`
- **OpenAPI JSON:** `/openapi.json`

---

## Troubleshooting

### Erro: "Modelo não encontrado"
Certifique-se que `model/artifacts/` está no repositório ou configure Railway para clonar:
```bash
railway link <projectId>
railway up
```

### Timeout no Railway
Se a predição demora > 30s, optimize:
1. Usar batch predictions em vez de single
2. Adicionar cache de modelos
3. Aumentar plano do Railway

### Tamanho do build > 500MB
Excluir arquivos desnecessários em `.railwayignore`:
```
__pycache__/
*.pyc
.pytest_cache/
```

---

## Monitoramento

### Logs locais
```bash
python -m uvicorn tools.api_server:app --log-level debug
```

### Logs Railway
Painel do Railway → Logs → Visualizar em tempo real

---

## Próximos Passos

- [ ] Adicionar autenticação (Bearer token)
- [ ] Implementar rate limiting
- [ ] Adicionar cache Redis
- [ ] Monitoramento com Sentry
- [ ] CI/CD com GitHub Actions

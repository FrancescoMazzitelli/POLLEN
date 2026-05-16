# POLLEN — Plan Orchestration with LLM-based ENgine

POLLEN is an AI-powered API orchestration system that translates natural language queries into executable multi-step API execution plans. It combines a semantic service catalog, an LLM-based planner with guided JSON decoding, a multi-stage retrieval pipeline, and a distributed model inference backend running on a Petals swarm of Raspberry Pis.

## Architecture Overview

The system is composed of 15+ Docker services orchestrated via Docker Compose, divided into three layers:

- **Data Layer**: MongoDB + Qdrant + Redis for service metadata, vector search, and health status
- **Business Logic Layer**: Control Unit (planner/executor), API Importer, Catalog Gateway (semantic search), Healthcheck Service
- **Infrastructure Layer**: Consul (service discovery), Microcks (mock API server), Petals (distributed LLM inference)

## Getting Started

To automatically deploy and configure all the needed services you can just run:
```bash
sudo docker compose -f Compose.yaml -f docker-compose.pi.yml up -d
```
Keep in mind that you first need to configure your .env file
---

## 1. Control Unit (`control-unit/`)

**Port: 5500** — Flask + Flask-RESTX application serving as the central orchestrator.

### `POST /api/control/invoke`
Main entry point. Accepts `{"input": "user query"}` and returns an execution plan + results.

### Components

#### `service/controlService.py` — Controller
The core planning-execution pipeline:
1. **Service Discovery** — Queries Consul for registered service instances
2. **Semantic Retrieval** — Sends query to the Catalog Gateway (`/index/search`) for two-stage retrieval
3. **LLM Planning** — Sends system prompt + catalog to Ollama/Petals with structured output (guided JSON decoding). The model produces a JSON execution plan with a `reasoning` field and a `tasks` array
4. **Plan Validation** — Post-parse validation via `PlanValidator`:
   - Checks required fields, valid operations (`GET/POST/PUT/DELETE/SQL`)
   - Validates service IDs against available catalog entries
   - Detects unresolved path parameter placeholders
   - Schema enforcement: input keys must match documented request schemas from the catalog
   - HC-12 rule (MOCK mode only): GET URLs must not combine multiple query params in AND
5. **Auto-fix** — Attempts to correct common LLM mistakes (wrong service_id, missing base URL)
6. **Execution** — Asynchronously executes tasks via `aiohttp`, resolving `{{...}}` JMESPath placeholders from prior task results
7. **SQL Processing** — In-process DuckDB execution for SQL tasks that join/aggregate prior results

#### `service/designerService.py` — Designer
Fallback triage agent. When the planner produces an empty plan (`tasks=[]`):
- Classifies the reason as `OUT_OF_DOMAIN`, `AMBIGUOUS`, `INVALID`, or `UNKNOWN`
- If `OUT_OF_DOMAIN`, designs a conceptual REST API contract suggesting what service would fill the gap
- Uses a single LLM call with `anyOf(null | contract)` guided decoding

#### `service/discoveryService.py` — Discovery
Interfaces with HashiCorp Consul to retrieve registered services and their metadata.

### Prompt Engineering
The system prompt (1600+ lines) includes:
- 6 worked examples covering distinct structural patterns (single GET, JMESPath chaining, multi-zone joins, SQL aggregation, set difference, qualitative sorting)
- 12 hard constraints (HC-1 through HC-12) including closed-world assumption, no unresolved placeholders, no AND-combination of query params in MOCK mode
- JMESPath reference with 3 canonical patterns
- SQL reference with DuckDB dialect and common patterns
- Chain-of-draft reasoning protocol

---

## 2. Catalog Gateway (`db-gateway/`)

**Port: 5000** — Flask application serving as the unified database gateway for MongoDB + Qdrant.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Healthcheck (waits for ML model loading) |
| `/service` | POST | Create/update a service in MongoDB + index in Qdrant |
| `/services` | GET | List all services |
| `/services/<id>` | GET/DELETE | Get or delete a specific service |
| `/services/<id>/schemas` | PATCH | Update schemas (called by API Importer) |
| `/index/search` | POST | Two-stage semantic search |
| `/index/reindex` | POST | Rebuild the Stage 1 index |

### Two-Stage Retrieval Pipeline (`/index/search`)

**Stage 0 — LLM Query Decomposition (optional)**
- Uses an LLM (default: same model as planner) to decompose a composite user query into 1–4 atomic sub-queries
- Each sub-query targets a single information type (e.g., "parking spots", "air quality measurements")
- Fallback: original query as single sub-query on any error

**Stage 1 — Bi-encoder Semantic Retrieval**
- Model: [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) (SentenceTransformers)
- Index: Qdrant collection `services_index` — one vector per service, computed from `description` + `generated_description`
- Top-K = 7 per sub-query, merged with max-score deduplication
- Uses asymmetric prefixing: `query:` for queries, `passage:` for documents

**Stage 2 — Cross-encoder Reranking**
- Model: [BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base) via ONNX runtime (CPU-optimized)
- Single-batch inference across all (query, enriched_text) pairs
- Enriched text combines: capability description, parameter names, response schema fields, request schema fields
- Virtual boost (+5.0) applied to GET list endpoints (no path params) for ordinal ranking
- Top-4 endpoints per service retained

**Ranking — Reciprocal Rank Fusion (RRF)**
- Combines Stage 1 ranking (cosine similarity) and Stage 2 ranking (logit) with K=60 (Cormack et al., 2009)
- Zero hyperparameters — canonical IR technique

**Token Budget**
- Max 5000 tokens for the full context sent to the LLM
- Graceful trimming: endpoints per service are progressively reduced (from top-N down to top-1) until the service fits in the budget

### ML Model Loading
- Embedding model loaded at startup (PyTorch, CPU)
- Cross-encoder loaded via `optimum.onnxruntime` for 2–4x faster CPU inference

---

## 3. API Importer (`api-importer/`)

**Port: 7500** — Flask + Flask-RESTX application for importing OpenAPI specifications.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/importer/import` | POST | Import ALL providers from apis.guru |
| `/api/importer/import/<key>` | POST | Import a single provider from apis.guru |
| `/api/importer/import/url` | POST | Import from a public OpenAPI URL |
| `/api/importer/import/file` | POST | Upload an OpenAPI YAML/JSON file |
| `/api/importer/enrich` | POST | Enrich existing services with response/request schemas |

### `service/importService.py` — Service

**Schema Extraction Pipeline**:
1. Downloads OpenAPI YAML via `requests` + `prance` (resolves all `$ref` references)
2. Extracts **capabilities**: endpoint descriptions used for semantic search
3. Extracts **endpoints**: full URLs for each HTTP operation
4. Extracts **response schemas**: compact field-type strings (e.g., `{id:int, location:str, status:enum}`) from success responses (200/201). Fallback to examples when schema is absent
5. Extracts **request schemas**: input body fields for POST/PUT with `*` marking required fields
6. Extracts **parameters**: query/path parameters with enum values in compact format
7. Generates **generated_description**: synthetic description combining title + endpoint summaries + enum values for improved semantic retrieval
8. Flattens nested objects into dotted keys (e.g., `address.street`)

**Registration** (in parallel via ThreadPoolExecutor):
- **Redis**: Health status via healthcheck-service
- **Consul**: Service registration with HTTP health check (10s interval)
- **MongoDB**: Full document via catalog-gateway

---

## 4. Petals Proxy (`petals-proxy/`)

**Port: 11434** — FastAPI service translating Ollama-compatible `/api/chat` requests to OpenAI-compatible `/v1/chat/completions` for the Petals inference server.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/chat` | POST | Ollama-compatible chat completion |
| `/api/tags` | GET | List available models |
| `/health` | GET | Healthcheck |

### Structured Output Enforcement
When a `format` schema is provided (JSON guided decoding):
1. Injects a system prompt: "You must respond with ONLY valid JSON matching the schema below"
2. Calls Petals with standard OpenAI API format
3. Extracts JSON from the response (handles markdown fences)
4. Validates against the provided JSON schema via `jsonschema`
5. Retries up to `MAX_RETRIES` (default: 3) with progressively stricter prompts on failure

---

## 5. Petals Inference (`petals-inference/`)

**Port: 31337** — Docker container running the Petals decentralized inference server.

- Uses `ghcr.io/bigscience-workshop/petals:main`
- Can run in **standalone mode** (single node, no swarm) when `INITIAL_PEERS` is empty
- Connects to a **distributed swarm** of Raspberry Pis when `INITIAL_PEERS` is configured
- Model: `meta-llama/Meta-Llama-3.1-8B-Instruct` (requires `HF_TOKEN`)
- Persistent HuggingFace cache volume

---

## 6. Redis Connector / Healthcheck Service (`redis-connector-quarkus/`)

**Port: 5600** — Quarkus (Java 21, native binary) service for service health status management.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/status/{key}` | GET | Get a service's health status |
| `/status/register` | POST | Register/update service status in Redis |

### `StatusService`
Uses Quarkus Redis Client to perform key-value operations on a Redis Stack instance. The service is compiled to a native binary via GraalVM Mandrel for fast startup and low memory footprint.

---

## 7. Supporting Infrastructure (Docker Compose)

### `registry` — HashiCorp Consul
- Service discovery and health checking
- All services register with Consul upon startup
- Health checks at 10s intervals; deregister after 30s of failure

### `catalog-data` — MongoDB 8.0
- Primary document store for all service metadata (capabilities, endpoints, schemas)
- Persistent volume: `mongo-data`

### `catalog-vector` — Qdrant
- Vector database for semantic search
- Two collections: `services` (per-endpoint Stage 2) and `services_index` (per-service Stage 1)
- Vector dimension: 1024 (Qwen3-Embedding), distance: Cosine
- Persistent volume: `qdrant-data`

### `catalog-gui` — Mongo Express
- Web UI at port 27018 for browsing MongoDB contents

### `healthcheck-catalog` — Redis Stack 7.2
- In-memory store for service health status
- Web UI at port 6380 (RedisInsight)

### `mock-server` — Microcks
- Mock API server for the Smart City domain APIs
- 682 services, 34873 endpoints
- Uses custom SCRIPT dispatchers for POST /register and GET list endpoints

### `yaml-preprocessor` — Python YAML fixer
- One-shot container that validates and fixes OpenAPI YAML files before Microcks import
- Runs `fix_yaml.py` from the scripts directory

### `mock-deployer` — Microcks CLI
- Orchestrates the full mock deployment pipeline:
  1. Imports all YAML files into Microcks
  2. Patches dispatchers (custom Groovy scripts for dynamic responses)
  3. Invokes POST /register on each service
  4. Triggers API schema enrichment via API Importer

---

## 8. Test & Evaluation (`test_validation/`)

### `testRunner.py` (v4.1)
Comprehensive test runner with multi-layer evaluation:

**4-Layer Oracle Evaluation** (for executable queries):
- **L1 — Plan**: Compares generated tasks against oracle ground truth (method + path). Reports precision, recall, F1, TP/FP/FN
- **L2 — Execution**: Checks HTTP status codes (2xx = SUCCESS)
- **L3 — Chaining**: Verifies `{{...}}` placeholders in URLs are resolved (reads `url_resolved` field)
- **L4 — Schema**: Validates POST/PUT body fields are non-empty

**Out-of-Plan Evaluation** (for queries that should produce empty plans):
- Evaluates `empty_plan_category` returned by the Designer against expected category from CSV
- Validates `suggested_api_contracts` coherence (HC-2, HC-3, HC-5)

**Metrics**: Micro F1, Macro F1, Pass@1, Pass@K, per-category accuracy, latency stats, noise robustness breakdown

### `plan_validator.py`
Standalone validation script that compares execution plans against CSV-based oracles (ground truth). Computes precision, recall, F1, Jaccard, accuracy, coverage, overprediction/underprediction rates.

### `submitter.py`
Legacy batch submission script that sends queries to the control unit and logs results.

---

## 9. Deployment & Configuration

### `deploy.sh`
Imports each OpenAPI YAML from `MOCKS-smart-city/apis/` into Microcks, then patches two dispatchers per service:
- **POST /register**: Custom Groovy script that dynamically registers the service in Consul + MongoDB
- **GET list**: Custom Groovy script providing dynamic list responses

### `register.sh`
Calls `POST /register` on each deployed mock service to trigger the registration pipeline.

### `Compose.yaml`
Full Docker Compose stack with 15+ services, all dependency-ordered with health checks. Environment variables control backend mode (`MOCK`/`REAL`), model selection, and API authentication headers.

---

## 10. Configuration

| Env Variable | Default | Service | Purpose |
|---|---|---|---|
| `BACKEND_MODE` | `MOCK` | control-unit | `MOCK` for Microcks, `REAL` for production APIs |
| `PETALS_MODEL` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | multiple | LLM model for planning |
| `USE_LLM_DECOMPOSITION` | `true` | db-gateway | Enable LLM query decomposition in Stage 0 |
| `HF_TOKEN` | — | petals-inference | HuggingFace token for gated models |
| `INITIAL_PEERS` | — | petals-inference | Petals swarm peer multiaddrs |
| `API_AUTH_HEADERS` | — | control-unit | JSON object of auth headers for REAL mode |
| `PROXY_MAX_RETRIES` | `3` | petals-proxy | Max retries for structured output |
| `PROXY_TIMEOUT` | `180` | petals-proxy | Request timeout |

---

## 11. Datasets (`datasets/`)

Sample OpenAPI specifications:
- `spotify.json` — Spotify Web API
- `tmdb.json` — The Movie Database API

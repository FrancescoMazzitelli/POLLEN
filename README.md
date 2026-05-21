# POLLEN: Prompt-based Orchestration of LLM-decomposed services over Edge Nodes

POLLEN is an AI-powered API orchestration system that translates natural language queries into executable multi-step API execution plans. It combines a semantic service catalog, an LLM-based planner with guided JSON decoding, a multi-stage retrieval pipeline, and a distributed model inference backend running on a llama.cpp RPC cluster of Raspberry Pis.

## Architecture Overview

The system is composed of 15+ Docker services orchestrated via Docker Compose, divided into three layers:

- **Data Layer**: MongoDB + Qdrant + Redis for service metadata, vector search, and health status
- **Business Logic Layer**: Control Unit (planner/executor), API Importer, Catalog Gateway (semantic search), Healthcheck Service
- **Infrastructure Layer**: Consul (service discovery), Microcks (mock API server), llama.cpp RPC (distributed LLM inference)

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
3. **LLM Planning** — Sends system prompt + catalog to the llama.cpp RPC proxy (`localhost:52415`) with structured output (guided JSON decoding). The model produces a JSON execution plan with a `reasoning` field and a `tasks` array
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

## 4. Distributed LLM Inference — llama.cpp RPC Cluster

**Exposed API port: 52415** (mapped externally as `192.168.241.2:2131` per master, `2132` per worker, etc.)

The inference backend runs across a cluster of Raspberry Pis using llama.cpp's native RPC mechanism. A single **master** node runs `llama-server` + nginx; **worker** nodes run `rpc-server` to share compute and RAM.

Due to WiFi client isolation, nodes cannot communicate directly over the LAN. Traffic between master and workers passes through **SSH reverse tunnels** (port 22, always open).

### Architecture

```
                    ┌──────────────────────┐
                    │  External client     │
                    │  192.168.241.2:2131  │
                    └────────┬─────────────┘
                             │
                    ┌────────▼─────────────┐
                    │   nginx (master)     │
                    │   :52415 → :11434    │
                    └────────┬─────────────┘
                             │
                    ┌────────▼─────────────┐
                    │  llama-server        │
                    │  (inference master)  │
                    │  :11434              │
                    └────────┬─────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼─────┐ ┌──────▼──────┐ ┌─────▼────────┐
     │ SSH tunnel   │ │ SSH tunnel  │ │ SSH tunnel   │
     │ 127.0.0.1    │ │ 127.0.0.1   │ │ 127.0.0.1    │
     │ :50164       │ │ :50165      │ │ :50166       │
     └──────┬───────┘ └──────┬──────┘ └──────┬───────┘
            │                │                │
     ┌──────▼───────┐ ┌──────▼──────┐ ┌──────▼───────┐
     │ rpc-server   │ │ rpc-server  │ │ rpc-server   │
     │ (worker 1)   │ │ (worker 2)  │ │ (worker N)   │
     │ :50052       │ │ :50052      │ │ :50052       │
     └──────────────┘ └─────────────┘ └──────────────┘
```

Each node loads a portion of the model layers via RPC. Workers are reachable through SSH reverse tunnels: each worker opens an outbound SSH connection to the master and exposes its `rpc-server` on a unique localhost port (e.g. `50164` for IP `.164`).

### Prerequisites — SSH Key Setup

Before installing, each worker must be able to SSH to the master **without password**:

```bash
# On each worker:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -p 2121 pi@192.168.241.2
```

Test it:

```bash
ssh -p 2121 pi@192.168.241.2 echo ok    # must print "ok" without asking password
```

### Full Node Deployment

**Step 1 — Master:**

```bash
# Copy script & install
scp -P 2121 ./llama_pi.sh pi@192.168.241.2:~/
ssh -p 2121 pi@192.168.241.2
./llama_pi.sh install master
sudo systemctl start llama-rpc
```

**Step 2 — Each worker (first, set up SSH key to master):**

```bash
# On each worker — one-time SSH key setup:
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -p 2121 pi@192.168.241.2           # inserisci password master un'ultima volta
ssh -p 2121 pi@192.168.241.2 echo ok            # deve funzionare senza password

# Then install & start:
scp -P 2122 ./llama_pi.sh pi@192.168.241.2:~/   # 2122, 2123, ... (worker SSH port)
ssh -p 2122 pi@192.168.241.2
./llama_pi.sh install worker
sudo systemctl start llama-rpc
./llama_pi.sh tunnel 192.168.241.2 2121           # SSH tunnel + auto-registration
```

The `tunnel` command does three things automatically:
1. Calculates a unique tunnel port from the worker's IP (`50000 + last octet`)
2. Creates and starts `llama-tunnel.service` (persistent SSH reverse tunnel)
3. Registers the worker on the master via SSH as `127.0.0.1:<tunnel-port>`

**Step 3 — Load a model & start inference (on master):**

```bash
ssh -p 2121 pi@192.168.241.2
./llama_pi.sh load https://huggingface.co/bartowski/...gguf
sudo systemctl restart llama-server
```

### Checking Node Connectivity

**From the master — verify tunnel ports are listening:**

```bash
ss -tlnp | grep 5016
```

Each worker appears as `127.0.0.1:50164`, `127.0.0.1:50165`, etc. — one per node.

**From the master — check registered workers:**

```bash
./llama_pi.sh list-workers
```

**Overall cluster status:**

```bash
~/llama-status.sh
```

Shows running services, model name, connected workers, and common commands.

**Test inference from outside:**

```bash
curl http://192.168.241.2:2131/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "max_tokens": 200
  }'
```

### Adding a New Worker Later

```bash
# First, set up SSH key (one-time):
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -p 2121 pi@192.168.241.2
ssh -p 2121 pi@192.168.241.2 echo ok

# Then install & tunnel:
scp -P <ssh-port> ./llama_pi.sh pi@192.168.241.2:~/
ssh -p <ssh-port> pi@192.168.241.2
./llama_pi.sh install worker
sudo systemctl start llama-rpc
./llama_pi.sh tunnel 192.168.241.2 2121     # auto-registers on master
```

Then on the master, restart inference to include the new worker:

```bash
ssh -p 2121 pi@192.168.241.2
sudo systemctl restart llama-server
```

### Removing a Worker

```bash
# On the master:
ssh -p 2121 pi@192.168.241.2
./llama_pi.sh remove-worker 127.0.0.1:50164
sudo systemctl restart llama-server

# On that worker (tear down tunnel):
./llama_pi.sh clean-tunnel
```

---

## 5. Inference API — nginx + llama-server

**Port: 52415** — OpenAI-compatible `/v1/chat/completions` endpoint served through nginx, forwarding to `llama-server` on port 11434.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completion (OpenAI-compatible) |

The Control Unit sends requests to nginx on port 52415 (externally `192.168.241.2:2131`), which forwards to `llama-server` on port 11434. `llama-server` distributes inference across RPC workers via their SSH tunnel endpoints.

### Supported Models

Any GGUF model from HuggingFace can be loaded dynamically:

```bash
./llama_pi.sh load <hf-gguf-url>
```

The model is downloaded, validated (GGUF magic bytes), and configured in `/etc/llama-server.env`. `llama-server` restarts automatically after loading.

### Testing

```bash
curl http://192.168.241.2:2131/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "max_tokens": 200
  }'
```

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

### `Compose.yaml`
Full Docker Compose stack with 15+ services, all dependency-ordered with health checks. Environment variables control backend mode (`MOCK`/`REAL`), model selection, and API authentication headers.

---

## 10. Configuration

| Env Variable | Default | Service | Purpose |
|---|---|---|---|
| `BACKEND_MODE` | `MOCK` | control-unit | `MOCK` for Microcks, `REAL` for production APIs |
| `LLAMA_MODEL` | — | llama_pi.sh | GGUF model name for inference |
| `PUBLIC_PORT` | `52415` | nginx (master) | External-facing API port |
| `RPC_PORT` | `50052` | rpc-server | Inter-node RPC communication |
| `API_PORT` | `11434` | llama-server | Local inference API port |
| `THREADS` | `4` | llama-server | CPU threads per node |
| `USE_LLM_DECOMPOSITION` | `true` | db-gateway | Enable LLM query decomposition in Stage 0 |
| `API_AUTH_HEADERS` | — | control-unit | JSON object of auth headers for REAL mode |

---

## 11. Datasets (`datasets/`)

Sample OpenAPI specifications:
- `spotify.json` — Spotify Web API
- `tmdb.json` — The Movie Database API

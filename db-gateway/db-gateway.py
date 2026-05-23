from flask import Flask, request, jsonify
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from bson.json_util import dumps
from cheroot.wsgi import Server as WSGIServer
import uuid
import os
import sys
import logging
import json
import signal
import re
import time
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
from transformers import AutoTokenizer


def handle_sigterm(*args):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

log_file_path = "test.txt"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='a', encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("app")

app = Flask(__name__)

MONGO_USER = os.environ.get("MONGO_USER", "admin")
MONGO_PASS = os.environ.get("MONGO_PASS", "admin")
MONGO_HOST = os.environ.get("MONGO_HOST", "catalog-data")
MONGO_PORT = os.environ.get("MONGO_PORT", "27017")
MONGO_DB   = os.environ.get("MONGO_DB", "microcks")
MONGO_URI  = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"

QDRANT_HOST             = os.environ.get("QDRANT_HOST", "catalog-vector")
QDRANT_PORT             = os.environ.get("QDRANT_PORT", "6333")
QDRANT_COLLECTION       = os.environ.get("QDRANT_COLLECTION", "services")
QDRANT_COLLECTION_INDEX = os.environ.get("QDRANT_COLLECTION_INDEX", "services_index")
QDRANT_URI              = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

# -- LLM query decomposition (optional) -----------------------------------------
# If USE_LLM_DECOMPOSITION=true, the query is decomposed into atomic
# sub-queries via an LLM before Stage 1.
# Supports two backends:
#   - "llamacpp" (default): OpenAI-compatible API (Distributed-Llama / llama.cpp)
#   - "ollama":             Ollama /api/chat endpoint
# Each sub-query produces its own top-K set; the deduplicated union becomes
# the input for Stage 2. On error, it falls back to the original query as a
# single sub-query -- the system never breaks.
USE_LLM_DECOMPOSITION     = os.environ.get("USE_LLM_DECOMPOSITION", "true").lower() == "true"
DECOMPOSITION_BACKEND     = os.environ.get("DECOMPOSITION_BACKEND", "llamacpp")
OLLAMA_URL                = os.environ.get("OLLAMA_URL", "http://localhost:52415")

# Model context limit from dllama-api --max-seq-len (see dllama_command_4.txt)
DECOMPOSITION_CTX_LENGTH  = int(os.environ.get("DECOMPOSITION_CTX_LENGTH", "4096"))
DECOMPOSITION_MAX_SUBQ    = int(os.environ.get("DECOMPOSITION_MAX_SUBQ", "4"))

mongo_client = MongoClient(MONGO_URI)
qdrant_client = QdrantClient(QDRANT_URI)

db         = mongo_client[MONGO_DB]
collection = db["services"]

is_server_ready = False
embedding_model = None
reranker_model  = None
tokenizer       = None


class ONNXCrossEncoder:
    """
    Custom wrapper for running a CrossEncoder in ONNX format.
    Massively optimizes CPU inference compared to standard PyTorch.
    """
    def __init__(self, model_name):
        logger.info(f"Converting and loading {model_name} in ONNX format (this may take a minute)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # The export=True parameter downloads the PyTorch model and converts it to ONNX on the fly
        self.model = ORTModelForSequenceClassification.from_pretrained(
            model_name, 
            export=True, 
            provider="CPUExecutionProvider"
        )
        logger.info(f"{model_name} successfully loaded in ONNX.")

    def predict(self, sentences, batch_size=4):
        all_scores = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            # max_length=512 prevents crashes on endpoints with anomalous descriptions
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            # Inference via ONNX runtime (C++)
            outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            logits = logits.detach().numpy()
            
            # Handle results (single or multiple)
            if logits.ndim == 0:
                all_scores.append(float(logits))
            else:
                all_scores.extend([float(x) for x in logits])
                
        return all_scores


def load_model():
    global embedding_model, reranker_model, tokenizer
    logger.info("Loading models...")
    
    # 1. Load Embedding Model (kept in standard PyTorch)
    embedding_model = SentenceTransformer(
        model_name_or_path='Qwen/Qwen3-Embedding-0.6B',
        device='cpu',
        trust_remote_code=True
    )
    tokenizer = embedding_model.tokenizer
    logger.info("Embedding model loaded.")
    
    # 2. Load Reranker Model (using the new ONNX engine)
    reranker_model = ONNXCrossEncoder('BAAI/bge-reranker-base')
    logger.info("Reranker model loaded.")


def embed_query(text: str) -> list:
    """
    Embedding for the user query at runtime.
    'query:' prefix -- Qwen3-Embedding is trained with query/passage asymmetry:
    using the same prefix for queries and documents degrades retrieval.
    """
    embedding = embedding_model.encode(
        f"query: {text}", convert_to_tensor=False, normalize_embeddings=True
    )
    return embedding.tolist()


def embed_passage(text: str) -> list:
    """
    Embedding for documents to be indexed (capabilities, descriptions).
    'passage:' prefix -- matches the document role in Qwen3-Embedding training.
    Use in all indexing phases.
    """
    embedding = embedding_model.encode(
        f"passage: {text}", convert_to_tensor=False, normalize_embeddings=True
    )
    return embedding.tolist()


def count_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return len(tokens)


def select_stage1_text(doc: dict) -> tuple[str, str]:
    """
    Selects text for the Stage 1 index with the following priority:
      1) description + generated_description concatenated (if both present)
      2) description alone (if at least 100 characters long)
      3) generated_description alone (fallback)
      4) short description alone (last fallback)
    """
    description = (doc.get("description") or "").strip()
    generated   = (doc.get("generated_description") or "").strip()

    if description and generated:
        return f"{description} {generated}", "description+generated"
    if len(description) >= 100:
        return description, "description"
    if generated:
        return generated, "generated_description"
    if description:
        return description, "description"
    return "", "none"


# ------------------------------------------------------------------------------
# ENRICHED RERANK TEXT
#
# The CrossEncoder has poor lexical overlap between user queries and capability
# text because capabilities describe REST behaviors, not domain concepts.
# Enriching the text with parameters and response schema fields exposes
# domain terms (e.g., "temperature", "Celsius", "sensorType") that the
# CrossEncoder can match directly against the query.
#
# The enriched text is used ONLY for reranking -- it is not saved in Qdrant
# nor sent to the LLM. The LLM context always receives the original
# capability text (more readable and concise).
# ------------------------------------------------------------------------------

def _build_enriched_text(
    http_op: str,
    capabilities: dict,
    parameters: dict,
    response_schemas: dict,
    request_schemas: dict,
) -> str:
    """
    Builds the enriched text for CrossEncoder reranking by combining:
      - capability text (functional description of the endpoint)
      - parameters (query param names and enum values)
      - response schema fields (returned field names)
      - request schema fields (input field names, for POST/PUT)
    """
    parts = [capabilities.get(http_op, "")]

    param_str = parameters.get(http_op, "") or ""
    if param_str:
        parts.append(f"| parameters: {param_str}")

    resp_str = response_schemas.get(http_op, "") or ""
    if resp_str:
        fields = re.findall(r'(\w+):', resp_str)
        if fields:
            parts.append(f"| response fields: {', '.join(fields)}")

    req_str = request_schemas.get(http_op, "") or ""
    if req_str and not http_op.startswith("GET"):
        fields = re.findall(r'(\w+):', req_str)
        if fields:
            parts.append(f"| request fields: {', '.join(fields)}")

    return " ".join(parts)


# ------------------------------------------------------------------------------
# QUERY DECOMPOSITION (Stage 1 recall enhancer)
#
# Splits a composite user query into atomic sub-queries, each targeting
# a single type of information/service. Improves Stage 1 recall on large,
# cross-domain catalogs where a single embedding of the composite query
# is a semantic compromise that penalizes "marginal" services.
#
# Decomposition is DOMAIN-AGNOSTIC: the prompt contains no catalog-specific
# knowledge; the LLM reasons about the logical form of the query.
#
# On error (toggle off, timeout, malformed JSON, empty list), it falls back
# to the original query as a single sub-query -- identical behavior to the
# pre-decomposition pipeline, zero regression risk.
# ------------------------------------------------------------------------------

DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant for a service discovery system.
Given a user query, decompose it into atomic information needs.
Each sub-query targets ONE type of information that can be satisfied by a single type of API/service.

RULES:
- Output 1 to {max_subq} sub-queries.
- If the query asks for ONE type of information, output exactly one sub-query.
- Each sub-query is a short noun phrase (2-6 words), describing the resource/data type.
- Strip references to specific entities, locations, filters, conditions -- keep only the resource type.
- Sub-queries must be independent (no references between them).
- Use the same language as the input query.
- Do NOT invent information not implied by the query.

EXAMPLES:
Input: "list all temperature sensors"
Output: {{"sub_queries": ["temperature sensors"]}}

Input: "find car parks near Arco di Traiano with charging stations nearby"
Output: {{"sub_queries": ["parking spots", "tourist attractions", "charging stations"]}}

Input: "show me air quality in zones with heavy traffic"
Output: {{"sub_queries": ["air quality measurements", "traffic data by zone"]}}

Input: "create a new user account"
Output: {{"sub_queries": ["user account management"]}}
"""

DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "sub_queries": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": DECOMPOSITION_MAX_SUBQ,
        }
    },
    "required": ["sub_queries"],
}


# The model name sent to dllama-api is ignored by the server.
_OLLAMA_MODEL  = os.environ.get("DECOMPOSITION_MODEL", "model")
_LLAMACPP_MODEL = "default"


def _extract_json(text: str) -> str:
    """Extract JSON object from text that may contain markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("```"):
                start = i + 1
                break
        end = len(lines)
        for i in range(len(lines) - 1, start - 1, -1):
            if lines[i].strip().startswith("```"):
                end = i
                break
        text = "\n".join(lines[start:end]).strip()
    # find first { and last }
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e > s:
        return text[s : e + 1]
    return text


_DECOMPOSITION_TIMEOUT = None
_DECOMPOSITION_MAX_TOKENS = 4000


def decompose_query(query_text: str) -> tuple[list, dict]:
    """
    Decomposes the query into atomic sub-queries via LLM.
    Supports llamacpp (OpenAI-compatible) and Ollama backends.

    Returns:
        (sub_queries, meta) where meta contains:
          - source: "llm" | "fallback" | "disabled"
          - latency_ms: int
          - reason: reason for fallback (if applicable)
    """
    if not USE_LLM_DECOMPOSITION:
        return [query_text], {"source": "disabled", "latency_ms": 0, "reason": None}

    logger.info(
        f"[DECOMPOSE] backend={DECOMPOSITION_BACKEND} ctx_limit={DECOMPOSITION_CTX_LENGTH}"
    )

    t0 = time.perf_counter()

    if DECOMPOSITION_BACKEND == "llamacpp":
        # ---- llama.cpp / Distributed-Llama (OpenAI-compatible) ----
        system_prompt = DECOMPOSITION_SYSTEM_PROMPT.format(max_subq=DECOMPOSITION_MAX_SUBQ)
        payload = {
            "model": _LLAMACPP_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query_text},
            ],
            "temperature": 0.0,
            "max_tokens":  _DECOMPOSITION_MAX_TOKENS,
            "stream":      False,
        }
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/v1/chat/completions",
                json=payload,
                timeout=_DECOMPOSITION_TIMEOUT,
            )
            resp.raise_for_status()
            latency_ms = int((time.perf_counter() - t0) * 1000)

            data = resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            if not content:
                logger.warning("[DECOMPOSE] Empty response from llamacpp")
                return [query_text], {"source": "fallback",
                                      "latency_ms": latency_ms,
                                      "reason": "empty_response"}

            finish_reason = data["choices"][0].get("finish_reason", "")
            if finish_reason == "length":
                logger.warning("[DECOMPOSE] Response truncated (finish_reason=length)")

            extracted = _extract_json(content)
            parsed = json.loads(extracted)
            sub_queries = parsed.get("sub_queries", [])

            seen, clean = set(), []
            for sq in sub_queries:
                sq = (sq or "").strip()
                if not sq or sq.lower() in seen:
                    continue
                seen.add(sq.lower())
                clean.append(sq)
            clean = clean[:DECOMPOSITION_MAX_SUBQ]

            if not clean:
                return [query_text], {"source": "fallback",
                                      "latency_ms": latency_ms,
                                      "reason": "empty_sub_queries"}

            return clean, {"source": "llm", "latency_ms": latency_ms, "reason": None}

        except (requests.exceptions.RequestException,
                json.JSONDecodeError, ValueError, KeyError, IndexError) as e:
            logger.warning(f"[DECOMPOSE] llamacpp error: {type(e).__name__}: {e}")
            return [query_text], {"source": "fallback",
                                  "latency_ms": int((time.perf_counter() - t0) * 1000),
                                  "reason": type(e).__name__}

    else:
        # ---- Ollama backend ----
        payload = {
            "model": _OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT.format(max_subq=DECOMPOSITION_MAX_SUBQ)},
                {"role": "user",   "content": query_text},
            ],
            "format": DECOMPOSITION_SCHEMA,
            "options": {
                "temperature": 0.0,
                "num_predict": _DECOMPOSITION_MAX_TOKENS,
            },
            "think":  False,
            "stream": False,
        }

        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
                timeout=_DECOMPOSITION_TIMEOUT,
            )
            resp.raise_for_status()
            latency_ms = int((time.perf_counter() - t0) * 1000)

            content = resp.json().get("message", {}).get("content", "")
            if not content:
                logger.warning("[DECOMPOSE] Empty response from Ollama")
                return [query_text], {"source": "fallback",
                                      "latency_ms": latency_ms,
                                      "reason": "empty_response"}

            parsed  = json.loads(content)
            sub_queries = parsed.get("sub_queries", [])

            seen, clean = set(), []
            for sq in sub_queries:
                sq = (sq or "").strip()
                if not sq or sq.lower() in seen:
                    continue
                seen.add(sq.lower())
                clean.append(sq)
            clean = clean[:DECOMPOSITION_MAX_SUBQ]

            if not clean:
                return [query_text], {"source": "fallback",
                                      "latency_ms": latency_ms,
                                      "reason": "empty_sub_queries"}

            return clean, {"source": "llm", "latency_ms": latency_ms, "reason": None}

        except (requests.exceptions.RequestException,
                json.JSONDecodeError, ValueError, KeyError) as e:
            return [query_text], {"source": "fallback",
                                  "latency_ms": int((time.perf_counter() - t0) * 1000),
                                  "reason": type(e).__name__}


def create_vector_collection():
    existing_collections = {col.name for col in qdrant_client.get_collections().collections}

    if QDRANT_COLLECTION not in existing_collections:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{QDRANT_COLLECTION}' created")
    else:
        logger.info(f"Collection '{QDRANT_COLLECTION}' already exists")

    if QDRANT_COLLECTION_INDEX not in existing_collections:
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_INDEX,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
        )
        logger.info(f"Collection '{QDRANT_COLLECTION_INDEX}' created")
    else:
        logger.info(f"Collection '{QDRANT_COLLECTION_INDEX}' already exists")


@app.route("/health")
def index():
    if is_server_ready is True:
        return jsonify({"status": "ok", "message": "Gateway Server is ready", "model_loaded": True}), 200
    else:
        logger.error("Model not yet loaded or broken")
        return jsonify({"status": "error", "message": "Model not yet loaded or broken", "model_loaded": False}), 500


@app.route("/index/search", methods=["POST"])
def vector_search():
    """
    Two-stage retrieval pipeline with optional decomposition and RRF ranking:

    Stage 0 (optional) -- Query decomposition via LLM:
      If USE_LLM_DECOMPOSITION=true, the query is split into atomic sub-queries.
      Each sub-query produces its own candidate set in Stage 1; the
      deduplicated union (max-score) becomes the input for Stage 2.
      The ORIGINAL query is always passed to the CrossEncoder to preserve
      the operational context (entities, filters, conditions).

    Stage 1 -- Bi-encoder on descriptions (services_index):
      Retrieves top-K services by semantic similarity on description.

    Stage 2 -- Cross-encoder on capabilities (reranker BAAI/bge-reranker-base):
      For each service retrieved in Stage 1, loads endpoints from MongoDB
      and reranks them with the CrossEncoder using the original query.

    Final ranking -- Reciprocal Rank Fusion (RRF):
      Combines Stage 1 (s1_score) and Stage 2 (best_ep_score) rankings
      by summing 1/(K + rank_i) for each ranking. Standard IR technique
      (Cormack, Clarke, Buettcher 2009) that avoids score normalization
      across incompatible scales (cosine similarity vs logit). No empirical
      hyperparameters: K=60 is the canonical value in the literature.

    Token budget with graceful trimming:
      Instead of discarding an entire service when it does not fit the budget,
      progressively scales the number of endpoints keeping the top-N
      by ep_score until the minimum fitting configuration is found.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query_text = data["query"]

    # -- Pipeline parameters ------------------------------------------------------
    STAGE1_K                  = 7
    TOP_ENDPOINTS_PER_SERVICE = 4
    K_RRF                     = 60   # canonical parameter (Cormack et al., 2009)

    # =============================================================================
    # STAGE 0: QUERY DECOMPOSITION (optional)
    # =============================================================================
    sub_queries, decomp_meta = decompose_query(query_text)
    logger.info(
        f"[DECOMPOSITION] source={decomp_meta['source']} "
        f"latency={decomp_meta['latency_ms']}ms "
        f"reason={decomp_meta['reason']} | "
        f"{len(sub_queries)} sub-queries: {sub_queries}"
    )

    # =============================================================================
    # STAGE 1: union of top-K for each sub-query (max-score per service)
    # =============================================================================
    stage1_scores: dict = {}
    per_subquery_log = []

    for sq in sub_queries:
        sq_embedding = embed_query(sq)
        # query_points is the successor of search() (deprecated in qdrant-client 1.10+).
        # Returns a QueryResponse with .points containing ScoredPoint identical
        # to those returned by search(): same .id, .score, .payload attributes.
        sq_response = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_INDEX,
            query=sq_embedding,
            limit=STAGE1_K,
        )
        sq_results = sq_response.points
        ids_for_log = []
        for r in sq_results:
            sid = r.payload["mongo_id"]
            if sid not in stage1_scores or r.score > stage1_scores[sid]:
                stage1_scores[sid] = r.score
            ids_for_log.append(f"{sid}({r.score:.3f})")
        per_subquery_log.append(f"  '{sq}' \u2192 {ids_for_log}")

    if not stage1_scores:
        logger.warning("[SEARCH] Stage 1: nessun servizio trovato in services_index")
        return jsonify({"results": []}), 200

    logger.info(
        f"[STAGE 1] union di {len(sub_queries)} sub-query \u2192 "
        f"{len(stage1_scores)} servizi unici:\n" +
        "\n".join(per_subquery_log) +
        f"\n  UNION: {list(stage1_scores.keys())}"
    )

    # =============================================================================
    # STAGE 2: reranking endpoints -- single batch for all services
    #
    # Load services from MongoDB with projection (no BSON roundtrip),
    # build all (query, enriched_text) pairs in a single list,
    # and make ONE call to predict().
    # NOTE: always uses the original query_text, not the sub-queries.
    # =============================================================================
    services_data = {}
    for doc_id in stage1_scores:
        # Projection: exclude _id (we already have it) and select only
        # the necessary fields. No dumps/loads roundtrip, all sub-fields
        # are string dicts (no nested ObjectId).
        retrieved = collection.find_one(
            {"_id": doc_id},
            {"_id": 0, "name": 1, "description": 1, "capabilities": 1,
             "endpoints": 1, "response_schemas": 1, "request_schemas": 1,
             "parameters": 1}
        )
        if not retrieved:
            logger.warning(f"[STAGE 2] Servizio '{doc_id}' non trovato in MongoDB")
            continue

        capabilities = retrieved.get("capabilities", {}) or {}
        # Excluded: POST /register (mock internal registration) and any
        # /health endpoint (useful for liveness check, useless for LLM planner).
        ops = [op for op in capabilities.keys()
               if op != "POST /register" and not op.endswith("/health")]
        if not ops:
            continue

        services_data[doc_id] = {
            "s1_score":         stage1_scores[doc_id],
            "name":             retrieved.get("name"),
            "description":      retrieved.get("description"),
            "capabilities":     capabilities,
            "endpoints":        retrieved.get("endpoints", {}) or {},
            "response_schemas": retrieved.get("response_schemas", {}) or {},
            "request_schemas":  retrieved.get("request_schemas", {}) or {},
            "parameters":       retrieved.get("parameters", {}) or {},
            "ops":              ops,
        }

    if not services_data:
        return jsonify({"results": []}), 200

    # -- Build all pairs in a single list -----------------------------------------
    all_pairs = []
    pair_map  = []  # (doc_id, op) for each element in all_pairs

    for doc_id, sdata in services_data.items():
        for op in sdata["ops"]:
            enriched = _build_enriched_text(
                op,
                sdata["capabilities"],
                sdata["parameters"],
                sdata["response_schemas"],
                sdata["request_schemas"],
            )
            all_pairs.append((query_text, enriched))
            pair_map.append((doc_id, op))

    logger.info(f"[STAGE 2] Single-batch reranking: {len(all_pairs)} coppie "
                f"da {len(services_data)} servizi")

    # -- Single predict() call for all pairs (with batching) ----------------------
    # Set batch_size=4 to force the CrossEncoder to compute small blocks,
    # preventing RAM exhaustion and Swap Disk usage on CPU in Docker.
    all_scores = reranker_model.predict(all_pairs, batch_size=4)

    # -- Aggregate scores by doc_id -----------------------------------------------
    scores_by_service: dict = {doc_id: {} for doc_id in services_data}
    for (doc_id, op), score in zip(pair_map, all_scores):
        scores_by_service[doc_id][op] = float(score)

    merged: dict = {}
    for doc_id, sdata in services_data.items():
        ops = sdata["ops"]

        # -- HEURISTIC: Virtual boost for GET list endpoints -----------------------
        # GET endpoints without a path parameter are the only ones that list
        # resources, and are operationally necessary to orchestrate subsequent
        # calls. The boost is purely ordinal (raises GET list endpoints to the
        # top of the slice without altering the best_endpoint_score used in RRF
        # ranking). /health is excluded: it provides no useful data.
        GET_LIST_BOOST = 5.0

        def sorting_heuristic(op_score_tuple):
            op_name, original_score = op_score_tuple
            is_get_list = (op_name.startswith("GET ")
                           and "{" not in op_name
                           and not op_name.endswith("/health"))
            return original_score + (GET_LIST_BOOST if is_get_list else 0.0)

        scored_ops = sorted(
            [(op, scores_by_service[doc_id][op]) for op in ops],
            key=sorting_heuristic,
            reverse=True
        )

        relevant_ops = scored_ops[:TOP_ENDPOINTS_PER_SERVICE]

        # Real best semantic score -- use max() on raw scores, NOT relevant_ops[0]
        # which may be inflated by the boost.
        best_endpoint_score = max(scores_by_service[doc_id].values())

        merged[doc_id] = {
            "_id":              doc_id,
            "name":             sdata["name"],
            "description":      sdata["description"],
            "capabilities":     {op: sdata["capabilities"][op] for op, _ in relevant_ops},
            "endpoints":        {op: sdata["endpoints"].get(op) for op, _ in relevant_ops},
            "response_schemas": {op: sdata["response_schemas"].get(op) for op, _ in relevant_ops},
            "request_schemas":  {op: sdata["request_schemas"].get(op) for op, _ in relevant_ops},
            "parameters":       {op: sdata["parameters"].get(op) for op, _ in relevant_ops},
            "_stage1_score":    sdata["s1_score"],
            "_best_ep_score":   best_endpoint_score,
        }

        ep_rows = "\n            ".join(
            f"{op:<40} score={score:.4f}" for op, score in relevant_ops
        )
        discarded_ops = [op for op, _ in scored_ops[TOP_ENDPOINTS_PER_SERVICE:]]
        disc_str = f"\n          SCARTATI : {discarded_ops}" if discarded_ops else ""
        logger.info(
            f"[STAGE 2] {doc_id}\n"
            f"          s1={sdata['s1_score']:.4f} | best_ep={best_endpoint_score:.4f}\n"
            f"          SELEZIONATI ({len(relevant_ops)}/{len(ops)}):\n"
            f"            {ep_rows}"
            f"{disc_str}"
        )

    if not merged:
        return jsonify({"results": []}), 200

    # =============================================================================
    # RECIPROCAL RANK FUSION (RRF)
    #
    # Combines two service rankings without the need to normalize scales:
    #   - ranking by s1_score (cosine similarity on description, range 0-1)
    #   - ranking by best_ep_score (CrossEncoder logit, range approx +/-10)
    #
    # Formula: rrf(s) = SUM 1/(K + rank_i(s)) for each ranking i
    # K=60 is the canonical value in the literature (Cormack et al. 2009),
    # large enough to smooth top rank differences and small enough not to
    # flatten the entire ranking.
    #
    # Advantages over composite scoring ALPHA*s1 + BETA*sigmoid(z):
    #   - zero empirical hyperparameters to justify
    #   - robust to outliers (a very high best_ep_score does not destroy ranking)
    #   - solid bibliographic citation for thesis defense
    # =============================================================================
    services_list = list(merged.values())

    ranked_by_s1 = sorted(services_list, key=lambda x: x["_stage1_score"],   reverse=True)
    ranked_by_ep = sorted(services_list, key=lambda x: x["_best_ep_score"], reverse=True)

    # Rank 1-based as in the original formulation by Cormack et al. (2009):
    # the top-ranked item has rank=1, contributing 1/(K+1) to the score.
    s1_rank = {s["_id"]: i for i, s in enumerate(ranked_by_s1, start=1)}
    ep_rank = {s["_id"]: i for i, s in enumerate(ranked_by_ep, start=1)}

    for s in services_list:
        s["_rrf_score"] = (1.0 / (K_RRF + s1_rank[s["_id"]])
                         + 1.0 / (K_RRF + ep_rank[s["_id"]]))

    ordered_services = sorted(services_list, key=lambda x: x["_rrf_score"], reverse=True)

    rrf_rows = "\n    ".join(
        f"  {s['_id']:<50} s1_rank={s1_rank[s['_id']]:<2} "
        f"ep_rank={ep_rank[s['_id']]:<2} rrf={s['_rrf_score']:.5f}"
        for s in ordered_services
    )
    logger.info(
        f"[RRF] K={K_RRF} | ranking finale di {len(ordered_services)} servizi:\n"
        f"    {rrf_rows}"
    )

    # Clean up internal fields before token budget
    for s in ordered_services:
        s.pop("_stage1_score", None)
        s.pop("_best_ep_score", None)
        s.pop("_rrf_score", None)

    # =============================================================================
    # TOKEN BUDGET with graceful trimming
    #
    # For each service, first attempts insertion with all endpoints.
    # If it does not fit in the remaining budget, progressively scales the
    # number of endpoints (top-N by ep_score, already sorted) until the
    # minimum fitting configuration is found. Only if even the single best
    # endpoint does not fit in the budget is the service discarded.
    # =============================================================================
    _DICT_FIELDS = ("capabilities", "endpoints", "response_schemas",
                    "request_schemas", "parameters")

    def _trim_service(s: dict, keep_ops: list) -> dict:
        """Returns a copy of the service with only the endpoints in keep_ops."""
        trimmed = {}
        for k, v in s.items():
            if k in _DICT_FIELDS and isinstance(v, dict):
                trimmed[k] = {op: v[op] for op in keep_ops if op in v}
            else:
                trimmed[k] = v
        return trimmed

    max_tokens     = 5000
    current_tokens = 0
    top_results    = []
    budget_log     = []  # (service_id, n_inseriti, n_totali, nomi_endpoint)

    for s in ordered_services:
        ops = list(s.get("capabilities", {}).keys())

        inserted = False
        for n in range(len(ops), 0, -1):
            candidate     = _trim_service(s, ops[:n])
            candidate_tok = count_tokens(json.dumps(candidate))

            if current_tokens + candidate_tok <= max_tokens:
                top_results.append(candidate)
                current_tokens += candidate_tok
                budget_log.append((s["_id"], n, len(ops), ops[:n]))
                inserted = True
                break

        if not inserted:
            logger.info(
                f"[BUDGET] {s['_id']} escluso: nemmeno il top-1 endpoint "
                f"({count_tokens(json.dumps(_trim_service(s, ops[:1])))} tok) "
                f"entra nel budget residuo ({max_tokens - current_tokens} tok)"
            )

    budget_rows = "\n    ".join(
        f"  {sid:<50} ({n}/{tot} ep) \u2192 {ep_names}"
        for sid, n, tot, ep_names in budget_log
    )
    logger.info(
        f"[SEARCH] Stage1={len(stage1_scores)} \u2192 Stage2={len(merged)} \u2192 "
        f"RRF_ranked={len(ordered_services)} \u2192 "
        f"{len(top_results)} nel budget | token usati: {current_tokens}/{max_tokens}\n"
        f"    {budget_rows}"
    )
    return jsonify({"results": top_results}), 200


@app.route("/service", methods=["POST"])
def create_or_update_service():
    data = request.get_json()
    if not data or "id" not in data:
        return jsonify({"error": "Missing 'id' field"}), 400

    doc_id = data["id"]
    data["_id"] = doc_id
    data.pop("id", None)

    # -- Stage 2 index: 1 vector per endpoint (capability text) -------------------
    capabilities = data.get("capabilities", {})
    for http_op, capability in capabilities.items():
        embedding = embed_passage(capability)
        vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, capability))
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={"mongo_id": doc_id, "http_operation": http_op}
                )
            ]
        )

    # -- Stage 1 index: 1 vector per service (description text) -------------------
    # Prefers description. Uses generated_description only if description
    # is absent or too short (<100 characters).
    text_to_index, source = select_stage1_text(data)
    if text_to_index:
        desc_vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"desc:{doc_id}"))
        desc_embedding = embed_passage(text_to_index)
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_INDEX,
            points=[
                PointStruct(
                    id=desc_vector_id,
                    vector=desc_embedding,
                    payload={"mongo_id": doc_id}
                )
            ]
        )
        logger.info(f"[INDEX] Service '{doc_id}' indexed in services_index (fonte: {source})")

    collection.replace_one({"_id": doc_id}, data, upsert=True)
    return jsonify({"status": "ok", "id": doc_id}), 200


@app.route("/services", methods=["GET"])
def list_services():
    docs = list(collection.find())
    return dumps(docs), 200


@app.route("/services/<string:service_id>", methods=["GET"])
def get_service(service_id):
    doc = collection.find_one({"_id": service_id})
    if not doc:
        return jsonify({"error": "Service not found"}), 404
    return dumps(doc), 200


@app.route("/services/<string:service_id>", methods=["DELETE"])
def delete_service(service_id):
    result = collection.delete_one({"_id": service_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Service not found"}), 404
    return jsonify({"status": "deleted", "id": service_id}), 200


@app.route("/index/reindex", methods=["POST"])
def reindex_descriptions():
    """
    Re-indexes descriptive texts of all services in services_index (Stage 1).

    Prefers description; uses generated_description only as fallback when
    description is absent or shorter than 100 characters.
    Uses embed_passage() for the correct asymmetric role.
    """
    docs    = list(collection.find())
    indexed = 0
    skipped = 0
    for doc in docs:
        doc_id = doc.get("_id")
        text_to_index, source = select_stage1_text(doc)
        if not text_to_index:
            logger.warning(f"[REINDEX] {doc_id}: nessuna description disponibile, saltato")
            skipped += 1
            continue
        try:
            desc_vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"desc:{doc_id}"))
            desc_embedding = embed_passage(text_to_index)
            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_INDEX,
                points=[
                    PointStruct(
                        id=desc_vector_id,
                        vector=desc_embedding,
                        payload={"mongo_id": doc_id}
                    )
                ]
            )
            logger.info(f"[REINDEX] {doc_id} indicizzato (fonte: {source})")
            indexed += 1
        except Exception as e:
            logger.error(f"[REINDEX] {doc_id} failed: {e}")
            skipped += 1

    logger.info(f"[REINDEX] Completato: {indexed} indicizzati, {skipped} saltati")
    return jsonify({"indexed": indexed, "skipped": skipped}), 200


@app.route("/services/<string:service_id>/schemas", methods=["PATCH"])
def update_service_schemas(service_id):
    """
    Updates response_schemas, request_schemas, parameters, and/or
    generated_description of a service.
    Called by the api-importer after extracting schemas with prance.
    """
    data = request.get_json()
    update_fields = {}
    if "response_schemas" in data:
        update_fields["response_schemas"] = data["response_schemas"]
    if "request_schemas" in data:
        update_fields["request_schemas"] = data["request_schemas"]
    if "parameters" in data:
        update_fields["parameters"] = data["parameters"]
    if "generated_description" in data:
        update_fields["generated_description"] = data["generated_description"]

    if not update_fields:
        return jsonify({"error": "No valid schema fields provided"}), 400

    result = collection.update_one(
        {"_id": service_id},
        {"$set": update_fields}
    )

    if result.matched_count == 0:
        return jsonify({"error": f"Service '{service_id}' not found"}), 404

    logger.info(f"[SCHEMAS] Updated {list(update_fields.keys())} for {service_id}")
    return jsonify({"status": "ok", "id": service_id}), 200


if __name__ == "__main__":
    try:
        with app.app_context():
            logger.info("\U0001f6e0\ufe0f Creating Qdrant collection...")
            create_vector_collection()
            logger.info("\U0001f4e6 Loading embedding model...")
            load_model()
            is_server_ready = True
            logger.info("\u2705 Server is ready.")
    except Exception as e:
        logger.exception("\u274c Failed to initialize application")
        sys.exit(1)

    server = WSGIServer(('0.0.0.0', 5000), app)
    try:
        print("\U0001f680 Starting Flask app with Cheroot on http://0.0.0.0:5000")
        server.start()
    except KeyboardInterrupt:
        print("\U0001f6d1 Shutting down server...")
        server.stop()

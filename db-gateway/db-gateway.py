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

# ── LLM query decomposition (opzionale) ──────────────────────────────────────
# Se USE_LLM_DECOMPOSITION=true, prima dello Stage 1 la query viene decomposta
# in sotto-query atomiche tramite un LLM (default glm-4.7-flash via Ollama).
# Ogni sotto-query produce un proprio set top-K; l'unione deduplicata è il
# nuovo input dello Stage 2. In caso di errore si fa fallback alla query
# originale come singola sotto-query — il sistema non si rompe mai.
USE_LLM_DECOMPOSITION  = os.environ.get("USE_LLM_DECOMPOSITION", "true").lower() == "true"
OLLAMA_URL             = os.environ.get("OLLAMA_URL", "http://ollama:11434")
DECOMPOSITION_MODEL    = os.environ.get("DECOMPOSITION_MODEL", "glm-4.7-flash:q4_K_M")
DECOMPOSITION_TIMEOUT  = int(os.environ.get("DECOMPOSITION_TIMEOUT", "15"))
DECOMPOSITION_MAX_SUBQ = int(os.environ.get("DECOMPOSITION_MAX_SUBQ", "4"))

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
    Wrapper personalizzato per eseguire un CrossEncoder in formato ONNX.
    Ottimizza massivamente l'inferenza CPU rispetto a PyTorch standard.
    """
    def __init__(self, model_name):
        logger.info(f"Converting and loading {model_name} in ONNX format (this may take a minute)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Il parametro export=True scarica il modello PyTorch e lo converte in ONNX al volo
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
            # max_length=512 previene crash su endpoint con descrizioni anomale
            inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            
            # Inferenza tramite runtime ONNX (C++)
            outputs = self.model(**inputs)
            logits = outputs.logits
            if logits.ndim > 1:
                logits = logits.squeeze(-1)
            logits = logits.detach().numpy()
            
            # Gestione dei risultati (singoli o multipli)
            if logits.ndim == 0:
                all_scores.append(float(logits))
            else:
                all_scores.extend([float(x) for x in logits])
                
        return all_scores


def load_model():
    global embedding_model, reranker_model, tokenizer
    logger.info("Loading models...")
    
    # 1. Caricamento Embedding Model (Lasciato in PyTorch standard)
    embedding_model = SentenceTransformer(
        model_name_or_path='Qwen/Qwen3-Embedding-0.6B',
        device='cpu',
        trust_remote_code=True
    )
    tokenizer = embedding_model.tokenizer
    logger.info("Embedding model loaded.")
    
    # 2. Caricamento Reranker Model (Usando il nuovo motore ONNX)
    reranker_model = ONNXCrossEncoder('BAAI/bge-reranker-base')
    logger.info("Reranker model loaded.")


def embed_query(text: str) -> list:
    """
    Embedding per la query utente a runtime.
    Prefisso 'query:' — Qwen3-Embedding è addestrato con asimmetria query/passage:
    usare lo stesso prefisso per query e documenti degrada il retrieval.
    """
    embedding = embedding_model.encode(
        f"query: {text}", convert_to_tensor=False, normalize_embeddings=True
    )
    return embedding.tolist()


def embed_passage(text: str) -> list:
    """
    Embedding per documenti da indicizzare (capabilities, descriptions).
    Prefisso 'passage:' — corrisponde al ruolo dei documenti nel training
    di Qwen3-Embedding. Da usare in tutte le fasi di indexing.
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
    Seleziona il testo per l'indice Stage 1 con la seguente priorità:
      1) description + generated_description concatenate (se entrambe presenti)
      2) description da sola (se lunga almeno 100 caratteri)
      3) generated_description da sola (fallback)
      4) description corta da sola (ultimo fallback)
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


# ─────────────────────────────────────────────────────────────────────────────
# ENRICHED RERANK TEXT
#
# Il CrossEncoder ha scarso overlap lessicale tra query utente e capability text
# perché le capability descrivono comportamenti REST, non concetti del dominio.
# Arricchire il testo con i parametri e i campi del response schema espone
# termini di dominio (es. "temperature", "Celsius", "sensorType") che il
# CrossEncoder può confrontare direttamente con la query.
#
# Il testo arricchito è usato SOLO per il reranking — non viene salvato in
# Qdrant né inviato all'LLM. Il contesto LLM riceve sempre la capability
# originale (più leggibile e concisa).
# ─────────────────────────────────────────────────────────────────────────────

def _build_enriched_text(
    http_op: str,
    capabilities: dict,
    parameters: dict,
    response_schemas: dict,
    request_schemas: dict,
) -> str:
    """
    Costruisce il testo arricchito per il reranking CrossEncoder combinando:
      - capability text (descrizione funzionale dell'endpoint)
      - parameters (nomi e valori enum dei query param)
      - response schema fields (nomi dei campi restituiti)
      - request schema fields (nomi dei campi in input, per i POST/PUT)
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


# ─────────────────────────────────────────────────────────────────────────────
# QUERY DECOMPOSITION (Stage 1 recall enhancer)
#
# Spezza una query utente composita in sotto-query atomiche, ognuna mirata
# a un singolo tipo di informazione/servizio. Migliora il recall di Stage 1
# su cataloghi grandi e cross-domain dove un singolo embedding della query
# composta è un compromesso semantico che penalizza i servizi "marginali".
#
# La decomposizione è DOMAIN-AGNOSTIC: il prompt non contiene knowledge
# specifica del catalogo, l'LLM ragiona sulla forma logica della query.
#
# In caso di errore (toggle off, timeout, JSON malformato, lista vuota) si
# fa fallback alla query originale come singola sotto-query — comportamento
# identico al pipeline pre-decomposizione, zero rischio di regressione.
# ─────────────────────────────────────────────────────────────────────────────

DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition assistant for a service discovery system.
Given a user query, decompose it into atomic information needs.
Each sub-query targets ONE type of information that can be satisfied by a single type of API/service.

RULES:
- Output 1 to 4 sub-queries.
- If the query asks for ONE type of information, output exactly one sub-query.
- Each sub-query is a short noun phrase (2-6 words), describing the resource/data type.
- Strip references to specific entities, locations, filters, conditions — keep only the resource type.
- Sub-queries must be independent (no references between them).
- Use the same language as the input query.
- Do NOT invent information not implied by the query.

EXAMPLES:
Input: "list all temperature sensors"
Output: {"sub_queries": ["temperature sensors"]}

Input: "find car parks near Arco di Traiano with charging stations nearby"
Output: {"sub_queries": ["parking spots", "tourist attractions", "charging stations"]}

Input: "show me air quality in zones with heavy traffic"
Output: {"sub_queries": ["air quality measurements", "traffic data by zone"]}

Input: "create a new user account"
Output: {"sub_queries": ["user account management"]}
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


def decompose_query(query_text: str) -> tuple[list, dict]:
    """
    Decompone la query in sotto-query atomiche tramite LLM.

    Returns:
        (sub_queries, meta) dove meta contiene:
          - source: "llm" | "fallback" | "disabled"
          - latency_ms: int
          - reason: motivo del fallback (se applicabile)
    """
    if not USE_LLM_DECOMPOSITION:
        return [query_text], {"source": "disabled", "latency_ms": 0, "reason": None}

    payload = {
        "model": DECOMPOSITION_MODEL,
        "messages": [
            {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
            {"role": "user",   "content": query_text},
        ],
        "format": DECOMPOSITION_SCHEMA,
        "options": {
            "temperature": 0.0,
            "num_predict": 200,
        },
        "think":  False,
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=DECOMPOSITION_TIMEOUT,
        )
        resp.raise_for_status()
        latency_ms = int((time.perf_counter() - t0) * 1000)

        content = resp.json().get("message", {}).get("content", "")
        parsed  = json.loads(content)
        sub_queries = parsed.get("sub_queries", [])

        # Sanitize: strip, dedupe (case-insensitive), drop empty, cap a MAX_SUBQ
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
    Two-stage retrieval pipeline con decomposizione opzionale e ranking RRF:

    Stage 0 (opzionale) — Query decomposition via LLM:
      Se USE_LLM_DECOMPOSITION=true, la query viene spezzata in sotto-query
      atomiche. Ogni sotto-query produce un proprio set di candidati nello
      Stage 1; l'unione deduplicata (max-score) diventa l'input dello Stage 2.
      La query ORIGINALE viene comunque passata al CrossEncoder per preservare
      il contesto operativo (entità, filtri, condizioni).

    Stage 1 — Bi-encoder su descriptions (services_index):
      Recupera i top-K servizi per similarità semantica sulla description.

    Stage 2 — Cross-encoder su capabilities (reranker BAAI/bge-reranker-base):
      Per ogni servizio recuperato nel Stage 1, carica gli endpoint da MongoDB
      e li rerankerizza con il CrossEncoder usando la query originale.

    Ranking finale — Reciprocal Rank Fusion (RRF):
      Combina i ranking di Stage 1 (s1_score) e Stage 2 (best_ep_score)
      sommando 1/(K + rank_i) per ogni ranking. Tecnica IR standard
      (Cormack, Clarke, Buettcher 2009) che evita la normalizzazione di
      score su scale incompatibili (cosine similarity vs logit). Nessun
      iperparametro empirico: K=60 è il valore canonico in letteratura.

    Token budget con trimming graceful:
      Invece di scartare un intero servizio quando non entra nel budget,
      scala progressivamente il numero di endpoint mantenendo i top-N
      per ep_score fino a trovare la configurazione minima che entra.
    """
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query_text = data["query"]

    # ── Parametri pipeline ────────────────────────────────────────────────────
    STAGE1_K                  = 7
    TOP_ENDPOINTS_PER_SERVICE = 4
    K_RRF                     = 60   # parametro canonico (Cormack et al., 2009)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 0: QUERY DECOMPOSITION (opzionale)
    # ════════════════════════════════════════════════════════════════════════
    sub_queries, decomp_meta = decompose_query(query_text)
    logger.info(
        f"[DECOMPOSITION] source={decomp_meta['source']} "
        f"latency={decomp_meta['latency_ms']}ms "
        f"reason={decomp_meta['reason']} | "
        f"{len(sub_queries)} sub-queries: {sub_queries}"
    )

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1: union dei top-K per ogni sotto-query (max-score per servizio)
    # ════════════════════════════════════════════════════════════════════════
    stage1_scores: dict = {}
    per_subquery_log = []

    for sq in sub_queries:
        sq_embedding = embed_query(sq)
        # query_points è il successore di search() (deprecato in qdrant-client 1.10+).
        # Restituisce un QueryResponse con .points contenente ScoredPoint identici
        # a quelli ritornati da search(): stessi attributi .id, .score, .payload.
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
        per_subquery_log.append(f"  '{sq}' → {ids_for_log}")

    if not stage1_scores:
        logger.warning("[SEARCH] Stage 1: nessun servizio trovato in services_index")
        return jsonify({"results": []}), 200

    logger.info(
        f"[STAGE 1] union di {len(sub_queries)} sub-query → "
        f"{len(stage1_scores)} servizi unici:\n" +
        "\n".join(per_subquery_log) +
        f"\n  UNION: {list(stage1_scores.keys())}"
    )

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2: reranking degli endpoint — single-batch per tutti i servizi
    #
    # Carichiamo i servizi da MongoDB con projection (no roundtrip BSON),
    # costruiamo tutte le coppie (query, enriched_text) in una lista unica
    # e facciamo UNA SOLA chiamata a predict().
    # NOTA: si usa SEMPRE la query_text originale, non le sotto-query.
    # ════════════════════════════════════════════════════════════════════════
    services_data = {}
    for doc_id in stage1_scores:
        # Projection: escludiamo _id (lo abbiamo già) e selezioniamo solo
        # i campi necessari. Niente roundtrip dumps/loads, tutti i sotto-campi
        # sono dict di stringhe (no ObjectId nidificati).
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
        # Esclusi: POST /register (registrazione interna del mock) e qualsiasi
        # endpoint /health (utile per liveness check, inutile per il planner LLM).
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

    # ── Costruisci tutte le coppie in una lista unica ────────────────────────
    all_pairs = []
    pair_map  = []  # (doc_id, op) per ogni elemento di all_pairs

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

    # ── Unica chiamata a predict() per tutte le coppie (con partizionamento) ──
    # Impostiamo batch_size=4 per forzare il CrossEncoder a calcolare piccoli blocchi,
    # prevenendo l'esaurimento della RAM e l'utilizzo dello Swap Disk su CPU in Docker.
    all_scores = reranker_model.predict(all_pairs, batch_size=4)

    # ── Aggrega gli score per doc_id ─────────────────────────────────────────
    scores_by_service: dict = {doc_id: {} for doc_id in services_data}
    for (doc_id, op), score in zip(pair_map, all_scores):
        scores_by_service[doc_id][op] = float(score)

    merged: dict = {}
    for doc_id, sdata in services_data.items():
        ops = sdata["ops"]

        # ── HEURISTIC: Virtual boost per GET list endpoints ───────────────────
        # I GET senza path parameter sono gli unici endpoint che permettono di
        # listare le risorse, e sono operativamente necessari per orchestrare
        # le chiamate successive. Il boost è puramente ordinale (alza i GET
        # list in cima allo slice senza alterare il best_endpoint_score usato
        # nel ranking RRF). /health è escluso: non porta dati utili.
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

        # Best score semantico reale — usa max() sui grezzi, NON relevant_ops[0]
        # che potrebbe essere drogato dal boost.
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

    # ════════════════════════════════════════════════════════════════════════
    # RECIPROCAL RANK FUSION (RRF)
    #
    # Combina due ranking di servizi senza necessità di normalizzare le scale:
    #   - ranking per s1_score (cosine similarity sulla description, range 0-1)
    #   - ranking per best_ep_score (logit del CrossEncoder, range circa ±10)
    #
    # Formula: rrf(s) = Σ 1/(K + rank_i(s))  per ogni ranking i
    # K=60 è il valore canonico in letteratura (Cormack et al. 2009),
    # sufficientemente grande da smussare le differenze di rank top e
    # sufficientemente piccolo da non appiattire tutto il ranking.
    #
    # Vantaggi rispetto allo scoring composito ALPHA*s1 + BETA*sigmoid(z):
    #   - zero iperparametri empirici da giustificare
    #   - robusto a outlier (un best_ep_score molto alto non distrugge il ranking)
    #   - citazione bibliografica solida per la difesa di tesi
    # ════════════════════════════════════════════════════════════════════════
    services_list = list(merged.values())

    ranked_by_s1 = sorted(services_list, key=lambda x: x["_stage1_score"],   reverse=True)
    ranked_by_ep = sorted(services_list, key=lambda x: x["_best_ep_score"], reverse=True)

    # Rank 1-based come da formulazione originale di Cormack et al. (2009):
    # il primo classificato ha rank=1, contribuendo con 1/(K+1) allo score.
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

    # Cleanup campi interni prima del token budget
    for s in ordered_services:
        s.pop("_stage1_score", None)
        s.pop("_best_ep_score", None)
        s.pop("_rrf_score", None)

    # ════════════════════════════════════════════════════════════════════════
    # TOKEN BUDGET con trimming graceful
    #
    # Per ogni servizio si tenta prima l'inserimento con tutti gli endpoint.
    # Se non ci sta nel budget residuo, si scala progressivamente il numero
    # di endpoint (top-N per ep_score, già ordinati) fino a trovare la
    # configurazione minima che entra. Solo se nemmeno il singolo endpoint
    # migliore entra nel budget, il servizio viene scartato.
    # ════════════════════════════════════════════════════════════════════════
    _DICT_FIELDS = ("capabilities", "endpoints", "response_schemas",
                    "request_schemas", "parameters")

    def _trim_service(s: dict, keep_ops: list) -> dict:
        """Restituisce una copia del servizio con solo gli endpoint in keep_ops."""
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
        f"  {sid:<50} ({n}/{tot} ep) → {ep_names}"
        for sid, n, tot, ep_names in budget_log
    )
    logger.info(
        f"[SEARCH] Stage1={len(stage1_scores)} → Stage2={len(merged)} → "
        f"RRF_ranked={len(ordered_services)} → "
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

    # ── Stage 2 index: 1 vettore per endpoint (capability text) ──────────────
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

    # ── Stage 1 index: 1 vettore per servizio (description text) ─────────────
    # Preferisce description. Usa generated_description solo se description
    # è assente o troppo corta (<100 caratteri).
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
    Reindicizza i testi descrittivi di tutti i servizi in services_index (Stage 1).

    Preferisce description; usa generated_description solo come fallback quando
    description è assente o inferiore a 100 caratteri.
    Usa embed_passage() per il corretto ruolo asimmetrico.
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
    Aggiorna response_schemas, request_schemas, parameters e/o
    generated_description di un servizio.
    Chiamato dall'api-importer dopo aver estratto gli schema con prance.
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
            logger.info("🛠️ Creating Qdrant collection...")
            create_vector_collection()
            logger.info("📦 Loading embedding model...")
            load_model()
            is_server_ready = True
            logger.info("✅ Server is ready.")
    except Exception as e:
        logger.exception("❌ Failed to initialize application")
        sys.exit(1)

    server = WSGIServer(('0.0.0.0', 5000), app)
    try:
        print("🚀 Starting Flask app with Cheroot on http://0.0.0.0:5000")
        server.start()
    except KeyboardInterrupt:
        print("🛑 Shutting down server...")
        server.stop()
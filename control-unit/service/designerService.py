"""
designerService.py
==================

Triage + API Designer fallback in UNA sola chiamata LLM.

Quando il planner principale produce un piano vuoto (tasks=[]), il Controller
delega a questa classe una SECONDA chiamata LLM che fa due cose in un colpo:

  1) CLASSIFICA il motivo del piano vuoto scegliendo una categoria da un
     enum chiuso (OUT_OF_DOMAIN / AMBIGUOUS / INVALID / UNKNOWN).
  2) SE e solo se la categoria e' OUT_OF_DOMAIN, progetta anche un
     contratto REST concettuale che coprirebbe il gap; altrimenti
     il campo service_contract viene lasciato a null.

Il guided decoding di Ollama garantisce, tramite anyOf(null | contract),
che il modello non possa produrre un contratto vuoto/malformato quando
non serve, e che quando serve abbia tutti i campi richiesti.

La classe e' stateless (nessun effetto collaterale, solo chiamate HTTP a
Ollama) e dipende solo da 'requests' + stdlib.
"""

import json
import os
import re
import time

import requests


class Designer:
    """
    Triage + designer. Metodo pubblico unico: analyze().

    Uso tipico dal Controller:

        designer = Designer(fallback_model=controller.model_name)
        result = designer.analyze(
            query=user_query,
            plan_reasoning=plan.get("reasoning", ""),
            discovered_services=disc_services,
            discovered_capabilities=disc_caps,
            input_files=analyzed_files,
        )
        # result == {
        #   "category":         "OUT_OF_DOMAIN" | "AMBIGUOUS" | "INVALID" | "UNKNOWN",
        #   "justification":    str,
        #   "service_contract": dict | None,   # popolato solo se OUT_OF_DOMAIN
        # }
        # In caso di errore/JSON invalido: result == {} (dict vuoto).
    """

    # ------------------------------------------------------------------
    # INIT
    # ------------------------------------------------------------------

    def __init__(self, fallback_model: str = "glm-4.7-flash:q4_K_M"):
        """
        fallback_model:  modello da usare se la env DESIGNER_LLM_MODEL
                         non e' settata. Tipicamente lo stesso del planner
                         principale, passato dal Controller.
        """
        self.model_name = os.environ.get("DESIGNER_LLM_MODEL", fallback_model)

    # ------------------------------------------------------------------
    # PROMPT BUILDERS
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """
        System prompt unico per triage + design.

        Struttura:
          <role>/<task>/<categories>/<output_contract>/<design_principles>/
          <rationale_protocol>/<hard_constraints>/<examples>

        Le sezioni di design si applicano SOLO quando category == OUT_OF_DOMAIN.
        """
        # Esempio di contratto progettato: dominio (energy) volutamente
        # diverso da quelli reali del progetto, per evitare imitazione.
        contract_example = {
            "rationale": (
                "GAP: user asks for per-building energy consumption | "
                "CATALOG: environment sensors, logistics warehouses, hospital wards — no energy data | "
                "MISSING: no endpoint returns kWh readings per building | "
                "PROPOSAL: smart-building-energy-mock with GET /building-meter"
            ),
            "service_name": "Smart Building Energy API",
            "service_id": "smart-building-energy-mock",
            "description": (
                "Exposes real-time and historical energy consumption data per building, "
                "reusing the zoneId convention of the existing catalog so it can be "
                "joined with environment and logistics services."
            ),
            "suggested_endpoints": [
                {
                    "method": "GET",
                    "path": "/building-meter",
                    "purpose": "List buildings with their latest kWh reading and alert flag.",
                    "parameters": "zoneId(Z-NORD,Z-SUD,Z-EST,Z-OVEST), alertActive(true,false)",
                    "request_schema": "",
                    "response_schema": "{buildingId:str, zoneId:str, kWhLast24h:float, alertActive:bool, lastReadingAt:str}"
                }
            ],
            "integration_notes": (
                "Uses the same zoneId enum as smart-environment-sensors so SQL "
                "joins on zoneId are trivial."
            )
        }
        ood_example = {
            "category": "OUT_OF_DOMAIN",
            "justification": "Catalog has no service exposing book/library data.",
            "service_contract": contract_example,
        }
        amb_example = {
            "category": "AMBIGUOUS",
            "justification": "Query 'show me something' is too vague to decompose.",
            "service_contract": None,
        }

        return f"""<role>
You are a triage agent AND an Expert REST API Architect.
</role>

<task>
An upstream planner received a user query and a service catalog, and produced
an EMPTY plan (tasks=[]). Your job is to do TWO things in a single response:

  (1) Classify WHY the plan is empty, picking exactly one label from
      <categories>.
  (2) IF AND ONLY IF the chosen category is OUT_OF_DOMAIN, design one
      conceptual REST service that would close the gap and return it in
      the 'service_contract' field.
      For every other category, set 'service_contract' to null.
</task>

<categories>
OUT_OF_DOMAIN - No service in the catalog covers the domain of the query.
                A new service would need to be designed and registered.
                (This is the ONLY case in which you also design a contract.)
AMBIGUOUS     - The query is too vague or underspecified to decompose
                deterministically (e.g. "show me something", "help me").
INVALID       - The query requires a parameter value NOT documented in the
                catalog's enums, OR is logically impossible / self-contradictory.
UNKNOWN       - None of the above, or insufficient information to decide.
</categories>

<output_contract>
Output ONLY a valid JSON object. Zero prose, zero markdown fences.
Top-level schema:
  {{
    "category":         "OUT_OF_DOMAIN" | "AMBIGUOUS" | "INVALID" | "UNKNOWN",
    "justification":    "string — ONE short sentence explaining the choice",
    "service_contract": null  OR  <full contract object, see below>
  }}

When category == "OUT_OF_DOMAIN", service_contract MUST be the full object:
  {{
    "rationale":          "string — chain of draft, see <rationale_protocol>",
    "service_name":       "string — human-readable name",
    "service_id":         "string — machine id, kebab-case, suffix '-mock'",
    "description":        "string — one-paragraph purpose of the service",
    "suggested_endpoints":[ ...at least ONE endpoint... ],
    "integration_notes":  "string — how this service joins with existing ones"
  }}
Each endpoint has: method, path, purpose, parameters, request_schema, response_schema.
Use empty string "" for parameters / request_schema when not applicable.

When category != "OUT_OF_DOMAIN", service_contract MUST be null. Do not
invent a contract "just in case" — it is explicitly forbidden.
</output_contract>

<design_principles>
These principles apply ONLY to the service_contract you produce when the
category is OUT_OF_DOMAIN. Ignore them otherwise.

DP-1  DO NOT DUPLICATE
      Inspect the EXISTING CATALOG in the user prompt. If a service already
      partially covers the request, the gap is smaller than it looks. Call
      it out in 'rationale' and design ONLY the missing piece.

DP-2  COHERENT NAMING
      - service_id:  lowercase kebab-case, suffix '-mock'
                     (e.g. smart-building-energy-mock).
      - path:        kebab-case, singular resource noun (e.g. /building-meter).
      - field names: camelCase in schemas (e.g. zoneId, lastReadingAt).

DP-3  REUSE DOMAIN CONVENTIONS
      Reuse conventions already present in the existing catalog whenever
      applicable: zoneId enum values, timestamp field names, status enums,
      alertActive flags. This lets the new service SQL-join with existing ones.

DP-4  MINIMAL SURFACE
      Expose the fewest endpoints sufficient to answer the user query.
      Prefer GET for discovery, POST for actions, PUT for updates.

DP-5  SCHEMA RICHNESS
      response_schema and request_schema MUST enumerate concrete field names
      and types in the catalog's compact format:
        "{{field1:str, field2:int, field3:bool, field4:float, field5:arr}}"
      Supported types: str, int, float, bool, arr, obj. Mark required request
      fields with '*' (e.g. "{{name:str*, zoneId:str*, quota:int}}").

DP-6  PARAMETER ENUMS
      When a query parameter is an enum, list its values in parentheses:
        "zoneId(Z-NORD,Z-SUD), status(active,inactive)"
      Mark required parameters with '*'.
</design_principles>

<rationale_protocol>
Inside 'service_contract.rationale', use CHAIN OF DRAFT — one phase per line,
keywords only, separated by ' | '. Never full sentences. Never "wait",
"actually", "or perhaps".

  GAP:      <what the user asks that the catalog cannot provide>
  CATALOG:  <what the closest existing services already offer>
  MISSING:  <the specific capability the gap boils down to>
  PROPOSAL: <service_id and primary endpoint in one line>
</rationale_protocol>

<hard_constraints>
HC-1  The designed service MUST NOT duplicate an existing service_id.
HC-2  When service_contract is NOT null, it MUST contain at least ONE endpoint.
HC-3  service_id MUST match the regex ^[a-z][a-z0-9-]*-mock$.
HC-4  method MUST be exactly one of GET, POST, PUT, DELETE.
HC-5  response_schema MUST be non-empty for every GET endpoint.
</hard_constraints>

<examples>
Example 1 — out-of-domain (catalog has no library-style service):
{json.dumps(ood_example, indent=2)}

Example 2 — ambiguous query:
{json.dumps(amb_example, indent=2)}
</examples>
"""

    def _build_user_prompt(self, query: str,
                           plan_reasoning: str,
                           discovered_services: list,
                           discovered_capabilities: list,
                           input_files=None) -> str:
        """
        User prompt: include catalogo compatto, ragionamento del planner e query.
        Il plan_reasoning e' fondamentale: da' al triage l'indizio piu' forte
        (il planner ha gia' scritto "no service provides X" o simili).
        """
        lines = ["EXISTING CATALOG (do NOT duplicate any of these services):"]
        if discovered_services:
            for i, s in enumerate(discovered_services):
                lines.append(f"\nSERVICE_ID:  {s.get('_id')}")
                lines.append(f"NAME:        {s.get('name')}")
                desc = (s.get("description") or "").strip()
                if desc:
                    lines.append(f"DESCRIPTION: {desc}")
                caps = discovered_capabilities[i] \
                       if i < len(discovered_capabilities or []) else {}
                if caps:
                    lines.append("ENDPOINTS:")
                    for key, cap_desc in caps.items():
                        if key == "POST /register":
                            continue
                        lines.append(f"  {key}  —  {cap_desc}")
        else:
            lines.append("(no services available in the current retrieval)")

        return f"""{chr(10).join(lines)}

PLANNER'S REASONING (why it produced no tasks):
{plan_reasoning or '(none provided)'}

FILES ATTACHED BY USER: {str(input_files) if input_files else 'none'}

USER QUERY:
{query}

Classify and, if the category is OUT_OF_DOMAIN, design the missing service."""

    # ------------------------------------------------------------------
    # OUTPUT SCHEMA (guided decoding)
    # ------------------------------------------------------------------

    def _build_output_schema(self) -> dict:
        """
        Schema unificato: category + justification + service_contract (null|obj).

        L'uso di anyOf su service_contract forza il modello a scegliere:
          - o null (quando la categoria non e' OUT_OF_DOMAIN)
          - o l'oggetto completo con tutti i required popolati.
        Non e' possibile produrre un contract parziale/malformato.
        """
        contract_inner = {
            "type": "object",
            "properties": {
                "rationale":    {"type": "string"},
                "service_name": {"type": "string"},
                "service_id":   {"type": "string"},
                "description":  {"type": "string"},
                "suggested_endpoints": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {
                                "type": "string",
                                "enum": ["GET", "POST", "PUT", "DELETE"],
                            },
                            "path":            {"type": "string"},
                            "purpose":         {"type": "string"},
                            "parameters":      {"type": "string"},
                            "request_schema":  {"type": "string"},
                            "response_schema": {"type": "string"},
                        },
                        "required": [
                            "method", "path", "purpose",
                            "parameters", "request_schema", "response_schema",
                        ],
                        "additionalProperties": False,
                    },
                },
                "integration_notes": {"type": "string"},
            },
            "required": [
                "rationale", "service_name", "service_id",
                "description", "suggested_endpoints", "integration_notes",
            ],
            "additionalProperties": False,
        }

        return {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["OUT_OF_DOMAIN", "AMBIGUOUS", "INVALID", "UNKNOWN"],
                },
                "justification": {"type": "string"},
                "service_contract": {
                    "anyOf": [
                        {"type": "null"},
                        contract_inner,
                    ],
                },
            },
            "required": ["category", "justification", "service_contract"],
            "additionalProperties": False,
        }

    # ------------------------------------------------------------------
    # VALIDAZIONE POST-PARSE
    # ------------------------------------------------------------------

    def _validate_contract(self, contract: dict,
                           existing_ids: set) -> list[str]:
        """
        Check leggero su HC-1, HC-3, HC-5 dopo il parsing.
        HC-2 (almeno un endpoint) e HC-4 (method enum) sono gia' garantiti
        dal guided decoding (minItems=1 e enum nello schema).
        Ritorna lista di warning; lista vuota = contratto pulito.
        """
        warnings: list[str] = []

        sid = contract.get("service_id", "")
        if not re.match(r"^[a-z][a-z0-9-]*-mock$", sid):
            warnings.append(f"service_id '{sid}' non rispetta HC-3 "
                            f"(atteso kebab-case con suffisso '-mock')")

        if sid in existing_ids:
            warnings.append(f"service_id '{sid}' duplica un servizio "
                            f"gia' presente nel catalogo (viola HC-1)")

        for idx, ep in enumerate(contract.get("suggested_endpoints", [])):
            if ep.get("method") == "GET" and not ep.get("response_schema"):
                warnings.append(
                    f"endpoint #{idx} ({ep.get('method')} "
                    f"{ep.get('path')}): response_schema vuoto per GET "
                    f"(viola HC-5)"
                )

        return warnings

    # ------------------------------------------------------------------
    # ENTRY POINT
    # ------------------------------------------------------------------

    def analyze(self, query: str,
                plan_reasoning: str = "",
                discovered_services: list = None,
                discovered_capabilities: list = None,
                input_files=None) -> dict:
        """
        UNICA chiamata LLM per il fallback:
          - classifica il motivo del piano vuoto
          - se OUT_OF_DOMAIN, progetta il servizio mancante
          - altrimenti ritorna service_contract = None

        Parametri:
            query:                    la query utente originale
            plan_reasoning:           il reasoning del planner (prima chiamata)
            discovered_services:      servizi correntemente nel catalogo
            discovered_capabilities:  capabilities per ciascun servizio
            input_files:              allegati utente (se presenti)

        Ritorna un dict:
            {
              "category":         str,   # enum
              "justification":    str,
              "service_contract": dict | None,
            }
        Oppure {} in caso di errore / JSON invalido.
        """
        system_prompt = self._build_system_prompt()
        user_prompt   = self._build_user_prompt(
            query, plan_reasoning,
            discovered_services or [], discovered_capabilities or [],
            input_files,
        )
        output_schema = self._build_output_schema()

        url = os.environ.get("OLLAMA_API_URL", "http://localhost:11434")

        try:
            t0 = time.perf_counter()
            response = requests.post(
                f"{url}/api/chat",
                json={
                    "model":    self.model_name,
                    "format":   output_schema,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "think":   False,
                    "options": {
                        "temperature": 0.2,    # leggera creativita' solo per il design
                        "num_ctx":     8192,
                    },
                    "stream":  False,
                },
                timeout=60,
            )
            response.raise_for_status()
            raw     = response.json()["message"]["content"].strip()
            latency = time.perf_counter() - t0
            print(f"[LATENCY] Designer analyze completato in {latency:.2f}s")

            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                print(f"[DESIGNER WARN] JSON non valido. Anteprima: {raw[:200]}")
                return {}

            if not isinstance(parsed, dict):
                print(f"[DESIGNER WARN] Output non è un oggetto JSON: {type(parsed)}")
                return {}

            category      = parsed.get("category", "UNKNOWN")
            justification = parsed.get("justification", "")
            contract      = parsed.get("service_contract")

            print(f"[DESIGNER] category:      {category}")
            print(f"[DESIGNER] justification: {justification}")

            # Coerenza: se categoria != OUT_OF_DOMAIN, ignoriamo qualunque
            # contract spurio prodotto dal modello (difesa in profondita').
            if category != "OUT_OF_DOMAIN":
                if contract is not None:
                    print("[DESIGNER] category non-OOD ma contract presente → ignoro.")
                parsed["service_contract"] = None
                return parsed

            # category == OUT_OF_DOMAIN → il contract deve esserci
            if not isinstance(contract, dict):
                print("[DESIGNER WARN] category=OUT_OF_DOMAIN ma contract "
                      "mancante o non-dict.")
                return parsed

            # Validazione leggera post-parse sul contract
            existing_ids = {s.get("_id") for s in (discovered_services or [])}
            warnings     = self._validate_contract(contract, existing_ids)
            if warnings:
                print(f"[DESIGNER VALIDATOR] {len(warnings)} warning/s:")
                for w in warnings:
                    print(f"  - {w}")
            else:
                print("[DESIGNER VALIDATOR] ✅ Contratto coerente.")

            print(f"[DESIGNER] service_id:    {contract.get('service_id')}")
            print(f"[DESIGNER] service_name:  {contract.get('service_name')}")
            print(f"[DESIGNER] endpoints:     "
                  f"{len(contract.get('suggested_endpoints', []))}")

            return parsed

        except requests.exceptions.RequestException as e:
            print(f"[DESIGNER ERROR] HTTP: {e}")
            return {}
        except Exception as e:
            print(f"[DESIGNER ERROR] {e}")
            return {}
from service.discoveryService import Discovery
from service.designerService  import Designer
from service.llm_provider     import build_provider, build_schema_for_format
from flask import Response
import json
import requests
import re
import aiohttp
import asyncio
import os
import mimetypes
import shutil
import time
import math
import jmespath
import duckdb
import pandas as pd
from jmespath import exceptions as jmespath_exc
from urllib.parse import urlparse, parse_qs


class PlanValidator:
    """Validates LLM-generated execution plans for structural and semantic correctness."""

    VALID_OPERATIONS = {"GET", "POST", "PUT", "DELETE", "SQL"}

    @staticmethod
    def validate(plan: dict, available_service_ids: list) -> tuple[bool, list[str]]:
        errors = []
        if not isinstance(plan, dict) or "tasks" not in plan:
            return False, ["Missing or invalid 'tasks' array"]
        tasks = plan["tasks"]
        if not isinstance(tasks, list):
            return False, ["'tasks' must be an array"]
        if len(tasks) == 0:
            return True, []

        service_id_set = set(available_service_ids)
        defined_task_names = set()

        for i, task in enumerate(tasks):
            prefix = f"Task {i} ({task.get('task_name', 'unnamed')})"
            if not isinstance(task, dict):
                errors.append(f"{prefix}: not a dictionary")
                continue
            for field in ("task_name", "service_id", "url", "operation", "input"):
                if field not in task:
                    errors.append(f"{prefix}: missing required field '{field}'")

            sid = task.get("service_id", "")
            op = task.get("operation", "")

            if sid and sid not in service_id_set and op != "SQL":
                errors.append(f"{prefix}: unknown service_id '{sid}'")

            url = str(task.get("url", ""))

            if op and op not in PlanValidator.VALID_OPERATIONS:
                errors.append(f"{prefix}: invalid operation '{op}'")

            if op != "SQL":
                if not url:
                    errors.append(f"{prefix}: empty url")
                elif not url.startswith("http") and "{{" not in url:
                    errors.append(f"{prefix}: url must start with http:// (got '{url[:60]}')")

                clean_url = re.sub(r'\{\{.*?\}\}', '', url)
                if re.search(r'(?<!\{)\{(?!\{)[^{]*\}(?!\})', clean_url):
                    errors.append(f"{prefix}: unresolved path parameter in url '{url[:60]}'")

                if op == "GET" and isinstance(task.get("input"), dict) and task.get("input"):
                    errors.append(
                        f"{prefix}: GET request has non-empty 'input' dict — "
                        f"query params must be in the url, not in input field"
                    )

            task_str = json.dumps(task)
            current_name = task.get("task_name", "")
            for ref in re.findall(r'\{\{\s*([a-zA-Z0-9_]+)', task_str):
                if ref == current_name:
                    errors.append(f"{prefix}: self-reference — task cannot reference itself in chaining")
                elif ref not in defined_task_names:
                    errors.append(f"{prefix}: references undefined task '{ref}'")

            defined_task_names.add(current_name)

        return len(errors) == 0, errors


class Controller:

    def __init__(self):
        self.model_name = os.environ.get("LLM_MODEL", "qwen3.5:27b")
        self.backend_mode = "MOCK"
        self.designer = Designer(fallback_model=self.model_name)
        self.llm = build_provider(model_name=self.model_name)
        os.makedirs("Files", exist_ok=True)

    def analyze_files(self, files: list):
        analyzed = []
        for f in files:
            filename = os.path.basename(f.filename)
            content_type = f.mimetype or mimetypes.guess_type(filename)[0]
            path = os.path.join("Files", filename)
            f.save(path)
            category = "unknown"
            if content_type == "application/pdf":
                category = "document"
            elif content_type and content_type.startswith("image/"):
                category = "image"
            elif content_type in ["text/csv", "application/vnd.ms-excel"]:
                category = "tabular"
            analyzed.append({
                "filename": filename,
                "content_type": content_type,
                "size": f.content_length or 0,
                "path": path,
                "category": category,
            })
        return analyzed

    def _build_input_format_schema(self, discovered_request_schemas: list) -> dict:
        """
        Aggregate all valid field keys from every registered request_schema
        and produce a JSON schema with additionalProperties: false.

        This schema is injected as the guided-decoding 'input' field schema:
        the LLM is structurally unable to generate keys not present in any
        registered service's request schema.

        Each field is typed as string | <native type> so that JMESPath
        placeholders (strings at generation time) pass validation while
        native types are accepted for direct values.

        Expected compact schema format from the database (e.g. smart-environment-sensors):
            "{data:arr*, by:str*, order:enum(asc,desc), top:int}"
        """
        field_pattern = re.compile(r'(\w+):([\w]+)(?:\([^)]*\))?\*?')

        type_map = {
            "arr":   {"type": ["array",   "string"]},
            "str":   {"type": "string"},
            "int":   {"type": ["integer", "string"]},
            "float": {"type": ["number",  "string"]},
            "bool":  {"type": ["boolean", "string"]},
            "obj":   {"type": ["object",  "string"]},
            "enum":  {"type": "string"},
            "any":   {},
        }

        all_properties = {}

        for schemas_per_service in discovered_request_schemas:
            if not isinstance(schemas_per_service, dict):
                continue
            for endpoint, schema_str in schemas_per_service.items():
                if not schema_str:
                    continue
                for match in field_pattern.finditer(schema_str):
                    field_name = match.group(1)
                    field_type = match.group(2)
                    if field_name not in all_properties:
                        all_properties[field_name] = type_map.get(field_type, {})

        if not all_properties:
            print("[INPUT SCHEMA] No request_schema found — using permissive fallback.")
            return {"type": ["string", "object", "null"]}

        all_properties["sql_query"] = {"type": "string"}

        print(f"[INPUT SCHEMA] Allowed keys for 'input': {sorted(all_properties.keys())}")
        return {
            "type":                 "object",
            "properties":          all_properties,
            "additionalProperties": False,
        }

    def _validate_plan(self,
                       plan: dict,
                       discovered_services: list,
                       discovered_request_schemas: list,
                       discovered_parameters: list | None = None,
                       backend_mode: str = "MOCK") -> list[str]:
        """
        Post-parse validation of the LLM-generated plan.

        Checks per task:
          1. (POST/PUT/PATCH) Input body keys must be documented in the
             endpoint's request_schema from the catalog.
          2. (GET) Every query parameter in the URL must be in the documented
             parameter set for that endpoint.
          3. (GET, MOCK only) A URL must not combine two or more query params
             in AND, because Microcks matches one parameter at a time.

        Returns a list of warnings; an empty list means a clean plan.
        """
        warnings = []
        field_pattern = re.compile(r'(\w+):')
        endpoint_valid_keys = {}
        endpoint_valid_params = {}

        for i in range(len(discovered_services)):
            schemas = discovered_request_schemas[i] \
                      if i < len(discovered_request_schemas) else {}
            params = discovered_parameters[i] \
                     if discovered_parameters and i < len(discovered_parameters) else {}

            for ep_key, schema_str in schemas.items():
                if not schema_str:
                    continue
                path = ep_key.split(" ")[-1]
                endpoint_valid_keys[path] = set(field_pattern.findall(schema_str))

            for ep_key, params_str in params.items():
                if not params_str:
                    continue
                path = ep_key.split(" ")[-1]
                endpoint_valid_params[path] = set(field_pattern.findall(str(params_str)))

        for task in plan.get("tasks", []):
            task_name = task.get("task_name", "?")
            url = task.get("url", "")
            input_val = task.get("input")
            operation = str(task.get("operation") or "GET").upper()

            if isinstance(input_val, dict) and input_val:
                matched_valid_keys = None
                for ep_path, valid_keys in endpoint_valid_keys.items():
                    if ep_path in url:
                        matched_valid_keys = valid_keys
                        break

                if matched_valid_keys is not None:
                    hallucinated = set(input_val.keys()) - matched_valid_keys
                    if hallucinated:
                        warnings.append(
                            f"[SCHEMA VIOLATION] Task '{task_name}' → "
                            f"invalid keys: {sorted(hallucinated)} | "
                            f"allowed keys: {sorted(matched_valid_keys)}"
                        )

            if operation == "GET" and url:
                try:
                    parsed = urlparse(url)
                    qs = parse_qs(parsed.query, keep_blank_values=True)
                except Exception:
                    qs = {}

                param_names = set(qs.keys())
                matched_valid_params = None
                for ep_path, valid_names in endpoint_valid_params.items():
                    if ep_path in parsed.path:
                        matched_valid_params = valid_names
                        break

                if matched_valid_params:
                    hallucinated_params = param_names - matched_valid_params
                    if hallucinated_params:
                        warnings.append(
                            f"[PARAM VIOLATION] Task '{task_name}' → "
                            f"undocumented query params: {sorted(hallucinated_params)} | "
                            f"documented: {sorted(matched_valid_params)}"
                        )

                if backend_mode == "MOCK" and len(param_names) >= 2:
                    warnings.append(
                        f"[HC-12 VIOLATION] Task '{task_name}' → "
                        f"GET combines {len(param_names)} query params in AND: "
                        f"{sorted(param_names)} | "
                        f"url: {url[:120]}"
                    )

        return warnings

    def query_ollama(self, system_prompt: str, user_prompt: str,
                     input_schema: dict | None = None) -> tuple[str, float]:
        schema = build_schema_for_format(input_schema)
        return self.llm.chat(system_prompt, user_prompt, schema)

    def _build_system_prompt(self, backend_mode: str = "MOCK") -> str:

        if backend_mode == "REAL":
            hc12_block = ""
            self_check_hc12 = (
                "  \u25a1 Every query param in a GET url is documented in the "
                "endpoint's parameters list (HC-1)."
            )
        else:
            hc12_block = """

HC-12  NO AND-COMBINATION OF QUERY PARAMS
      A GET url MUST NOT carry two or more query parameters in AND (e.g. ?a=x&b=y)
      unless the endpoint's description EXPLICITLY documents that the combination
      is supported.
      Why: mock servers (Microcks) match each parameter in isolation against
      registered examples. AND-combining two independently-documented parameters
      returns the first matched example, NOT the intersection — silently
      producing wrong data.
      Correct pattern when two independent filters are needed and no single
      endpoint documents them together: issue ONE GET per filter, combine
      the results in a terminal SQL task (e.g. JOIN on zoneId, or WHERE IN)."""
            self_check_hc12 = (
                "  \u25a1 No GET url combines two or more query params in AND (HC-12) "
                "\u2014 use one GET per filter + a terminal SQL task."
            )

        ex_a = {
            "reasoning": (
                "DECOMPOSE: available books | "
                "MAP: smart-library-mock / GET /book | "
                "CHAIN: none | "
                "COMBINE: single task | "
                "FILTER: status=available (one param, catalog enum) | "
                "VALIDATE: \u2713"
            ),
            "tasks": [{
                "task_name":  "get_available_books",
                "service_id": "smart-library-mock",
                "url":        "http://mock-server:8080/rest/Smart+Library+Management+API/1.0/book?status=available",
                "operation":  "GET",
                "input":      ""
            }]
        }

        ex_b = {
            "reasoning": (
                "DECOMPOSE: find patient Rossi \u2192 set discharged | "
                "MAP: smart-hospital-mock / GET /patient + PUT /patient/{id} | "
                "CHAIN: PUT path \u2190 get_all_patients[?surname==\'Rossi\'] | [0].id | "
                "COMBINE: chain: discharge_patient consumes get_all_patients via JMESPath | "
                "FILTER: surname match via JMESPath, not query param | "
                "VALIDATE: \u2713 id in path, \u2713 no bare placeholders"
            ),
            "tasks": [
                {
                    "task_name":  "get_all_patients",
                    "service_id": "smart-hospital-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Hospital+Management+API/1.0/patient",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "discharge_patient",
                    "service_id": "smart-hospital-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Hospital+Management+API/1.0/patient/{{get_all_patients[?surname=='Rossi'] | [0].id}}",
                    "operation":  "PUT",
                    "input": {
                        "zoneId":    "Z-SUD",
                        "surname":   "Rossi",
                        "status":    "discharged",
                        "wardId":    12,
                        "updatedAt": "2025-09-25T14:00:00Z"
                    }
                }
            ]
        }

        ex_c = {
            "reasoning": (
                "DECOMPOSE: canteens near occupied halls | "
                "MAP: smart-campus-mock / GET /lecture-hall + GET /canteen | "
                "CHAIN: canteen?zoneIds \u2190 get_occupied_halls[*].zoneId | join(\',\',@) | "
                "COMBINE: chain: get_canteens_near_halls consumes get_occupied_halls via JMESPath | "
                "FILTER: lecture-hall \u2192 status=occupied; canteen \u2192 zoneIds param documented | "
                "VALIDATE: \u2713 join on string field, \u2713 zoneIds in catalog"
            ),
            "tasks": [
                {
                    "task_name":  "get_occupied_halls",
                    "service_id": "smart-campus-mock",
                    "url":        "http://mock-server:8080/rest/Smart+University+Campus+API/1.0/lecture-hall?status=occupied",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "get_canteens_near_halls",
                    "service_id": "smart-campus-mock",
                    "url":        "http://mock-server:8080/rest/Smart+University+Campus+API/1.0/canteen?zoneIds={{get_occupied_halls[*].zoneId | join(',', @)}}",
                    "operation":  "GET",
                    "input":      ""
                }
            ]
        }

        ex_d = {
            "reasoning": (
                "DECOMPOSE: rank warehouses by avg temp per zone | "
                "MAP: smart-logistics-mock / GET /warehouse + GET /thermometer | "
                "CHAIN: SQL joins both on zoneId | "
                "COMBINE: sql join+aggregate over get_warehouses and get_thermometers | "
                "FILTER: no query params; avg+rank \u2192 SQL | "
                "VALIDATE: \u2713 table names = task names, \u2713 no {{}} in SQL"
            ),
            "tasks": [
                {
                    "task_name":  "get_warehouses",
                    "service_id": "smart-logistics-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Logistics+API/1.0/warehouse",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "get_thermometers",
                    "service_id": "smart-logistics-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Logistics+API/1.0/thermometer",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "rank_by_avg_temp",
                    "service_id": "sql-processor",
                    "url":        "",
                    "operation":  "SQL",
                    "input":      {"sql_query": "SELECT w.id, w.name, w.zoneId, AVG(t.lastReading) AS avg_temp FROM get_warehouses w JOIN get_thermometers t ON w.zoneId = t.zoneId GROUP BY w.id, w.name, w.zoneId ORDER BY avg_temp DESC"}
                }
            ]
        }

        ex_e = {
            "reasoning": (
                "DECOMPOSE: available hotels | avoid noisy districts | "
                "MAP: smart-hospitality-mock / GET /hotel + smart-acoustics-mock / GET /noise-sensor | "
                "CHAIN: SQL set difference on districtId | "
                "COMBINE: sql set_difference get_available_hotels minus get_noisy_sensors | "
                "FILTER: hotel available=true; noise-sensor level=high | "
                "VALIDATE: \u2713 two GETs + terminating SQL, \u2713 exclusion via NOT IN"
            ),
            "tasks": [
                {
                    "task_name":  "get_available_hotels",
                    "service_id": "smart-hospitality-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Hospitality+API/1.0/hotel?available=true",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "get_noisy_sensors",
                    "service_id": "smart-acoustics-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Acoustics+API/1.0/noise-sensor?level=high",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "hotels_in_quiet_districts",
                    "service_id": "sql-processor",
                    "url":        "",
                    "operation":  "SQL",
                    "input":      {"sql_query": "SELECT * FROM get_available_hotels WHERE districtId NOT IN (SELECT districtId FROM get_noisy_sensors)"}
                }
            ]
        }

        ex_f = {
            "reasoning": (
                "DECOMPOSE: find the quietest apartments | "
                "MAP: smart-real-estate-mock / GET /apartment + smart-acoustics-mock / GET /noise-sensor | "
                "CHAIN: SQL join and rank | "
                "COMBINE: sql join get_apartments and get_noise_sensors, order by noise | "
                "FILTER: user asks for 'quietest' (qualitative). I MUST NOT invent a threshold like '< 40'. I will use ORDER BY decibelLevel ASC LIMIT 3 | "
                "VALIDATE: \u2713 no invented thresholds, used sorting instead"
            ),
            "tasks": [
                {
                    "task_name":  "get_apartments",
                    "service_id": "smart-real-estate-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Real+Estate+API/1.0/apartment",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "get_noise_sensors",
                    "service_id": "smart-acoustics-mock",
                    "url":        "http://mock-server:8080/rest/Smart+Acoustics+API/1.0/noise-sensor",
                    "operation":  "GET",
                    "input":      ""
                },
                {
                    "task_name":  "quietest_apartments",
                    "service_id": "sql-processor",
                    "url":        "",
                    "operation":  "SQL",
                    "input":      {"sql_query": "SELECT a.*, n.decibelLevel FROM get_apartments a JOIN get_noise_sensors n ON a.zoneId = n.zoneId ORDER BY n.decibelLevel ASC LIMIT 3"}
                }
            ]
        }

        examples_str = (
            f"EXAMPLE A \u2014 single GET with enum filter:\n{json.dumps(ex_a, indent=2)}\n\n"
            f"EXAMPLE B \u2014 GET list \u2192 JMESPath id extraction \u2192 PUT path param:\n{json.dumps(ex_b, indent=2)}\n\n"
            f"EXAMPLE C \u2014 GET with filter \u2192 collect zoneIds \u2192 multi-zone GET:\n{json.dumps(ex_c, indent=2)}\n\n"
            f"EXAMPLE D \u2014 two GETs \u2192 SQL join/rank/aggregate:\n{json.dumps(ex_d, indent=2)}\n\n"
            f"EXAMPLE E \u2014 two GETs \u2192 SQL set difference (exclusion):\n{json.dumps(ex_e, indent=2)}\n\n"
            f"EXAMPLE F \u2014 Qualitative constraint \u2192 SQL Sort (NO magic numbers):\n{json.dumps(ex_f, indent=2)}"
        )

        return f"""<role>
You are an API orchestrator for a distributed system.
Given a user query and a service catalog, produce ONLY a valid JSON execution plan.
</role>

<output_contract>
- Output ONLY raw JSON. Zero prose, zero markdown fences, nothing outside the JSON object.
- Top-level schema: {{"reasoning": "string", "tasks": [...]}}
- Every task must have exactly these five keys: task_name, service_id, url, operation, input.
- CONSTRAINT RULE: Before building the plan, scan the full query for constraint
  phrases ("without X", "avoiding Y", "where Z is [condition]", "near Z",
  "somewhere [adjective]"). Each constraint implies a real-time data requirement.
  Find the service in the catalog that provides that data and add a GET task for it,
  even if the user did not explicitly ask for that data.
  A plan that ignores a constraint phrase FAILS validation \u2014 add the missing task
  before marking VALIDATE as \u2713.
- If the query cannot be satisfied with the available services from the catalog, output:
  {{"reasoning": "No available service can fulfil this request.", "tasks": []}}
</output_contract>


<grounding_rule> 
The catalog below is the ONLY source of truth for services, endpoints, parameters, and field names.
Treat it as a closed world: if something is not in the catalog, it does not exist.
Never invent, guess, or extrapolate service IDs, endpoint paths, parameter names, or field names.
</grounding_rule>

<reasoning_protocol>
Write the "reasoning" value BEFORE the tasks array.
Use CHAIN OF DRAFT format: one line per phase, keywords only, no full sentences.

  DECOMPOSE: <what data is needed>
  MAP:       <service-id / method endpoint> for each need
  CHAIN:     how tasks depend on each other, one of:
              - "none" (independent tasks)
              - "<target_task>.<slot> \u2190 <source_task><jmespath>" (JMESPath chaining)
              - "SQL <op> over <source_task>[, <source_task>]" (SQL reference)
  COMBINE:   <how the final answer is produced \u2014 one of:
              "single task" /
              "chain: last task consumes task_X via JMESPath" /
              "sql filter task_X" /
              "sql join task_X and task_Y" /
              "sql set_difference task_X minus task_Y" /
              "sql set_intersection task_X and task_Y" /
              "sql aggregate/rank over task_X">
  FILTER:    <which param per GET, threshold logic if any>
  VALIDATE:  \u2713 / list any issue found and how it is fixed

COMMIT RULE: write each phase once and move on.
Do not use "wait", "actually", "or perhaps", "however", "but".
If the correct interpretation is ambiguous, pick the most literal reading and commit.

COMBINE CONSISTENCY: the value you write for COMBINE must match the tasks array.
  - "single task"  \u2192 exactly one task.
  - "chain: ..."   \u2192 N HTTP tasks; the last task's url or input contains a
                     {{{{...}}}} placeholder; its raw result IS the answer (no SQL).
  - "sql ..."      \u2192 the LAST task is an SQL task.
A plan whose tasks don't match the declared COMBINE strategy FAILS validation \u2014
fix it before writing the JSON.
</reasoning_protocol>

<hard_constraints>
These rules are absolute and may never be violated.

HC-1  CLOSED WORLD
      Use only services, endpoints, parameters, and field names present in the catalog.

HC-2  NO UNRESOLVED PLACEHOLDERS
      Every url must be fully resolved. {{id}}, {{zoneId}} and similar bare placeholders
      are forbidden. Use chaining expressions {{{{task<expr>}}}} or literal values only.

HC-3  REQUIRED KEYS
      Every task must have: task_name, service_id, url, operation, input.
      service_id must be the exact SERVICE_ID string from the catalog (not the name).

HC-4  REQUIRED PARAMETERS
      Parameters marked * in the catalog are required and must appear in every url.

HC-5  OPERATION VALUES
      operation must be exactly one of: GET  POST  PUT  DELETE  SQL

HC-6  URL RULES
      - Non-SQL tasks: url must start with http:// and be non-empty.
      - SQL tasks: url must be an empty string "".

HC-7  NO DUPLICATE CALLS
      Never build a task that calls the same url+service as an earlier task.
      Reuse earlier results via JMESPath or a SQL task instead.

HC-8  NO CONCATENATED PLACEHOLDERS
      Never write ?param={{{{task1[*].f | join(',',@)}}}},{{{{task2[*].f | join(',',@)}}}}
      Collect all needed data in one prior task and filter with a JMESPath OR expression.

HC-9  PARAMETER VALUES FROM CATALOG
      When setting a query parameter value, copy it verbatim from the catalog's parameter
      examples or enum list \u2014 never from the user's query text. The user may use different
      casing, abbreviations, or synonyms. The catalog value is always authoritative.
      Example: if the catalog shows categoryId example "NARRATIVE" and the user writes
      "narrative" or "Narrative", use "NARRATIVE".

HC-10  NO INVENTED THRESHOLDS
    Never invent numeric thresholds in WHERE clauses (e.g. "< 50", "> 100") UNLESS the
    user explicitly specifies an exact number in their query.
    If the user asks for qualitative states (e.g. "clean x", "quiet x", "cheap x")
    without providing numbers:
    - To find extremes, use an SQL task with ORDER BY field ASC/DESC LIMIT N.
    - To filter by "good/bad/safe" conditions, use catalog-documented boolean or
      enum fields (e.g. alertActive=false, status='ok').

HC-11  JMESPATH IS URL/INPUT SUBSTITUTION ONLY
      JMESPath placeholders {{{{task<expr>}}}} serve EXACTLY ONE purpose: injecting
      values from a prior task's result into the url or input of a subsequent
      HTTP task. This is the CHAIN mechanism \u2014 valid and expected across any
      number of chained HTTP tasks (see Examples B, C).

      JMESPath is NOT a result-combination mechanism. Use an SQL terminal task
      when the final answer requires ANY of the following over prior results:
        - merging rows from TWO OR MORE task results (join, set intersect/diff)
        - aggregation (sum, avg, count, min, max, group by)
        - ranking or sorting across a dataset (order by + limit)
        - post-filtering a single task when the HTTP API could not filter it
          server-side

      Therefore: min_by, max_by, sort_by MUST NOT appear inside {{{{...}}}} \u2014
      express them as SQL (ORDER BY ... LIMIT, MIN, MAX, GROUP BY) in a
      terminal SQL task.

      Quick decision table:
        one GET, API does it all                    \u2192 COMBINE "single task",   no SQL
        GET \u2192 GET/POST/PUT/DELETE via JMESPath url  \u2192 COMBINE "chain: ...",    no SQL
        two+ GETs whose results must be merged      \u2192 COMBINE "sql ...",       last task SQL
        one GET + post-aggregation/ranking          \u2192 COMBINE "sql ...",       last task SQL{hc12_block}
</hard_constraints>

<soft_constraints>
SC-1  MULTI-ZONE QUERIES
      When an endpoint's description documents a ?zoneIds= parameter for multi-zone queries,
      pass zones from a prior task as: ?zoneIds={{{{prev[*].zoneId | join(',', @)}}}}

SC-2  PATH PARAMETER INJECTION
      Inject ids and keys directly into the url string using chaining syntax.
      Never put them in the input field.
</soft_constraints>

<self_check>
Before writing the final JSON, verify every item below:

  \u25a1 Every service_id is copied verbatim from the catalog's SERVICE_ID field.
  \u25a1 Every endpoint path is copied verbatim from the catalog.
  \u25a1 Every parameter name is copied verbatim from the catalog (no guessing synonyms).
  \u25a1 No url contains bare {{...}} unless it is a valid chaining expression {{{{task<expr>}}}}.
{self_check_hc12}
  \u25a1 All required (*) parameters are present in every url.
  \u25a1 No two tasks call the same url+service.
  \u25a1 SQL tasks have url="" and input={{"sql_query":"..."}}.
  \u25a1 Non-SQL tasks have a non-empty url starting with http://.
  \u25a1 COMBINE phase matches tasks:
      "single task" \u2192 exactly 1 task.
      "chain: ..."  \u2192 N HTTP tasks; last task's url or input contains {{...}}; no SQL.
      "sql ..."     \u2192 last task is SQL.
  \u25a1 If any catalog lookup failed in PHASE 2, tasks is an empty array [].
</self_check>

<jmespath_reference>
JMESPath serves ONE purpose: inject values from a prior task's result into the
url or input of a subsequent HTTP task. Syntax: {{{{task_name<expr>}}}}.

Three canonical patterns \u2014 nothing else is permitted inside {{{{...}}}}:

  1. Single value  (filter + pick first):  {{{{t[?k=='v'] | [0].field}}}}     \u2190 pipe required
  2. All values    (array):                 {{{{t[*].field}}}}
  3. Joined string (comma-joined):          {{{{t[*].field | join(',', @)}}}} \u2190 string fields only

Pattern 1 is combinable with any filter expression; pattern 3 accepts an
optional filter before the pipe: {{{{t[?k=='v'].field | join(',', @)}}}}.

Filter expressions inside [?...]:
  operators  ==  !=  <  >  <=  >=  &&  ||
  functions  contains(field, 'text')
  literals   'strings'   `numbers`   `true`   `false`   `null`

Examples of valid filters (plug into pattern 1 or 3):
  [?status=='open']                 [?alertActive==`true`]
  [?price<`100`]                    [?contains(name, 'Rossi')]
  [?a=='x' || a=='y']               [?field==`null`]

CRITICAL RULES:
  - join(',', @) works ONLY on string fields. Never on integers or arrays.
  - [?k=='v'] | [0].field \u2014 the pipe is mandatory to extract a single value.
  - Ranking, sorting, aggregation, grouping, and combining two prior tasks
    are ALWAYS SQL \u2014 never JMESPath. min_by, max_by, sort_by MUST NOT
    appear inside {{{{...}}}}. Use SQL's ORDER BY, LIMIT, MIN, MAX, GROUP BY.
</jmespath_reference>

<examples>
Study the WHY comment after each example. It states the abstract principle.
Apply the principle to any domain \u2014 do not imitate the specific services or field names.

{examples_str}
</examples>

<sql_reference>
SQL tasks use DuckDB dialect. Reference prior task results by task_name as table name.
Never use {{{{}}}} placeholders inside the sql_query string.
NEVER use SQL reserved words as task_name (e.g. order, group, select, index, table, user).

Common patterns:
  Post-filter:    SELECT * FROM t WHERE field = 'value'
                  SELECT col1, col2 FROM t WHERE cond1 AND cond2
  Sort + top-N:   SELECT * FROM t ORDER BY field ASC LIMIT 1
                  SELECT * FROM t ORDER BY field DESC LIMIT 1
                  SELECT * FROM t ORDER BY field ASC LIMIT N
  Aggregate:      SELECT MIN(field), MAX(field), AVG(field) FROM t
  Join:           SELECT a.*, b.field FROM task_a a JOIN task_b b ON a.zoneId = b.zoneId
  Intersect:      SELECT * FROM task_a WHERE zoneId IN (SELECT zoneId FROM task_b WHERE cond)
  Difference:     SELECT * FROM task_a WHERE zoneId NOT IN (SELECT zoneId FROM task_b)
</sql_reference>"""

    def _build_user_prompt(self, discovered_services, discovered_capabilities,
                           discovered_endpoints, discovered_schemas,
                           discovered_request_schemas, discovered_parameters,
                           query, input_files=None) -> str:
        lines = ["SERVICES AND ENDPOINTS:"]
        for i, service in enumerate(discovered_services):
            lines.append(f"\nSERVICE_ID: {service.get('_id')}")
            lines.append(f"NAME: {service.get('name')}")
            caps   = discovered_capabilities[i]   if i < len(discovered_capabilities)   else {}
            eps    = discovered_endpoints[i]       if i < len(discovered_endpoints)       else {}
            rsch   = discovered_schemas[i]         if i < len(discovered_schemas)         else {}
            qsch   = discovered_request_schemas[i] if i < len(discovered_request_schemas) else {}
            params = discovered_parameters[i]      if i < len(discovered_parameters)      else {}
            for key in caps:
                if key == "POST /register":
                    continue
                lines.append(f"  {key}")
                lines.append(f"    URL: {eps.get(key, 'N/A')}")
                lines.append(f"    DESC: {caps[key]}")
                if params.get(key):
                    lines.append(f"    PARAMETERS (* = required): {params[key]}")
                if rsch.get(key):
                    lines.append(f"    RESPONSE SCHEMA: {rsch[key]}")
                if qsch.get(key):
                    lines.append(f"    REQUEST SCHEMA (* = required): {qsch[key]}")

        return f"""{chr(10).join(lines)}

FILES:
{str(input_files) if input_files else "none"}

QUERY:
{query}"""

    def decompose_task(self, discovered_services, discovered_capabilities,
                       discovered_endpoints, discovered_schemas,
                       discovered_request_schemas, discovered_parameters,
                       query, input_files=None):
        system_prompt = self._build_system_prompt(self.backend_mode)
        user_prompt = self._build_user_prompt(
            discovered_services, discovered_capabilities, discovered_endpoints,
            discovered_schemas, discovered_request_schemas, discovered_parameters,
            query, input_files
        )
        input_schema = self._build_input_format_schema(discovered_request_schemas)

        response, latency = self.query_ollama(system_prompt, user_prompt, input_schema)
        print(f"[LLM RESPONSE] {response}")
        print("=" * 100)
        return response, latency

    def _empty_plan_detected(self, plan: dict) -> bool:
        if not isinstance(plan, dict):
            return False
        tasks = plan.get("tasks")
        return isinstance(tasks, list) and len(tasks) == 0

    def extract_agents(self, agents_json: str) -> dict:
        def try_parse(text):
            s, e = text.find('{'), text.rfind('}') + 1
            if s != -1 and e > s:
                try:
                    return json.loads(text[s:e])
                except json.JSONDecodeError:
                    pass
            return None

        try:
            result = json.loads(agents_json)
            if isinstance(result, dict):
                print("[PARSE] Direct parse succeeded.")
                return result
        except json.JSONDecodeError:
            pass

        m = re.search(r'</think>', agents_json, flags=re.IGNORECASE)
        if m:
            result = try_parse(agents_json[m.end():].strip())
            if result is not None:
                print("[PARSE] Extracted after </think>.")
                return result

        result = try_parse(agents_json)
        if result is not None:
            print("[PARSE] Raw extraction.")
            return result

        print(f"[FORMAT ERROR] No valid JSON found. Preview: {agents_json[:200]}")
        return {}

    def _resolve_expression(self, expr: str, context: dict):
        expr = re.sub(r'\.(output|response|data)\b', '', expr)
        expr = re.sub(r'(?<=\w)(\[)(\d+\])', r'.\1\2', expr)
        m = re.match(r'^(\w+)(.*)', expr, re.DOTALL)
        if not m:
            print(f"[JMESPATH] Cannot extract task_name from '{expr}'")
            return ""

        task_name = m.group(1)
        remainder = m.group(2).strip()

        JMESPATH_FUNCTIONS = {"min_by", "max_by", "sort_by", "length", "keys",
                              "values", "contains", "starts_with", "ends_with",
                              "reverse", "to_array", "to_string", "to_number", "type"}
        if task_name in JMESPATH_FUNCTIONS:
            try:
                result = jmespath.search(expr, context)
                if result is None or result == [] or result == "":
                    print(f"[JMESPATH] Function '{task_name}' \u2014 no result for '{expr}'")
                    return ""
                return result
            except jmespath_exc.JMESPathError as e:
                print(f"[JMESPATH ERROR] function '{task_name}': {e}")
                return ""

        val = context.get(task_name)
        if val is None:
            print(f"[JMESPATH] Task '{task_name}' not found in context")
            return ""

        if not remainder:
            return val

        if remainder.startswith('|'):
            remainder = remainder[1:].strip()

        jmespath_expr = remainder[1:] if remainder.startswith('.') else remainder

        if not jmespath_expr:
            return val

        try:
            result = jmespath.search(jmespath_expr, val)
            if result is None:
                print(f"[JMESPATH] No match for '{jmespath_expr}' on '{task_name}'")
                return ""
            if isinstance(result, list) and len(result) == 0:
                print(f"[JMESPATH WARN] Empty list for '{jmespath_expr}' on '{task_name}' \u2014 resulting URL param will be empty")
                return ""
            if isinstance(result, list) and all(isinstance(x, str) for x in result):
                seen = set()
                result = [x for x in result if not (x in seen or seen.add(x))]
            return result
        except jmespath_exc.JMESPathError as e:
            print(f"[JMESPATH ERROR] expr='{expr}' jmespath='{jmespath_expr}': {e}")
            if isinstance(val, dict):
                return val.get(jmespath_expr.split('.')[0], "")
            return ""

    def resolve_placeholders(self, data, context: dict):
        if isinstance(data, str):
            fixed = re.sub(r'(?<!\})\},(\{\{)', r'}},\1', data)
            if fixed != data:
                print(f"[PLACEHOLDER FIX] Malformed concatenation normalized in: {data[:80]}")
                data = fixed

            matches = re.findall(r'\{\{(.*?)\}\}', data)
            if not matches:
                return data

            if len(matches) == 1 and data.strip() == f"{{{{{matches[0]}}}}}":
                return self._resolve_expression(matches[0].strip(), context)

            for match in matches:
                val = self._resolve_expression(match.strip(), context)
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                elif val is None:
                    val = ""
                else:
                    val = str(val)
                data = data.replace(f"{{{{{match}}}}}", val)
            return data

        if isinstance(data, dict):
            return {k: self.resolve_placeholders(v, context) for k, v in data.items()}
        if isinstance(data, list):
            return [self.resolve_placeholders(item, context) for item in data]
        return data

    def _build_auth_headers(self) -> dict:
        raw = os.environ.get("API_AUTH_HEADERS", "")
        if not raw:
            return {}
        try:
            headers = json.loads(raw)
            if isinstance(headers, dict):
                return {str(k): str(v) for k, v in headers.items()}
            print(f"[AUTH] API_AUTH_HEADERS is not a JSON object, ignoring")
        except json.JSONDecodeError as e:
            print(f"[AUTH] API_AUTH_HEADERS is not valid JSON ({e}), ignoring")
        return {}

    async def call_agent(self, session, task, discovered_services):
        task_name = task.get("task_name") or "unnamed_task"
        endpoint = task.get("endpoint") or task.get("url") or ""
        input_data = task.get("input", "")
        operation = str(task.get("operation") or "GET").upper()

        if " " in endpoint:
            endpoint = endpoint.split(" ")[-1]
        if self.backend_mode == "MOCK" and endpoint and not endpoint.startswith("http"):
            mock_url = os.environ.get("MOCK_SERVER_URL", "http://mock-server:8080")
            endpoint = f"{mock_url}{endpoint}" if endpoint.startswith("/") else f"{mock_url}/{endpoint}"

        response_result = {
            "task_name": task_name,
            "operation": operation,
            "url_template": task.get("url", ""),
            "url_resolved": endpoint,
        }

        if not endpoint or not endpoint.strip():
            print(f"[WARN] Phantom task '{task_name}' skipped.")
            response_result.update({"status": "SUCCESS", "status_code": 200, "result": {}})
            return response_result

        auth_headers = self._build_auth_headers()

        try:
            tag_pattern = r"\[(\w+)\](.*?)\[/\1\]"
            tag_matches = re.findall(tag_pattern, str(input_data), re.DOTALL) \
                          if isinstance(input_data, str) else []

            if operation == "GET":
                async with session.get(endpoint, headers=auth_headers) as resp:
                    status = resp.status
                    try:
                        result = await resp.json()
                    except Exception:
                        result = await resp.text()
                    response_result.update({
                        "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                        "status_code": status,
                        "result": result,
                    })

            elif operation in ("POST", "PUT", "PATCH"):
                if tag_matches:
                    form_data = aiohttp.FormData()
                    open_files = []
                    for tag_type, tag_content in tag_matches:
                        tag_content = tag_content.strip()
                        if tag_type == "FILE":
                            file_path = os.path.join("Files", tag_content)
                            if os.path.exists(file_path):
                                file_obj = open(file_path, "rb")
                                open_files.append(file_obj)
                                form_data.add_field(
                                    "file", file_obj,
                                    filename=tag_content,
                                    content_type=mimetypes.guess_type(file_path)[0]
                                                 or "application/octet-stream"
                                )
                        elif tag_type == "TEXT":
                            form_data.add_field("data", tag_content, content_type="application/json")
                    try:
                        async with session.request(operation, endpoint, data=form_data, headers=auth_headers) as resp:
                            status = resp.status
                            try:
                                result = await resp.json()
                            except Exception:
                                result = await resp.text()
                            response_result.update({
                                "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                                "status_code": status, "result": result,
                            })
                    finally:
                        for f in open_files:
                            f.close()
                else:
                    payload = input_data if isinstance(input_data, dict) else {}
                    async with session.request(operation, endpoint, json=payload, headers=auth_headers) as resp:
                        status = resp.status
                        try:
                            result = await resp.json()
                        except Exception:
                            result = await resp.text()
                        response_result.update({
                            "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                            "status_code": status, "result": result,
                        })

            elif operation == "DELETE":
                async with session.delete(endpoint, headers=auth_headers) as resp:
                    status = resp.status
                    try:
                        result = await resp.json()
                    except Exception:
                        result = await resp.text()
                    response_result.update({
                        "status": "SUCCESS" if status in (200, 201, 204) else "ERROR",
                        "status_code": status, "result": result,
                    })

        except Exception as e:
            print(f"[EXCEPTION] '{task_name}': {e}")
            response_result.update({"status": "EXCEPTION", "status_code": 500, "result": str(e)})

        return response_result

    def _execute_sql_task(self, task: dict, context: dict) -> dict:
        task_name = task.get("task_name") or "sql_task"
        input_data = task.get("input", {})

        if isinstance(input_data, dict):
            sql_query = input_data.get("sql_query", "")
        else:
            sql_query = ""

        if not sql_query:
            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "ERROR",
                "status_code": 400,
                "result":      "SQL task requires input={'sql_query': 'SELECT ...'}",
            }

        tail = "..." if len(sql_query) > 120 else ""
        print(f"[SQL] Task '{task_name}': {sql_query[:120]}{tail}")

        try:
            conn = duckdb.connect()
        except Exception as e:
            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "ERROR",
                "status_code": 500,
                "result":      f"DuckDB connect error: {e}",
            }

        try:
            for tbl, data in context.items():
                escaped_tbl = str(tbl).replace('"', '""')

                if isinstance(data, list):
                    if not data:
                        conn.execute(f'CREATE TABLE "{escaped_tbl}" (dummy VARCHAR)')
                    else:
                        df = pd.DataFrame(data)
                        conn.register(str(tbl), df)
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                    conn.register(str(tbl), df)

            rel = conn.execute(sql_query)
            rows = rel.fetchall()
            cols = [desc[0] for desc in rel.description]

            result = []
            for row in rows:
                row_dict = {}
                for col, val in zip(cols, row):
                    if isinstance(val, float) and math.isnan(val):
                        val = None
                    elif hasattr(val, 'isoformat'):
                        val = val.isoformat()
                    row_dict[col] = val
                result.append(row_dict)

            print(f"[SQL] Task '{task_name}' completed \u2014 {len(result)} rows.")
            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "SUCCESS",
                "status_code": 200,
                "result":      result,
            }

        except Exception as e:
            error_msg = str(e)
            print(f"[SQL ERROR] Task '{task_name}': {error_msg}")

            if "Binder Error" in error_msg or "dummy" in error_msg:
                print(f"[SQL WARN] Task '{task_name}': Binder Error on empty table \u2014 returning empty result")
                return {
                    "task_name":   task_name,
                    "operation":   "SQL",
                    "status":      "SUCCESS",
                    "status_code": 200,
                    "result":      [],
                }

            return {
                "task_name":   task_name,
                "operation":   "SQL",
                "status":      "ERROR",
                "status_code": 500,
                "result":      error_msg,
            }

        finally:
            conn.close()

    async def trigger_agents_async(self, agents: dict, discovered_services):
        results = []
        context = {}
        async with aiohttp.ClientSession() as session:
            for task in agents.get("tasks", []):
                operation = str(task.get("operation") or "GET").upper()

                if operation == "SQL":
                    result = self._execute_sql_task(task, context)
                else:
                    task["endpoint"] = self.resolve_placeholders(task.get("url") or "", context)
                    task["url_resolved"] = task["endpoint"]
                    if task.get("input"):
                        task["input"] = self.resolve_placeholders(task["input"], context)
                    result = await self.call_agent(session, task, discovered_services)

                if result.get("status") == "FILE":
                    return result
                results.append(result)
                name = task.get("task_name") or "unnamed_task"
                if result.get("status") == "SUCCESS":
                    raw = result.get("result", {})
                    if isinstance(raw, (dict, list)):
                        context[name] = raw
                    else:
                        print(f"[CONTEXT] '{name}' returned non-JSON ({type(raw).__name__}) \u2014 not stored in chain context")
                        context[name] = {}
                else:
                    print(f"[CHAIN BROKEN] '{name}' failed. Pipeline halted.")
                    break
        return results

    def trigger_agents(self, agents: dict, discovered_services):
        return asyncio.run(self.trigger_agents_async(agents, discovered_services))

    def replace_endpoints(self, endpoints_list, mock_server_address):
        if self.backend_mode == "REAL":
            return endpoints_list
        return [
            {k: re.sub(r"http://localhost:8585", mock_server_address, v)
             if isinstance(v, str) else v
             for k, v in ep.items()}
            for ep in endpoints_list
        ]

    def _attempt_auto_fix(self, plan: dict, available_ids: list = None, name_to_id: dict = None) -> dict:
        mock_url = os.environ.get("MOCK_SERVER_URL", "http://mock-server:8080")
        id_set = set(available_ids or [])
        fixed_tasks = []
        for task in plan.get("tasks", []):
            if not isinstance(task, dict):
                continue

            sid = task.get("service_id", "")
            if sid and sid not in id_set and name_to_id and sid in name_to_id:
                corrected = name_to_id[sid]
                print(f"[AUTO-FIX] service_id '{sid}' \u2192 '{corrected}'")
                task["service_id"] = corrected

            url = str(task.get("url", ""))
            if self.backend_mode == "MOCK":
                if url and re.search(r'http://localhost:\d+', url):
                    fixed = re.sub(r'http://localhost:\d+', mock_url, url)
                    print(f"[AUTO-FIX] localhost \u2192 mock-server: {fixed}")
                    task["url"] = fixed
                    url = fixed
                if url and not url.startswith("http") and not url.startswith("{{"):
                    task["url"] = (mock_url if url.startswith("/") else mock_url + "/") + url.lstrip("/")
                    print(f"[AUTO-FIX] URL: {task['url']}")
            if isinstance(task.get("operation"), str):
                task["operation"] = task["operation"].upper()
            if all(f in task for f in ("task_name", "service_id", "url", "operation")):
                fixed_tasks.append(task)
            else:
                print(f"[AUTO-FIX] Task discarded: {task.get('task_name', 'unnamed')}")
        plan["tasks"] = fixed_tasks
        return plan

    def control(self, query, files=None):
        analyzed_files = self.analyze_files(files or [])
        planning_latency_s = None

        self.backend_mode = os.environ.get("BACKEND_MODE", "MOCK").upper()
        if self.backend_mode not in ("MOCK", "REAL"):
            print(f"[BACKEND_MODE] Unrecognized value '{self.backend_mode}', falling back to MOCK")
            self.backend_mode = "MOCK"
        print(f"[BACKEND_MODE] {self.backend_mode}")

        catalog_url  = os.environ.get("CATALOG_URL",    "http://catalog-gateway:5000")
        registry_url = os.environ.get("REGISTRY_URL",   "http://registry:8500")
        mock_url     = os.environ.get("MOCK_SERVER_URL", "http://mock-server:8080")

        registry = Discovery(registry_url)
        services = registry.services()
        service_data = requests.post(f"{catalog_url}/index/search", json={"query": query}).json()
        service_list = service_data.get("results", [])

        if not service_list:
            return {
                "execution_plan": {},
                "execution_results": [],
                "error": "No services matched the query",
                "planning_latency_s": planning_latency_s,
            }

        registry_ids = {s["id"] for s in services}
        filtered_service_list = [s for s in service_list if s["_id"] in registry_ids]
        orphaned = [s for s in service_list if s["_id"] not in registry_ids]

        if orphaned:
            print("[WARNING] Services no longer in registry:", [s.get("_id") for s in orphaned])

        if not filtered_service_list:
            return {
                "execution_plan": {},
                "execution_results": [],
                "error": "None of the discovered services are currently available",
                "planning_latency_s": planning_latency_s,
            }

        print("\n" + "=" * 60)
        print(f"[RETRIEVAL] Query: {query}")
        for s in filtered_service_list:
            caps = s.get("capabilities", {})
            print(f"  - {s.get('_id')} | {s.get('name')} | {len(caps)} endpoints")
        print("=" * 60 + "\n")

        disc_services, disc_caps, disc_eps = [], [], []
        disc_schemas, disc_req_schemas, disc_params = [], [], []

        for s in filtered_service_list:
            if isinstance(s.get("capabilities"), dict): s["capabilities"].pop("POST /register", None)
            if isinstance(s.get("endpoints"), dict):    s["endpoints"].pop("POST /register", None)
            disc_services.append({
                "_id":         s.get("_id"),
                "name":        s.get("name"),
                "description": s.get("description"),
            })
            disc_caps.append(s.get("capabilities", {}))
            disc_eps.append(s.get("endpoints", {}))
            disc_schemas.append(s.get("response_schemas", {}))
            disc_req_schemas.append(s.get("request_schemas", {}))
            disc_params.append(s.get("parameters", {}))

        disc_eps = self.replace_endpoints(disc_eps, mock_url)

        plan_json, latency = self.decompose_task(
            disc_services, disc_caps, disc_eps,
            disc_schemas, disc_req_schemas, disc_params,
            query, analyzed_files
        )
        planning_latency_s = round(latency, 3)
        print(f"[LATENCY] Plan generated in {latency:.2f}s")

        plan = self.extract_agents(plan_json)

        schema_warnings = self._validate_plan(plan, disc_services, disc_req_schemas, disc_params, self.backend_mode)
        if schema_warnings:
            print(f"\n{'!' * 40}")
            print(f"[SCHEMA VALIDATOR] {len(schema_warnings)} violation(s) detected:")
            for w in schema_warnings:
                print(f"  {w}")
            print(f"{'!' * 40}\n")
        else:
            print("[SCHEMA VALIDATOR] No violations detected.")

        if self._empty_plan_detected(plan):
            print("\n[ROUTING] Empty plan detected. "
                  "Activating Designer (triage + design)...")
            analysis = self.designer.analyze(
                query=query,
                plan_reasoning=plan.get("reasoning", ""),
                discovered_services=disc_services,
                discovered_capabilities=disc_caps,
                input_files=analyzed_files,
            )
            category = analysis.get("category", "UNKNOWN")
            contract = analysis.get("service_contract")

            if category == "OUT_OF_DOMAIN" and isinstance(contract, dict):
                error_msg = ("Query out of domain. The system requires "
                             "services not currently available.")
            else:
                error_msg = (f"Empty plan classified as {category}: "
                             f"{analysis.get('justification', '')}")

            return {
                "execution_plan":           plan,
                "execution_results":        [],
                "error":                    error_msg,
                "empty_plan_category":      category,
                "empty_plan_justification": analysis.get("justification", ""),
                "suggested_api_contracts":  [contract] if isinstance(contract, dict) else [],
                "planning_latency_s":       planning_latency_s,
            }

        available_ids = [s["_id"] for s in disc_services]
        name_to_id = {s.get("name"): s.get("_id") for s in disc_services}

        is_valid, val_errors = PlanValidator.validate(plan, available_ids)

        if not is_valid:
            print(f"[VALIDATION] {len(val_errors)} error(s):")
            for e in val_errors:
                print(f"  - {e}")
            plan = self._attempt_auto_fix(plan, available_ids, name_to_id)
            is_valid, val_errors = PlanValidator.validate(plan, available_ids)
            if not is_valid:
                print("[VALIDATION] Plan unrecoverable \u2014 execution cancelled.")
                plan["tasks"] = []

        results = self.trigger_agents(plan, disc_services)

        if isinstance(results, dict) and results.get("status") == "FILE":
            return Response(results["body"], status=results["status_code"],
                            headers=results["headers"])

        if os.path.exists("Files"):
            for filename in os.listdir("Files"):
                file_path = os.path.join("Files", filename)
                try:
                    if os.path.isfile(file_path): os.unlink(file_path)
                    elif os.path.isdir(file_path): shutil.rmtree(file_path)
                except Exception as e:
                    print(f"[WARN] Cleanup failed: {e}")

        return {
            "execution_plan": plan,
            "execution_results": results,
            "planning_latency_s": planning_latency_s,
        }

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
        else:
            hc12_block = "\nHC-12 no AND-combined GET params (?a=x&b=y). Microcks matches each param alone."

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

        examples_str = (
            f"EXAMPLE chain:\n{json.dumps(ex_b, separators=(',', ':'))}"
        )

        return f"""<role>
Output ONLY a JSON execution plan. Schema: {{"reasoning": "string", "tasks": [{{task_name, service_id, url, operation, input}}]}}
Zero prose, zero fences. Constraint phrases (without X, near Z, ...) need a GET task.
If impossible: {{"reasoning": "No available service can fulfil this request.", "tasks": []}}
</role>

<rules>
HC-1  Use only catalog services/endpoints/params.
HC-2  No bare {{{{placeholder}}}}. Use {{{{task<expr>}}}} chaining.
HC-3  task: task_name, service_id, url, operation, input. service_id = SERVICE_ID from catalog.
HC-4  Marked * params required in url.
HC-5  operation: GET|POST|PUT|DELETE|SQL
HC-6  Non-SQL url starts http://. SQL url is "".
HC-7  No duplicate url+service calls. Reuse via JMESPath/SQL.
HC-8  No concatenated placeholders. Use one prior task + JMESPath OR.
HC-9  Param values from catalog examples/enums, not user query.
HC-10 No invented thresholds. ORDER BY+LIMIT for extremes, boolean fields for states.
HC-11 {{{{task<expr>}}}} for injection only, not aggregation. Use SQL for that.{hc12_block}
</rules>

<reasoning>
DECOMPOSE: <data> | MAP: <svc/endpoint> | CHAIN: none|JMESPath|SQL | COMBINE: single|chain|sql | FILTER: <params> | VALIDATE: ok
Commit once. No wait/but/perhaps.
</reasoning>

<JMESPath>
{{{{task[?k=='v']|[0].field}}}} single | {{{{task[*].field}}}} array | {{{{task[*].field|join(',',@)}}}} joined
</JMESPath>

<examples>
{examples_str}
</examples>

<sql>
DuckDB. Prior tasks as tables. No {{{{}}}} inside SQL. SELECT/ORDER BY/AVG/COUNT/JOIN on zoneId.
</sql>"""

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

        # ---- Token budget: trim catalog to fit model context -----------------
        planner_ctx = int(os.environ.get("PLANNER_CTX_LENGTH", "4096"))
        base_prompt = self._build_system_prompt(self.backend_mode)

        def _est_tok(text: str) -> int:
            return len(text) // 4

        base_tok = _est_tok(base_prompt) + _est_tok(f"\nFILES:\nnone\n\nQUERY:\n{query}\n")
        margin = int(planner_ctx * 0.85)

        def _svc_tok(i: int) -> int:
            c = len(str(disc_services[i].get("_id", ""))) + len(str(disc_services[i].get("name", "")))
            caps = disc_caps[i] if i < len(disc_caps) else {}
            for k in caps:
                if k == "POST /register":
                    continue
                c += len(k) + len(str(disc_eps[i].get(k, ""))) + len(str(caps[k]))
                c += len(str(disc_params[i].get(k, ""))) if i < len(disc_params) else 0
                c += len(str(disc_schemas[i].get(k, ""))) if i < len(disc_schemas) else 0
                c += len(str(disc_req_schemas[i].get(k, ""))) if i < len(disc_req_schemas) else 0
            return c

        total_est = base_tok + sum(_svc_tok(i) for i in range(len(disc_services)))
        if total_est > margin:
            print(f"[BUDGET] Est. {total_est} tok > {margin} (85% ctx) — trimming")
            keep, acc = [], base_tok
            for i in range(len(disc_services)):
                t = _svc_tok(i)
                if acc + t <= margin:
                    acc += t
                    keep.append(i)
            print(f"[BUDGET] Keeping {len(keep)}/{len(disc_services)} services ({acc} est. tok)")
            disc_services = [disc_services[i] for i in keep]
            disc_caps = [disc_caps[i] for i in keep]
            disc_eps = [disc_eps[i] for i in keep]
            disc_schemas = [disc_schemas[i] for i in keep]
            disc_req_schemas = [disc_req_schemas[i] for i in keep]
            disc_params = [disc_params[i] for i in keep]

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

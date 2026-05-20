from concurrent.futures import ThreadPoolExecutor
import requests
import yaml
import prance          # library to resolve $ref in OpenAPI YAML files
import os
import json
import re
from urllib.parse import quote, unquote, urlparse
import time


class Service:
    # -----------------------------------------------------------------------
    # Environment variables: allow configuring hosts without
    # modifying code (they differ between local dev and Docker)
    # -----------------------------------------------------------------------
    CONSUL_HOST = os.environ.get("CONSUL_HOST", "registry")
    CONSUL_PORT = int(os.environ.get("CONSUL_PORT", 8500))

    GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "catalog-gateway")
    GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", 5000))

    HEALTHCHECK_SERVICE_HOST = os.environ.get("HEALTHCHECK_SERVICE_HOST", "healthcheck-service")
    HEALTHCHECK_SERVICE_PORT = os.environ.get("HEALTHCHECK_SERVICE_PORT", 5600)

    MOCK_SERVER_URL = os.environ.get("MOCK_SERVER_URL", "http://mock-server:8080")

    # -----------------------------------------------------------------------
    # Methods for importing external APIs from apis.guru
    # -----------------------------------------------------------------------

    def fetch_providers(self):
        # fetches the list of all providers from apis.guru
        url = "https://api.apis.guru/v2/providers.json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["data"]

    def fetch_api_details(self, provider):
        # fetches details for a specific provider (versions, YAML URL, etc.)
        url = f"https://api.apis.guru/v2/{provider}.json"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def extract_swagger_url(self, api_data, provider):
        # looks for the OpenAPI YAML URL in the provider metadata
        # apis.guru can have different structures (dict or list) — handles both
        apis = api_data.get("apis")

        if isinstance(apis, dict):
            for version_key, version_data in apis.items():
                swagger_url = version_data.get("swaggerYamlUrl")
                if swagger_url:
                    return swagger_url

        elif isinstance(apis, list):
            for entry in apis:
                if isinstance(entry, dict):
                    swagger_url = entry.get("swaggerYamlUrl") or entry.get("swaggerUrl")
                    if swagger_url:
                        return swagger_url

        print(f"[WARN] swaggerYamlUrl not found for provider {provider}")
        return None

    def extract_endpoints_from_swagger(self, swagger_url):
        # extracts only the paths (e.g. /bin, /bin/{id}) without processing details
        # used for quick analysis, not for schema extraction
        response = requests.get(swagger_url)
        if response.status_code != 200:
            return []
        try:
            swagger_data = yaml.safe_load(response.text)
            return list(swagger_data.get("paths", {}).keys())
        except yaml.YAMLError:
            return []

    # -----------------------------------------------------------------------
    # Helpers for schema extraction — dotted keys with compact types
    # These methods are the core of the chaining system:
    # they transform a complex OpenAPI schema into a compact string
    # that the LLM can read and use to build correct placeholders
    # -----------------------------------------------------------------------

    def _map_type(self, prop_schema):
        """
        Converts the OpenAPI type of a single property into the compact type
        used in the schema string passed to the LLM.

        OpenAPI uses "type: integer", "type: string", etc.
        We use: int, float, str, bool, enum, arr, obj, any

        Example: {"type": "string", "enum": ["open","closed"]} -> "enum"
                 {"type": "integer"} -> "int"
        """
        if not isinstance(prop_schema, dict):
            return "any"

        # enum takes priority over type: even "type: string" with enum is "enum"
        if prop_schema.get("enum"):
            return "enum"

        oa_type = prop_schema.get("type", "")

        if oa_type == "integer":
            return "int"
        if oa_type == "number":
            return "float"
        if oa_type == "boolean":
            return "bool"
        if oa_type == "array":
            return "arr"
        if oa_type == "string":
            return "str"
        # object with nested properties -> "obj" (but _flatten_dotted expands it)
        if oa_type == "object" or "properties" in prop_schema or "allOf" in prop_schema:
            return "obj"

        return "any"

    def _flatten_dotted(self, schema, prefix=""):
        """
        Receives an OpenAPI schema already dereferenced by prance (all $ref
        have been replaced with their actual content) and returns
        a flat dictionary with dotted keys.

        Example input (Bin schema after prance):
          {allOf: [{properties: {location: {type: str}, fillLevel: {type: int}}},
                   {properties: {id: {type: int}}}]}

        Example output:
          {"location": "str", "fillLevel": "int", "id": "int"}

        The prefix is used for nested objects:
          if "address" contains "street" and "city",
          the result is {"address.street": "str", "address.city": "str"}
        """
        if not isinstance(schema, dict):
            return {}

        result = {}

        # --- allOf: pattern used for inheritance in OpenAPI ---
        # Example: Bin = allOf[NewBin, {properties: {id: int}}]
        # Iterates all sub-schemas and merges results
        for sub in schema.get("allOf", []):
            result.update(self._flatten_dotted(sub, prefix))

        # --- anyOf / oneOf: polymorphic schemas ---
        # Instead of choosing a single variant, merges them all
        # to avoid missing fields that might be present
        for combinator in ("anyOf", "oneOf"):
            for sub in schema.get(combinator, []):
                result.update(self._flatten_dotted(sub, prefix))

        # --- direct properties: the most common case ---
        for prop_name, prop_schema in schema.get("properties", {}).items():
            if not isinstance(prop_schema, dict):
                prop_schema = {}

            full_key = f"{prefix}{prop_name}"  # e.g. "address.street"

            # recurse for nested objects (e.g. prop_schema itself has properties)
            sub_props = self._flatten_dotted(prop_schema, f"{full_key}.")
            if sub_props:
                # nested object -> expand with dotted keys
                result.update(sub_props)
            else:
                # leaf: map to compact type
                result[full_key] = self._map_type(prop_schema)

        # --- array: only enters if no properties were found ---
        # (if not result avoids overwriting already found results)
        if not result and schema.get("type") == "array":
            items = schema.get("items", {})

            # Guard: in external APIs (apis.guru) items can be a list
            # (OpenAPI 3.1 typed tuples) or a malformed string.
            # In those cases we cannot descend — return what we have.
            if not isinstance(items, dict):
                return result

            # distinguishes arrays of complex objects from arrays of scalars
            has_object_items = (
                items.get("type") == "object" or
                "properties" in items or
                "allOf" in items
            )

            if has_object_items:
                field_name = prefix.rstrip(".")
                if field_name:
                    # CASE 1: Nested array inside another object -> mark as "arr"
                    result[field_name] = "arr"
                else:
                    # CASE 2: Array at the ROOT (the API response is a list of objects).
                    # We must descend into 'items' and extract the fields!
                    sub = self._flatten_dotted(items, prefix)
                    if sub:
                        result.update(sub)
            else:
                # array of scalars (e.g. availableLanguages: ["it", "en"])
                sub = self._flatten_dotted(items, prefix)
                if sub:
                    result.update(sub)
                else:
                    # simple scalar (e.g. array of string): represent as "arr"
                    field_name = prefix.rstrip(".")
                    if field_name:
                        result[field_name] = "arr"

        return result

    def _infer_type_from_value(self, value):
        """
        Infers the compact type from a concrete example value.
        Used as fallback when the OpenAPI schema is missing.

        Example: value=42 -> "int", value="open" -> "str", value=[...] -> "arr"
        """
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            return "arr"
        if isinstance(value, dict):
            return "obj"
        return "any"

    def _flatten_dotted_from_example(self, example_value, prefix=""):
        """
        Alternative version of _flatten_dotted that works on a concrete
        example value instead of an OpenAPI schema.

        Used as fallback when the schema is missing or empty.
        Infers types from actual values rather than declarations.

        Example input (first element of GET /bin):
          {"id": 1, "location": "City Square", "fillLevel": 35, "status": "normal"}

        Example output:
          {"id": "int", "location": "str", "fillLevel": "int", "status": "str"}
        """
        if isinstance(example_value, dict):
            result = {}
            for key, val in example_value.items():
                full_key = f"{prefix}{key}"
                if isinstance(val, dict):
                    # nested object: recurse with dotted prefix
                    sub = self._flatten_dotted_from_example(val, f"{full_key}.")
                    result.update(sub) if sub else result.update({full_key: "obj"})
                else:
                    result[full_key] = self._infer_type_from_value(val)
            return result

        elif isinstance(example_value, list) and example_value:
            # for arrays, only analyze the first element as representative
            return self._flatten_dotted_from_example(example_value[0], prefix)

        return {}

    def _schema_to_string(self, flat_dict, is_array):
        """
        Converts the flat dictionary {field: type} into the compact string
        that is saved in MongoDB and passed to the LLM in the prompt.

        is_array=False -> "{id:int, location:str, fillLevel:int, status:enum}"
        is_array=True  -> "[{id:int, location:str, fillLevel:int, status:enum}]"

        Square brackets tell the LLM that the response is an array,
        so it must use the index or FIND to access elements.
        """
        if not flat_dict:
            return None
        inner = ", ".join(f"{k}:{v}" for k, v in flat_dict.items())
        return f"[{{{inner}}}]" if is_array else f"{{{inner}}}"

    def _collect_required_fields(self, schema):
        """
        Collects the names of all required fields from a schema.
        Also handles allOf because the common pattern in OpenAPI is:

          NewBin:
            allOf:
              - $ref: BaseModel   (which has its own required fields)
              - properties: {id}
                required: [id]    (additional required fields)

        Returns a set of strings with the names of required fields.
        Used by _extract_request_schema_from_details to add the * marker
        """
        required = set(schema.get("required", []))
        # also collects required fields from allOf sub-schemas
        for sub in schema.get("allOf", []):
            required.update(sub.get("required", []))
        return required

    def _extract_schema_from_details(self, details):
        """
        Extracts the success response schema (200 or 201) of an endpoint.
        Used to build the RESPONSE SCHEMAS passed to the LLM.

        Two-level strategy:
          1. Declarative OpenAPI schema (highest priority — the "contractual truth")
          2. Examples as fallback (when the schema is missing or empty)

        HTTP 204 is excluded because it has no body (DELETE returns 204).

        Output: string like "[{id:int, location:str, status:enum}]"
                or None if nothing can be extracted
        """
        responses = details.get("responses", {})

        # look for the success response: 200 (GET/PUT) or 201 (POST)
        # 204 (DELETE) has no body — intentionally excluded
        success_response = None
        for code in ["200", "201", 200, 201]:
            if code in responses:
                success_response = responses[code]
                break

        if not success_response or not isinstance(success_response, dict):
            return None

        # supports both Swagger 2.0 and OpenAPI 3.0 (different structures)
        swagger2_schema   = success_response.get("schema", {})
        swagger2_examples = success_response.get("examples", {})
        swagger2_json_ex  = swagger2_examples.get("application/json") if isinstance(swagger2_examples, dict) else None

        content      = success_response.get("content", {})
        json_content = content.get("application/json", {})

        # --- Strategy 1: Declarative schema (source of truth) ---
        # prefers OpenAPI 3.0 schema, then Swagger 2.0
        final_schema = json_content.get("schema") or swagger2_schema

        if isinstance(final_schema, dict) and final_schema:
            is_array  = final_schema.get("type") == "array"
            flat_dict = self._flatten_dotted(final_schema)
            if flat_dict:
                return self._schema_to_string(flat_dict, is_array)

        # --- Strategy 2: Examples as fallback ---
        # used when the schema is missing or has no properties

        # 2a. OpenAPI 3.0 — examples (plural, dictionary of named examples)
        examples = json_content.get("examples", {})
        if examples:
            # takes the first available example
            first_example = next(iter(examples.values()), None)
            if isinstance(first_example, dict) and first_example.get("value") is not None:
                ex_val    = first_example["value"]
                is_array  = isinstance(ex_val, list)
                flat_dict = self._flatten_dotted_from_example(ex_val)
                if flat_dict:
                    return self._schema_to_string(flat_dict, is_array)

        # 2b. OpenAPI 3.0 — example (singular)
        if "example" in json_content and json_content["example"] is not None:
            ex_val    = json_content["example"]
            is_array  = isinstance(ex_val, list)
            flat_dict = self._flatten_dotted_from_example(ex_val)
            if flat_dict:
                return self._schema_to_string(flat_dict, is_array)

        # 2c. Swagger 2.0 — explicit JSON example
        if swagger2_json_ex is not None:
            is_array  = isinstance(swagger2_json_ex, list)
            flat_dict = self._flatten_dotted_from_example(swagger2_json_ex)
            if flat_dict:
                return self._schema_to_string(flat_dict, is_array)

        return None

    def _extract_request_schema_from_details(self, details, method):
        """
        Extracts the input body schema for POST and PUT.
        GET and DELETE have no body -> returns None directly.

        Required fields are marked with * in the output format,
        so the LLM knows which fields it must include
        in the request body.

        Output: "{location:str*, fillLevel:int*, binType:enum*, status:enum*}"
                (fields without * are optional)
        """
        # only POST and PUT have a request body
        if method.upper() not in {"POST", "PUT"}:
            return None

        schema = None

        # OpenAPI 3.0: schema is in requestBody.content.application/json.schema
        request_body = details.get("requestBody", {})
        if isinstance(request_body, dict):
            content      = request_body.get("content", {})
            json_content = content.get("application/json", {})
            schema       = json_content.get("schema")

        # Swagger 2.0: schema is in parameters[in=body].schema
        if not schema:
            for param in details.get("parameters", []):
                if isinstance(param, dict) and param.get("in") == "body":
                    schema = param.get("schema")
                    break

        if not schema or not isinstance(schema, dict):
            return None

        # collects required fields (also handles allOf recursively)
        required_fields = self._collect_required_fields(schema)

        flat_dict = self._flatten_dotted(schema)
        if not flat_dict:
            return None

        # serializes by adding * on required fields
        parts = []
        for key, type_label in flat_dict.items():
            # for dotted keys (e.g. "address.street"), only checks the root part
            # because required is declared at the parent object level
            root_field = key.split(".")[0]
            marker = "*" if root_field in required_fields else ""
            parts.append(f"{key}:{type_label}{marker}")

        inner = ", ".join(parts)
        return f"{{{inner}}}"

    def _extract_parameters_from_details(self, details):
        """
        Extracts query parameters and path parameters from an endpoint.
        Returns a compact string, e.g. "{zoneId:str*, status:enum}"
        """
        params = details.get("parameters", [])
        if not params:
            return None

        extracted = {}
        for p in params:
            if isinstance(p, dict) and p.get("in") in ["query", "path"]:
                name = p.get("name")
                is_required = p.get("required", False)
                schema = p.get("schema", {})

                # Uses the same _map_type used elsewhere
                param_type  = self._map_type(schema)
                marker      = "*" if is_required or p.get("in") == "path" else ""

                # Includes enum values in compact format: enum(v1,v2,v3)
                # Saves tokens compared to describing values in prose,
                # and prevents the LLM from inventing variants (e.g. airQuality vs air_quality)
                # Caps at 8 values to avoid bloating the prompt on very large enums
                enum_values = schema.get("enum")
                if enum_values and isinstance(enum_values, list) and len(enum_values) <= 8:
                    type_label = "enum(" + ",".join(str(v) for v in enum_values) + ")"
                else:
                    type_label = param_type

                extracted[name] = f"{type_label}{marker}"

        if not extracted:
            return None

        # Formats as compact JSON-like (TOON)
        inner = ", ".join(f"{k}:{v}" for k, v in extracted.items())
        return f"{{{inner}}}"

    def _build_swagger_url_from_endpoint(self, endpoint_url, mock_server_url):
        """
        Reconstructs the Microcks YAML URL from an endpoint
        already saved in MongoDB.

        Serves as fallback when swagger_url was not saved directly
        (e.g. services registered before the swagger_url field was added).

        Example transformation:
          input:  "http://localhost:8585/rest/Smart+Bins+API/1.0/bin"
          output: "http://mock-server:8080/api/resources/Smart%20Bins%20API-1.0.yaml"

        The Microcks mock server exposes original YAML files under /api/resources/
        with the format: {ApiName}-{version}.yaml (spaces encoded as %20)
        """
        # extracts the part after /rest/
        match = re.search(r"/rest/(.+)", endpoint_url)
        if not match:
            return None

        rest_path = match.group(1)
        parts     = rest_path.split("/")
        if len(parts) < 2:
            return None

        api_name_raw     = parts[0]            # e.g. "Smart+Bins+API"
        version          = parts[1]            # e.g. "1.0"
        api_name         = unquote(api_name_raw.replace("+", " "))  # -> "Smart Bins API"
        api_name_encoded = quote(api_name)     # -> "Smart%20Bins%20API"

        filename = f"{api_name_encoded}-{version}.yaml"
        return f"{mock_server_url.rstrip('/')}/api/resources/{filename}"

    def _generate_description(self, swagger: dict) -> str:
        """
        Generates a concise description for Stage 1 indexing of retrieval.

        Built from:
          - API title (semantic anchor)
          - Summary of each endpoint (functional text, denser than the description)
          - Enum values of parameters (precise domain vocabulary)

        The original description (info.description) is often abstract and has
        low lexical overlap with user queries. The summaries and enums
        contain the actual terms used in queries: "Find best available
        parking", "spotType: standard, electric, disabled", etc.

        Saved in generated_description field (separate from description)
        and used exclusively for Qdrant indexing, not sent to the LLM.
        """
        info  = swagger.get("info", {})
        paths = swagger.get("paths", {})
        parts = []

        title = info.get("title", "").strip()
        if title:
            parts.append(title)

        seen_summaries = set()
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, details in methods.items():
                if method.lower() not in {"get", "post", "put", "delete"}:
                    continue
                if not isinstance(details, dict):
                    continue
                summary = details.get("summary", "").strip()
                if summary and summary not in seen_summaries:
                    parts.append(summary)
                    seen_summaries.add(summary)

        seen_params = set()
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, details in methods.items():
                if not isinstance(details, dict):
                    continue
                for param in details.get("parameters", []):
                    if not isinstance(param, dict):
                        continue
                    schema     = param.get("schema", {})
                    enum_vals  = schema.get("enum", [])
                    param_name = param.get("name", "").strip()
                    if enum_vals and param_name and param_name not in seen_params:
                        enum_str = ", ".join(str(v) for v in enum_vals[:10])
                        parts.append(f"{param_name}: {enum_str}")
                        seen_params.add(param_name)

        if not parts:
            return info.get("description", "").strip()

        return ". ".join(parts)

    def _read_spec_text(self, swagger_url):
        if swagger_url.startswith("file://"):
            with open(swagger_url[len("file://"):], "r", encoding="utf-8") as f:
                return f.read()
        resp = requests.get(swagger_url)
        resp.encoding = 'utf-8'
        return resp.text

    def _extract_schemas_from_yaml(self, swagger_url):
        try:
            text = self._read_spec_text(swagger_url)
            parser = prance.ResolvingParser(spec_string=text, lazy=False, strict=False)
            swagger = parser.specification
        except Exception as e:
            print(f"Failed to load/resolve YAML from {swagger_url}: {e}")
            return {"response_schemas": {}, "request_schemas": {}, "parameters": {}}

        response_schemas = {}
        request_schemas  = {}
        parameters       = {}

        paths = swagger.get("paths", {})
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, details in methods.items():
                if method.lower() not in {"get", "post", "put", "delete"}:
                    continue
                if not isinstance(details, dict):
                    continue

                key = f"{method.upper()} {path}"

                resp_schema = self._extract_schema_from_details(details)
                if resp_schema:
                    response_schemas[key] = resp_schema

                req_schema = self._extract_request_schema_from_details(details, method)
                if req_schema:
                    request_schemas[key] = req_schema

                params_schema = self._extract_parameters_from_details(details)
                if params_schema:
                    parameters[key] = params_schema

        # Generates the enriched description from the already-parsed swagger.
        # No second download — uses parser.specification already in memory.
        generated_description = self._generate_description(swagger)

        return {
            "response_schemas":      response_schemas,
            "request_schemas":       request_schemas,
            "parameters":            parameters,
            "generated_description": generated_description,
        }
    # -----------------------------------------------------------------------
    # Main enrichment method
    # -----------------------------------------------------------------------

    def enrich_schemas(self, mock_server_url=None, service_id=None):
        """
        Entry point for enrichment: called by mock-deployer after
        service deployment via POST /api/importer/enrich.

        Flow:
          1. Fetches all services from MongoDB via catalog-gateway
          2. For each, finds the YAML URL (from swagger_url or by reconstruction)
          3. Extracts schemas with _extract_schemas_from_yaml
          4. Saves to MongoDB via PATCH /services/{id}/schemas

        Optional service_id: if specified, enriches only that service.
        """
        mock_server_url = mock_server_url or self.MOCK_SERVER_URL
        gateway_base    = f"http://{self.GATEWAY_HOST}:{self.GATEWAY_PORT}"

        try:
            resp = requests.get(f"{gateway_base}/services", timeout=10)
            resp.raise_for_status()
            all_services = resp.json()
        except Exception as e:
            print(f"[ENRICH] Cannot fetch services from gateway: {e}")
            return {"enriched": 0, "skipped": 0, "errors": 1}

        # optional filter for a single service
        if service_id:
            all_services = [s for s in all_services if s.get("_id") == service_id]

        enriched = 0
        skipped  = 0
        errors   = 0

        for svc in all_services:
            doc_id = svc.get("_id")

            # idempotent skip: does not reprocess already enriched services
            # (avoids overwriting correct schemas with a re-run)
            if ("response_schemas" in svc and "request_schemas" in svc
                    and "parameters" in svc and "generated_description" in svc):
                print(f"[ENRICH] Skipping {doc_id} (already enriched)")
                skipped += 1
                continue

            # finds the YAML URL: first looks for the direct field in MongoDB,
            # then attempts reconstruction from the endpoint as fallback
            swagger_url = svc.get("swagger_url")

            if not swagger_url:
                endpoints = svc.get("endpoints", {})
                for ep_url in endpoints.values():
                    if isinstance(ep_url, str) and "/rest/" in ep_url:
                        swagger_url = self._build_swagger_url_from_endpoint(ep_url, mock_server_url)
                        break

            if not swagger_url:
                print(f"[ENRICH] Cannot find or reconstruct swagger URL for {doc_id}")
                skipped += 1
                continue

            # extracts response_schemas and request_schemas from the YAML
            schemas = self._extract_schemas_from_yaml(swagger_url)

            try:
                # updates the MongoDB document with the extracted schemas
                # PATCH instead of PUT: adds fields without overwriting the rest
                patch_resp = requests.patch(
                    f"{gateway_base}/services/{doc_id}/schemas",
                    json=schemas,
                    timeout=10
                )
                patch_resp.raise_for_status()
                n_resp   = len(schemas.get("response_schemas", {}))
                n_req    = len(schemas.get("request_schemas", {}))
                n_params = len(schemas.get("parameters", {}))
                has_desc = bool(schemas.get("generated_description"))
                print(f"[ENRICH] {doc_id} -> {n_resp} response, {n_req} request, {n_params} parameters, generated_description={'yes' if has_desc else 'no'}")
                enriched += 1
            except Exception as e:
                print(f"[ENRICH] Failed to update {doc_id}: {e}")
                errors += 1

        return {"enriched": enriched, "skipped": skipped, "errors": errors}

    # -----------------------------------------------------------------------
    # Original registration methods
    # Used by import_apis() for external services from apis.guru
    # parse_swagger is similar to _extract_schemas_from_yaml but also includes
    # capabilities, endpoints and swagger_url in the return payload
    # -----------------------------------------------------------------------

    def parse_swagger(self, service, swagger_url, fallback_base_url=None):
        """
        Parses an OpenAPI YAML and builds the complete document
        to save in MongoDB for a service.

        Unlike _extract_schemas_from_yaml (which only extracts schemas),
        this method also extracts:
          - capabilities: textual endpoint descriptions (used by Qdrant)
          - endpoints: full endpoint URLs
          - swagger_url: YAML URL (for future re-enrichment)
        """
        try:
            text = self._read_spec_text(swagger_url)
            parser = prance.ResolvingParser(
                spec_string=text,
                lazy=False,
                strict=False,
            )
            swagger = parser.specification
        except Exception as e:
            print(f"Failed to load/resolve YAML from {swagger_url}: {e}")
            return None

        try:
            info         = swagger.get("info", {})
            service_name = info.get("title", service)
            paths        = swagger.get("paths", {})
            servers      = swagger.get("servers")

            # --- ORIGINAL LOGIC ---
            if servers and isinstance(servers, list) and len(servers) > 0 and isinstance(servers[0], dict):
                host_url = servers[0].get("url", "http://localhost")
            else:
                host      = swagger.get("host", "localhost")
                schemes   = swagger.get("schemes", ["http"])
                base_path = swagger.get("basePath", "")
                scheme    = schemes[0] if isinstance(schemes, list) and schemes else "http"
                host_url  = f"{scheme}://{host}{base_path}"

            # --- NEW FALLBACK LOGIC ---
            # If after standard checks we are still stuck on localhost...
            if host_url.startswith("http://localhost"):
                # 1. Priority to the manual parameter (if provided)
                if fallback_base_url:
                    host_url = fallback_base_url.rstrip('/')
                # 2. Otherwise deduce it from the download URL (if it is a remote link)
                elif swagger_url.startswith("http"):
                    parsed = urlparse(swagger_url)
                    host_url = f"{parsed.scheme}://{parsed.netloc}"

        except Exception as e:
            print(f"Failed to get information from parsed swagger: {e}")
            return None

        capabilities     = {}
        endpoints        = {}
        response_schemas = {}
        request_schemas  = {}
        parameters       = {} # <--- NEW DICTIONARY

        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, details in methods.items():
                if method.lower() not in {"get", "post", "put", "delete"}:
                    continue
                if not isinstance(details, dict):
                    continue

                key      = f"{method.upper()} {path}"
                desc     = details.get("description") or details.get("summary") or details.get("operationId") or method
                full_url = f"{host_url.rstrip('/')}{path}"

                capabilities[key] = desc
                endpoints[key]    = full_url

                resp_schema = self._extract_schema_from_details(details)
                if resp_schema:
                    response_schemas[key] = resp_schema

                req_schema = self._extract_request_schema_from_details(details, method)
                if req_schema:
                    request_schemas[key] = req_schema

                # --- NEW CALL FOR PARAMETERS ---
                params_schema = self._extract_parameters_from_details(details)
                if params_schema:
                    parameters[key] = params_schema

        # returns the complete document ready for MongoDB
        return {
            "id":                    service,
            "name":                  service_name,
            "description":           swagger.get("info", {}).get("description", "No description"),
            "generated_description": self._generate_description(swagger),
            "swagger_url":           swagger_url,
            "capabilities":          capabilities,
            "endpoints":             endpoints,
            "response_schemas":      response_schemas,
            "request_schemas":       request_schemas,
            "parameters":            parameters,
        }

    def register_to_redis(self, service_name, service_status):
        # registers the service health status in Redis
        # used by the healthcheck service to know if the service is active
        headers      = {"Content-Type": "application/json"}
        json_payload = {"key": service_name, "value": service_status}
        response = requests.post(
            f"http://{self.HEALTHCHECK_SERVICE_HOST}:{self.HEALTHCHECK_SERVICE_PORT}/status/register",
            headers=headers,
            data=json.dumps(json_payload)
        )
        if response.status_code == 200:
            print(f"Registered {service_name} to Redis with status {service_status}")
        else:
            print(f"Failed to register {service_name} to Redis: HTTP {response.status_code}")

    def register_to_consul(self, service_id, service_name):
        # registers the service in Consul with an automatic health check
        # Consul uses this to know if the service is reachable
        # and to exclude it from the catalog if unresponsive
        payload = {
            "Name": service_name,
            "Id":   service_id,
            "Meta": {"service_doc_id": service_id},
            "Check": {
                "TlsSkipVerify":                 True,
                "Method":                         "GET",
                "Http":                           f"http://{self.HEALTHCHECK_SERVICE_HOST}:{self.HEALTHCHECK_SERVICE_PORT}/status/{service_id}",
                "Interval":                       "10s",   # checks every 10 seconds
                "Timeout":                        "5s",
                "DeregisterCriticalServiceAfter": "30s"    # removes if unresponsive for 30s
            }
        }
        try:
            url      = f"http://{self.CONSUL_HOST}:{self.CONSUL_PORT}/v1/agent/service/register"
            response = requests.put(url, json=payload)
            print(f"Consul registration ({service_id}): HTTP {response.status_code}")
        except Exception as e:
            print(f"Failed to register to Consul: {e}")

    def _verify_service_in_mongo(self, service_id, retries=3, delay_sec=1.0):
        verify_url = f"http://{self.GATEWAY_HOST}:{self.GATEWAY_PORT}/services/{service_id}"
        for attempt in range(1, retries + 1):
            try:
                r = requests.get(verify_url, timeout=10)
                if r.status_code == 200:
                    return True
                print(f"[MONGO][VERIFY][attempt {attempt}/{retries}] HTTP {r.status_code} for id={service_id}")
            except Exception as e:
                print(f"[MONGO][VERIFY][attempt {attempt}/{retries}] error for id={service_id}: {e}")
            time.sleep(delay_sec)
        return False

    def register_to_mongo(self, catalog_payload):
        service_id = catalog_payload.get("id", "UNKNOWN")
        post_url = f"http://{self.GATEWAY_HOST}:{self.GATEWAY_PORT}/service"

        try:
            print(f"[MONGO][ATTEMPT] POST {post_url} id={service_id}")
            response = requests.post(post_url, json=catalog_payload, timeout=15)
            response.raise_for_status()
            print(f"[MONGO][HTTP_OK] id={service_id} HTTP {response.status_code}")

            if service_id == "UNKNOWN":
                raise RuntimeError("[MONGO][INCONSISTENT] Missing 'id' in payload, cannot verify persistence")

            if not self._verify_service_in_mongo(service_id):
                raise RuntimeError(f"[MONGO][INCONSISTENT] HTTP success but service not found after write id={service_id}")

            print(f"[MONGO][VERIFIED] id={service_id} persisted in Mongo")
            return True

        except requests.exceptions.RequestException as e:
            status = e.response.status_code if getattr(e, "response", None) is not None else "N/A"
            body = e.response.text if getattr(e, "response", None) is not None else ""
            raise RuntimeError(f"[MONGO][FAILED] id={service_id} HTTP={status} body={body}") from e

    def import_apis(self):
        """
        Imports all public services from apis.guru.
        For each provider: downloads the YAML, extracts capabilities/endpoints/schema,
        then registers in parallel to Redis, Consul and MongoDB.

        Used to enrich the catalog with real external services
        (not local Smart City mocks).
        """
        all_endpoints = {}
        providers     = self.fetch_providers()
        print(f"Found {len(providers)} providers.")
        for provider in providers:
            try:
                api_data    = self.fetch_api_details(provider)
                swagger_url = self.extract_swagger_url(api_data, provider)
                all_endpoints[provider] = swagger_url
            except Exception as e:
                print(f"Error processing {provider}: {e}")

        print(f"\nFound: {len(all_endpoints)} endpoints.")
        for service, swagger_url in all_endpoints.items():
            if not swagger_url:
                print(f"[SKIP] {service}: swagger URL not found.")
                continue

            openapi = self.parse_swagger(service, swagger_url)
            if not openapi:
                print(f"[SKIP] {service}: error parsing swagger.")
                continue

            # the three registrations run in parallel with ThreadPoolExecutor
            # then wait for all to finish before proceeding (BARRIER)
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(self.register_to_redis, service, "true"),
                    executor.submit(self.register_to_consul, service, openapi.get("id", service)),
                    executor.submit(self.register_to_mongo, openapi)
                ]

                # BARRIER: waits for all threads before moving to the next service
                for future in futures:
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[ERROR] Register function failed: {e}")
